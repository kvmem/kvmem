#!/usr/bin/env python3
"""Initialize pinned backend submodules and apply the repository's integration patches.

hipfire is an external checkout rather than a submodule: backends/hipfire/upstream.json
pins its repository, base commit and verified patch series, and the checkout lives
beside this repository (override with --hipfire-dir).
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HIPFIRE_DIR = ROOT / 'backends' / 'hipfire'


def git(directory, *args, check=True, env=None):
    return subprocess.run(['git', '-c', 'core.longpaths=true', '-c', 'core.autocrlf=false',
                           '-C', str(directory), *args], check=check,
                          capture_output=True, text=True, env=env)


@contextmanager
def temporary_index():
    handle, filename = tempfile.mkstemp(prefix='kvmem-backend-', suffix='.index')
    os.close(handle)
    os.unlink(filename)
    env = dict(os.environ, GIT_INDEX_FILE=filename)
    try:
        yield env
    finally:
        if Path(filename).exists():
            os.unlink(filename)


def source_tree(directory):
    # Include patch-added files without changing the developer's index.
    with temporary_index() as env:
        git(directory, 'read-tree', 'HEAD', env=env)
        git(directory, 'add', '-A', env=env)
        return git(directory, 'write-tree', env=env).stdout.strip()


def patch_series(name, entry, base=None):
    # ROOT is looked up at call time so source-bundle tests can point it at a fixture tree.
    base = ROOT if base is None else base
    patches = entry.get('patches')
    if not isinstance(patches, list) or not patches:
        raise RuntimeError(f'{name}: expected a nonempty ordered patches list')
    seen = set()
    paths = set()
    for item in patches:
        identity = item['id']
        if not isinstance(identity, str) or not identity or identity in seen:
            raise RuntimeError(f'{name}: duplicate or invalid patch id: {identity}')
        dependencies = item.get('requires', [])
        if not isinstance(dependencies, list) or any(dep not in seen for dep in dependencies):
            raise RuntimeError(f'{name}/{identity}: dependencies must precede the patch')
        relative = Path(item['path'])
        patch = (base / relative).resolve()
        if relative.is_absolute() or not patch.is_relative_to(ROOT.resolve()) or patch in paths:
            raise RuntimeError(f'{name}/{identity}: invalid or duplicate patch path')
        if hashlib.sha256(patch.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError(f'{name}/{identity}: patch hash differs from the recorded version')
        if not isinstance(item['after_tree'], str) or len(item['after_tree']) != 40:
            raise RuntimeError(f'{name}/{identity}: invalid prepared tree hash')
        seen.add(identity)
        paths.add(patch)
    if patches[-1]['after_tree'] != entry['patched_tree']:
        raise RuntimeError(f'{name}: final patch tree differs from patched_tree')
    return patches


def verify_patch_series(directory, entry, patches, base=None):
    base = ROOT if base is None else base
    # Preflight the entire series in an isolated index before changing any source.
    with temporary_index() as env:
        git(directory, 'read-tree', entry['base_commit'], env=env)
        for item in patches:
            git(directory, 'apply', '--cached', '--whitespace=nowarn',
                str(base / item['path']), env=env)
            actual = git(directory, 'write-tree', env=env).stdout.strip()
            if actual != item['after_tree']:
                raise RuntimeError(f'{item["id"]}: patch produces {actual}, '
                                   f'expected {item["after_tree"]}')


def prepare(name, entry, check_only):
    directory = ROOT / entry['path']
    patches = patch_series(name, entry)
    # Exported source bundles contain prepared sources and a per-file manifest, without Git metadata.
    if not (ROOT / '.git').exists():
        manifest = json.loads((ROOT / 'SOURCE-MANIFEST.json').read_text(encoding='utf-8'))
        if manifest['backends'][name] != entry:
            raise RuntimeError(f'{name}: source bundle backend version differs from versions.json')
        files = {path: value for path, value in manifest['files'].items()
                 if path.startswith(entry['path'] + '/')}
        if not files:
            raise RuntimeError(f'{name}: source bundle has no backend files')
        for path, expected in files.items():
            source = (ROOT / path).resolve()
            if not source.is_relative_to(directory.resolve()):
                raise RuntimeError(f'{name}: invalid source manifest path: {path}')
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'{name}: source bundle file differs from manifest: {path}')
        print(f'{name}: verified prepared source bundle', flush=True)
        return
    if not check_only and not (directory / '.git').exists():
        git(ROOT, 'submodule', 'update', '--init', '--recursive', '--depth', '1', '--', entry['path'])
    if not (directory / '.git').exists():
        raise RuntimeError(f'{name}: run scripts/prepare-backends.py --backend {name}')
    finish_prepare(name, directory, entry, patches, check_only)


def finish_prepare(name, directory, entry, patches, check_only, base=None):
    base = ROOT if base is None else base
    head = git(directory, 'rev-parse', 'HEAD').stdout.strip()
    if head != entry['base_commit']:
        raise RuntimeError(f'{name}: expected {entry["base_commit"]}, got {head}')
    verify_patch_series(directory, entry, patches, base)
    current = source_tree(directory)
    states = [git(directory, 'rev-parse', 'HEAD^{tree}').stdout.strip()]
    states.extend(item['after_tree'] for item in patches)
    if current not in states:
        raise RuntimeError(f'{name}: source has changes outside the recorded patch series; '
                           'preserving the working directory')
    if current != entry['patched_tree']:
        if check_only:
            raise RuntimeError(f'{name}: patch series is incomplete')
        completed = max(i for i, tree in enumerate(states) if tree == current)
        for item in patches[completed:]:
            if source_tree(directory) != current:
                raise RuntimeError(f'{name}: source changed during preparation; stopping')
            git(directory, 'apply', '--check', str(base / item['path']))
            git(directory, 'apply', '--whitespace=nowarn', str(base / item['path']))
            current = source_tree(directory)
            if current != item['after_tree']:
                raise RuntimeError(f'{name}/{item["id"]}: prepared source differs from expected tree')
    print(f'{name}: {head[:12]} + {len(patches)} patches ready', flush=True)


def backend_entry(versions, name):
    entry = versions.get(name)
    if not isinstance(entry, dict):
        raise RuntimeError(f'{name}: missing backend entry')
    manifest_value = entry.get('manifest')
    if not manifest_value:
        return entry
    manifest_rel = Path(manifest_value)
    if manifest_rel.is_absolute():
        raise RuntimeError(f'{name}: manifest path must be relative: {manifest_value}')
    manifest = (ROOT / manifest_rel).resolve()
    if not manifest.is_relative_to(ROOT.resolve()):
        raise RuntimeError(f'{name}: manifest path escapes the repository: {manifest}')
    try:
        data = json.loads(manifest.read_text(encoding='utf-8'))
    except OSError as error:
        raise RuntimeError(f'{name}: cannot read manifest {manifest}: {error}') from error
    if not isinstance(data, dict):
        raise RuntimeError(f'{name}: manifest is not an object: {manifest}')
    if entry.get('layout') != data.get('layout'):
        raise RuntimeError(
            f'{name}: versions.json layout {entry.get("layout")!r} differs from '
            f'{manifest} layout {data.get("layout")!r}')
    for key in ('repository', 'base_commit', 'patched_tree', 'patches'):
        if key not in data:
            raise RuntimeError(f'{name}: {manifest} missing {key}')
    return data


def hipfire_entry(versions=None):
    if versions is None:
        versions = json.loads((ROOT / 'backends/versions.json').read_text(encoding='utf-8'))
    if 'hipfire' in versions:
        return backend_entry(versions, 'hipfire')
    # Compatibility for one release: an unindexed upstream.json still prepares.
    entry = json.loads((HIPFIRE_DIR / 'upstream.json').read_text(encoding='utf-8'))
    if entry.get('layout') != 'external-checkout-with-patches':
        raise RuntimeError('hipfire: unsupported upstream.json layout')
    for key in ('repository', 'base_commit', 'patched_tree'):
        if not isinstance(entry.get(key), str) or not entry[key]:
            raise RuntimeError(f'hipfire: upstream.json missing {key}')
    return entry


def hipfire_manifest_dir(versions):
    entry = versions.get('hipfire')
    if isinstance(entry, dict) and entry.get('manifest'):
        return (ROOT / Path(entry['manifest'])).resolve().parent
    return HIPFIRE_DIR


def hipfire_checkout(versions, override):
    if override is not None:
        return override
    entry = versions.get('hipfire')
    if isinstance(entry, dict) and entry.get('checkout'):
        checkout = Path(entry['checkout'])
        return checkout if checkout.is_absolute() else (ROOT / checkout).resolve()
    return ROOT.parent / 'hipfire'


def hipfire_ready(entry, directory):
    if not (directory / '.git').exists():
        return False, f'{directory} is not a git checkout'
    head = git(directory, 'rev-parse', 'HEAD', check=False)
    if head.returncode != 0:
        return False, f'{directory} has no readable HEAD'
    if head.stdout.strip() != entry['base_commit']:
        return False, f'{directory} HEAD is not the pinned base commit'
    try:
        tree = source_tree(directory)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        return False, f'{directory} tree cannot be read: {error}'
    if tree != entry['patched_tree']:
        return False, f'{directory} working tree is not the recorded patched tree'
    return True, ''


def normalized_remote(url):
    return url.rstrip('/').removesuffix('.git')


def prepare_hipfire(entry, directory, check_only, base=None):
    base = HIPFIRE_DIR if base is None else base
    patches = patch_series('hipfire', entry, base)
    if not (directory / '.git').exists():
        if check_only:
            raise RuntimeError(f'hipfire: {directory} is missing; '
                               'run scripts/prepare-backends.py --backend hipfire')
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f'hipfire: {directory} exists but is not a git checkout; '
                               'preserving it')
        directory.mkdir(parents=True, exist_ok=True)
        git(directory, 'init', '--quiet')
        git(directory, 'remote', 'add', 'origin', entry['repository'])
        fetch = git(directory, 'fetch', '--depth', '1', 'origin', entry['base_commit'], check=False)
        if fetch.returncode != 0:
            # Some servers refuse direct commit fetches; fall back to full history.
            git(directory, 'fetch', '--quiet', 'origin')
        git(directory, 'checkout', '--quiet', '--detach', entry['base_commit'])
    remote = git(directory, 'remote', 'get-url', 'origin').stdout.strip()
    if normalized_remote(remote) != normalized_remote(entry['repository']):
        raise RuntimeError(f'hipfire: {directory} has remote {remote}, '
                           f'expected {entry["repository"]}; preserving it')
    finish_prepare('hipfire', directory, entry, patches, check_only, base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('llamacpp', 'ninfer', 'polystrata', 'hipfire', 'all'), default='all')
    parser.add_argument('--check', action='store_true',
                        help='verify exact prepared sources without fetching or applying patches')
    parser.add_argument('--hipfire-dir', type=Path, default=None,
                        help='external hipfire checkout (default: hipfire beside this repository)')
    args = parser.parse_args()
    versions = json.loads((ROOT / 'backends/versions.json').read_text(encoding='utf-8'))
    hipfire_dir = hipfire_checkout(versions, args.hipfire_dir)
    try:
        if args.backend == 'hipfire':
            prepare_hipfire(hipfire_entry(versions), hipfire_dir, args.check,
                            hipfire_manifest_dir(versions))
            return
        names = tuple(name for name, entry in versions.items()
                      if isinstance(entry, dict) and 'path' in entry) if args.backend == 'all' else (args.backend,)
        for name in names:
            prepare(name, versions[name], args.check)
        if args.backend == 'all':
            if 'hipfire' not in versions:
                print('hipfire: skipped (not indexed)', flush=True)
            elif args.hipfire_dir is not None:
                prepare_hipfire(hipfire_entry(versions), hipfire_dir, args.check,
                                hipfire_manifest_dir(versions))
            else:
                entry = hipfire_entry(versions)
                ready, reason = hipfire_ready(entry, hipfire_dir)
                if ready:
                    prepare_hipfire(entry, hipfire_dir, args.check,
                                    hipfire_manifest_dir(versions))
                else:
                    print(f'hipfire: skipped ({reason})', flush=True)
    except (OSError, RuntimeError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        parser.exit(1, f'Backend preparation failed: {detail}\n')


if __name__ == '__main__':
    main()
