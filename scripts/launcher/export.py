"""Versioned Linux scripts generated from the same plan shown in the console."""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
import uuid

from .config import without_secrets
from .plan import build_plan
from .worker import probe_worker, require_launch_features
from .systemd import profile_id as validate_id, render_unit, unit_name


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def _write(path, data, mode=0o600):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        os.chmod(path, mode)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _generation_lock(directory):
    import fcntl
    flags = os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(directory / '.generate.lock', flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('This profile is already being generated') from error
        yield
    finally:
        os.close(descriptor)


def render_script(plan, lock_path, secret_ref=None):
    if plan['target_platform'] != 'linux':
        raise ValueError('Shell exports require a Linux launch plan')
    args = list(plan['argv'])
    secret_index = args.index('--api-key') + 1 if '--api-key' in args else None
    if secret_index is not None and not secret_ref:
        raise ValueError('An API key needs a runtime credential reference')
    lines = ['#!/bin/bash', '# Generated KVMem launch script. Regenerate to apply UI changes.',
             'set -euo pipefail', 'umask 077',
             'fail() { printf "%s\\n" "$1" >&2; exit 78; }',
             'if (( $# != 0 )); then fail "This script takes no arguments; regenerate the profile to change settings."; fi',
             'cd -- ' + shlex.quote(plan['cwd']) + ' || fail "Package directory is unavailable."']
    for name, value in plan['environment'].items():
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*', name):
            raise ValueError('Invalid launch environment name')
        lines.append('export ' + name + '=' + shlex.quote(value))
    worker = shlex.quote(args[0])
    lines += [f'[[ -f {worker} && -x {worker} ]] || fail "Worker is missing or not executable."',
              '[[ -x /usr/bin/flock ]] || fail "Install util-linux (flock) before running this script."']
    digest = plan.get('worker', {}).get('executable_sha256')
    if digest:
        if not re.fullmatch('[a-f0-9]{64}', digest):
            raise ValueError('Invalid worker SHA-256')
        lines += ['kvmem_worker_hash=$(/usr/bin/sha256sum < ' + worker + ') || fail "Cannot verify worker."',
                  '[[ ${kvmem_worker_hash:0:64} == ' + digest + ' ]] || fail "Worker changed; regenerate this version."']
    for key in ('Model', 'DraftModel', 'ChatTemplateFile', 'Mmproj', 'Lora'):
        value = plan['effective_parameters'].get(key)
        if value:
            lines.append('[[ -f ' + shlex.quote(value) + ' && -r ' + shlex.quote(value)
                         + ' ]] || fail ' + shlex.quote(key + ' is missing or unreadable.'))
    if secret_index is not None:
        if secret_ref['kind'] == 'file':
            path = shlex.quote(secret_ref['path'])
            lines += [f'[[ -f {path} && -r {path} ]] || fail "API key file is missing or unreadable."',
                      f'kvmem_api_key=$(cat -- {path})']
        elif secret_ref['kind'] == 'environment':
            lines.append('kvmem_api_key=${KVMEM_API_KEY:-}')
        else:
            raise ValueError('Unsupported credential reference')
        lines += ['[[ -n "$kvmem_api_key" ]] || fail "Provide the API key before starting this service."',
                  '[[ ${#kvmem_api_key} -le 512 && "$kvmem_api_key" =~ ^[A-Za-z0-9._~+/-]+=*$ ]] || fail "API key must be one ASCII token."']
    lines.append('kvmem_command=(')
    for index, arg in enumerate(args):
        lines.append('  ' + ('"${kvmem_api_key}"' if index == secret_index else shlex.quote(arg)))
    lines += [')', '# Replace this shell without backgrounding; the supervisor owns the process.',
              'exec /usr/bin/flock --no-fork --exclusive --nonblock --conflict-exit-code 75 '
              + shlex.quote(str(lock_path)) + ' "${kvmem_command[@]}"', '']
    return '\n'.join(lines)


def export_profile(data, backend_root, profiles_root, profile_id, *,
                   secret_file=None, persist_secret=False, scope='user', user=None, _validated_plan=None):
    if os.name == 'nt':
        raise ValueError('Generate Linux scripts on the target Linux machine')
    validate_id(profile_id)
    profiles_root = Path(profiles_root).resolve()
    profiles_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory = profiles_root / profile_id
    if directory.is_symlink():
        raise ValueError('A profile directory must not be a symlink')
    directory.mkdir(exist_ok=True, mode=0o700)
    revisions = directory / 'revisions'
    if revisions.is_symlink():
        raise ValueError('A revisions directory must not be a symlink')
    revisions.mkdir(exist_ok=True, mode=0o700)
    data = copy.deepcopy(data)
    if secret_file:
        secret_file = Path(secret_file).resolve()
        if not secret_file.is_file() or not os.access(secret_file, os.R_OK):
            raise ValueError('API key file is missing or unreadable')
        data['parameters']['ApiKey'] = data['parameters'].get('ApiKey') or 'credential-reference'
    plan = copy.deepcopy(_validated_plan) if _validated_plan is not None else build_plan(data, backend_root, platform='linux')
    if plan.get('target_platform') != 'linux':
        raise ValueError('A Linux plan is required')
    if _validated_plan is None:
        plan['worker'] = probe_worker(plan['argv'][0], plan['backend'], plan['environment'])
        require_launch_features(plan['worker'], data['parameters'])
    revision = uuid.uuid4().hex
    target = revisions / revision
    secret_ref = None
    if plan['api_key_configured']:
        secret_ref = dict(kind='file', path=str(secret_file)) if secret_file else dict(kind='environment', name='KVMEM_API_KEY')
    if persist_secret and secret_file:
        raise ValueError('Choose a supplied key file or a stored key, not both')
    if persist_secret and not data['parameters'].get('ApiKey'):
        raise ValueError('No API key was supplied to store')
    with _generation_lock(directory):
        staging = Path(tempfile.mkdtemp(prefix='.generating-', dir=revisions))
        stored_secret = None
        try:
            if persist_secret:
                credentials = directory / 'credentials'
                if credentials.is_symlink():
                    raise ValueError('Credentials directory must not be a symlink')
                credentials.mkdir(exist_ok=True, mode=0o700)
                stored_secret = credentials / (revision + '.key')
                _write(stored_secret, data['parameters']['ApiKey'] + '\n')
                secret_ref = dict(kind='file', path=str(stored_secret))
            plan['secret_ref'] = secret_ref
            plan.update(profile_id=profile_id, revision=revision)
            files = {'profile.json': _json(without_secrets(data)), 'launch-plan.json': _json(plan),
                     'run.sh': render_script(plan, directory / 'run.lock', secret_ref),
                     unit_name(profile_id): render_unit(plan, target / 'run.sh', scope=scope, user=user)}
            manifest = dict(schema_version=1, profile_id=profile_id, revision=revision,
                            scope=scope, user=user, files={})
            for name, body in files.items():
                _write(staging / name, body, 0o700 if name == 'run.sh' else 0o600)
                manifest['files'][name] = hashlib.sha256(body.encode('utf-8')).hexdigest()
            _write(staging / 'manifest.json', _json(manifest))
            os.rename(staging, target)
        except BaseException:
            if stored_secret is not None and stored_secret.is_file():
                stored_secret.unlink()
            raise
        finally:
            if staging.exists() and staging.parent == revisions:
                shutil.rmtree(staging)
    return dict(profile_id=profile_id, revision=revision, directory=str(target),
                script=str(target / 'run.sh'), unit=str(target / unit_name(profile_id)),
                scope=scope,
                install_commands=(['sudo install -m 644 -- ' + shlex.quote(str(target / unit_name(profile_id)))
                                   + ' ' + shlex.quote('/etc/systemd/system/' + unit_name(profile_id)),
                                   'sudo systemctl daemon-reload'] if scope == 'system' else []),
                requires_api_key_environment=bool(secret_ref and secret_ref['kind'] == 'environment'))


def read_revision(profiles_root, profile_id, revision):
    validate_id(profile_id)
    if not isinstance(revision, str) or not re.fullmatch(r'[a-f0-9]{32}', revision):
        raise ValueError('Invalid profile revision')
    profiles_root = Path(profiles_root).resolve()
    directory = profiles_root / profile_id / 'revisions' / revision
    if any(path.is_symlink() for path in (directory.parent.parent, directory.parent, directory)):
        raise ValueError('A generated revision must not be a symlink')
    if not directory.resolve().is_relative_to(profiles_root):
        raise ValueError('Profile revision escapes the configured directory')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    expected = {'profile.json', 'launch-plan.json', 'run.sh', unit_name(profile_id)}
    if (manifest.get('schema_version') != 1 or manifest.get('profile_id') != profile_id
            or manifest.get('revision') != revision or set(manifest.get('files', {})) != expected):
        raise ValueError('Invalid generated manifest')
    for name, digest in manifest['files'].items():
        path = directory / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Generated file was modified: ' + name + '; keep it as a custom script or generate a new revision')
    plan = json.loads((directory / 'launch-plan.json').read_text(encoding='utf-8'))
    if plan.get('profile_id') != profile_id or plan.get('revision') != revision:
        raise ValueError('Generated plan identity mismatch')
    return directory, manifest, plan
