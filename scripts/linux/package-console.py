#!/usr/bin/env python3
"""Build a Linux x86_64 console-only folder and tarball from verified inputs.

Requires system Python 3.10+. Does not download or compile a native worker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(source, output, chat, wheel, expected, backend='ninfer'):
    if backend not in ('ninfer','radiance','all'):raise ValueError('Unsupported console backend')
    source, output, chat, wheel = (Path(p).resolve() for p in (source, output, chat, wheel))
    if output.exists() or Path(str(output) + '.tar.gz').exists():
        raise ValueError('Output already exists; choose a new versioned package path')
    if not re.fullmatch(r'psutil-7\.2\.2-cp3[0-9]+-abi3-manylinux[0-9_a-z.]*x86_64\.whl', wheel.name):
        raise ValueError('Supply the pinned psutil 7.2.2 manylinux x86_64 ABI3 wheel')
    if digest(wheel) != expected.lower():
        raise ValueError('psutil wheel SHA-256 mismatch')
    chat_info = json.loads((chat / 'ui-build.json').read_text(encoding='utf-8'))
    if chat_info.get('console_chat_protocol') != 1 or not (chat / 'index.html').is_file():
        raise ValueError('Compatible built unified chat assets are required')
    files = ['START-HERE-CN.sh', 'START-HERE-EN.sh', 'scripts/linux/start-ui.sh',
             'scripts/local-console.py', 'scripts/launcher-cli.py', 'scripts/console_chat.py', 'scripts/console_radiance.py',
             'scripts/console_hardware.py', 'scripts/console_memory.py', 'scripts/model_health.py', 'scripts/requirements-console.txt',
             'scripts/launcher-capabilities.json', 'scripts/speculative-capabilities.json',
             'docs/linux-launcher.md']
    if backend in ('radiance','all'):
        files.extend(('backends/radiance/tools/inspect_rad.py',
                      'backends/radiance/tools/draft_control.py'))
    for folder in ('scripts/launcher', 'scripts/model_health_ninfer', 'ui/monitor'):
        files.extend(str(p.relative_to(source)).replace('\\', '/') for p in (source / folder).iterdir()
                     if p.is_file() and p.suffix not in ('.pyc', '.map'))
    for name in files:
        if not (source / name).is_file(): raise ValueError('Missing package input: ' + name)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.console-package-', dir=output.parent))
    try:
        for name in files:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = (source / name).read_bytes()
            if target.suffix in ('.sh', '.py'):
                data = data.replace(b'\r\n', b'\n')
            target.write_bytes(data)
            target.chmod(0o755 if target.suffix == '.sh' else 0o644)
        shutil.copytree(chat, staging / 'ui/dist')
        runtime = staging / 'runtime/python-libs'
        runtime.mkdir(parents=True)
        with zipfile.ZipFile(wheel) as archive:
            for entry in archive.infolist():
                name = PurePosixPath(entry.filename)
                if (name.is_absolute() or '..' in name.parts or '\\' in entry.filename or ':' in entry.filename
                        or not name.parts or name.parts[0] not in ('psutil', 'psutil-7.2.2.dist-info')
                        or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError('Unexpected wheel member: ' + entry.filename)
            archive.extractall(runtime)
        (staging / 'START-HERE.txt').write_text(
            'Linux x86_64 console preview; system Python 3.10+ is required.\n'
            f'No worker or model is included. Select a compatible Linux worker for {backend}.\n'
            '中文: bash START-HERE-CN.sh\nEnglish: bash START-HERE-EN.sh\n'
            'Headless: bash scripts/linux/start-ui.sh --no-browser\n'
            'Check: bash scripts/linux/start-ui.sh --check\n'
            'Guide: docs/linux-launcher.md\n', encoding='utf-8', newline='\n')
        controller_files = {name: digest(staging / name) for name in files if name.startswith(('scripts/', 'ui/monitor/', 'backends/radiance/tools/'))}
        info = dict(schema_version=1, package_kind='console-only-preview', platform='linux-x86_64',
                    launch_contract_version=3, supported_backends=['ninfer','llamacpp','radiance'] if backend=='all' else [backend], python_minimum='3.10',
                    psutil_version='7.2.2', psutil_wheel=wheel.name, psutil_sha256=expected,
                    native_worker_included=False, native_boot_validated=False, controller_files=controller_files,
                    chat_manifest_sha256=digest(chat / 'ui-build.json'))
        (staging / 'console-build.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
        paths = sorted(p for p in staging.rglob('*') if p.is_file())
        (staging / 'SHA256SUMS').write_text(''.join(digest(p) + '  ' + p.relative_to(staging).as_posix() + '\n' for p in paths), encoding='utf-8', newline='\n')
        os.rename(staging, output)
    finally:
        if staging.exists(): shutil.rmtree(staging)
    archive_path = Path(str(output) + '.tar.gz')
    def portable_mode(member):
        member.uid = member.gid = 0; member.uname = member.gname = ''
        member.mode = 0o755 if member.isdir() or member.name.endswith('.sh') else 0o644
        return member
    with tarfile.open(archive_path, 'w:gz') as archive:
        archive.add(output, arcname=output.name, filter=portable_mode)
    return dict(directory=str(output), archive=str(archive_path), sha256=digest(archive_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chat-ui', type=Path, required=True)
    parser.add_argument('--psutil-wheel', type=Path, required=True)
    parser.add_argument('--psutil-sha256', required=True)
    parser.add_argument('--backend', choices=('ninfer','radiance','all'), default='ninfer')
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.output, args.chat_ui, args.psutil_wheel, args.psutil_sha256,args.backend), indent=2))
