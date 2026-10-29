#!/usr/bin/env python3
"""Package a chat-only overlay for one verified Windows release archive."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

CHAT = 'workers/share/kvmem/ui/'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stream_digest(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(chunk)
    return result.hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def safe_name(name):
    if (not name or '\\' in name or ':' in name or name.startswith('/') or
            any(p in ('', '.', '..') for p in name.split('/'))):
        raise ValueError('Unsafe package path: ' + name)
    return name


def chat_assets(directory):
    directory = directory.resolve()
    assets = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError('Chat assets must not contain links')
        if not path.is_file():
            continue
        name = safe_name(path.relative_to(directory).as_posix())
        if name.startswith('monitor/') or name == 'monitor-build.json':
            continue  # The release's independent monitor is not part of this update.
        assets[name] = path.read_bytes()
    if not {'index.html', 'ui-build.json'}.issubset(assets):
        raise ValueError('Built chat index and manifest are required')
    # The bundled Vite build references hashed assets by literal filename.
    # Follow JS/CSS references transitively, leaving retained historical builds
    # out of the download. Existing files on the destination are never deleted.
    candidates = {n for n in assets if n.startswith('_app/') and n != '_app/version.json'}
    pending = [assets['index.html'].decode('utf-8')]
    used = set()
    while pending:
        text = pending.pop()
        for name in candidates - used:
            if PurePosixPath(name).name in text:
                used.add(name)
                if name.endswith(('.js', '.css', '.json')):
                    pending.append(assets[name].decode('utf-8'))
    if not any(n.endswith('.js') for n in used):
        raise ValueError('Chat index does not reference a bundled script')
    return {n: data for n, data in assets.items() if n not in candidates or n in used}


def package(base_archive, base_sha256, chat_ui, output):
    base_archive, chat_ui, output = map(Path, (base_archive, chat_ui, output))
    checksum_output = output.with_suffix(output.suffix + '.sha256')
    if output.exists() or checksum_output.exists():
        raise ValueError('Output already exists')
    with base_archive.open('rb') as stream:
        actual = stream_digest(stream)
    if actual != base_sha256.lower():
        raise ValueError('Base archive SHA-256 mismatch')
    assets = chat_assets(chat_ui)
    after = json.loads(assets['ui-build.json'].decode('utf-8-sig'))
    with zipfile.ZipFile(base_archive) as source:
        entries, folded = {}, set()
        roots = {i.filename.split('/')[0] for i in source.infolist()}
        if len(roots) != 1:
            raise ValueError('Base archive must contain one installation root')
        prefix = roots.pop() + '/'
        for entry in source.infolist():
            safe_name(entry.filename.rstrip('/'))
            if entry.is_dir():
                continue
            name = safe_name(entry.filename.removeprefix(prefix))
            if name.casefold() in folded:
                raise ValueError('Duplicate package path: ' + name)
            folded.add(name.casefold())
            entries[name] = entry
        checksums = {}
        for line in source.read(entries['SHA256SUMS']).decode('utf-8-sig').splitlines():
            expected, name = line.split('  ', 1)
            safe_name(name)
            if name in checksums:
                raise ValueError('Duplicate checksum: ' + name)
            with source.open(entries[name]) as stream:
                if stream_digest(stream) != expected:
                    raise ValueError('Base file checksum mismatch: ' + name)
            checksums[name] = expected
        manifest = json.loads(source.read(entries['console-build.json']))
        before = json.loads(source.read(entries[CHAT + 'ui-build.json']))
        for name, expected in manifest['controller_files'].items():
            if checksums.get(name) != expected:
                raise ValueError('Controller checksum mismatch: ' + name)
        if manifest['chat_manifest_sha256'] != checksums.get(CHAT + 'ui-build.json'):
            raise ValueError('Base chat manifest checksum mismatch')
        if any(v.get('console_chat_protocol') != 1 for v in (manifest, before, after)):
            raise ValueError('Unsupported chat protocol')
        def api_hash(value):
            return value.get('source_entry_files', value.get('entry_files', {})).get('api.ts')
        if not api_hash(before) or api_hash(before) != api_hash(after):
            raise ValueError('Chat API source changed; validate the new contract separately')

    updates = {CHAT + name: data for name, data in assets.items()}
    manifest['chat_manifest_sha256'] = digest(assets['ui-build.json'])
    manifest['chat_assets_update'] = {
        'kind': 'chat-only-overlay', 'base_archive': base_archive.name,
        'base_archive_sha256': actual,
    }
    updates['console-build.json'] = encode(manifest)
    updates['CHAT-UPDATE.txt'] = (f'''KVMem chat-only update / 聊天界面更新

Applies only to / 仅适用原包: {base_archive.name}
Base archive SHA-256: {actual}

Close the original launcher and chat page. Back up workers/share/kvmem/ui,
console-build.json and SHA256SUMS. Extract this update into the original
installation root, beside START-HERE-CN.bat, allowing file replacement.
Restart with the original launcher and refresh the browser with Ctrl+F5.

关闭原启动器和聊天页，备份 workers/share/kvmem/ui、console-build.json、SHA256SUMS。
解压到原 START-HERE-CN.bat 所在目录并覆盖同名文件，重新启动后 Ctrl+F5 刷新。
沿用原浏览器和控制台地址，已有聊天数据库保留。

Only chat assets and their package manifests are updated. Launch scripts,
monitor, inference engines, DLLs, Python, models and user configuration stay
unchanged. Other release variants require their own matching overlay.
仅更新聊天资源和配套清单；启动器、监控、引擎、运行库、模型和配置保持原样。
其他架构/版本的包应单独生成匹配的更新包。
''').encode('utf-8-sig')
    for name, data in updates.items():
        checksums[name] = digest(data)
    updates['SHA256SUMS'] = ''.join(h + '  ' + n + '\n' for n, h in sorted(checksums.items())).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    owned_output = output.open('xb')
    try:
        with owned_output:
            with zipfile.ZipFile(owned_output, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
                for name, data in sorted(updates.items()):
                    info = zipfile.ZipInfo(name)  # Stable timestamps for reproducible output.
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, data, compresslevel=9)
        with zipfile.ZipFile(output) as archive:
            if archive.testzip() is not None or any(archive.read(n) != data for n, data in updates.items()):
                raise ValueError('Output archive verification failed')
        output_sha = digest(output.read_bytes())
        with checksum_output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(output_sha + '  ' + output.name + '\n')
    except Exception:
        # The output did not exist at entry and is owned by this invocation.
        output.unlink(missing_ok=True)
        raise
    return {'archive': str(output), 'bytes': output.stat().st_size, 'sha256': output_sha,
            'base_archive_sha256': actual, 'chat_files': len(assets),
            'controller_files_unchanged': len(manifest['controller_files'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-archive', type=Path, required=True)
    parser.add_argument('--base-sha256', required=True, help='SHA-256 from the release publisher')
    parser.add_argument('--chat-ui', type=Path, required=True, help='Validated built lightweight UI')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.base_archive, args.base_sha256, args.chat_ui, args.output), indent=2))


if __name__ == '__main__':
    main()
