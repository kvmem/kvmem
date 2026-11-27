"""Compose verified NInfer and PolyStrata release ZIPs without native rebuilds.

The output directory must be new. Run the package acceptance checks before
archiving or distributing it. Input workers remain byte-identical.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile


CUDA_DLLS = ('cublas64_13.dll', 'cublasLt64_13.dll')
METADATA = {'BUILD-INFO.json', 'console-build.json', 'SHA256SUMS',
            'README.md', 'RELEASE-NOTES.md', 'START-HERE.txt', 'VERSION'}
OVERLAYS = ('scripts/local-console.py', 'scripts/windows/native-process.ps1',
            'scripts/windows/start-ui.ps1', 'START-HERE-CN.bat', 'START-HERE-EN.bat',
            'scripts/console_chat.py', 'scripts/console_radiance.py', 'scripts/launcher/config.py', 'scripts/launcher/plan.py',
            'scripts/launcher/radiance.py', 'scripts/launcher/worker.py', 'scripts/windows/start-backend.ps1',
            'ui/monitor/launcher.js',
            'ui/monitor/monitor.js', 'ui/monitor/i18n.js')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class Package:
    def __init__(self, path, expected):
        self.path = Path(path).resolve()
        if sha(self.path) != expected.lower():
            raise ValueError('Input ZIP SHA-256 mismatch: ' + str(self.path))
        self.sha256 = expected.lower()
        self.zip = zipfile.ZipFile(self.path)
        self.files = {}
        roots = set()
        for entry in self.zip.infolist():
            parts = PurePosixPath(entry.filename).parts
            if (len(parts) < 2 or '..' in parts or entry.filename.startswith('/')
                    or '\\' in entry.filename or ':' in entry.filename):
                raise ValueError('Unsafe input archive path: ' + entry.filename)
            roots.add(parts[0])
            if entry.is_dir():
                continue
            name = '/'.join(parts[1:])
            if name.casefold() in {p.casefold() for p in self.files}:
                raise ValueError('Duplicate archive entry: ' + name)
            self.files[name] = entry
        if len(roots) != 1:
            raise ValueError('Input must have one package root')
        self.sums = {}
        for line in self.read('SHA256SUMS').decode('utf-8-sig').splitlines():
            digest, name = line.split('  ', 1)
            if name in self.sums or not re.fullmatch('[a-f0-9]{64}', digest):
                raise ValueError('Invalid input checksum manifest')
            self.sums[name] = digest
        if set(self.sums) != set(self.files) - {'SHA256SUMS'}:
            raise ValueError('Input manifest does not cover the whole package')
        for name, digest in self.sums.items():
            with self.zip.open(self.files[name]) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError('Input payload checksum mismatch: ' + name)
        self.console = json.loads(self.read('console-build.json'))
        self.version = self.read('VERSION').decode('utf-8-sig').strip()

    def read(self, name):
        return self.zip.read(self.files[name])

    def copy(self, name, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.zip.open(self.files[name]) as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8 << 20)


def prepare(args):
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+(?:[-.][A-Za-z0-9]+)*', args.version):
        raise ValueError('Invalid unified release version')
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise ValueError('Output directory must be new')
    source = Path(args.launcher_source).resolve()
    ninfer = Package(args.ninfer_zip, args.ninfer_sha256)
    poly = Package(args.polystrata_zip, args.polystrata_sha256)
    try:
        if ninfer.console['supported_backends'] != ['ninfer'] or poly.console['supported_backends'] != ['polystrata']:
            raise ValueError('Expected two qualified single-backend packages')
        for key in ('launch_contract_version', 'console_chat_protocol', 'chat_manifest_sha256'):
            if ninfer.console[key] != poly.console[key]:
                raise ValueError('Frontend contracts differ: ' + key)
        for name in CUDA_DLLS:
            if ninfer.sums['workers/ninfer/' + name] != poly.sums['workers/polystrata/' + name]:
                raise ValueError('CUDA runtime mismatch; cannot share ' + name)
        out.mkdir(parents=True)
        # Use the full PolyStrata Python runtime as one unit, never overlay it
        # with the embedded NInfer runtime's differently versioned DLLs/stdlib.
        for name in poly.files:
            if name in METADATA or name.startswith('provenance/'):
                target = out / 'provenance/components/polystrata' / name
            elif name in {'workers/polystrata/' + n for n in CUDA_DLLS}:
                target = out / 'runtime/cuda' / Path(name).name
            else:
                target = out / name
            poly.copy(name, target)
        for name in ninfer.files:
            if name in METADATA or name.startswith('provenance/'):
                target = out / 'provenance/components/ninfer' / name
            elif name.startswith('workers/ninfer/'):
                if Path(name).name in CUDA_DLLS:
                    continue
                target = out / name
            elif name.startswith('licenses/') and not (out / name).exists():
                target = out / name
            elif name == 'scripts/windows/start-ninfer.ps1':
                target = out / name
            else:
                continue
            ninfer.copy(name, target)
        for name in OVERLAYS:
            shutil.copyfile(source / name, out / name)
        # Keep the monitor embedded in Chat in sync with the console monitor.
        for name in ('launcher.js', 'monitor.js', 'i18n.js'):
            shutil.copyfile(out / 'ui/monitor' / name, out / 'ui/dist/monitor' / name)
        launcher_text = (out / 'ui/monitor/launcher.js').read_text(encoding='utf-8-sig')
        preset = re.search(r'const ninferDefaultPreset=\{([^}]+)\}', launcher_text)
        if not preset or "ThinkingMode:'on'" not in preset[1] or 'ThinkingBudget:0' not in preset[1]:
            raise ValueError('NInfer presets must explicitly enable thinking with no separate cap')
        (out / 'VERSION').write_text(args.version + '\n', encoding='utf-8')
        worker_paths = {'ninfer': 'workers/ninfer/ninfer-kvmem-server.exe',
                        'polystrata': 'workers/polystrata/strata-poly.exe'}
        components = {}
        for backend, package in [('ninfer', ninfer), ('polystrata', poly)]:
            worker = worker_paths[backend]
            if sha(out / worker) != package.sums[worker]:
                raise ValueError('Worker changed during composition')
            components[backend] = dict(version=package.version, worker=worker,
                worker_sha256=package.sums[worker], source_commit=package.console.get('source_commit'),
                source_archive=package.path.name, source_archive_sha256=package.sha256,
                prior_validation=f'provenance/components/{backend}/provenance/validation.json')
        info = dict(schema_version=1, version=args.version, backend='multi',
            supported_backends=['ninfer', 'polystrata'], components=components,
            target='windows-x64-nvidia-50-sm120a', models_included=False,
            kv_cross_backend_transfer=False, native_compiles=0, cuda_compiles=0,
            python_runtime='PolyStrata complete portable Python 3.12.13 runtime',
            shared_cuda={name: sha(out / 'runtime/cuda' / name) for name in CUDA_DLLS},
            launcher_overlays={name: sha(out / name) for name in OVERLAYS},
            validation='provenance/validation.json; component evidence is historical, not a full rerun')
        write_json(out / 'BUILD-INFO.json', info)
        controllers = set(poly.console['controller_files']) | {'scripts/windows/start-ninfer.ps1'}
        console = dict(schema_version=1, version=args.version, supported_backends=['ninfer', 'polystrata'],
            components={k: v['version'] for k, v in components.items()},
            launch_contract_version=poly.console['launch_contract_version'],
            console_chat_protocol=poly.console['console_chat_protocol'],
            chat_manifest_sha256=sha(out / 'ui/dist/ui-build.json'),
            offline_console=True, models_included=False, python_version='3.12.13',
            controller_files={name: sha(out / name) for name in sorted(controllers)})
        write_json(out / 'console-build.json', console)
        readme = f'''# KVMem {args.version} · Windows RTX 50 · 双后端统一版

同一个控制台选择 NInfer（稠密模型）或 PolyStrata（MoE 模型）。

1. 将完整 ZIP 解压到新目录。安装兼容 CUDA 13.2 的 NVIDIA 驱动和 Microsoft Visual C++ x64 运行库。
2. 双击 START-HERE-CN.bat，确认控制台 IP 和端口（回车保留默认；0.0.0.0 允许局域网访问），然后选择后端、本地模型和显卡，检查配置后启动。
3. NInfer 使用转换后的 .ninfer 模型；PolyStrata 使用 qwen35moe GGUF 模型，首次启动在本机导出 tokenizer。
4. 同一个控制台切换后端前，先停止当前模型服务，再选择另一后端并启动。

组件：NInfer {ninfer.version}；PolyStrata {poly.version}。两个引擎程序保持原发布包字节不变。
NInfer 极速档、质量档不设独立 thinking token 上限。PolyStrata 默认 B=36K、R=4K，thinking 不设固定上限。
模型、显卡能力和 KV 缓存按后端独立；合包不会扩大各后端支持的模型范围，也不跨后端转移 KV。
本包只适用于 Windows x64 / RTX 50（sm_120a）；PolyStrata 需要 AVX2 CPU，仅单 GPU。
Python、聊天界面及 CUDA 运行库离线随包提供；不含模型，无需安装 Python、Node.js 或 CUDA Toolkit。
两份完全相同的 cuBLAS 合为 runtime/cuda 中的一份，由控制台/启动脚本为子进程加载。
请使用随包启动器；直接运行 worker 时需自行将本包 runtime/cuda 加入该进程的 PATH。
普通启动脚本仍可使用 scripts/windows/start-backend.ps1 -Backend ninfer 或 -Backend polystrata。

## English

Extract into a new directory and run START-HERE-EN.bat. Confirm the console IP and
port (Enter keeps defaults; 0.0.0.0 allows LAN access). Select NInfer for supported
dense .ninfer models, or PolyStrata for qwen35moe GGUF models. Supply models separately.
Stop the current model service before changing backend. One console manages one
active backend at a time; switching unloads the model and its process-local KV cache.
The complete portable Python runtime, Chat and shared CUDA libraries are bundled.
Use the included launchers, which configure the private runtime path automatically.
Windows x64, RTX 50 (sm_120a), a CUDA 13.2-compatible driver and the Microsoft Visual
C++ x64 Runtime are required; PolyStrata requires AVX2 and supports a single GPU.
See BUILD-INFO.json for component identities and provenance/validation.json for
unified-package acceptance. Original component reports are preserved separately.
'''
        (out / 'README.md').write_text(readme, encoding='utf-8')
        (out / 'START-HERE.txt').write_text(readme, encoding='utf-8')
        write_json(out / 'provenance/composition.json', info)
        print(json.dumps(dict(package=str(out), components=console['components'],
                             native_compiles=0, cuda_compiles=0), ensure_ascii=True), flush=True)
    finally:
        ninfer.zip.close()
        poly.zip.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ninfer-zip', 'ninfer-sha256', 'polystrata-zip', 'polystrata-sha256',
                 'output-dir', 'version', 'launcher-source'):
        parser.add_argument('--' + name, required=True)
    prepare(parser.parse_args())
