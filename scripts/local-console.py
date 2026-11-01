"""Local, independently hosted configuration UI and owned inference-process controller.

Uses structured launch plans and owned process controllers. Windows standalone
launchers remain compatible; Linux runs an explicitly generated script revision.
"""
import argparse
import copy
import ctypes
import getpass
import hashlib
import ipaddress
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import subprocess
import sys
import tempfile
import time
import threading
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
sys.path.insert(0, str(Path(__file__).resolve().parent))
from console_chat import ChatBridge, MAX_BODY
from console_hardware import SystemDiskSampler
from console_memory import diagnose as diagnose_memory, validate_budget_plan
from model_health import request_parameters, unchecked

ROOT = Path(__file__).resolve().parents[1]
AUTO_BUDGET_SUPPORTED = os.name == 'nt'
# Product assets must not inherit OS MIME overrides (notably Windows .js=text/plain).
# Route allowlists still decide which files may be served.
CONTENT_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.txt': 'text/plain; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.ico': 'image/x-icon',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
}


def media_type(name):
    return CONTENT_TYPES.get(Path(name).suffix.lower(), 'application/octet-stream')


from launcher.config import (SPECULATIVE_PROFILES, DEFAULTS, ENGINE_DEFAULTS,
    STRINGS, INTEGERS, BOOLEANS, DECIMALS, NATIVE, NATIVE_SWITCHES,
    without_secrets, normalize)
from launcher.plan import build_plan, capabilities as launcher_capabilities, scoring_settings
from launcher.export import export_profile, read_revision
from launcher.worker import require_launch_features


def private_log(path, *, truncate=False):
    """Create or repair a private regular log without following links."""
    if os.name == 'nt':
        return Path(path).open('wb' if truncate else 'ab')
    flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if truncate:
        flags |= os.O_CREAT
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
                or (os.geteuid() != 0 and metadata.st_uid != os.geteuid())):
            raise ValueError('The console log must be a private regular file owned by this account')
        os.fchmod(descriptor, 0o600)
        if truncate:
            os.ftruncate(descriptor, 0)
        stream = os.fdopen(descriptor, 'wb')
        descriptor = None
        return stream
    finally:
        if descriptor is not None:
            os.close(descriptor)


def prepare_state_directory(path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != 'nt':
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(descriptor)
            if (path in (Path(path.anchor), Path.home().resolve()) or metadata.st_mode & stat.S_ISVTX
                    or (os.geteuid() != 0 and metadata.st_uid != os.geteuid())):
                raise ValueError('Use a dedicated state directory owned by this account')
            os.fchmod(descriptor, 0o700)
        finally:
            os.close(descriptor)
        log = path / 'service.log'
        if log.exists() or log.is_symlink():
            with private_log(log):
                pass


class WindowsJob:
    """Keep wrappers and their workers in an owned, kill-on-close process tree."""
    def __init__(self):
        class BASIC(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                        ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                        ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                        ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                        ('SchedulingClass', wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ('ReadOperationCount', 'WriteOperationCount',
                        'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]
        class EXTENDED(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', BASIC), ('IoInfo', IO),
                        ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                        ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {'CreateJobObjectW': ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
                      'SetInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
                      'AssignProcessToJobObject': ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
                      'TerminateJobObject': ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
                      'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL)}
        for name, (argtypes, restype) in signatures.items():
            fn = getattr(self.kernel, name); fn.argtypes = argtypes; fn.restype = restype
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = EXTENDED(); limits.BasicLimitInformation.LimitFlags = 0x2000
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close(); raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            process.kill(); process.wait()
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.TerminateJobObject(self.handle, 1)
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class Controller:
    def __init__(self, backend_root, state_dir, port, prefer_bundled_worker=False):
        self.backend_root = Path(backend_root).resolve()
        self.state_dir = Path(state_dir).resolve()
        prepare_state_directory(self.state_dir)
        self.prefer_bundled_worker = bool(prefer_bundled_worker)
        if os.name == 'nt':
            source = (self.backend_root / 'scripts/windows/start-backend.ps1').read_text(encoding='utf-8-sig')
            match = re.search(r"\[ValidateSet\(([^\n]+)\)\]\[string\]\$Backend", source)
            if not match:
                raise ValueError('Cannot detect supported backends from start-backend.ps1')
            self.backends = [name for name in re.findall(r"'([^']+)'", match[1]) if name in NATIVE]
            if ((self.backend_root/'scripts/launcher-capabilities.json').is_file() and
                    'radiance' in launcher_capabilities(self.backend_root)['backends']):
                self.backends.append('radiance')
            self.launcher_query_contract = all(name in source for name in ('QueryScorer', 'GraphAlpha', 'QueryMaxTokens', 'QueryReplayMaxTokens'))
        else:
            manifest = launcher_capabilities(self.backend_root)
            self.backends = [name for name in manifest['backends'] if name in ('ninfer', 'llamacpp', 'radiance')]
            self.launcher_query_contract = True
        build_info=self.backend_root/'BUILD-INFO.json'
        if build_info.is_file():
            package=json.loads(build_info.read_text(encoding='utf-8-sig'))
            if package.get('backend') in NATIVE:
                self.backends=[name for name in self.backends if name==package['backend']]
        console_build=self.backend_root/'console-build.json'
        if console_build.is_file():
            allowed=json.loads(console_build.read_text(encoding='utf-8-sig')).get('supported_backends')
            if allowed is not None:
                if not isinstance(allowed,list) or not allowed or any(name not in NATIVE for name in allowed):
                    raise ValueError('Invalid console package backend list')
                self.backends=[name for name in self.backends if name in allowed]
        profile_file = self.backend_root / 'scripts/speculative-capabilities.json'
        self.speculative_profiles = json.loads(profile_file.read_text(encoding='utf-8-sig')) if profile_file.is_file() else {}
        if not self.backends:
            raise ValueError('This console requires a NInfer or llama.cpp launcher')
        self.port, self.token = port, secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.inspection_slots = threading.BoundedSemaphore(2)
        self.picker_lock = threading.Lock()
        self.process = self.job = self.log = None
        self.budget_busy = False
        self.budget_job = None
        self.budget_cancelled = False
        self.active = None
        self.active_plan = None
        self.worker_cache = {}
        self.observed_scoring = {}
        self.observed_service = None
        self.observed_offset = 0
        self.observed_pending = ''
        self.instance = secrets.token_hex(12)
        self.chat = ChatBridge(self)
        self.redactions = set()
        self.exit_code = None
        self.memory_trial = None
        self.launch_was_ready = False
        self.memory_diagnosis = None
        self.memory_diagnosis_instance = None
        self.draft_path = self.state_dir / 'launch-draft.json'
        self.log_path = self.state_dir / 'service.log'
        self.profiles_root = self.state_dir / 'profiles'
        self.generated_secrets = {}
        self.active_revision = None
        self.service_selection_path = self.state_dir / 'service-selection.json'
        self.service_selection = None
        if os.name != 'nt' and self.service_selection_path.is_file():
            self.service_selection = json.loads(self.service_selection_path.read_text(encoding='utf-8'))
        # This describes the file, not the editing draft or the live process.
        # Keep it after stop so the final output keeps its original identity.
        self.log_session = secrets.token_hex(12)
        self.log_metadata = {}
        manifest_path = ROOT / 'console-build.json'
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
            if manifest.get('launch_contract_version') not in ((2, 3) if os.name == 'nt' else (3,)):
                raise ValueError('控制台组件版本不匹配，请完整更新启动包。 Console components do not match; update the complete package.')
            for name, digest in manifest.get('controller_files', {}).items():
                path = (ROOT / name).resolve()
                if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError('控制台文件不匹配，请完整更新启动包。 Console file mismatch: ' + name)

    def worker_capabilities(self, worker, backend, environment=None):
        path = Path(worker).resolve()
        stat = path.stat()
        key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
               backend, tuple(sorted((environment or {}).items())))
        if key in self.worker_cache:
            return copy.deepcopy(self.worker_cache[key])
        from launcher.worker import probe_worker
        value = probe_worker(path, backend, environment)
        self.worker_cache = {key: value}
        return copy.deepcopy(value)

    @staticmethod
    def scoring_from_argv(argv):
        return scoring_settings(argv)

    def observe_scoring(self):
        if self.service_selection:
            log = self.service_log(scoring=True)
            identity = (self.service_selection['profile'], self.service_selection['scope'], log['session'])
            if identity != self.observed_service:
                self.observed_scoring = {}
                self.observed_service = identity
            self.parse_scoring(log['text'].splitlines())
            return
        if not self.active_plan or not self.log_path.is_file():
            return
        with self.log_path.open('rb') as stream:
            stream.seek(self.observed_offset)
            data = stream.read(1 << 20)
            self.observed_offset = stream.tell()
        text = self.observed_pending + data.decode('utf-8', errors='replace')
        lines = text.split('\n')
        self.observed_pending = lines.pop()
        self.parse_scoring(lines)

    def parse_scoring(self, lines):
        for line in lines:
            for marker in ('KVMEM_SCORING ', 'KVMEM_LIMITS ', 'KVMEM_BLOCKS '):
                if marker not in line:
                    continue
                for key, value in re.findall(r'(\w+)=([^\s]+)', line.split(marker, 1)[1]):
                    if key in {'scorer', 'agg'}:
                        self.observed_scoring[key] = value
                    elif key == 'lrfu':
                        self.observed_scoring[key] = value == '1'
                    elif key in {'lambda', 'top_frac', 'graph_alpha'}:
                        self.observed_scoring[key] = float(value)
                    elif key in {'query_max_tokens', 'query_replay_max_tokens', 'block_tokens'}:
                        self.observed_scoring[key] = int(value)

    def worker_defaults(self):
        # A source checkout and a runnable installation need not share a directory.
        # A packaged BAT launch prefers its own complete worker set so a stale
        # registration or browser draft cannot silently select another package.
        # Direct/source launches retain the explicit-registration precedence.
        registration = self.state_dir / 'worker-paths.json'
        paths = json.loads(registration.read_text(encoding='utf-8-sig')) if registration.exists() else {}
        if (not isinstance(paths, dict) or set(paths) - set(NATIVE)
                or any(not isinstance(p, str) or not Path(p).is_absolute() or '\x00' in p
                       for p in paths.values())):
            raise ValueError('Invalid worker-paths.json: map backend names to absolute executable paths')
        names = {'ninfer': ('ninfer-kvmem-server.exe', 'ninfer-serve.exe'),
                 'llamacpp': ('llama-kvmem-server.exe',), 'polystrata':('strata-poly.exe',),
                 'radiance':('radiance-kvmem-server.exe',)}
        if os.name != 'nt':
            names = {backend: tuple(name.removesuffix('.exe') for name in entries) for backend, entries in names.items()}
        result = {}
        for backend in self.backends:
            # Search only known names in bounded installation locations; never
            # execute discovered files, search PATH, or recurse into other builds.
            roots = dict.fromkeys((self.backend_root, ROOT.resolve()))
            candidates = [directory / name
                          for root in roots
                          for directory in (root, root / 'bin', root / 'workers' / backend)
                          for name in names[backend]]
            fallback = self.backend_root / 'workers' / backend / names[backend][0]
            discovered = next((candidate for candidate in candidates if candidate.is_file()), None)
            if self.prefer_bundled_worker and discovered is not None:
                path = discovered
            elif backend in paths:
                path = Path(paths[backend])
            else:
                path = discovered or fallback
            result[backend] = {'path': str(path), 'available': path.is_file()}
        return result

    def prefer_packaged_worker(self, parameters):
        parameters = copy.deepcopy(parameters)
        if self.prefer_bundled_worker:
            worker = self.worker_defaults().get(parameters.get('Backend'))
            if worker and worker['available']:
                parameters['Worker'] = worker['path']
        return parameters

    def draft(self):
        if self.draft_path.exists():
            saved = without_secrets(json.loads(self.draft_path.read_text(encoding='utf-8')))
            if saved.get('parameters', {}).get('Backend') in self.backends:
                saved['parameters'] = self.prefer_packaged_worker(saved['parameters'])
                return saved
        backend = DEFAULTS['Backend'] if DEFAULTS['Backend'] in self.backends else self.backends[0]
        return {'schema_version': 1, 'parameters': {**DEFAULTS, **copy.deepcopy(ENGINE_DEFAULTS[backend])}}

    def save(self, data):
        clean = without_secrets(normalize(data, self.backends))
        clean['parameters'] = self.prefer_packaged_worker(clean['parameters'])
        with self.lock:
            temp = self.draft_path.with_suffix('.tmp')
            temp.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding='utf-8')
            temp.replace(self.draft_path)
        return clean

    def redact(self, text):
        for secret in sorted(self.redactions, key=len, reverse=True):
            text = text.replace(secret, '<redacted>')
        return text

    def plan(self, data, *, reveal_secrets=False):
        clean = normalize(data, self.backends)
        if not self.launcher_query_contract:
            raise ValueError('启动脚本版本不匹配，请完整更新控制台和启动脚本。 Launcher version mismatch; update the complete console package.')
        api_key = clean['parameters'].pop('ApiKey', '')
        if api_key:
            self.redactions.add(api_key)
        params = self.prefer_packaged_worker(clean['parameters'])
        if params.get('Port', 18200) == self.port:
            raise ValueError('Inference service and console must use different ports')
        params = copy.deepcopy(params)
        if not params.get('Worker'):
            worker = self.worker_defaults()[params['Backend']]
            if not worker['available']:
                raise ValueError('未找到可用的后端程序。请在“引擎专属参数”的“后端程序”中填写可执行文件路径，'
                                 '或在控制台 worker-paths.json 中登记安装位置。')
            params['Worker'] = worker['path']
        elif not Path(params['Worker']).is_file():
            raise ValueError('指定的后端程序不存在，请检查“引擎专属参数 → 后端程序”：' + params['Worker'])
        if os.name != 'nt' or params['Backend']=='radiance':
            if api_key:
                params['ApiKey'] = api_key
            result = build_plan({'schema_version': 1, 'parameters': params}, self.backend_root,
                                reveal_secrets=reveal_secrets)
            result['worker'] = self.worker_capabilities(result['argv'][0], params['Backend'], result['environment'])
            if params['Backend']=='radiance' and params.get('Vision') and not result['worker'].get('multimodal'):
                raise ValueError('该 Radiance worker 未编入图片解码支持，请使用带 FFmpeg 的视觉构建。 This Radiance worker has no image decoder; use a vision build with FFmpeg.')
            require_launch_features(result['worker'], params)
            result['resolved_scoring'] = self.scoring_from_argv(result['argv'])
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind((params.get('ListenHost', '127.0.0.1'), params.get('Port', 18200)))
                    available = True
                except OSError:
                    available = False
            result['preflight'] = {'port': params.get('Port', 18200),
                                   'listen_host': params.get('ListenHost', '127.0.0.1'),
                                   'port_available': available}
            return result
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8',
                                         dir=self.state_dir, delete=False) as stream:
            json.dump(params, stream, ensure_ascii=False); name = stream.name
        try:
            powershell = str(Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe')
            # PolyStrata validates non-loopback authentication during the dry run.
            # Pass the key before that check, without putting it in JSON or argv.
            plan_env = None
            if api_key and params['Backend'] == 'polystrata':
                plan_env = dict(os.environ, STRATA_API_KEY=api_key)
            run = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                                  '-File', str(ROOT / 'scripts/windows/describe-launch-json.ps1'),
                                  '-BackendRoot', str(self.backend_root), '-ConfigFile', name],
                                 capture_output=True, encoding='utf-8', errors='replace', timeout=45,
                                 cwd=self.backend_root, env=plan_env,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
            if run.returncode:
                message = run.stderr.strip()[-6000:] or run.stdout.strip()[-6000:]
                concise = re.search(r'\[KVMEM_CONFIG\]\s*([^\r\n]+)', message)
                raise ValueError(self.redact(concise.group(1).strip() if concise else message))
            # PowerShell warnings may precede the JSON result.
            start = run.stdout.find('{')
            result = json.loads(run.stdout[start:])
            argv = result.get('argv')
            if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or '\x00' in x for x in argv):
                raise ValueError('Launcher returned invalid argv')
            if api_key:
                if params['Backend']=='polystrata':
                    result.setdefault('environment',{})['STRATA_API_KEY']=api_key if reveal_secrets else '<redacted>'
                else:
                    argv.extend(['--api-key', api_key if reveal_secrets else '<redacted>'])
            result['warnings'] = self.redact(run.stdout[:start].strip())
            native_argv = result.get('native_argv', argv)
            result['worker'] = self.worker_capabilities(native_argv[0], params['Backend'])
            require_launch_features(result['worker'], params)
            result['resolved_scoring'] = self.scoring_from_argv(native_argv)
            if params['Backend'] == 'polystrata':
                block = params.get('BlockTokens', 32)
                if block not in result['worker'].get('block_tokens', {}).get('supported', []):
                    raise ValueError('PolyStrata worker does not support BlockTokens; update the backend')
                if '--kvmem-block-tokens' not in native_argv or int(native_argv[native_argv.index('--kvmem-block-tokens') + 1]) != block:
                    raise ValueError('PolyStrata BlockTokens differs from the native launch plan')
                result['resolved_scoring']['block_tokens'] = block
            return result
        finally:
            Path(name).unlink(missing_ok=True)

    def browse_model(self, data, folder=False, template=False, projector=False):
        if not isinstance(data, dict) or set(data) - {'backend', 'current_path'}:
            raise ValueError('Expected backend and optional current_path')
        backend, current = data.get('backend'), data.get('current_path', '')
        if backend not in self.backends or not isinstance(current, str) or '\x00' in current or len(current) > 8192:
            raise ValueError('Invalid model picker request')
        if projector and backend not in ('llamacpp', 'polystrata'):
            raise ValueError('This backend does not use an external GGUF projector')
        if os.name != 'nt':
            raise ValueError('The native model picker currently requires Windows')
        picker = ROOT / 'scripts/windows/select-model.ps1'
        if not picker.is_file():
            raise ValueError('选择窗口组件缺失，请更新完整的控制台包；也可以直接粘贴模型文件或目录的完整路径。')
        if not self.picker_lock.acquire(blocking=False):
            raise ValueError('已有一个系统选择窗口，请先完成或关闭它。')
        try:
            try:
                remembered = self._model_directory_paths() if not current else []
            except ValueError:
                remembered = []
            initial = (Path(current.strip().strip('"')).expanduser() if current else
                       next((path for path in remembered if path.is_dir()), self.backend_root / 'models'))
            if not initial.is_dir():
                initial = initial.parent
            while not initial.is_dir() and initial.parent != initial:
                initial = initial.parent
            powershell = str(Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe')
            result = subprocess.run([powershell, '-NoProfile', '-STA', '-ExecutionPolicy', 'Bypass',
                                     '-File', str(picker),
                                     '-Backend', backend, '-InitialDirectory', str(initial.resolve())] + (['-Folder'] if folder else ['-Template'] if template else ['-Projector'] if projector else []),
                                    capture_output=True, encoding='utf-8', errors='replace', timeout=300,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode:
                # Startup errors occur before the helper can set UTF-8. Raw PowerShell
                # diagnostics may use a different encoding and do not belong in the UI.
                raise ValueError('无法打开系统选择窗口。请重试，或直接粘贴模型文件或目录的完整路径。')
            try:
                selected = json.loads(result.stdout).get('path')
            except (ValueError, AttributeError):
                raise ValueError('选择窗口返回异常，请重试或直接粘贴完整路径。') from None
            if selected is None:
                return {'path': None}
            if not isinstance(selected, str):
                raise ValueError('Invalid model selection result')
            path = Path(selected).resolve(strict=True)
            if folder:
                if not path.is_dir():
                    raise ValueError('Select a model directory')
                return {'path': str(path)}
            suffix = '.rad' if backend=='radiance' else '.ninfer' if backend == 'ninfer' else '.gguf'
            if not path.is_file() or (not template and path.suffix.lower() != suffix):
                raise ValueError(f'{backend} requires a {suffix} model file')
            return {'path': str(path), 'name': path.name}
        except subprocess.TimeoutExpired:
            raise ValueError('选择窗口已超时，请重新打开，或直接粘贴完整路径。') from None
        except OSError:
            raise ValueError('无法打开系统选择窗口，请直接粘贴完整路径。') from None
        finally:
            self.picker_lock.release()

    @staticmethod
    def model_directory(value, must_exist=True):
        if not isinstance(value, str) or '\x00' in value or not 0 < len(value) <= 8192:
            raise ValueError('请输入模型文件夹的完整路径。')
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError('请输入完整的文件夹路径，例如 D:\\Models。')
        try:
            path = path.resolve() if must_exist else Path(os.path.abspath(path))
            if path == Path(path.anchor) or str(path).startswith('\\\\'):
                raise ValueError('请选择本地的模型文件夹，例如 D:\\Models；暂不扫描整个盘符或网络目录。')
            if must_exist and not path.is_dir():
                raise ValueError('文件夹不存在或无法访问，请检查路径：' + str(path))
        except OSError:
            raise ValueError('无法访问该文件夹，请检查路径和访问权限。') from None
        return path

    def _model_directory_paths(self):
        path = self.state_dir / 'model-directories.json'
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            if (not isinstance(data, dict) or data.get('schema_version') != 1
                    or not isinstance(data.get('directories'), list) or len(data['directories']) > 16):
                raise ValueError()
            return list(dict.fromkeys(self.model_directory(item, must_exist=False)
                                      for item in data['directories']))
        except (OSError, ValueError, TypeError):
            raise ValueError('保存的模型目录记录无法读取；仍可直接选择模型文件。') from None

    def model_directories(self):
        with self.lock:
            paths = self._model_directory_paths()
        # Filesystem availability checks must not block owned-process status/control.
        directories = []
        for path in paths:
            try:
                available = path.is_dir()
            except OSError:
                available = False
            directories.append({'path': str(path), 'available': available})
        return {'directories': directories}

    def update_model_directories(self, data):
        if (not isinstance(data, dict) or set(data) != {'action', 'path'}
                or data['action'] not in ('add', 'remove')):
            raise ValueError('Invalid model directory request')
        path = self.model_directory(data['path'], must_exist=data['action'] == 'add')
        with self.lock:
            paths = [item for item in self._model_directory_paths() if item != path]
            if data['action'] == 'add':
                if len(paths) >= 16:
                    raise ValueError('最多记住 16 个模型目录，请先移除不再使用的目录记录。')
                paths.insert(0, path)
            target = self.state_dir / 'model-directories.json'
            temp = target.with_suffix('.tmp')
            try:
                temp.write_text(json.dumps({'schema_version': 1, 'directories': list(map(str, paths))},
                                           ensure_ascii=False, indent=2), encoding='utf-8')
                temp.replace(target)
            except OSError:
                raise ValueError('无法保存模型目录，请检查控制台状态目录的写入权限。') from None
        return self.model_directories()

    def scan_models(self, data):
        if not isinstance(data, dict) or set(data) - {'backend', 'current_path', 'directory', 'worker_path'}:
            raise ValueError('Invalid model scan request')
        backend = data.get('backend')
        if backend not in self.backends:
            raise ValueError('Unknown backend')
        for key in ('current_path', 'directory', 'worker_path'):
            value = data.get(key, '')
            if not isinstance(value, str) or '\x00' in value or len(value) > 8192:
                raise ValueError('Invalid search path')
        requested = []
        if data.get('directory'):
            folder = self.model_directory(data['directory'])
            self.update_model_directories({'action': 'add', 'path': str(folder)})
            requested.append((folder, True))
        else:
            if data.get('current_path'):
                current = Path(data['current_path'].strip().strip('"')).expanduser()
                if current.is_absolute():
                    requested.append((current if current.is_dir() else current.parent, True))
            requested.extend((folder, True) for folder in self._model_directory_paths())
            bases = list(dict.fromkeys((self.backend_root, ROOT.resolve())))
            worker = data.get('worker_path')
            if not worker:
                try:
                    worker = self.worker_defaults()[backend]['path']
                except ValueError:
                    worker = ''  # Model discovery remains usable before worker configuration is valid.
            if worker and Path(worker).is_absolute() and Path(worker).is_file():
                bases.append(Path(worker).parent)
            for base in dict.fromkeys(bases):
                # Models often sit beside the EXE. Inspect only that directory's files;
                # recurse into named model libraries, never entire build/workspace trees.
                requested.append((base, False))
                for parent in (base, *list(base.parents)[:4]):
                    for name in ('models', 'model'):
                        candidate = parent / name
                        if candidate.is_dir():
                            requested.append((candidate, True))
        roots = {}
        for folder, recursive in requested:
            folder = folder.resolve()
            if (folder == Path(folder.anchor) or str(folder).startswith('\\\\')
                    or not folder.is_absolute()):
                continue
            roots[folder] = roots.get(folder, False) or recursive
        locations = [{'path': str(folder), 'recursive': recursive, 'status': 'pending'}
                     for folder, recursive in roots.items()]
        suffix = '.rad' if backend=='radiance' else '.ninfer' if backend == 'ninfer' else '.gguf'
        found, visited, candidates = {}, {}, 0
        other = {'.ninfer': set(), '.gguf': set(), '.rad':set()}
        reasons = set()
        skipped, limited = 0, False
        deadline = time.monotonic() + 3
        stack = [(folder, 6 if recursive else 0, index)
                 for index, (folder, recursive) in reversed(list(enumerate(roots.items())))]
        while stack:
            if time.monotonic() >= deadline or candidates >= 10000 or len(found) >= 200:
                reasons.add('time' if time.monotonic() >= deadline else 'entries' if candidates >= 10000 else 'results')
                limited = True; break
            folder, remaining, location = stack.pop()
            identity = os.path.normcase(str(folder))
            if visited.get(identity, -1) >= remaining:
                if locations[location]['status'] == 'pending':
                    locations[location]['status'] = 'covered'
                continue
            visited[identity] = remaining
            try:
                with os.scandir(folder) as entries:
                    if locations[location]['status'] == 'pending':
                        locations[location]['status'] = 'scanned'
                    for entry in entries:
                        candidates += 1
                        if candidates > 10000 or time.monotonic() >= deadline or len(found) >= 200:
                            reasons.add('time' if time.monotonic() >= deadline else 'entries' if candidates > 10000 else 'results')
                            locations[location]['status'] = 'partial'
                            limited = True; break
                        try:
                            stat = entry.stat(follow_symlinks=False)
                            # Junctions and links can lead outside the chosen folder or into loops.
                            if entry.is_symlink() or getattr(stat, 'st_file_attributes', 0) & 0x400:
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                if entry.name.startswith('.') or entry.name.lower() in ('node_modules', '__pycache__'):
                                    continue
                                if remaining:
                                    stack.append((Path(entry.path), remaining - 1, location))
                                elif locations[location]['recursive']:
                                    limited = True; reasons.add('depth')
                                    locations[location]['status'] = 'partial'
                            elif entry.is_file(follow_symlinks=False) and Path(entry.name).suffix.lower() in other:
                                # Auxiliary vision projectors and non-first GGUF shards aren't standalone models.
                                if re.search(r'(?:mmproj|projector)', entry.name, re.I):
                                    continue
                                shard = re.search(r'-(\d{5})-of-\d{5}\.gguf$', entry.name, re.I)
                                if shard and shard[1] != '00001':
                                    continue
                                # Windows DirEntry.stat may omit the file ID. Read it explicitly
                                # so legacy hard links appear only once in the model library.
                                file_stat = os.stat(entry.path, follow_symlinks=False)
                                key = ((file_stat.st_dev, file_stat.st_ino) if file_stat.st_ino
                                       else os.path.normcase(entry.path))
                                actual_suffix = Path(entry.name).suffix.lower()
                                if actual_suffix != suffix:
                                    other[actual_suffix].add(key)
                                    continue
                                candidate = {'path': entry.path, 'name': entry.name,
                                             'bytes': file_stat.st_size, 'format': suffix[1:],
                                             'warnings': ['empty'] if file_stat.st_size == 0 else []}
                                def preference(item):
                                    path = Path(item['path'])
                                    canonical = (path.parent.name.casefold() == path.stem.casefold()
                                                 and path.parent.parent.name.casefold() == suffix[1:])
                                    return (not canonical, len(path.parts), item['path'].casefold())
                                if key not in found or preference(candidate) < preference(found[key]):
                                    found[key] = candidate
                        except OSError:
                            skipped += 1
                            locations[location]['status'] = 'partial'
            except OSError:
                skipped += 1
                locations[location]['status'] = 'unavailable' if folder in roots else 'partial'
        return {'models': sorted(found.values(), key=lambda x: (bool(x['warnings']), x['name'].casefold(), x['path'])),
                'roots': [str(folder) for folder in roots], 'locations': locations,
                'limited': limited, 'limit_reasons': sorted(reasons), 'skipped': skipped,
                'other_formats': {key[1:]: len(value) for key, value in other.items()}}

    def status(self):
        with self.lock:
            service = None
            service_error = None
            if self.service_selection:
                try:
                    service = self.services().status(**self.service_selection)
                    _, _, self.active_plan, self.active = self.revision_config(service['profile_id'], service['revision'])
                    self.active_revision = {key: service[key] for key in ('profile_id', 'revision')}
                    self.instance = service['invocation'] or 'systemd-' + service['profile_id']
                    self.exit_code = service['exit_code']
                except (OSError, ValueError) as error:
                    service_error = str(error)
                    self.active = self.active_plan = None
            if not service_error:
                self.observe_scoring()
            running = self.process is not None and self.process.poll() is None
            if self.process and not running:
                self.exit_code = self.process.returncode
                self._release()
            if service:
                running = service['active'] in ('active', 'activating')
            result = {'state': 'starting' if running else ('exited' if self.exit_code is not None else 'stopped'),
                      'pid': self.process.pid if running and self.process else None, 'exit_code': self.exit_code, 'instance':self.instance,
                      'active_config': without_secrets(self.active) if self.active else None,
                      'api_key_configured': bool(self.active and self.active['parameters'].get('ApiKey')), 'url': None}
            result['owner'] = 'temporary' if self.active else None
            if self.service_selection:
                result.update(owner='systemd', service=service, service_error=service_error,
                              pid=service['pid'] if service else None)
                if service and service['active'] == 'inactive':
                    result['state'] = 'stopped'
            result['generated_revision'] = copy.deepcopy(self.active_revision)
            result['planned_scoring'] = copy.deepcopy(self.active_plan.get('resolved_scoring')) if self.active_plan else None
            result['budget_planning'] = self.budget_busy
            result['effective_scoring'] = dict(self.observed_scoring) or None
            result['worker'] = copy.deepcopy(self.active_plan.get('worker')) if self.active_plan else None
            result['token_settings'] = copy.deepcopy(self.active_plan.get('token_settings')) if self.active_plan else None
            if AUTO_BUDGET_SUPPORTED and result['state'] == 'exited' and self.exit_code != 0 and self.active and not self.launch_was_ready:
                if self.memory_diagnosis_instance != self.instance:
                    plan = self.active_plan or {}
                    self.memory_diagnosis = diagnose_memory(self.log_tail(), self.active['parameters'],
                        plan.get('token_settings'))
                    self.memory_diagnosis_instance = self.instance
                result['memory_diagnosis'] = copy.deepcopy(self.memory_diagnosis)
            planned = result['planned_scoring']
            result['scoring_confirmed'] = bool(planned and all(
                key in self.observed_scoring and self.observed_scoring[key] == value for key, value in planned.items()))
            result['scoring_mismatch'] = bool(planned and set(planned).issubset(self.observed_scoring)
                                              and not result['scoring_confirmed'])
            if running and self.active and (not service or service['worker_matches']):
                port = self.active['parameters'].get('Port', 18200)
                result['url'] = f'http://127.0.0.1:{port}'
                try:
                    target = result['url'] + '/health'
                    api_key = self.active['parameters'].get('ApiKey')
                    if api_key:
                        target = urllib.request.Request(target, headers={'Authorization': 'Bearer ' + api_key})
                    with urllib.request.urlopen(target, timeout=0.5) as response:
                        if response.status == 200:
                            result['state'] = 'ready'
                            self.launch_was_ready = True
                except urllib.error.HTTPError as error:
                    error.close()
                except (OSError, urllib.error.URLError):
                    pass
            if self.memory_trial:
                result['memory_trial'] = {**self.memory_trial, 'startup_verified': result['state'] == 'ready'}
            return result

    def services(self):
        if os.name == 'nt':
            raise ValueError('systemd management requires Linux')
        from launcher.services import Services
        return Services(self.profiles_root)

    def service_list(self):
        manager = self.services()
        return {'capabilities': manager.available(), 'services': manager.list(),
                'selected': self.service_selection}

    def service_log(self, *, scoring=False):
        with self.lock:
            if not self.service_selection:
                return dict(text='', available=False, session='', running=False, truncated=False)
            manager = self.services()
            registered, _, _, _ = manager.read(**self.service_selection)
            # Logs may be the first request after reopening the console. Restore
            # redaction credentials before reading any journal content.
            self.revision_config(registered['profile_id'], registered['revision'])
            return manager.journal(**self.service_selection, redactions=tuple(self.redactions), scoring=scoring)

    def service_action(self, data):
        if not isinstance(data, dict) or set(data) - {'action', 'profile_id', 'revision', 'scope'}:
            raise ValueError('Expected a registered service and action')
        action, profile, scope = data.get('action'), data.get('profile_id'), data.get('scope', 'user')
        manager = self.services()
        with self.lock:
            if action == 'disconnect':
                # Disconnecting only changes this console's selection. It must
                # also work when the registration or system unit was removed.
                manager._record_path(profile, scope)
                selected = self.service_selection == dict(profile=profile, scope=scope)
                if selected:
                    self.chat.cancel_all()
                    self.clear_service_selection()
                return dict(disconnected=selected)
            if action in ('stop', 'restart'):
                self.chat.cancel_all()
            if action == 'register':
                return manager.register(profile, data.get('revision'), scope)
            if action in ('select', 'start', 'restart'):
                if self.process and self.process.poll() is None:
                    raise ValueError('Stop the temporary service before switching to systemd')
                state = manager.status(profile, scope)
                _, _, plan, config = self.revision_config(profile, state['revision'])
                if action != 'select':
                    native = self.worker_capabilities(plan['argv'][0], plan['backend'], plan['environment'])
                    require_launch_features(native, config['parameters'])
                    if native.get('executable_sha256') != plan.get('worker', {}).get('executable_sha256'):
                        raise ValueError('Worker changed; generate and register a new version')
                    if not self.inspect_model(config, for_launch=True)['can_start']:
                        raise ValueError('Model inspection blocked service launch')
                    if state['active'] not in ('active', 'activating'):
                        p = config['parameters']
                        with socket.socket() as probe:
                            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                            try: probe.bind((p.get('ListenHost', '127.0.0.1'), p.get('Port', 18200)))
                            except OSError as error:
                                raise ValueError('Inference port is occupied; the existing service was left running') from error
                    manager.action(profile, scope, action)
                from launcher.services import atomic_json
                self.service_selection = dict(profile=profile, scope=scope)
                atomic_json(self.service_selection_path, self.service_selection)
                self._release()
                self.active, self.active_plan = config, plan
                self.observed_scoring = {}
                self.observed_service = None
                return self.status()
            result = manager.action(profile, scope, action)
            if action == 'remove' and self.service_selection == dict(profile=profile, scope=scope):
                self.clear_service_selection()
            return result

    def clear_service_selection(self):
        self.service_selection = None
        self.service_selection_path.unlink(missing_ok=True)
        self.active = self.active_plan = self.active_revision = None
        self.observed_scoring = {}
        self.observed_service = None
        self.exit_code = None
        self.instance = secrets.token_hex(12)

    def close(self):
        # Persistent services belong to systemd and survive closing the UI.
        with self.lock:
            self.chat.cancel_all()
            if not self.service_selection:
                self.stop()
            else:
                self._release()

    def _run_budget_plan(self, plan):
        env = dict(os.environ); env.update(plan.get('environment', {}))
        argv = list(plan.get('native_argv', plan['argv'])) + ['--plan-budget']
        job, process = WindowsJob(), None
        try:
            with self.lock:
                if self.budget_cancelled:
                    raise ValueError('Budget 测算已取消 / Budget planning cancelled')
                self.budget_job = job
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker-gate'],
                    cwd=self.backend_root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                job.assign(process)
            output = process.communicate((json.dumps(argv)+'\n').encode('utf-8'), timeout=180)[0]
            text = output.decode('utf-8', errors='replace')
            if process.returncode:
                raise ValueError('Budget 测算失败 / Budget planning failed: '+self.redact(text[-4000:]))
            return text
        except subprocess.TimeoutExpired:
            raise ValueError('Budget 测算超时，已停止测算进程 / Budget planning timed out; worker stopped') from None
        finally:
            with self.lock:
                job.close()
                self.budget_job = None
            if process is not None:
                process.wait(timeout=10)

    def adjust_budget(self, data):
        if not AUTO_BUDGET_SUPPORTED:
            raise ValueError('自动调整暂支持 Windows 控制台 / Automatic budget adjustment currently requires Windows')
        clean = normalize(data, self.backends)
        p = clean['parameters']
        if p['Backend'] != 'ninfer' or not p.get('Gpu') or ',' in p['Gpu'] or p.get('TensorParallel'):
            raise ValueError('自动调整暂支持 NInfer 单卡，请选择一张显卡 / Auto budget requires single-GPU NInfer')
        # Budget is the output of this operation, not an input constraint. Use a
        # legal seed so an oversized or zero current value cannot block planning.
        p['Budget'] = max(128, (max(1,p.get('SinkTokens',DEFAULTS['SinkTokens'])//64)
                              + p.get('RecentTokens',DEFAULTS['RecentTokens'])//64)*64,
                          max(0,256-p.get('Reserve',DEFAULTS['Reserve'])+63)//64*64)
        with self.lock:
            if self.budget_busy or (self.process and self.process.poll() is None):
                raise ValueError('请等待测算完成或停止运行中的服务 / Wait for planning or stop the running service')
            self.budget_busy = True
            self.budget_cancelled = False
        started = time.monotonic()
        try:
            health = self.inspect_model(clean, for_launch=True)
            if not health['can_start']:
                raise ValueError('模型体检未通过，请先检查配置 / Model inspection blocked budget planning')
            clean['parameters'].pop('ApiKey', None)
            plan = self.plan(clean)
            if plan['worker'].get('budget_planning_version') != 1:
                raise ValueError('后端尚不支持自动调整 budget，请更新后端 / Update worker to support automatic budget planning')
            effective = plan['token_settings']['effective']
            # Omitted fields use the launcher's defaults. Only explicit values
            # must remain unchanged; report validation uses the effective plan.
            if any(key in p and effective[key] != p[key] for key in ('Reserve', 'Prefill')):
                raise ValueError('请先通过检查配置对齐 gen reserve 和 prefill；自动调整只修改 budget / Align reserve and prefill first; auto adjustment changes budget only')
            result = validate_budget_plan(self._run_budget_plan(plan), p, plan['token_settings'])
            if self.budget_cancelled:
                raise ValueError('Budget 测算已取消 / Budget planning cancelled')
            if not result['available']:
                raise ValueError('当前设置下最低合法 budget 仍无法容纳；请调整其他设置 / Minimum budget does not fit; adjust other settings')
            return dict(result, elapsed_ms=round((time.monotonic()-started)*1000))
        finally:
            with self.lock:
                self.budget_busy = False

    def retry_memory(self, data):
        if not isinstance(data, dict) or set(data) != {'instance', 'option', 'configuration'}:
            raise ValueError('Expected memory recommendation, launch instance and configuration')
        with self.lock:
            status = self.status()
            if status['state'] != 'exited' or data['instance'] != self.instance:
                raise ValueError('启动状态已改变，请刷新推荐。 / Launch changed; refresh recommendations.')
            current = normalize(data['configuration'], self.backends)
            current['parameters'] = self.prefer_packaged_worker(current['parameters'])
            if without_secrets(current) != without_secrets(self.active):
                raise ValueError('配置已改变，请使用当前配置重新启动；旧推荐已失效。 / Settings changed; old recommendations no longer apply.')
            option = next((x for x in (self.memory_diagnosis or {}).get('options', [])
                           if x['id'] == data['option'] and x['available']), None)
            if option is None:
                raise ValueError('此推荐不可用。 / Recommendation unavailable.')
            updated = copy.deepcopy(self.active)
            updated['parameters'].update(option['changes'])
            # Normal start rechecks model health and canonical launcher constraints.
            self.start(updated)
            self.memory_trial = {'format': option['id'], 'changes': option['changes']}
            return self.status()

    def _release(self):
        if self.job:
            self.job.close(); self.job = None
        if self.log:
            self.log.close(); self.log = None
        self.process = None

    def inspect_model(self, data, *, for_launch=False):
        parameters = self.prefer_packaged_worker(request_parameters(data))
        if parameters['Backend'] not in self.backends:
            raise ValueError('Backend is not installed')
        if not parameters.get('Worker'):
            parameters['Worker'] = self.worker_defaults().get(parameters['Backend'], {}).get('path', '')
        acquired = self.inspection_slots.acquire(timeout=11) if for_launch else self.inspection_slots.acquire(blocking=False)
        if not acquired:
            if for_launch:
                raise ValueError('模型体检正在忙，请稍后重试启动 / Model inspection busy; retry launch shortly')
            return unchecked('Inspection is busy; retry shortly')
        try:
            run = subprocess.run([sys.executable, str(ROOT / 'scripts/model_health.py')],
                                 input=json.dumps({'schema_version': 1, 'parameters': parameters}),
                                 capture_output=True, text=True, encoding='utf-8',
                                 cwd=self.backend_root, timeout=10,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if run.returncode != 0 or len(run.stdout) > 262144:
                return unchecked('Inspector did not return a bounded report')
            report = json.loads(run.stdout)
            if not isinstance(report, dict) or report.get('schema_version') != 1 or type(report.get('can_start')) is not bool:
                return unchecked('Unsupported inspector response')
            return report
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return unchecked('Inspector unavailable or timed out; retry or attempt launch')
        finally:
            self.inspection_slots.release()

    def export_config(self, data, *, for_launch=False):
        if os.name == 'nt':
            raise ValueError('Linux script generation requires the target Linux machine')
        if not isinstance(data, dict) or set(data) - {'config', 'profile_id', 'scope', 'user', 'store_api_key'}:
            raise ValueError('Expected a configuration and profile export settings')
        if type(data.get('store_api_key', False)) is not bool:
            raise ValueError('store_api_key must be boolean')
        clean = normalize(data.get('config'), self.backends)
        health = self.inspect_model(clean, for_launch=for_launch)
        if not health['can_start']:
            raise ValueError('Model inspection blocked export: ' + '; '.join(
                c.get('zh', c.get('en', '')) + ': ' + c.get('detail', '')
                for c in health['checks'] if c['status'] == 'error'))
        plan = self.plan(clean)
        resolved = {'schema_version': 1, 'parameters': dict(plan['requested_parameters'])}
        key = clean['parameters'].get('ApiKey')
        if key:
            resolved['parameters']['ApiKey'] = key
        result = export_profile(resolved, self.backend_root, self.profiles_root,
            data.get('profile_id', 'default-' + clean['parameters']['Backend']),
            persist_secret=data.get('store_api_key', False), scope=data.get('scope', 'user'),
            user=data.get('user'), _validated_plan=plan)
        if key and result['requires_api_key_environment']:
            self.generated_secrets[(result['profile_id'], result['revision'])] = key
            while len(self.generated_secrets) > 64:
                self.generated_secrets.pop(next(iter(self.generated_secrets)))
        return result

    def revision_config(self, profile_id, revision):
        directory, manifest, plan = read_revision(self.profiles_root, profile_id, revision)
        config = json.loads((directory / 'profile.json').read_text(encoding='utf-8'))
        key = ''
        ref = plan.get('secret_ref')
        if ref:
            if ref['kind'] == 'file':
                key = Path(ref['path']).read_text(encoding='utf-8').rstrip('\n')
            else:
                key = self.generated_secrets.get((profile_id, revision), '')
            if not key:
                raise ValueError('This generated version needs its API key. Re-enter it and generate a new version, or use a stored credential.')
            config['parameters']['ApiKey'] = key
            self.redactions.add(key)
        config = normalize(config, self.backends)
        return directory, manifest, plan, config

    def start_revision(self, data):
        if os.name == 'nt':
            raise ValueError('Linux script execution requires Linux')
        if not isinstance(data, dict) or set(data) != {'profile_id', 'revision'}:
            raise ValueError('Expected profile_id and revision')
        from launcher.process_linux import OwnedProcess
        directory, _, plan, clean = self.revision_config(data['profile_id'], data['revision'])
        native = self.worker_capabilities(plan['argv'][0], plan['backend'], plan['environment'])
        require_launch_features(native, clean['parameters'])
        if native.get('executable_sha256') != plan.get('worker', {}).get('executable_sha256'):
            raise ValueError('Worker changed after script generation; check the configuration and generate a new version')
        health = self.inspect_model(clean, for_launch=True)
        if not health['can_start']:
            raise ValueError('Model inspection blocked launch: ' + '; '.join(
                c.get('zh', c.get('en', '')) + ': ' + c.get('detail', '')
                for c in health['checks'] if c['status'] == 'error'))
        with self.lock:
            if self.service_selection:
                current = self.services().status(**self.service_selection)
                if current['active'] in ('active', 'activating', 'deactivating'):
                    raise ValueError('Stop the selected systemd service before starting a temporary service')
                self.clear_service_selection()
            if self.process and self.process.poll() is None:
                raise ValueError('A managed service is already running; stop it before starting another')
            self._release()
            p = clean['parameters']
            if p.get('Port', 18200) == self.port:
                raise ValueError('Inference service and console must use different ports')
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind((p.get('ListenHost', '127.0.0.1'), p.get('Port', 18200)))
                except OSError as error:
                    raise ValueError('Inference port is occupied; the existing service was left running') from error
            self.observed_scoring = {}; self.observed_offset = 0; self.observed_pending = ''
            self.log = private_log(self.log_path, truncate=True)
            for change in plan['token_settings'].get('adjustments', []):
                self.log.write((f"KVMEM_CONFIG {change['parameter']}: {change['requested']} -> {change['effective']} tokens ({change['reason']})\n").encode())
            self.log.flush()
            env = dict(os.environ)
            if p.get('ApiKey'):
                env['KVMEM_API_KEY'] = p['ApiKey']
            try:
                owned = OwnedProcess(directory / 'run.sh', self.log, env)
                self.process = self.job = owned
            except BaseException:
                self._release()
                raise
            self.active, self.active_plan, self.exit_code = clean, plan, None
            self.active_revision = dict(data)
            self.instance = secrets.token_hex(12)
            self.log_session = secrets.token_hex(12)
            self.log_metadata = {'backend': p['Backend'], 'model': Path(p['Model']).name,
                                 'started_at': int(time.time() * 1000)}
            return self.status()

    def start(self, data):
        if os.name != 'nt':
            exported = self.export_config({'config': data}, for_launch=True)
            return self.start_revision({key: exported[key] for key in ('profile_id', 'revision')})
        clean = normalize(data, self.backends)
        clean['parameters'] = self.prefer_packaged_worker(clean['parameters'])
        # Never trust a cached browser report. Inspect this exact launch request.
        health = self.inspect_model(clean, for_launch=True)
        if not health['can_start']:
            errors = [c['zh'] + ' / ' + c['en'] + ': ' + c.get('detail', '')
                      for c in health['checks'] if c['status'] == 'error']
            raise ValueError('Model inspection blocked launch: ' + '; '.join(errors))
        with self.lock:
            if self.budget_busy:
                raise ValueError('正在测算 budget，请稍后启动 / Budget planning is in progress')
            if self.process and self.process.poll() is None:
                raise ValueError('A managed service is already running; stop it before starting another')
            self._release()
            result = self.plan(clean, reveal_secrets=True)
            port = clean['parameters'].get('Port', 18200)
            with socket.socket() as probe:
                if os.name == 'nt':
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                try:
                    probe.bind((clean['parameters'].get('ListenHost', '127.0.0.1'), port))
                except OSError as error:
                    raise ValueError(f'端口 {port} 无法使用，可能已有服务占用。请在“服务与并发”中更换端口，'
                                     '或先停止原服务。此控制台不会停止外部服务。') from error
            env = dict(os.environ); env.update(result.get('environment', {}))
            plan_summary = {'resolved_scoring': result.get('resolved_scoring'), 'worker': result.get('worker'), 'token_settings': result.get('token_settings')}
            self.observed_scoring = {}; self.observed_offset = 0; self.observed_pending = ''
            self.log = self.log_path.open('wb')
            for change in (result.get('token_settings') or {}).get('adjustments', []):
                self.log.write((f"KVMEM_CONFIG {change['parameter']}: {change['requested']} -> {change['effective']} tokens ({change['reason']})\n").encode('utf-8'))
            self.log.flush()
            self.log_session = secrets.token_hex(12)
            self.log_metadata = {'backend': clean['parameters']['Backend'],
                                 'model': Path(clean['parameters'].get('Model', '')).name,
                                 'started_at': int(time.time() * 1000)}
            self.job = WindowsJob()
            try:
                # The gate waits for argv before creating any descendants. Assigning
                # it to the job first closes the spawn-before-job-assignment race.
                self.process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker-gate'],
                                                cwd=self.backend_root, env=env,
                                                stdin=subprocess.PIPE, stdout=self.log, stderr=self.log,
                                                creationflags=subprocess.CREATE_NO_WINDOW)
                self.job.assign(self.process)
                self.process.stdin.write((json.dumps(result['argv']) + '\n').encode('utf-8'))
                self.process.stdin.close()
            except Exception:
                self._release(); raise
            self.active, self.active_plan, self.exit_code = clean, plan_summary, None
            self.launch_was_ready = False
            self.memory_trial = None
            self.memory_diagnosis = None
            self.memory_diagnosis_instance = None
            self.instance = secrets.token_hex(12)
            return self.status()

    def stop(self):
        with self.lock:
            self.chat.cancel_all()
            if self.service_selection:
                self.services().action(**self.service_selection, action='stop')
                return self.status()

            self.budget_cancelled = True
            if self.budget_job:
                self.budget_job.close(); self.budget_job = None
            process = self.process
            if self.job:
                self.job.close(); self.job = None
            if process:
                process.wait(timeout=10)
            self._release(); self.exit_code = None; self.active = None; self.active_plan = None
            self.active_revision = None

            self.memory_trial = None
            self.memory_diagnosis = None
            self.memory_diagnosis_instance = None
            self.observed_scoring = {}; self.observed_offset = 0; self.observed_pending = ''
            self.instance = secrets.token_hex(12)
            return self.status()

    def log_tail(self):
        if self.service_selection:
            return self.redact(self.service_log()['text'])
        if not self.log_path.exists():
            return ''
        with self.log_path.open('rb') as stream:
            stream.seek(0, 2); stream.seek(max(0, stream.tell() - 32768))
            return self.redact(stream.read().decode('utf-8', errors='replace'))

    def log_snapshot(self, revision=''):
        """Bounded read of the common stdout/stderr file; no inference/health calls.

        Return complete terminal lines while the process is writing. In addition
        to keeping UTF-8 intact this prevents exposing a partially written key
        before the normal redactor can match it. A stopped process's last line
        is useful even without a newline. No client-supplied path is accepted.
        """
        if self.service_selection:
            log = self.service_log()
            log['text'] = self.redact(log['text'])
            stamp = log['session'] + ':' + hashlib.sha256(log['text'].encode()).hexdigest()
            return dict(protocol=1, source='journal', revision=stamp, bytes=len(log['text'].encode()),
                        limit_bytes=262144, updated_at=None,
                        pending=False, **{**log, 'text': None if revision == stamp else log['text']})
        limit = 256 * 1024
        with self.lock:
            code = self.process.poll() if self.process else self.exit_code
            running = self.process is not None and code is None
            result = {'protocol': 1, 'session': self.log_session, **self.log_metadata,
                      'running': running, 'exit_code': code, 'limit_bytes': limit,
                      'available': False, 'bytes': 0, 'updated_at': None,
                      'truncated': False, 'pending': False, 'text': ''}
            try:
                stream = self.log_path.open('rb')
            except FileNotFoundError:
                result['revision'] = self.log_session + ':missing'
                return result
            with stream:
                stat = os.fstat(stream.fileno())
                size = stat.st_size
                stamp = f'{self.log_session}:{stat.st_ino}:{size}:{stat.st_mtime_ns}:{running}:{len(self.redactions)}'
                result.update(available=True, bytes=size, updated_at=stat.st_mtime_ns // 1_000_000,
                              revision=stamp, truncated=size > limit)
                # Metadata is returned even when the text revision is unchanged.
                if running and size:
                    stream.seek(size - 1)
                    result['pending'] = stream.read(1) not in (b'\r', b'\n')
                if revision == stamp:
                    result['text'] = None
                    return result
                start = max(0, size - limit)
                stream.seek(start)
                data = stream.read(size - start)
            if start:
                # Discard the first partial line (including partial UTF-8/ANSI).
                boundary = re.search(b'[\r\n]', data)
                data = data[boundary.end():] if boundary else b''
            if running and data and data[-1:] not in (b'\r', b'\n'):
                end = max(data.rfind(b'\n'), data.rfind(b'\r')) + 1
                data = data[:end]
                result['pending'] = True
            result['text'] = self.redact(data.decode('utf-8', errors='replace'))
            return result

    def monitor_authorization(self):
        with self.lock:
            key = self.active['parameters'].get('ApiKey') if self.active else None
            return 'Bearer ' + key if key else None


def gpu_inventory():
    try:
        run = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        devices = [dict(zip(('uuid', 'name', 'memory_mib'), (x.strip() for x in line.split(','))), vendor='nvidia')
                   for line in run.stdout.splitlines() if len(line.split(',')) == 3]
    except (OSError, subprocess.TimeoutExpired):
        devices = []
    from console_hardware import amd_gpu_inventory
    return devices + amd_gpu_inventory()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != 'nt'

    def server_bind(self):
        # Windows SO_REUSEADDR permits a second process to bind the same address.
        # Exclusive ownership is required for reliable repeated-click detection.
        if os.name == 'nt':
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    @property
    def control(self):
        return self.server.control

    def reply(self, code, body, content_type='application/json; charset=utf-8'):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')
        try:
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "frame-ancestors 'self'")
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            # Navigation/refresh can abort either headers or body on Windows.
            self.close_connection = True

    def trusted(self, mutate=False):
        host = self.headers.get('Host', '')
        # A wildcard listener accepts this connection's actual destination IP,
        # never an arbitrary Host supplied by the client (DNS rebinding).
        address = self.connection.getsockname()[0]
        hosts = {f'{address}:{self.server.server_port}'}
        if ipaddress.ip_address(address).is_loopback:
            hosts.add(f'localhost:{self.server.server_port}')
        if self.server.server_port == 80:
            hosts.update({value.removesuffix(':80') for value in hosts})
        origin = self.headers.get('Origin')
        origins = {'http://' + host}
        if self.server.server_port == 80:
            origins = {'http://' + host.removesuffix(':80'), 'http://' + host.removesuffix(':80') + ':80'}
        if host not in hosts or (origin is not None and origin not in origins):
            self.reply(403, {'error': 'Untrusted host or origin'}); return False
        if self.headers.get('Sec-Fetch-Site') not in (None, 'same-origin', 'none'):
            self.reply(403, {'error': 'Cross-site request rejected'}); return False
        if mutate and not secrets.compare_digest(self.headers.get('X-KVMem-Token', ''), self.control.token):
            self.reply(403, {'error': 'Invalid console token'}); return False
        return True

    def do_GET(self):
        if not self.trusted():
            return
        path = urlsplit(self.path).path
        try:
            if path == '/api/chat/context':
                self.reply(200, self.control.chat.context())
            elif path.startswith('/api/chat/requests/'):
                record = self.control.chat.record(path.rsplit('/',1)[1])
                self.reply(200 if record else 404, record or {'error':'Unknown request'})
            elif path.startswith('/chat/view/'):
                self.chat_asset(path[len('/chat/view/'):])
            elif path == '/api/launcher/info':
                self.reply(200, {'token': self.control.token, 'backends': self.control.backends,
                                'listen_host': self.server.server_address[0],
                                'console_protocol': 1, 'console_root': str(ROOT.resolve()),
                                'prefer_bundled_worker': self.control.prefer_bundled_worker,
                                'model_library_version': 1,
                                'model_inspection_version': 1,
                                'kv_cache_version': 1,
                                'query_scoring_version': 2,
                                'platform': 'windows' if os.name == 'nt' else 'linux',
                                'budget_adjustment': AUTO_BUDGET_SUPPORTED,
                                'native_picker': os.name == 'nt',
                                'script_export_version': 0 if os.name == 'nt' else 1,
                                'profiles_root': str(self.control.profiles_root),
                                'runtime_user': getpass.getuser(),
                                'speculative_capabilities': self.control.speculative_profiles,
                                'backend_capabilities': launcher_capabilities(self.control.backend_root)['backends'],
                                'state_dir': str(self.control.state_dir),
                                'backend_root': str(self.control.backend_root), 'gpus': self.server.gpus,
                                'draft': self.control.draft(), 'status': self.control.status(),
                                'workers': self.control.worker_defaults(),
                                'native_options': {k: sorted(v | NATIVE_SWITCHES.get(k, set())) for k, v in NATIVE.items()}})
            elif path == '/api/launcher/model-directories':
                self.reply(200, self.control.model_directories())
            elif path == '/api/launcher/status':
                self.reply(200, self.control.status())
            elif path == '/api/launcher/services':
                self.reply(200, self.control.service_list())
            elif path == '/api/launcher/hardware':
                self.reply(200, self.server.hardware.snapshot())
            elif path == '/api/launcher/log':
                self.reply(200, {'text': self.control.log_tail()})
            elif path == '/api/launcher/logs':
                revision = parse_qs(urlsplit(self.path).query).get('revision', [''])[0]
                self.reply(200, self.control.log_snapshot(revision))
            elif path in ('/api/monitor/snapshot', '/props', '/slots', '/stats'):
                state = self.control.status()
                if not state['url']:
                    self.reply(503, {'error': 'No managed service is running'}); return
                # Fixed read-only routes to the owned service, never an arbitrary URL.
                request = urllib.request.Request(state['url'] + path)
                authorization = self.control.monitor_authorization() or self.headers.get('Authorization')
                if authorization:
                    request.add_header('Authorization', authorization)
                try:
                    with urllib.request.urlopen(request, timeout=5) as response:
                        body = response.read(4 * 1024 * 1024 + 1)
                        if len(body) > 4 * 1024 * 1024:
                            raise ValueError('Monitor response too large')
                        self.reply(response.status, body)
                except urllib.error.HTTPError as error:
                    self.reply(error.code, {'error': 'Inference endpoint unavailable'})
                    error.close()
            elif path in ('/', '/monitor'):
                self.send_response(302)
                self.send_header('Location', '/monitor/')
                self.send_header('Content-Length', '0')
                self.end_headers()
            elif path in ('/monitor/', '/chat/'):
                self.asset('index.html')
            elif path.startswith('/monitor/') and '/' not in path[len('/monitor/'):]:
                self.asset(path[len('/monitor/'):])
            else:
                self.reply(404, {'error': 'Not found'})
        except (ValueError, OSError, urllib.error.URLError) as error:
            self.reply(503, {'error': str(error)})

    def asset(self, name):
        allowed = {'index.html', 'monitor.css', 'monitor.js', 'i18n.js', 'launcher.css', 'launcher.js',
                   'Strata-LICENSE.txt', 'Outfit-OFL.txt', 'outfit-latin-wght.woff2', 'console-shell.js', 'console-shell.css', 'history-export.html', 'logs.js', 'logs.css'}
        if name not in allowed:
            self.reply(404, {'error': 'Not found'}); return
        file = ROOT / 'ui/monitor' / name
        self.reply(200, file.read_bytes(), media_type(file.name))

    def chat_asset(self, name):
        roots = [ROOT/'ui/dist', self.control.backend_root/'workers/share/kvmem/ui', ROOT/'build/share/kvmem/ui']
        root = next((p for p in roots if (p/'index.html').is_file()), None)
        if root is None:
            self.reply(503, {'error':'Unified chat assets missing; rebuild the UI package'}); return
        root = root.resolve()
        file = (root/(name or 'index.html')).resolve()
        if root not in file.parents or not file.is_file() or file.suffix not in {'.html','.js','.css','.svg','.png','.ico','.woff','.woff2','.json'}:
            self.reply(404, {'error':'Not found'}); return
        self.reply(200,file.read_bytes(),media_type(file.name))

    def do_POST(self):
        if not self.trusted(mutate=True):
            return
        try:
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                raise ValueError('Expected application/json')
            size = int(self.headers.get('Content-Length', '0'))
            is_chat = self.path == '/api/chat/completions'
            if not 0 < size <= (MAX_BODY if is_chat else 65536):
                raise ValueError('Invalid request size')
            data = json.loads(self.rfile.read(size))
            if is_chat:
                self.control.chat.stream(self, data); return
            if self.path == '/api/chat/cancel':
                self.reply(200,self.control.chat.cancel(str(data.get('id','')))); return
            actions = {'/api/launcher/save': self.control.save, '/api/launcher/preview': self.control.plan,
                       '/api/launcher/export': self.control.export_config,
                       '/api/launcher/start-revision': self.control.start_revision,
                       '/api/launcher/service': self.control.service_action,

                       '/api/launcher/adjust-budget': self.control.adjust_budget,
                       '/api/launcher/browse-model': self.control.browse_model,
                       '/api/launcher/browse-mmproj': lambda data: self.control.browse_model(data, projector=True),
                       '/api/launcher/browse-template': lambda data: self.control.browse_model(data, template=True),
                       '/api/launcher/browse-model-folder': lambda data: self.control.browse_model(data, folder=True),
                       '/api/launcher/scan-models': self.control.scan_models,
                       '/api/launcher/inspect-model': self.control.inspect_model,
                       '/api/launcher/model-directories': self.control.update_model_directories,
                       '/api/launcher/start': self.control.start, '/api/launcher/retry-memory': self.control.retry_memory,
                       '/api/launcher/stop': lambda _: self.control.stop()}
            action = actions.get(self.path)
            if action is None:
                self.reply(404, {'error': 'Not found'}); return
            self.reply(200, action(data))
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            self.reply(400, {'error': self.control.redact(str(error))})


def open_console_browser(url):
    try:
        if webbrowser.open(url, new=2):
            return
    except (OSError, webbrowser.Error):
        pass
    print(f'Open this address in your browser: {url}', flush=True)


def console_host(value):
    if value == 'localhost':
        return '127.0.0.1'
    try:
        address = ipaddress.IPv4Address(value)
        if address.is_multicast or str(address) == '255.255.255.255':
            raise ValueError()
        return str(address)
    except ValueError:
        raise argparse.ArgumentTypeError('Use an IPv4 address, such as 127.0.0.1 or 0.0.0.0') from None


def console_url(host, port, language='zh-CN'):
    address = '127.0.0.1' if host == '0.0.0.0' else host
    return f'http://{address}:{port}/monitor/' + ('?lang=en' if language == 'en' else '')


def choose_console_address(host, port, language='zh-CN'):
    english = language == 'en'
    print('Console address: 127.0.0.1 = this computer; 0.0.0.0 = all interfaces (LAN access).'
          if english else '控制台地址：127.0.0.1 仅本机；0.0.0.0 所有网卡（允许局域网访问）。', flush=True)
    print('Press Enter to keep each default, or type q to cancel.'
          if english else '直接回车保留默认值，输入 q 取消启动。', flush=True)
    try:
        while True:
            value = input(f'Listen IP [{host}]: ' if english else f'监听 IP [{host}]：').strip()
            if value.lower() in ('q', 'quit', 'exit'):
                return None
            try:
                host = console_host(value or host)
                break
            except argparse.ArgumentTypeError:
                print('Enter a valid IPv4 address.' if english else '请输入有效的 IPv4 地址。', flush=True)
        while True:
            value = input(f'Console port [{port}]: ' if english else f'控制台端口 [{port}]：').strip()
            if value.lower() in ('q', 'quit', 'exit'):
                return None
            try:
                selected = int(value or str(port))
                if 1 <= selected <= 65535:
                    return host, selected
            except ValueError:
                pass
            print('Port must be an integer from 1 to 65535.' if english else '端口必须是 1 到 65535 之间的整数。', flush=True)
    except (EOFError, KeyboardInterrupt):
        return None


def matching_console(port, backend_root, state_dir, prefer_bundled_worker=False, profiles_root=None, host='127.0.0.1'):
    """Only reuse a controller from this installation and the same state directory."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        address = '127.0.0.1' if host == '0.0.0.0' else host
        with opener.open(f'http://{address}:{port}/api/launcher/info', timeout=2) as response:
            body = response.read(262145)
            if len(body) > 262144:
                return False
            data = json.loads(body)
        if not isinstance(data, dict):
            return False
        return (data.get('console_protocol') == 1
                and data.get('listen_host', '127.0.0.1') == host
                and Path(data['console_root']).resolve() == ROOT.resolve()
                and Path(data['backend_root']).resolve() == Path(backend_root).resolve()
                and Path(data['state_dir']).resolve() == Path(state_dir).resolve()
                and Path(data.get('profiles_root', Path(data['state_dir']) / 'profiles')).resolve()
                    == Path(profiles_root or Path(state_dir) / 'profiles').resolve()
                and bool(data.get('prefer_bundled_worker')) == bool(prefer_bundled_worker))
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError):
        return False


def choose_console_port(current_port, language='zh-CN'):
    english = language == 'en'
    if english:
        label = 'Default port' if current_port == 18199 else 'Port'
        print(f'{label} {current_port} is unavailable. It may be in use or reserved by the system.', flush=True)
    else:
        label = '默认端口' if current_port == 18199 else '端口'
        print(f'{label} {current_port} 不可用，可能已被占用或被系统保留。', flush=True)
    prompt = ('Enter another port (1-65535, e.g. 18198; press Enter or type q to cancel): '
              if english else '请输入其他端口（1–65535，例如 18198；直接回车或输入 q 退出）：')
    cancelled = 'Startup cancelled.' if english else '已取消启动。'
    while True:
        try:
            value = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print('\n' + cancelled, flush=True)
            return None
        if not value or value.lower() in ('q', 'quit', 'exit'):
            print(cancelled, flush=True)
            return None
        try:
            port = int(value)
        except ValueError:
            port = 0
        if not 1 <= port <= 65535:
            print('Port must be an integer from 1 to 65535.' if english else '端口必须是 1 到 65535 之间的整数。', flush=True)
        elif port == current_port:
            print(f'Choose a port other than {current_port}.' if english else f'请选择与 {current_port} 不同的端口。', flush=True)
        else:
            return port


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18199)
    parser.add_argument('--host', type=console_host, default='127.0.0.1', help='console listen IPv4 address; 0.0.0.0 allows LAN access')
    parser.add_argument('--prompt-address', action='store_true', help='confirm console IP and port before starting')
    parser.add_argument('--backend-root', type=Path, default=ROOT)
    default_state = ROOT / 'build/local-console' if os.name == 'nt' else Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local/state')) / 'kvmem/console'
    parser.add_argument('--state-dir', type=Path, default=default_state)
    parser.add_argument('--profiles-root', type=Path, default=None if os.name == 'nt' else Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'kvmem/profiles', help='Generated versions; Linux defaults to XDG_CONFIG_HOME/kvmem/profiles')
    parser.add_argument('--open-browser', action='store_true', help='open the UI; reuse an identical running console')
    parser.add_argument('--prefer-bundled-worker', action='store_true',
                        help='prefer a worker shipped inside this installation over saved external paths')
    parser.add_argument('--prompt-port', action='store_true', help='ask for another console port when unavailable')
    parser.add_argument('--language', choices=('zh-CN', 'en'), default='zh-CN',
                        help='startup message language; en also opens the UI in English')
    args = parser.parse_args()
    if args.prompt_address:
        address = choose_console_address(args.host, args.port, args.language)
        if address is None:
            return
        args.host, args.port = address
    # A unified Windows package stores byte-identical CUDA DLLs once. The
    # console, capability probes and owned worker trees inherit this local path.
    cuda_runtime = args.backend_root.resolve() / 'runtime/cuda'
    if os.name == 'nt' and cuda_runtime.is_dir():
        os.environ['PATH'] = str(cuda_runtime) + os.pathsep + os.environ.get('PATH', '')
    control = Controller(args.backend_root, args.state_dir, args.port, args.prefer_bundled_worker)
    if args.profiles_root:
        control.profiles_root = args.profiles_root.resolve()
    while True:
        url = console_url(args.host, args.port, args.language)
        try:
            server = Server((args.host, args.port), Handler)
            break
        except OSError as error:
            if ((args.open_browser or args.prompt_port or args.prompt_address)
                    and matching_console(args.port, args.backend_root, args.state_dir,
                                         args.prefer_bundled_worker, control.profiles_root, host=args.host)):
                print(f'KVMem console is already running: {url}', flush=True)
                if args.open_browser:
                    open_console_browser(url)
                return
            if args.prompt_address:
                print((f'Cannot listen on {args.host}:{args.port}: {error}. Choose another IP or port.' if args.language == 'en'
                       else f'无法监听 {args.host}:{args.port}：{error}。请更换本机 IP 或端口。'), flush=True)
                address = choose_console_address(args.host, args.port, args.language)
                if address is None:
                    return
                args.host, args.port = address
                continue
            if not args.prompt_port:
                parser.exit(1, f'Cannot open console port {args.port}: {error}. '
                            'Close the other application or choose --port with a free port.\n')
            port = choose_console_port(args.port, args.language)
            if port is None:
                return
            args.port = port
    control.port = server.server_port
    server.control, server.gpus = control, gpu_inventory()
    server.hardware = SystemDiskSampler().start()
    url = console_url(args.host, server.server_port, args.language)
    print(f'KVMem console: {url}', flush=True)
    if args.host != '127.0.0.1':
        print(f'Listening: {args.host}:{server.server_port}', flush=True)
        if args.host == '0.0.0.0':
            print((f'LAN: use this computer\'s LAN IP and port {server.server_port} in your browser.' if args.language == 'en'
                   else f'局域网访问：请在浏览器中使用本机的局域网 IP 和端口 {server.server_port}。'), flush=True)
    if args.language == 'en':
        print('\nHow to stop:\n'
              '  Keep this window open while using the console.\n'
              '  Close this window or press Ctrl+C to stop the console and temporary services; systemd services continue running.\n'
              '  Click "Stop service" in the UI to stop inference while keeping the console open.\n'
              '  Closing only the browser leaves services running.\n'
              '  Backends started separately are unaffected.\n', flush=True)
    else:
        print('\n使用说明：\n'
              '  保持此窗口开启，控制台才能持续运行。\n'
              '  关闭此窗口或按 Ctrl+C：停止控制台和临时服务；systemd 后台服务保持运行。\n'
              '  页面点击“停止服务”：只停止推理，控制台继续保留。\n'
              '  只关闭浏览器：服务继续运行。\n'
              '  另外单独启动的后端服务不受影响。\n', flush=True)
    if args.open_browser:
        threading.Thread(target=open_console_browser, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.hardware.close()
        control.close(); server.server_close()


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker-gate']:
        argv = json.loads(sys.stdin.buffer.readline(65536))
        sys.exit(subprocess.call(argv, stdin=subprocess.DEVNULL,
                                 creationflags=subprocess.CREATE_NO_WINDOW))
    else:
        main()
