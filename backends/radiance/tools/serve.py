"""Supervised local launch with unchanged capacity on Windows or Linux/WSL."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from platform_dispatch import select_platform

GIB = 1024 ** 3
WSL_RUNTIME_SHA256 = '2b2a082ede46324a5e3771b4b14f2850fdb9bd38348abfbeedee83966bbd6088'


def profile():
    return json.loads((Path(__file__).resolve().parents[1] / 'windows-profile.json').read_text(encoding='utf-8'))


def command_for(a, config):
    return [str(a.exe.resolve()), '--model', str(a.model.resolve()), '--radiance-home', str(a.home.resolve()),
            '--host', '127.0.0.1', '--port', str(a.port), '--served-model-name', 'radiance-local', *config['arguments']]


def validate_runtime(directory):
    for name in ('librocdxg.so.1.2.0', 'librocdxg.so.1', 'librocdxg.so'):
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != WSL_RUNTIME_SHA256:
            raise ValueError(f'{path}: expected the historically qualified scratch runtime and valid aliases; see wsl-runtime/README.md')


def validate_inputs(a, host, wsl):
    suffix = 'dll' if host == 'windows' else 'so'
    required = [a.exe, a.model, a.home / 'kernels.rkb', a.home / f'kernels/libr4d.{suffix}',
                a.home / f'architectures/qwen35_fp8.{suffix}']
    required += ([a.rocm / 'bin/amdhip64_7.dll'] if host == 'windows' else
                 [a.rocm / 'lib/libamdhip64.so', a.rocm / 'lib/libhsa-runtime64.so'])
    for path in required:
        if not path.is_file():
            raise ValueError(f'missing input: {path}')
    if host == 'linux':
        for path in required:
            if path == a.model or path.name == 'kernels.rkb':
                continue
            with path.open('rb') as stream:
                if stream.read(4) != b'\x7fELF':
                    raise ValueError(f'not a Linux ELF executable/library: {path}')
    if wsl:
        if not a.wsl_runtime_dir:
            raise ValueError('WSL requires --wsl-runtime-dir with the qualified scratch runtime; system fallback is disabled')
        validate_runtime(a.wsl_runtime_dir)
    elif a.wsl_runtime_dir:
        raise ValueError('--wsl-runtime-dir is only valid on WSL (/dev/dxg)')


def child_environment(a, host, wsl, config, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    for key in list(env):
        if key.startswith(('RADIANCE_', 'KVMEM_', 'KV_DIAG_')) or key in config['environment_reset']:
            env.pop(key, None)
    for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'HSA_ENABLE_DXG_DETECTION'):
        env.pop(key, None)
    env.update(ROCM_PATH=str(a.rocm.resolve()), HIP_PATH=str(a.rocm.resolve()))
    if host == 'windows':
        env.update(config['environment'])
        env['PATH'] = str(a.rocm.resolve() / 'bin') + os.pathsep + env.get('PATH', '')
    else:
        env.update(KVMEM_RAD_MTP_HEAD='shared', KVMEM_CAPTURE_EARLY_SEAL='1', OMP_NUM_THREADS='2')
        libraries = [a.rocm.resolve() / 'lib', a.rocm.resolve() / 'llvm/lib']
        if wsl:
            env['HSA_ENABLE_DXG_DETECTION'] = '1'
            libraries.insert(0, a.wsl_runtime_dir.resolve())
        env['LD_LIBRARY_PATH'] = ':'.join(map(str, libraries))
    return env


def proc_kib(text, name):
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == name + ':':
            if len(parts) != 3 or parts[2] != 'kB' or int(parts[1]) < 0:
                break
            return int(parts[1]) * 1024
    raise RuntimeError(f'missing/invalid Linux resource counter: {name}')


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
        (key, ctypes.c_ulonglong) for key in
        ('total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended')]


def available_memory(host):
    if host == 'linux':
        return (proc_kib(Path('/proc/meminfo').read_text(), 'MemAvailable'),)
    state = MemoryStatus()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise ctypes.WinError()
    return state.avail_phys, state.avail_page


def child_swap(pid):
    return proc_kib(Path(f'/proc/{pid}/status').read_text(), 'VmSwap')


def check_resources(host, startup=False, pid=None):
    threshold = (10 if startup else 2) * GIB
    if min(available_memory(host)) < threshold:
        raise RuntimeError(f'available RAM/commit below {threshold // GIB} GiB; capacity is unchanged')
    if host == 'linux' and pid is not None and child_swap(pid) > 64 * 1024 ** 2:
        raise RuntimeError('model child swap exceeds 64 MiB; capacity is unchanged')


def supervise(command, env, host):
    check_resources(host, startup=True)
    process = subprocess.Popen(command, env=env)
    try:
        while process.poll() is None:
            try:
                check_resources(host, pid=process.pid)
            except (FileNotFoundError, RuntimeError):
                if process.poll() is not None:
                    break
                raise
            time.sleep(1)
        return process.returncode
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform', choices=('auto', 'windows', 'linux'), default='auto')
    for name in ('exe', 'model', 'home', 'rocm'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--wsl-runtime-dir', type=Path)
    p.add_argument('--port', type=int, default=18380)
    p.add_argument('--print-command', '--dry-run', action='store_true')
    a = p.parse_args(argv)
    try:
        host = select_platform(a.platform)
        if not 1 <= a.port <= 65535:
            raise ValueError('port must be in 1..65535')
        wsl = host == 'linux' and Path('/dev/dxg').exists()
        validate_inputs(a, host, wsl)
        config = profile()
        env = child_environment(a, host, wsl, config)
        command = command_for(a, config)
        print(subprocess.list2cmdline(command) if host == 'windows' else shlex.join(command), flush=True)
        if a.print_command:
            return 0
        return supervise(command, env, host)
    except (ValueError, RuntimeError, OSError) as error:
        p.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
