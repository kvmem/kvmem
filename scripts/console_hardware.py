"""Backend-independent system telemetry, sampled once for all UI clients.

Uses PolyStrata's psutil cumulative-byte delta approach. Durations use a monotonic
clock; wall time is only for snapshot freshness. Engine cache I/O stays separate.
"""
import threading
import time
import re
import os
import json
import subprocess
import sys
import copy
import queue
import math
from pathlib import Path
from threading import Thread

try:
    import psutil
except ImportError:
    psutil = None


def parse_rocr_agents(text):
    """Use the runtime's own UUID spelling, never an AMD-SMI UUID or a GPU index."""
    devices=[]
    for agent in re.split(r'(?m)^Agent\s+\d+\s*$',text):
        def value(name):
            found=re.search(r'(?m)^\s*'+re.escape(name)+r':\s*(.*?)\s*$',agent)
            return found.group(1) if found else ''
        uuid=value('Uuid')
        if value('Device Type')!='GPU' or not re.fullmatch(r'GPU-[0-9a-fA-F]{16}',uuid):continue
        pools=re.findall(r'Segment:\s*GLOBAL; FLAGS: COARSE GRAINED\s+Size:\s*(\d+)',agent)
        devices.append(dict(uuid=uuid,name=value('Marketing Name') or value('Name'),
            memory_mib=str(int(pools[0])//1024) if pools else None,vendor='amd'))
    return devices


def windows_hip_inventory(root=None):
    """Enumerate the unmasked SDK in a short-lived process; the console owns no HIP context."""
    root = root or os.environ.get('ROCM_PATH') or os.environ.get('HIP_PATH')
    if not root:
        return []
    library = Path(root).resolve() / 'bin/amdhip64_7.dll'
    if not library.is_file():
        return []
    env = dict(os.environ)
    for key in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
        env.pop(key, None)
    try:
        run = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--hip-devices', str(library)],
                             capture_output=True, env=env, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
        return json.loads(run.stdout) if run.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return []


def read_hip_devices(library):
    import ctypes as c
    with os.add_dll_directory(str(library.parent)):
        hip = c.WinDLL(str(library))
        count = c.c_int()
        devices = []
        if hip.hipGetDeviceCount(c.byref(count)) != 0:
            return devices
        for ordinal in range(count.value):
            uuid = (c.c_ubyte * 16)()
            name = c.create_string_buffer(256)
            size = c.c_size_t()
            if hip.hipDeviceGetUuid(c.byref(uuid), ordinal) or not any(uuid):
                continue
            if hip.hipDeviceGetName(name, len(name), ordinal):
                continue
            memory_ok = hip.hipDeviceTotalMem(c.byref(size), ordinal) == 0
            row = dict(uuid='GPU-' + bytes(uuid).hex(), name=name.value.decode(errors='replace'),
                                memory_mib=str(size.value // 2**20) if memory_ok else None,
                                vendor='amd', ordinal=ordinal)
            pci = c.create_string_buffer(64)
            if hip.hipDeviceGetPCIBusId(pci, len(pci), ordinal) == 0:
                value = pci.value.decode('ascii', errors='replace')
                if re.fullmatch(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', value):
                    row['pci_bus_id'] = value.lower()
            devices.append(row)
        return devices


def amd_gpu_inventory():
    # The optional ROCm inventory tool initializes no model or inference queue.
    # Runtime paths may be versioned; try the explicitly selected installation.
    root=os.environ.get('ROCM_PATH') or os.environ.get('HIP_PATH')
    if os.name == 'nt':
        devices = windows_hip_inventory(root)
        if devices:
            return devices
    executable=str(Path(root)/'bin'/('rocminfo.exe' if os.name=='nt' else 'rocminfo')) if root else 'rocminfo'
    try:
        result=subprocess.run([executable],capture_output=True,text=True,timeout=5,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return parse_rocr_agents(result.stdout) if result.returncode==0 else []
    except (OSError,subprocess.TimeoutExpired):return []


def linux_disk_counters():
    # Whole physical devices only: exclude partitions, loop devices and stacked
    # dm/md devices so one transfer is not counted at several layers. System scope.
    from types import SimpleNamespace
    devices=[p.name for p in Path('/sys/block').iterdir() if (p/'device').exists()]
    if not devices:return None
    read=write=0
    for device in devices:
        fields=(Path('/sys/block')/device/'stat').read_text().split()
        read+=int(fields[2])*512;write+=int(fields[6])*512
    return SimpleNamespace(read_bytes=read,write_bytes=write)


class SystemDiskSampler:
    def __init__(self, reader=None, clock=time.monotonic, wall=time.time):
        self.reader = reader or (psutil.disk_io_counters if psutil else linux_disk_counters if Path('/sys/block').is_dir() else None)
        self.source='linux-sysfs-physical-devices' if self.reader is linux_disk_counters else 'psutil'
        self.clock, self.wall = clock, wall
        self.previous = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.value = self._empty('warming_up')

    def _empty(self, status):
        return dict(schema_version=1, scope='system', source=self.source, status=status,
                    sampled_at_ms=None, interval_ms=None,
                    system_disk_read_bytes_per_second=None,
                    system_disk_write_bytes_per_second=None)

    def _sample_disk(self):
        value = self._empty('unavailable' if self.reader else 'dependency_missing')
        current = None
        try:
            counters = self.reader() if self.reader else None
            if counters is not None:
                current = (self.clock(), counters.read_bytes, counters.write_bytes)
        except (OSError, RuntimeError, NotImplementedError):
            pass
        previous, self.previous = self.previous, current
        value['sampled_at_ms'] = self.wall() * 1000
        if current is not None:
            value['status'] = 'warming_up'
            if previous is not None:
                dt = current[0] - previous[0]
                read, write = current[1] - previous[1], current[2] - previous[2]
                # A reset or invalid clock interval establishes a fresh baseline.
                if dt > 0 and read >= 0 and write >= 0:
                    value.update(status='ready', interval_ms=dt * 1000,
                                 system_disk_read_bytes_per_second=read / dt,
                                 system_disk_write_bytes_per_second=write / dt)
        return value

    def sample(self):
        value = self._sample_disk()
        with self.lock: self.value = value

    def snapshot(self):
        with self.lock:
            return dict(self.value)

    def start(self):
        if self.thread is None:
            self.thread = Thread(target=self._loop, name='console-system-disk', daemon=True)
            self.thread.start()
        return self

    def _loop(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(1.0)

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)


class WindowsGpuSampler:
    """Supervised helper: a blocked or crashed driver cannot block console HTTP."""
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.value = {}
        self.thread = None

    def snapshot(self):
        with self.lock: value = copy.deepcopy(self.value)
        stamp = value.get('sampled_at_ms')
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or not -1000 <= time.time()*1000-stamp < 5000:
            return {'gpus': [], 'gpu_status': 'unavailable', 'gpu_sampled_at_ms': None}
        return {'gpus': value.get('gpus', []), 'gpu_status': 'ready' if value.get('gpus') else 'unavailable',
                'gpu_sampled_at_ms': stamp}

    def start(self):
        if self.thread is None:
            self.thread = Thread(target=self._loop, name='console-gpu-supervisor', daemon=True)
            self.thread.start()
        return self

    def _loop(self):
        script = Path(__file__).with_name('windows_gpu_hardware.py')
        while not self.stop_event.is_set():
            process = None
            reader = None
            failed = False
            try:
                process = subprocess.Popen([sys.executable, '-I', str(script), '--parent-pipe'], stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, stdin=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                messages = queue.Queue(maxsize=2)
                def drain(stream, target):
                    while True:
                        line = stream.readline(65537)
                        if not line or len(line) > 65536: break
                        try: target.put_nowait(line)
                        except queue.Full: pass
                reader = Thread(target=drain, args=(process.stdout, messages), daemon=True)
                reader.start()
                last = time.monotonic()
                while not self.stop_event.is_set():
                    try: line = messages.get(timeout=.5)
                    except queue.Empty:
                        if process.poll() is not None:
                            failed = process.returncode != 0
                            break
                        if time.monotonic()-last > 8: raise TimeoutError('GPU sampler stalled')
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict) or not isinstance(value.get('gpus'), list) or len(value['gpus']) > 256:
                        raise ValueError('Invalid GPU sample')
                    with self.lock: self.value = value
                    last = time.monotonic()
            except (OSError, ValueError, TimeoutError):
                failed = True
            finally:
                if process:
                    process.stdin.close()
                    if process.poll() is None: process.kill()
                    try: process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        failed = True
                    if reader: reader.join(timeout=2)
                    if not reader or not reader.is_alive(): process.stdout.close()
            if failed:
                with self.lock: self.value = {}
                self.stop_event.wait(3)

    def close(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=6)


class SystemHardwareSampler(SystemDiskSampler):
    def __init__(self, gpu_sampler=None):
        super().__init__()
        self.gpu_sampler = gpu_sampler if gpu_sampler is not None else (WindowsGpuSampler() if os.name == 'nt' else None)

    def sample(self):
        value = self._sample_disk()
        metrics = {'cpu_percent': None, 'ram_used_bytes': None, 'ram_total_bytes': None, 'ram_available_bytes': None}
        if psutil:
            try:
                metrics['cpu_percent'] = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                metrics.update(ram_total_bytes=memory.total, ram_available_bytes=memory.available,
                               ram_used_bytes=memory.total-memory.available)
            except (OSError, RuntimeError): pass
        value.update(metrics)
        with self.lock: self.value = value

    def snapshot(self):
        with self.lock: value = copy.deepcopy(self.value)
        # GPU freshness is checked at read time, independently of CPU/disk ticks.
        value.update(self.gpu_sampler.snapshot() if self.gpu_sampler else {'gpus': []})
        return value

    def start(self):
        if self.gpu_sampler: self.gpu_sampler.start()
        return super().start()

    def close(self):
        super().close()
        if self.gpu_sampler: self.gpu_sampler.close()


if __name__ == '__main__' and len(sys.argv) == 3 and sys.argv[1] == '--hip-devices':
    print(json.dumps(read_hip_devices(Path(sys.argv[2]))))
