"""Backend-independent system disk telemetry, sampled once for all UI clients.

Uses PolyStrata's psutil cumulative-byte delta approach. Durations use a monotonic
clock; wall time is only for snapshot freshness. Engine cache I/O stays separate.
"""
import threading
import time
import re
import os
import subprocess
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


def amd_gpu_inventory():
    # The optional ROCm inventory tool initializes no model or inference queue.
    # Runtime paths may be versioned; try the explicitly selected installation.
    root=os.environ.get('ROCM_PATH') or os.environ.get('HIP_PATH')
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

    def sample(self):
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
        with self.lock:
            self.value = value

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
