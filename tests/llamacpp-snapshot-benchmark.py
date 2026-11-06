"""Measure the llama checkpoint workload on Windows without changing its requests.

Pass the arguments of checkpoint-coverage-regression.py except --backend and
--timing-only, which this wrapper fixes. Native prefill/decode timings are read
from the server log; GPU framebuffer peaks are sampled, Windows working-set
peaks are kernel high-water counters. This short-output workload is diagnostic,
not a long-decode throughput benchmark. Run identical arguments for both builds.
"""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import threading
import time
import urllib.request


class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            'peak_working_set', 'working_set', 'peak_paged_pool', 'paged_pool',
            'peak_nonpaged_pool', 'nonpaged_pool', 'pagefile', 'peak_pagefile', 'private_bytes')]


def main():
    if os.name != 'nt':
        raise SystemExit('This sampler is for the authorized Windows GPU host.')
    arguments = sys.argv[1:]
    if any(x in arguments for x in ('--backend', '--timing-only')):
        raise SystemExit('The wrapper fixes --backend llama --timing-only.')
    def option(name):
        return arguments[arguments.index(name) + 1]
    output = Path(option('--out')).resolve()
    executable = Path(option('--exe')).resolve()
    gpu = option('--gpu')
    original_popen = subprocess.Popen
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    monitor = []
    samplers = []

    class MeasuredProcess(original_popen):
        def __init__(self, args, *positional, **kwargs):
            super().__init__(args, *positional, **kwargs)
            self.measured = isinstance(args, (list, tuple)) and Path(args[0]).resolve() == executable
            if not self.measured:
                return
            self.finished = threading.Event()
            self.started = time.monotonic()
            port = args[args.index('--port') + 1]
            self.url = 'http://127.0.0.1:' + str(port) + '/api/monitor/snapshot'
            self.thread = threading.Thread(target=self.sample, daemon=True)
            samplers.append(self)
            self.thread.start()

        def collect(self, include_gpu=True):
            sample = {'seconds': time.monotonic() - self.started}
            memory = ProcessMemory()
            memory.cb = ctypes.sizeof(memory)
            if psapi.GetProcessMemoryInfo(int(self._handle), ctypes.byref(memory), memory.cb):
                sample.update(rss_bytes=memory.working_set, peak_rss_bytes=memory.peak_working_set,
                              private_bytes=memory.private_bytes, peak_commit_bytes=memory.peak_pagefile)
            else:
                sample['memory_error'] = ctypes.get_last_error()
            if include_gpu:
                try:
                    result = subprocess.check_output([
                        'nvidia-smi', '-i', gpu, '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                        text=True, timeout=3, creationflags=0x08000000)
                    sample['gpu_framebuffer_mib'] = int(result.strip())
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    sample['gpu_error'] = str(error)
            try:
                with opener.open(self.url, timeout=.5) as response:
                    sample['monitor'] = json.load(response).get('kvmem', {})
            except (OSError, ValueError) as error:
                sample['monitor_unavailable'] = str(error)
            monitor.append(sample)

        def sample(self):
            while not self.finished.is_set() and self.poll() is None:
                self.collect()
                self.finished.wait(.2)

        def terminate(self):
            if self.measured:
                # Capture final physical I/O/account counters before shutdown.
                self.finished.set()
                self.thread.join(5)
                if self.poll() is None:
                    self.collect()
            return super().terminate()

    subprocess.Popen = MeasuredProcess
    workload = Path(__file__).with_name('checkpoint-coverage-regression.py')
    sys.argv = [str(workload), '--backend', 'llama', '--timing-only', *arguments]
    success = False
    try:
        runpy.run_path(str(workload), run_name='__main__')
        success = True
    finally:
        for sampler in samplers:
            sampler.finished.set()
            sampler.thread.join(5)
        subprocess.Popen = original_popen
        if output.exists():
            native = (output / 'server.log').read_text(encoding='utf-8', errors='replace') if (output / 'server.log').exists() else ''
            timings = [dict(phase=phase, milliseconds=float(ms), tokens=int(tokens), tokens_per_second=float(rate))
                       for phase, ms, tokens, rate in re.findall(
                           r'slot\s+(prompt eval|eval) time\s*=\s*([\d.]+) ms /\s*(\d+) tokens \(\s*([\d.]+) tokens per second\)', native)]
            report = {'completed': success, 'sample_interval_floor_seconds': .2,
                      'gpu_peak_scope': 'sampled total device framebuffer; not per-process or exact high-water',
                      'rss_peak_scope': 'server Windows PeakWorkingSetSize; excludes subprocesses and driver allocations',
                      'peak_rss_bytes': max((s.get('peak_rss_bytes', 0) for s in monitor), default=0),
                      'peak_commit_bytes': max((s.get('peak_commit_bytes', 0) for s in monitor), default=0),
                      'sampled_peak_gpu_mib': max((s.get('gpu_framebuffer_mib', 0) for s in monitor), default=0),
                      'native_timings': timings, 'samples': monitor}
            (output / 'resources.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            if success and (not timings or not report['peak_rss_bytes'] or not report['sampled_peak_gpu_mib']):
                raise RuntimeError('Benchmark completed but timing/resource evidence is incomplete')


if __name__ == '__main__':
    main()
