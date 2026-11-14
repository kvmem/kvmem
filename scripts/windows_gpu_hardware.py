"""Whole-adapter WDDM counters, joined to DXGI adapters by their exact LUID.

Windows SDK d3dkmthk.h / PDH / IDXGIAdapter1 ABI. WDDM Power is a percentage,
not watts, and is deliberately never exposed as board power.
"""
import ctypes as c
import math
import re
import uuid


class Luid(c.Structure):
    _fields_ = [('low', c.c_uint32), ('high', c.c_int32)]


class Description(c.Structure):
    _fields_ = [('name', c.c_wchar * 128), ('vendor', c.c_uint32), ('device', c.c_uint32),
                ('subsystem', c.c_uint32), ('revision', c.c_uint32),
                ('video', c.c_size_t), ('system', c.c_size_t), ('shared', c.c_size_t),
                ('luid', Luid), ('flags', c.c_uint32)]


class OpenAdapter(c.Structure):
    _fields_ = [('luid', Luid), ('handle', c.c_uint32)]


class Query(c.Structure):
    _fields_ = [('handle', c.c_uint32), ('kind', c.c_int), ('data', c.c_void_p), ('size', c.c_uint32)]


class PerfData(c.Structure):
    _fields_ = [('physical', c.c_uint32), ('memory_frequency', c.c_uint64),
                ('max_memory_frequency', c.c_uint64), ('max_memory_oc', c.c_uint64),
                ('memory_bandwidth', c.c_uint64), ('pcie_bandwidth', c.c_uint64),
                ('fan', c.c_uint32), ('power_percent', c.c_uint32),
                ('temperature', c.c_uint32), ('power_override', c.c_uint8)]


class CounterValue(c.Structure):
    _fields_ = [('status', c.c_uint32), ('value', c.c_double)]


class CounterItem(c.Structure):
    _fields_ = [('name', c.c_wchar_p), ('value', CounterValue)]


def counter_values(items):
    return {name.lower(): value for name, status, value in items
            if name and status in (0, 1) and math.isfinite(value) and value >= 0}


def adapter_usage(values, luid):
    engines = {}
    for name, value in values.items():
        match = re.fullmatch(r'pid_\d+_' + re.escape(luid) + r'_phys_0_eng_(\d+)_engtype_.+', name)
        if match: engines[match[1]] = engines.get(match[1], 0) + value
    return min(100.0, max(engines.values())) if engines else None


def com_call(obj, slot, types=(), *args):
    table = c.cast(obj, c.POINTER(c.POINTER(c.c_void_p))).contents
    return c.WINFUNCTYPE(c.c_int32, c.c_void_p, *types)(table[slot])(obj, *args)


class WddmReader:
    def __init__(self):
        self.adapters, self.query, self.counters = [], c.c_void_p(), {}
        self.gdi = c.WinDLL('gdi32.dll', winmode=0x800)
        self.gdi.D3DKMTOpenAdapterFromLuid.argtypes = [c.POINTER(OpenAdapter)]
        self.gdi.D3DKMTQueryAdapterInfo.argtypes = [c.POINTER(Query)]
        self.gdi.D3DKMTCloseAdapter.argtypes = [c.POINTER(c.c_uint32)]
        self.pdh = c.WinDLL('pdh.dll', winmode=0x800)
        self.pdh.PdhOpenQueryW.argtypes = [c.c_wchar_p, c.c_size_t, c.POINTER(c.c_void_p)]
        self.pdh.PdhAddEnglishCounterW.argtypes = [c.c_void_p, c.c_wchar_p, c.c_size_t, c.POINTER(c.c_void_p)]
        self.pdh.PdhCollectQueryData.argtypes = [c.c_void_p]
        self.pdh.PdhCloseQuery.argtypes = [c.c_void_p]
        self.pdh.PdhGetFormattedCounterArrayW.argtypes = [c.c_void_p, c.c_uint32, c.POINTER(c.c_uint32), c.POINTER(c.c_uint32), c.c_void_p]
        try:
            self.enumerate()
            if self.pdh.PdhOpenQueryW(None, 0, c.byref(self.query)) != 0:
                raise OSError('Windows GPU counters unavailable')
            for key, path in [('memory', r'\GPU Adapter Memory(*)\Dedicated Usage'),
                              ('engine', r'\GPU Engine(*)\Utilization Percentage')]:
                counter = c.c_void_p()
                if self.pdh.PdhAddEnglishCounterW(self.query, path, 0, c.byref(counter)) == 0:
                    self.counters[key] = counter
            self.pdh.PdhCollectQueryData(self.query)
        except BaseException:
            self.close(); raise

    def enumerate(self):
        dxgi = c.WinDLL('dxgi.dll', winmode=0x800)
        factory = c.c_void_p()
        iid = (c.c_ubyte * 16).from_buffer_copy(uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
        dxgi.CreateDXGIFactory1.argtypes = [c.c_void_p, c.POINTER(c.c_void_p)]
        if dxgi.CreateDXGIFactory1(c.byref(iid), c.byref(factory)) != 0:
            raise OSError('Windows GPU enumeration unavailable')
        try:
            for index in range(256):
                adapter = c.c_void_p()
                if com_call(factory, 12, (c.c_uint, c.POINTER(c.c_void_p)), index, c.byref(adapter)) != 0: break
                try:
                    desc = Description()
                    if com_call(adapter, 10, (c.POINTER(Description),), c.byref(desc)) != 0 or desc.vendor != 0x1002: continue
                    opened = OpenAdapter(desc.luid, 0)
                    if self.gdi.D3DKMTOpenAdapterFromLuid(c.byref(opened)) != 0: continue
                    handle = c.c_uint32(opened.handle)
                    address = (c.c_uint32 * 3)()
                    if not self.info(handle, 6, address) or not (address[0] <= 255 and address[1] <= 31 and address[2] <= 7):
                        self.gdi.D3DKMTCloseAdapter(c.byref(handle)); continue
                    self.adapters.append((handle, dict(name=desc.name,
                        luid=f'luid_0x{desc.luid.high & 0xffffffff:08x}_0x{desc.luid.low:08x}',
                        pci_bus_id=f'0000:{address[0]:02x}:{address[1]:02x}.{address[2]}',
                        total_bytes=desc.video, vendor='amd', source='windows-wddm')))
                finally: com_call(adapter, 2)
        finally: com_call(factory, 2)

    def info(self, handle, kind, data):
        query = Query(handle.value, kind, c.addressof(data), c.sizeof(data))
        return self.gdi.D3DKMTQueryAdapterInfo(c.byref(query)) == 0

    def read(self, key):
        counter = self.counters.get(key)
        if not counter: return {}
        size, count = c.c_uint32(), c.c_uint32()
        flags = 0x200 | 0x8000  # PDH_FMT_DOUBLE | PDH_FMT_NOCAP100
        status = self.pdh.PdhGetFormattedCounterArrayW(counter, flags, c.byref(size), c.byref(count), None)
        if (status & 0xffffffff) != 0x800007d2 or not 0 < size.value <= 4 * 1024**2: return {}
        buffer = c.create_string_buffer(size.value)
        if self.pdh.PdhGetFormattedCounterArrayW(counter, flags, c.byref(size), c.byref(count), buffer) != 0: return {}
        if count.value > len(buffer) // c.sizeof(CounterItem): return {}
        items = c.cast(buffer, c.POINTER(CounterItem))
        return counter_values((items[i].name, items[i].value.status, items[i].value.value) for i in range(count.value))

    def sample(self):
        # Missing PDH instances must not suppress independent D3DKMT temperature.
        collected = self.pdh.PdhCollectQueryData(self.query) == 0
        memory, engines = (self.read('memory'), self.read('engine')) if collected else ({}, {})
        rows = []
        for handle, identity in self.adapters:
            row = dict(identity, used_bytes=None, utilization_percent=adapter_usage(engines, identity['luid']),
                       temperature_c=None, power_w=None)
            used = memory.get(identity['luid'] + '_phys_0')
            if used is not None and used <= identity['total_bytes']: row['used_bytes'] = int(used)
            perf = PerfData()
            if self.info(handle, 62, perf) and 0 < perf.temperature <= 1500:
                row['temperature_c'] = perf.temperature / 10.0
            rows.append(row)
        return rows

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query); self.query = c.c_void_p()
        for handle, _ in self.adapters: self.gdi.D3DKMTCloseAdapter(c.byref(handle))
        self.adapters = []


def run_helper():
    import json,os,sys,time,threading
    from pathlib import Path
    if '--parent-pipe' in sys.argv:
        # EOF also arrives if the console is forcibly terminated. Keep this on a
        # separate thread so even a blocked driver call cannot orphan the helper.
        def parent_closed():
            sys.stdin.buffer.read(1)
            os._exit(0)
        threading.Thread(target=parent_closed, daemon=True).start()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from windows_amd_hardware import AmdReader
    readers = []
    try:
        for factory in (WddmReader, AmdReader):
            try: readers.append(factory())
            except OSError: pass
        for _ in range(60):  # Re-enumerate periodically, including hotplug and driver resets.
            rows = {}
            for reader in readers:
                try: samples = reader.sample()
                except OSError: continue
                for sample in samples:
                    pci = sample['pci_bus_id']
                    if pci not in rows: rows[pci] = sample
                    else:
                        for key, value in sample.items():
                            if value is not None: rows[pci][key] = value
            for row in rows.values():
                row['unavailable_metrics'] = [k for k in ('used_bytes', 'utilization_percent', 'temperature_c', 'power_w') if row.get(k) is None]
            print(json.dumps(dict(sampled_at_ms=time.time()*1000, gpus=list(rows.values()))), flush=True)
            time.sleep(1)
    finally:
        for reader in reversed(readers): reader.close()


if __name__ == '__main__':
    run_helper()
