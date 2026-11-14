"""Isolated, read-only AMD ADLX 1.5 hardware sampling on Windows.

C ABI: GPUOpen-LibrariesAndSDKs/ADLX, tag v1.5 (d9f04a9b).
Only the driver DLL in System32 is loaded. No SDK or tuning service is used.
"""
import ctypes as c
import json
import math


def call(obj, slot, types=(), *args, result=c.c_int):
    if not obj: raise OSError('Missing ADLX interface')
    table = c.cast(obj, c.POINTER(c.POINTER(c.c_void_p))).contents
    fn = c.WINFUNCTYPE(result, c.c_void_p, *types)(table[slot])
    return fn(obj, *args)


def release(obj):
    if obj: call(obj, 1)


def output(obj, slot, value_type, *prefix):
    value = value_type()
    types = tuple(c.c_void_p for _ in prefix) + (c.POINTER(value_type),)
    status = call(obj, slot, types, *prefix, c.byref(value))
    if status != 0: raise OSError(f'ADLX query {slot} failed ({status})')
    return value


class AmdReader:
    def __init__(self):
        self.dll = c.CDLL('amdadlx64.dll', winmode=0x800)
        self.system, self.mapping, self.performance = c.c_void_p(), c.c_void_p(), c.c_void_p()
        self.initialized = False
        terminate = getattr(self.dll, 'ADLXTerminate', None)
        # The normal entry can succeed while returning an empty GPU list.
        # Prefer AMD's compatibility entry for read-only, capability-checked
        # telemetry. Initialize only once: terminate/reinitialize in one process
        # can fault inside this driver, even after releasing every interface.
        init = getattr(self.dll, 'ADLXInitializeWithIncompatibleDriver2', None)
        if init is None: init = getattr(self.dll, 'ADLXInitialize2', None)
        if init is None or terminate is None: raise OSError('AMD ADLX entry points unavailable')
        terminate.argtypes, terminate.restype = [], c.c_int
        self.initialize(init)
        try:
            self.performance = output(self.system, 9, c.c_void_p)
        except BaseException:
            self.close(); raise

    def initialize(self, init):
        self.system, self.mapping = c.c_void_p(), c.c_void_p()
        init.argtypes = [c.c_uint64, c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)]
        init.restype = c.c_int
        status = init(0x000100050000007c, c.byref(self.system), c.byref(self.mapping))
        if status != 0: raise OSError(f'AMD ADLX initialization failed ({status})')
        self.initialized = True

    def sample(self):
        devices = output(self.system, 1, c.c_void_p)
        rows = []
        try:
            count = call(devices, 3, result=c.c_uint)
            if count > 256: raise OSError('Invalid ADLX GPU count')
            for index in range(count):
                gpu = c.c_void_p()
                if call(devices, 11, (c.c_uint, c.POINTER(c.c_void_p)), index, c.byref(gpu)) != 0:
                    continue
                try:
                    try: rows.append(self.gpu_sample(gpu))
                    except OSError: pass
                finally: release(gpu)
        finally: release(devices)
        return rows

    def gpu_sample(self, gpu):
        bus, device, function = c.c_int(), c.c_int(), c.c_int()
        status = call(self.mapping, 2, (c.c_void_p, c.POINTER(c.c_int), c.POINTER(c.c_int), c.POINTER(c.c_int)),
                      gpu, c.byref(bus), c.byref(device), c.byref(function))
        if status or not (0 <= bus.value <= 255 and 0 <= device.value <= 31 and 0 <= function.value <= 7):
            raise OSError('ADLX GPU PCI identity unavailable')
        name = (output(gpu, 7, c.c_char_p).value or b'').decode('utf8', errors='replace')
        total = output(gpu, 11, c.c_uint).value * 1024**2
        row = dict(pci_bus_id=f'0000:{bus.value:02x}:{device.value:02x}.{function.value}',
                   name=name, vendor='amd', source='amd-adlx', total_bytes=total or None,
                   used_bytes=None, utilization_percent=None, temperature_c=None, power_w=None)
        support, metrics = c.c_void_p(), c.c_void_p()
        try:
            support = output(self.performance, 21, c.c_void_p, gpu)
            metrics = output(self.performance, 18, c.c_void_p, gpu)
            # metric getter slot, support predicate slot, C type, scale, valid range.
            fields = {'utilization_percent': (4, 3, c.c_double, 1, 100),
                      'temperature_c': (7, 6, c.c_double, 1, 150),
                      'power_w': (10, 9, c.c_double, 1, 2000),
                      'used_bytes': (12, 11, c.c_int, 1024**2, total)}
            for key, (getter, predicate, kind, scale, maximum) in fields.items():
                try:
                    if not output(support, predicate, c.c_bool).value: continue
                    value = output(metrics, getter, kind).value * scale
                    if math.isfinite(value) and 0 <= value <= maximum: row[key] = value
                except OSError: pass
            if row['power_w'] is not None: row['power_source'] = 'adlx-total-board-power'
        finally:
            release(metrics); release(support)
        return row

    def close(self):
        release(self.performance); self.performance = c.c_void_p()
        if self.initialized:
            self.dll.ADLXTerminate(); self.initialized = False


if __name__ == '__main__':
    reader = AmdReader()
    try: print(json.dumps(reader.sample()), flush=True)
    finally: reader.close()
