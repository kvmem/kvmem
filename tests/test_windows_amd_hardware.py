import ctypes as c
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import windows_amd_hardware as amd


class AdlxInitializationTests(unittest.TestCase):
    def exercise(self, fallback=True, fallback_status=0):
        events = []
        def initialize(label, status):
            def run(version, system, mapping):
                events.append(label)
                system._obj.value = 10
                mapping._obj.value = 20
                return status
            return Mock(side_effect=run)
        dll = SimpleNamespace(
            ADLXInitialize2=initialize('normal', 0),
            ADLXTerminate=Mock(side_effect=lambda: events.append('terminate')))
        if fallback:
            dll.ADLXInitializeWithIncompatibleDriver2 = initialize('compatible', fallback_status)
        def output(obj, slot, kind):
            events.append('gpus' if slot == 1 else 'performance')
            return c.c_void_p(30 if slot == 1 else 40)
        with patch.object(amd.c, 'CDLL', return_value=dll), \
             patch.object(amd, 'output', side_effect=output), \
             patch.object(amd, 'release', side_effect=lambda p: events.append(f'release:{p.value}')):
            if fallback_status:
                with self.assertRaises(OSError): amd.AmdReader()
            else:
                reader = amd.AmdReader()
                reader.close()
        return events

    def test_compatibility_entry_is_used_once_without_standard_reinitialization(self):
        events = self.exercise()
        self.assertEqual([e for e in events if not e.startswith('release:')],
                         ['compatible', 'performance', 'terminate'])
        self.assertLess(events.index('release:40'), events.index('terminate'))
        self.assertEqual(events.count('performance'), 1)
        self.assertEqual(events.count('terminate'), 1)

    def test_missing_compatibility_export_keeps_standard_session(self):
        events = self.exercise(fallback=False)
        self.assertNotIn('compatible', events)
        self.assertEqual(events[0], 'normal')
        self.assertEqual(events.count('performance'), 1)

    def test_failed_compatibility_does_not_query_performance_or_double_terminate(self):
        events = self.exercise(fallback_status=1)
        self.assertNotIn('performance', events)
        self.assertNotIn('normal', events)
        self.assertNotIn('terminate', events)

    def test_incomplete_driver_dll_is_unavailable_without_breaking_wddm_reader(self):
        with patch.object(amd.c, 'CDLL', return_value=SimpleNamespace()):
            with self.assertRaises(OSError): amd.AmdReader()

    def test_only_supported_valid_total_board_watts_are_published(self):
        reader = amd.AmdReader.__new__(amd.AmdReader)
        reader.mapping, reader.performance = c.c_void_p(1), c.c_void_p(2)
        gpu = c.c_void_p(5)
        def pci(obj, slot, types, gpu, bus, device, function):
            bus._obj.value = 36
            return 0
        for supported, watts, expected in [(False,46,None),(True,46,46),(True,0,0),
                (True,float('nan'),None),(True,float('inf'),None),(True,-1,None),(True,2001,None)]:
            with self.subTest(supported=supported,watts=watts):
                queried = []
                def output(obj, slot, kind, *prefix):
                    if obj.value == 5:
                        return kind(b'AMD GPU' if slot == 7 else 16304)
                    if obj.value == 2: return kind(3 if slot == 21 else 4)
                    if obj.value == 3: return kind(supported if slot == 9 else False)
                    self.assertEqual((obj.value,slot),(4,10),'Only total board power, never ASIC power')
                    queried.append(slot)
                    return kind(watts)
                with patch.object(amd,'call',side_effect=pci), \
                     patch.object(amd,'output',side_effect=output), patch.object(amd,'release'):
                    row = reader.gpu_sample(gpu)
                self.assertEqual(row['pci_bus_id'],'0000:24:00.0')
                self.assertEqual(row['power_w'],expected)
                self.assertEqual('power_source' in row,expected is not None)
                if not supported: self.assertEqual(queried,[])


if __name__ == '__main__': unittest.main()
