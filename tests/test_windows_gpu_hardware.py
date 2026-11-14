import ctypes as c
import sys,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from windows_gpu_hardware import counter_values,adapter_usage,WddmReader
from console_hardware import WindowsGpuSampler,SystemHardwareSampler
from console_radiance import monitor_execution,enrich_hardware


class GpuTelemetryTests(unittest.TestCase):
    def test_invalid_counters_are_absent_but_real_idle_zero_survives(self):
        self.assertEqual(counter_values([('zero',0,0),('new',1,3),('failed',2,80),
            ('nan',0,float('nan')),('negative',0,-1)]),{'zero':0,'new':3})

    def test_busiest_engine_sums_processes_without_combining_other_gpus(self):
        luid='luid_0x00000000_0x00000001'
        values={f'pid_1_{luid}_phys_0_eng_0_engtype_Compute':30,
                f'pid_2_{luid}_phys_0_eng_0_engtype_Compute':40,
                f'pid_1_{luid}_phys_0_eng_1_engtype_Copy':50,
                'pid_1_luid_0x00000000_0x00000002_phys_0_eng_0_engtype_Compute':99}
        self.assertEqual(adapter_usage(values,luid),70)
        self.assertIsNone(adapter_usage({},luid))
        values[f'pid_3_{luid}_phys_0_eng_0_engtype_Compute']=80
        self.assertEqual(adapter_usage(values,luid),100)

    def test_stale_crashed_or_future_helper_never_displays_fresh_gpu_data(self):
        sampler=WindowsGpuSampler()
        for stamp in (None,float('nan'),0,time.time()*1000-6000,time.time()*1000+6000):
            sampler.value={'sampled_at_ms':stamp,'gpus':[{'used_bytes':123}]}
            self.assertEqual(sampler.snapshot()['gpus'],[])
        sampler.value={'sampled_at_ms':time.time()*1000,'gpus':[{'used_bytes':0}]}
        self.assertEqual(sampler.snapshot()['gpus'][0]['used_bytes'],0)
        sampler.snapshot()['gpus'][0]['used_bytes']=999
        self.assertEqual(sampler.snapshot()['gpus'][0]['used_bytes'],0)

    def test_gpu_samples_expire_at_http_read_time_without_waiting_for_system_tick(self):
        gpu = WindowsGpuSampler()
        gpu.value = {'sampled_at_ms':100000, 'gpus':[{'power_w':46}]}
        sampler = SystemHardwareSampler(gpu)
        sampler.value = {'sampled_at_ms':100000, 'cpu_percent':42}
        with patch('console_hardware.time.time', return_value=100):
            self.assertEqual(sampler.snapshot()['gpus'][0]['power_w'],46)
        with patch('console_hardware.time.time', return_value=106):
            result = sampler.snapshot()
        self.assertEqual(result['gpus'],[])
        self.assertEqual(result['cpu_percent'],42)

    def test_system_sample_does_not_publish_partial_cpu_ram_during_collection(self):
        gpu = Mock(snapshot=Mock(return_value={'gpus':[]}))
        sampler = SystemHardwareSampler(gpu)
        sampler.reader = lambda: None
        sampler.value = {'cpu_percent':17, 'ram_total_bytes':100}
        during = []
        def cpu_percent(**kwargs):
            during.append(sampler.snapshot())
            return 42
        with patch('console_hardware.psutil', SimpleNamespace(cpu_percent=cpu_percent,
                virtual_memory=lambda:SimpleNamespace(total=200,available=50))):
            sampler.sample()
        self.assertEqual(during[0]['cpu_percent'],17)
        self.assertEqual(during[0]['ram_total_bytes'],100)
        self.assertEqual(sampler.snapshot()['cpu_percent'],42)
        self.assertEqual(sampler.snapshot()['ram_used_bytes'],150)

    def test_missing_pdh_counters_preserve_independent_temperature(self):
        reader = WddmReader.__new__(WddmReader)
        reader.query = c.c_void_p(1)
        reader.pdh = SimpleNamespace(PdhCollectQueryData=lambda query:-1)
        reader.adapters = [(c.c_uint32(1),dict(luid='test',total_bytes=16*1024**3))]
        def info(handle, kind, perf):
            perf.temperature = 550
            return True
        reader.info = info
        row = reader.sample()[0]
        self.assertEqual(row['temperature_c'],55)
        self.assertIsNone(row['used_bytes'])
        self.assertIsNone(row['utilization_percent'])

    def test_only_frozen_matching_rank_identity_can_supply_missing_ids(self):
        native=dict(rank=0,device=0,name='AMD GPU',arch='gfx1201',vram_bytes=16*1024**3)
        engine=dict(tensor_parallel=1,devices=[native])
        binding=dict(rank=0,hip_ordinal=0,name='AMD GPU',uuid='GPU-'+'a'*32,pci_bus_id='0000:24:00.0')
        self.assertEqual(monitor_execution(engine,[binding])['monitor_devices'][0]['pci_bus_id'],'0000:24:00.0')
        for bad in (dict(binding,rank=1),dict(binding,hip_ordinal=2),dict(binding,name='Other GPU'),dict(binding,uuid='bad')):
            self.assertNotIn('pci_bus_id',monitor_execution(engine,[bad])['monitor_devices'][0])
        self.assertNotIn('pci_bus_id',monitor_execution(engine,[])['monitor_devices'][0])
        conflict=dict(tensor_parallel=1,devices=[dict(native,uuid='GPU-'+'b'*32)])
        self.assertNotIn('pci_bus_id',monitor_execution(conflict,[binding])['monitor_devices'][0])

    def test_only_missing_radiance_hardware_is_enriched(self):
        hardware={'sampled_at_ms':123,'gpus':[{'power_w':None,'used_bytes':0}]}
        original={'backend':'radiance','active':[]}
        result=enrich_hardware(original,hardware)
        self.assertEqual(result['hardware'],hardware)
        self.assertNotIn('hardware',original)
        result['hardware']['gpus'][0]['used_bytes']=123
        self.assertEqual(hardware['gpus'][0]['used_bytes'],0)
        native=dict(original,hardware={'native':True})
        self.assertEqual(enrich_hardware(native,hardware),native)
        for backend in ('ninfer','llamacpp','polystrata'):
            self.assertNotIn('hardware',enrich_hardware({'backend':backend},hardware))


if __name__=='__main__':unittest.main()
