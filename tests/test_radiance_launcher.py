"""Radiance shared-launcher contracts, without model loading or GPU access."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from launcher.plan import build_plan,token_settings
from launcher.radiance import build_radiance_plan
from console_hardware import parse_rocr_agents
sys.path.insert(0,str(ROOT/'backends/radiance/tests'))
from draft_control_test import container


class RadiancePlanTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.worker=self.root/'radiance-kvmem-server';self.worker.write_text('worker');self.worker.chmod(0o755)
        self.model=self.root/'model.rad'
        container(self.model,['mtp.fc.weight'],[('mtp_num_hidden_layers',1)])
        self.home=self.root/'radiance_home'
        for name in ('kernels','architectures'):(self.home/name).mkdir(parents=True)
        (self.home/'kernels.rkb').touch()

    def plan(self,**settings):
        parameters=dict(Backend='radiance',Model=str(self.model),Worker=str(self.worker),
            RadianceHome=str(self.home),Gpu='GPU-65471d75ce486b4f')
        parameters.update(settings)
        return build_plan(dict(schema_version=1,parameters=parameters))

    def test_prefill_default_and_explicit_override(self):
        for settings, expected in (({}, 256), ({'Prefill': 512}, 512)):
            with self.subTest(settings=settings):
                plan=self.plan(**settings)
                self.assertEqual(plan['effective_parameters']['Prefill'],expected)
                args=plan['argv']
                self.assertEqual(args[args.index('--max-num-batched-tokens')+1],str(expected))

    def test_cache_minimum(self):
        self.assertEqual(self.plan()['argv'],self.plan(CacheMinTokens=0)['argv'])
        for value in (4095,4096,4097,2147483647):
            plan=self.plan(CacheMinTokens=value);args=plan['argv']
            self.assertEqual(args[args.index('--kvmem-cache-min-tokens')+1],str(value))
            self.assertEqual(plan['effective_parameters']['CacheMinTokens'],value)
        for value in (-1,2147483648,True,1.5):
            with self.subTest(value=value),self.assertRaises(ValueError):self.plan(CacheMinTokens=value)

    def test_cpu_threads_and_image_cache_budget(self):
        plan=self.plan(CpuThreads=8,VisionCacheMiB=64)
        self.assertEqual(plan['environment']['OMP_NUM_THREADS'],'8')
        self.assertEqual(plan['environment']['KVMEM_RAD_VISION_CACHE_MIB'],'64')
        self.assertEqual(self.plan(VisionCacheMiB=0)['effective_parameters']['VisionCacheMiB'],0)
        for values in [dict(CpuThreads=0),dict(CpuThreads=257),dict(VisionCacheMiB=65537)]:
            with self.subTest(values=values),self.assertRaises(ValueError):self.plan(**values)

    def test_packed_kv_and_recurrent_state(self):
        for kv in ('fp8','bf16','fp8_v4','k6_v4'):
            for state in ('f32','f16'):
                args=self.plan(KvType=kv,StateCacheType=state)['argv']
                self.assertEqual(args[args.index('--kv-cache-dtype')+1],kv)
                self.assertEqual(args[args.index('--state-cache-dtype')+1],state)
        with self.assertRaises(ValueError):self.plan(StateCacheType='bf16')

    def test_units_identity_scoring_and_backend_flags(self):
        p=self.plan(Concurrency=8,SpeculativeMode='mtp',DraftTokens=3,RetainedSessions=0,
            QueryScorer='meanq-cosine',Lrfu=True,LrfuTopFraction=.75,QueryMaxTokens=128,
            QueryReplayMaxTokens=768,ModelAlias='model with spaces',ApiKey='local-secret')
        args=p['argv']
        for flag,value in [('--max-num-seqs','8'),('--num-speculative-tokens','3'),('--kvmem-sessions','0'),
            ('--kvmem-gen-reserve','1024'),('--served-model-name','model with spaces'),('--api-key','<redacted>'),
            ('--kvmem-query-max-tokens','128'),('--kvmem-query-replay-max-tokens','768')]:
            self.assertEqual(args.count(flag),1);self.assertEqual(args[args.index(flag)+1],value)
        self.assertNotIn('CUDA_VISIBLE_DEVICES',p['environment'])
        self.assertEqual(p['environment']['ROCR_VISIBLE_DEVICES'],'GPU-65471d75ce486b4f')
        self.assertEqual(p['resolved_scoring']['top_frac'],.75)
        self.assertNotIn('local-secret',json.dumps(p))
        self.assertNotIn('--default-max-tokens',args)
        self.assertEqual(args[args.index('--prefix-cache-mib')+1],'0')
        self.assertNotIn('--prefix-cache-host-mib',args)

    def test_unsupported_features_are_rejected_before_launch(self):
        for values in [dict(Vision=True),dict(DiskPath=str(self.root),DiskMiB=0),dict(DiskMiB=2),dict(StageLayers=[64]),
            dict(Gpu='GPU-65471d75ce486b4f,GPU-0000000000000001,GPU-0000000000000002'),
            dict(Gpu='GPU-65471d75ce486b4f,GPU-65471d75ce486b4f'),dict(KvType='nvfp4'),
            dict(FastPrefill=True),dict(Reserve=640,Prefill=512),dict(Prefill=1024),dict(SpeculativeMode='dflash2'),
            dict(SpeculativeMode='none',DraftPolicy='adaptive'),dict(RetainedSessions=1025)]:
            with self.subTest(values=values),self.assertRaises(ValueError):self.plan(**values)

    def test_two_devices_use_exact_native_tensor_parallelism(self):
        ids='GPU-9d36683ccd630821,GPU-dc5c3305d734c5fc'
        with mock.patch('launcher.radiance.vision_geometry',return_value=(16,2)):
            p=self.plan(Gpu=ids,Concurrency=8,SpeculativeMode='mtp',DraftTokens=7,
                        DraftPolicy='adaptive',Vision=True)
        args=p['argv']
        for flag,value in [('--tp','2'),('--tp-wire','exact'),('--max-num-seqs','8'),('--num-speculative-tokens','7')]:
            self.assertEqual(args.count(flag),1);self.assertEqual(args[args.index(flag)+1],value)
        self.assertEqual(p['environment']['ROCR_VISIBLE_DEVICES'],ids)
        self.assertEqual(p['environment']['HIP_VISIBLE_DEVICES'],'0,1')
        self.assertEqual(p['capabilities']['supported_combinations']['text']['max_devices'],2)
        self.assertNotIn('--stage-layers',args)
        with self.assertRaises(ValueError):self.plan(Gpu=ids,StageLayers=[32,32])

    def test_shared_ssd_budget_and_path(self):
        args=self.plan(DiskPath=str(self.root/'cache with spaces'),DiskMiB=4096,HostMiB=32)['argv']
        for flag,value in [('--kvmem-disk-path',str(self.root/'cache with spaces')),
                           ('--kvmem-disk-mib','4096'),('--kvmem-host-mib','32')]:
            self.assertEqual(args.count(flag),1);self.assertEqual(args[args.index(flag)+1],value)
        self.assertNotIn('--kvmem-disk-path',self.plan()['argv'])

    def test_native_adaptive_mtp(self):
        for depth in (3,5,7):
            for policy in ('fixed','adaptive'):
                with self.subTest(depth=depth,policy=policy):
                    args=self.plan(SpeculativeMode='mtp',DraftTokens=depth,DraftPolicy=policy)['argv']
                    self.assertEqual(args[args.index('--draft-window')+1],policy)
                    self.assertEqual(args[args.index('--num-speculative-tokens')+1],str(depth))
        for values in (dict(DraftTokens=8),dict(DraftTokens=15),dict(MtpDrafts=8)):
            with self.subTest(values=values),self.assertRaises(ValueError):
                self.plan(SpeculativeMode='mtp',**values)

    def test_vocab_weighted_mtp_selects_u2_without_changing_other_models(self):
        meta=[('mtp_num_hidden_layers',1)]
        names=['output.k3.weight','output.vrow','mtp.fc.weight','mtp.draft_head.weight']
        container(self.model,names,meta,quant='w8a8_fwht128',mtp_head_dtype=20)
        self.assertEqual(self.plan(SpeculativeMode='mtp')['environment']['KVMEM_RAD_MTP_HEAD'],'auto')
        self.assertEqual(self.plan(SpeculativeMode='none')['environment']['KVMEM_RAD_MTP_HEAD'],'shared')
        container(self.model,names,meta,quant='w8a8_fwht128',mtp_head_dtype=2)
        with self.assertRaisesRegex(ValueError,'requires a dedicated U2'):
            self.plan(SpeculativeMode='mtp')
        container(self.model,['output.weight','mtp.fc.weight','mtp.draft_head.weight'],meta,mtp_head_dtype=20)
        self.assertEqual(self.plan(SpeculativeMode='mtp')['environment']['KVMEM_RAD_MTP_HEAD'],'shared')

    def test_embedded_dflash_uses_its_trained_depth(self):
        container(self.model,['dflash.fc.weight'],[
            ('draft.architectures','DFlash2Model'),('draft.dflash_config.block_size',8)])
        for settings in ({},{'DraftTokens':7}):
            plan=self.plan(SpeculativeMode='dflash2',**settings)
            args=plan['argv']
            self.assertEqual(plan['effective_parameters']['DraftTokens'],7)
            self.assertEqual(args[args.index('--num-speculative-tokens')+1],'7')
            self.assertEqual(args[args.index('--kvmem-drafter')+1],'dflash2')
        for settings in ({'DraftTokens':5},{'DraftPolicy':'adaptive'},{'DraftModel':str(self.model)}):
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                self.plan(SpeculativeMode='dflash2',**settings)
        with self.assertRaises(ValueError):self.plan(SpeculativeMode='mtp')
        args=self.plan(SpeculativeMode='none')['argv']
        self.assertEqual(args[args.index('--num-speculative-tokens')+1],'0')
        self.assertEqual(args[args.index('--kvmem-drafter')+1],'off')

    def test_windows_single_request_capture_profile(self):
        container(self.model, ['dflash.fc.weight'], [
            ('draft.architectures', 'DFlash2Model'), ('draft.dflash_config.block_size', 8),
        ], quant='w8a8_fwht128')
        parameters = dict(Backend='radiance', Model=str(self.model), Worker=str(self.worker),
                          RadianceHome=str(self.home), Gpu='GPU-65471d75ce486b4f',
                          SpeculativeMode='dflash2', Vision=False)
        flags = ('KVMEM_CAPTURE_EARLY_SEAL', 'RADIANCE_HIP_SMALL_D2H')
        for platform, overrides, enabled in [
            ('windows', {}, True), ('linux', {}, False),
            ('windows', {'Concurrency': 2}, False), ('windows', {'SpeculativeMode': 'none'}, True),
            ('windows', {'Gpu': 'GPU-65471d75ce486b4f,GPU-0000000000000001'}, False),
            ('windows', {'Vision': True}, True),
        ]:
            with self.subTest(platform=platform, overrides=overrides), mock.patch(
                    'launcher.radiance.vision_geometry', return_value=(16, 2)):
                plan = build_radiance_plan({**parameters, **overrides}, self.root, platform, False, {})
                for key in flags:
                    self.assertEqual(plan['environment'].get(key), '1' if enabled else None)
        for block_size, quant in [(4, 'w8a8_fwht128'), (8, 'fp8_e4m3')]:
            with self.subTest(block_size=block_size, quant=quant):
                container(self.model, ['dflash.fc.weight'], [
                    ('draft.architectures', 'DFlash2Model'), ('draft.dflash_config.block_size', block_size),
                ], quant=quant)
                plan = build_radiance_plan(parameters, self.root, 'windows', False, {})
                self.assertTrue(all(plan['environment'].get(key) == '1' for key in flags))
        container(self.model, ['mtp.fc.weight'], [('mtp_num_hidden_layers', 1)], quant='w8a8_fwht128')
        plan = build_radiance_plan({**parameters, 'SpeculativeMode': 'mtp'}, self.root, 'windows', False, {})
        self.assertTrue(all(plan['environment'].get(key) == '1' for key in flags))

    def test_windows_capture_opt_out_and_recipe_omission(self):
        flags = ('KVMEM_CAPTURE_EARLY_SEAL', 'RADIANCE_HIP_SMALL_D2H')
        parameters = dict(Backend='radiance', Model=str(self.model), Worker=str(self.worker),
                          RadianceHome=str(self.home), Gpu='GPU-65471d75ce486b4f')
        for disabled in flags:
            with self.subTest(disabled=disabled), mock.patch.dict(os.environ, {disabled:'0'}):
                p = build_radiance_plan(parameters, self.root, 'windows', False, {})
                self.assertEqual(p['environment'][disabled], '0')
                self.assertEqual(p['environment'][next(k for k in flags if k != disabled)], '1')
        p = self.plan()
        self.assertNotIn('--reasoning-effort', p['argv'])
        self.assertFalse(p['effective_parameters']['Vision'])
        self.assertEqual(p['effective_parameters']['SpeculativeMode'], 'none')

    def test_explicit_speed_recipe_and_user_overrides(self):
        from launcher.config import ENGINE_DEFAULTS
        recipe = ENGINE_DEFAULTS['radiance']
        self.assertEqual((recipe['HostMiB'], recipe['SinkTokens'], recipe['RecentTokens']), (12288,12288,1024))
        container(self.model, ['dflash.fc.weight'], [
            ('draft.architectures', 'DFlash2Model'), ('draft.dflash_config.block_size', 8)])
        with mock.patch('launcher.radiance.vision_geometry', return_value=(16,2)):
            p = self.plan(**recipe)
            self.assertEqual(p['argv'][p['argv'].index('--reasoning-effort')+1], 'xhigh')
            self.assertEqual(p['effective_parameters']['VisionTokens'], 256)
            self.assertTrue(p['effective_parameters']['PreserveThinking'])
            saved = dict(recipe, ThinkingMode='off', PreserveThinking=False)
            saved.pop('ReasoningEffort')
            p = self.plan(**saved)
            self.assertEqual(p['argv'][p['argv'].index('--reasoning-effort')+1], 'none')
            self.assertFalse(p['effective_parameters']['PreserveThinking'])

    def test_windows_hip_selection_uses_current_physical_ordinals(self):
        first, second = ('GPU-' + x * 32 for x in ('a', 'b'))
        parameters = dict(Backend='radiance', Model=str(self.model), Worker=str(self.worker),
                          RadianceHome=str(self.home), Gpu=second + ',' + first)
        devices = [dict(uuid=first, ordinal=0, name='AMD GPU', pci_bus_id='0000:01:00.0'),
                   dict(uuid=second, ordinal=2, name='AMD GPU', pci_bus_id='0000:24:00.0')]
        with mock.patch('console_hardware.windows_hip_inventory', return_value=devices):
            plan = build_radiance_plan(parameters, self.root, 'windows', False, {})
            self.assertEqual(plan['environment']['HIP_VISIBLE_DEVICES'], '2,0')
            self.assertNotIn('ROCR_VISIBLE_DEVICES', plan['environment'])
            self.assertEqual([d['pci_bus_id'] for d in plan['monitor_devices']], ['0000:24:00.0','0000:01:00.0'])
            self.assertEqual([d['hip_ordinal'] for d in plan['monitor_devices']], [0,1])
            with self.assertRaisesRegex(ValueError, 'distinct AMD GPUs'):
                build_radiance_plan({**parameters, 'Gpu': first + ',' + first.upper()}, self.root, 'windows', False, {})
        for invalid in (devices[:1], devices + [devices[1]]):
            with mock.patch('console_hardware.windows_hip_inventory', return_value=invalid):
                with self.assertRaisesRegex(ValueError, 'missing or ambiguous'):
                    build_radiance_plan(parameters, self.root, 'windows', False, {})

    def test_embedded_drafters_accept_case_insensitive_rad_extension(self):
        # The shared launch contract and native magic reader accept .RAD too.
        # Metadata preflight must not newly reject that existing model path.
        self.model=self.root/'embedded.RAD'
        for mode,names,metadata,depth in [
            ('mtp',['mtp.fc.weight'],[('mtp_num_hidden_layers',1)],3),
            ('dflash2',['dflash.fc.weight'],[
                ('draft.architectures','DFlash2Model'),('draft.dflash_config.block_size',8)],7),
        ]:
            with self.subTest(mode=mode):
                container(self.model,names,metadata)
                args=self.plan(SpeculativeMode=mode)['argv']
                self.assertEqual(args[args.index('--num-speculative-tokens')+1],str(depth))

    def test_dflash_metadata_without_weights_and_truncation_are_rejected(self):
        container(self.model,['mtp.fc.weight'],[
            ('draft.architectures','DFlash2Model'),('draft.dflash_config.block_size',8),
            ('mtp_num_hidden_layers',1)])
        for mode in ('mtp','dflash2'):
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                self.plan(SpeculativeMode=mode)
        self.model.write_bytes(self.model.read_bytes()[:-1])
        with self.assertRaises(ValueError):self.plan(SpeculativeMode='dflash2')

    def test_native_image_geometry_and_residency(self):
        with mock.patch('launcher.radiance.vision_geometry',return_value=(16,2)):
            args=self.plan(Vision=True,VisionTokens=1024)['argv']
            self.assertEqual(args[args.index('--mm-max-patches')+1],'4096')
            self.assertEqual(args[args.index('--image-max-pixels')+1],'1048576')
            for residency in ('cpu','resident'):
                plan=self.plan(Vision=True,VisionResidency=residency)
                argv=plan['argv']
                self.assertEqual(argv[argv.index('--vision-residency')+1],residency)
                self.assertEqual(plan['effective_parameters']['VisionResidency'],residency)
            for values in (dict(VisionResidency='invalid'),dict(VisionTokens=63),dict(VisionTokens=16385)):
                with self.subTest(values=values),self.assertRaises(ValueError):self.plan(Vision=True,**values)
        args=self.plan(Vision=False)['argv']
        self.assertEqual(args[args.index('--mm-max-patches')+1],'0')
        self.assertNotIn('--image-max-pixels',args)
        self.assertNotIn('--vision-residency',args)

    def test_text_embedding_placement(self):
        for placement in ('ram','vram'):
            plan=self.plan(EmbeddingPlacement=placement)
            self.assertEqual(plan['argv'][plan['argv'].index('--embedding-placement')+1],placement)
            self.assertEqual(plan['effective_parameters']['EmbeddingPlacement'],placement)
        with self.assertRaises(ValueError):self.plan(EmbeddingPlacement='disk')

    def test_mtp_default_and_legacy_alias(self):
        for values in (dict(SpeculativeMode='mtp'),dict(MtpDrafts=3)):
            args=self.plan(**values)['argv'];self.assertEqual(args[args.index('--num-speculative-tokens')+1],'3')

    def test_alignment_and_generation_headroom(self):
        p=self.plan(Budget=4300,Reserve=1100,SinkTokens=160,RecentTokens=300)['token_settings']
        self.assertEqual(p['block_tokens'],128)
        self.assertEqual(p['effective']['Budget'],4224)
        self.assertEqual(p['effective']['Reserve'],1024)
        self.assertGreater(p['effective']['Reserve'],p['effective']['Prefill']+128)

    def test_rocr_uuid_is_not_management_uuid_or_cpu(self):
        data='''Agent 1
  Name: CPU
  Uuid: CPU-00000000
  Device Type: CPU
Agent 2
  Name: gfx1201
  Uuid: GPU-65471d75ce486b4f
  Marketing Name: AMD Radeon AI PRO R9700
  Device Type: GPU
  Segment: GLOBAL; FLAGS: COARSE GRAINED
  Size: 33406976 (0x1fdc000) KB
'''
        devices=parse_rocr_agents(data)
        self.assertEqual(len(devices),1)
        self.assertEqual(devices[0]['uuid'],'GPU-65471d75ce486b4f')
        self.assertEqual(devices[0]['memory_mib'],'32624')
        self.assertEqual(devices[0]['vendor'],'amd')


if __name__=='__main__':unittest.main()
