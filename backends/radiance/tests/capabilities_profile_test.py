"""Worker profile limits, using the same constructor that CMake executes."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

TOOLS=Path(__file__).resolve().parents[1]/'tools'
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(TOOLS))
from build_capabilities import for_profile


class ProfileCapabilitiesTest(unittest.TestCase):
    def setUp(self):
        self.path=ROOT/'scripts/launcher-capabilities.json'
        self.manifest=json.loads(self.path.read_text(encoding='utf-8-sig'))

    def test_general_build_payload_is_identical(self):
        self.assertEqual(for_profile(self.manifest,'all'),self.manifest['backends']['radiance'])

    def test_tp1_binary_limits_every_existing_device_capability(self):
        original=copy.deepcopy(self.manifest)
        result=for_profile(self.manifest,'qwen35-dense-tp1')
        for name in ('text','image_input'):
            self.assertEqual(result['supported_combinations'][name]['max_devices'],1)
        for mode in ('mtp','dflash2'):
            self.assertEqual(result['speculative']['modes'][mode]['max_devices'],1)
        self.assertEqual(result['supported_combinations']['text']['tensor_parallel'],[1])
        self.assertEqual(result['qualification']['status'],'not-requalified-on-v2')
        self.assertEqual(result['historical_qualification'],self.manifest['backends']['radiance']['historical_qualification'])
        self.assertEqual(result['build_profile'],'qwen35-dense-tp1')
        self.assertEqual(result['speculative']['modes']['mtp']['max_draft_tokens'],7)
        self.assertEqual(self.manifest,original)

    def test_cmake_constructor_command_emits_expected_json(self):
        result=subprocess.run([sys.executable,str(TOOLS/'build_capabilities.py'),
            '--manifest',str(self.path),'--kernel-profile','qwen35-dense-tp1'],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),for_profile(self.manifest,'qwen35-dense-tp1'))

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):for_profile(self.manifest,'unsupported-profile')

    def test_rdna3_profile_distinguishes_targets_from_hardware_evidence(self):
        result=for_profile(self.manifest,'qwen35-iqr-rdna3')
        q=result['qualification']
        self.assertEqual(q['status'],'per-device-qualification-required')
        self.assertEqual(q['target_family'],['gfx1100','gfx1101','gfx1102','gfx1103'])
        self.assertEqual(q['runtime_evidence']['gpu'],'gfx1101')
        self.assertEqual(set(result['speculative']['modes']),{'none','dflash2'})
        self.assertEqual(result['supported_combinations']['text']['max_devices'],1)

    def test_7800m_profile_does_not_advertise_unavailable_kernel_paths(self):
        original=copy.deepcopy(self.manifest)
        result=for_profile(self.manifest,'qwen35-iqr-gfx1101')
        self.assertEqual(set(result['speculative']['modes']),{'none','dflash2'})
        self.assertEqual(result['speculative']['modes']['dflash2']['max_draft_tokens'],7)
        self.assertEqual(result['speculative']['modes']['dflash2']['max_devices'],1)
        self.assertTrue(result['speculative_decoding'])
        self.assertFalse(result['multimodal'])
        text=result['supported_combinations']['text']
        self.assertEqual(text['kv'],['bf16','fp8'])
        self.assertEqual(text['tensor_parallel'],[1])
        self.assertEqual(text['max_concurrency'],1)
        self.assertNotIn('packed_kv_geometry',result)
        self.assertEqual(result['qualification']['gpu'],'gfx1101')
        self.assertEqual(result['qualification']['kv'],'bf16')
        self.assertEqual(self.manifest,original)


if __name__=='__main__':unittest.main()
