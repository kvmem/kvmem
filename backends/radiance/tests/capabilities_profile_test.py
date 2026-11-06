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
        self.assertEqual(result['qualification']['tensor_parallel'],[1])
        self.assertNotIn('tp2_mtp_depths',result['qualification'])
        self.assertEqual(result['build_profile'],'qwen35-dense-tp1')
        self.assertEqual(result['speculative']['modes']['mtp']['max_draft_tokens'],7)
        self.assertEqual(self.manifest,original)

    def test_cmake_constructor_command_emits_expected_json(self):
        result=subprocess.run([sys.executable,str(TOOLS/'build_capabilities.py'),
            '--manifest',str(self.path),'--kernel-profile','qwen35-dense-tp1'],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),for_profile(self.manifest,'qwen35-dense-tp1'))

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):for_profile(self.manifest,'unsupported-profile')


if __name__=='__main__':unittest.main()
