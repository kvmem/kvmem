import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr


SPEC = importlib.util.spec_from_file_location(
    'hipfire_pin',
    Path(__file__).resolve().parents[1] / 'scripts/hipfire_pin.py')
hipfire_pin = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hipfire_pin)


class HipfirePinTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        include = self.root / 'kvmem/include/kvmem'
        include.mkdir(parents=True)
        (include / 'memory_contract.hpp').write_text('#pragma once\n', encoding='utf-8')
        (self.root / 'kvmem/host-sources.txt').write_text(
            'src/host/store.cpp\n', encoding='utf-8')
        shim = self.root / 'backends/hipfire'
        shim.mkdir(parents=True)
        (shim / 'hipfire_kvmem_shim.h').write_text(
            '#define HK_ABI_VERSION 1u\n', encoding='utf-8')
        (shim / 'hipfire_kvmem_shim.cpp').write_text('// shim\n', encoding='utf-8')
        (self.root / 'docs').mkdir()
        (self.root / 'docs/readme.md').write_text('docs\n', encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_write_check_and_digest_scope(self):
        with redirect_stderr(io.StringIO()):
            values = hipfire_pin.write_pin(self.root)
        self.assertEqual(values['abi_version'], '1')
        self.assertEqual(values['commit'], 'unknown')
        self.assertEqual(hipfire_pin.check_pin(self.root, None)['abi_version'], '1')

        # Documentation changes do not invalidate the interface pin.
        (self.root / 'docs/readme.md').write_text('changed\n', encoding='utf-8')
        hipfire_pin.check_pin(self.root, None)

        (self.root / 'kvmem/include/kvmem/memory_contract.hpp').write_text(
            '#pragma once\n// changed\n', encoding='utf-8')
        with self.assertRaisesRegex(hipfire_pin.PinError, 'include_sha256 mismatch'):
            hipfire_pin.check_pin(self.root, None)

    def test_header_abi_version_must_match_pin(self):
        with redirect_stderr(io.StringIO()):
            hipfire_pin.write_pin(self.root)
        (self.root / 'backends/hipfire/hipfire_kvmem_shim.h').write_text(
            '#define HK_ABI_VERSION 2u\n', encoding='utf-8')
        with self.assertRaisesRegex(hipfire_pin.PinError, 'abi_version mismatch'):
            hipfire_pin.check_pin(self.root, None)

    def test_expected_commit_requires_a_readable_head(self):
        with redirect_stderr(io.StringIO()):
            hipfire_pin.write_pin(self.root)
        with self.assertRaisesRegex(hipfire_pin.PinError, 'no readable git HEAD'):
            hipfire_pin.check_pin(self.root, 'a' * 40)


if __name__ == '__main__':
    unittest.main()
