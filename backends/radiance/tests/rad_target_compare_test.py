import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from compare_rad_target import compare


class TargetCompareTest(unittest.TestCase):
    def check_pair(self, right_payload=b'AB', right_encoding='same', omit=False):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/'a.rad', Path(folder)/'b.rad'
            a.write_bytes(b'AB')
            b.write_bytes(b'xxxx' + right_payload)
            base = {'name': 'blk.0.weight', 'offset': 0, 'bytes': 2, 'shape': [1],
                    'encoding_sha256': 'same', 'relative_planes': [(0, 2)]}
            other = dict(base, offset=4, encoding_sha256=right_encoding)
            entries = [{'weight_entries': [base, dict(base, name='dflash.weight')]},
                       {'weight_entries': ([] if omit else [other]) + [dict(other, name='mtp.weight')]}]
            with patch('compare_rad_target.inspect', side_effect=entries):
                return compare(a, b)

    def test_offsets_may_change_and_draft_weights_are_separate(self):
        result = self.check_pair()
        self.assertTrue(result['all_reference_non_dflash_payloads_equal'])
        self.assertEqual(result['equal_count'], 1)
        self.assertEqual(result['candidate_extra'], ['mtp.weight'])

    def test_same_bytes_with_different_encoding_are_not_equivalent(self):
        self.assertFalse(self.check_pair(right_encoding='other')['all_reference_non_dflash_payloads_equal'])
        self.assertFalse(self.check_pair(right_payload=b'AC')['all_reference_non_dflash_payloads_equal'])

    def test_missing_target_weight_fails(self):
        result = self.check_pair(omit=True)
        self.assertFalse(result['all_reference_non_dflash_payloads_equal'])
        self.assertEqual(result['missing'], ['blk.0.weight'])


if __name__ == '__main__':
    unittest.main()
