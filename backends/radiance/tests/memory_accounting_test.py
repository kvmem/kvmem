import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from memory_accounting import compare_budgets, native_geometry, physical_tokens, weight_groups

LOG = '''
    kv_attn full 17 layers block 16 tok page 544.00 KiB 1536 blocks = 816.00 MiB (24576 tokens)
I KVMEM_POOL resident_tokens=24576 carved_bytes=1163526144 logical_context=262144
'''


class AccountingTest(unittest.TestCase):
    def test_matches_pinned_cpp_geometry(self):
        self.assertEqual(physical_tokens(1536, 128, 128, 1024, [16])['physical_tokens'], 6144)
        self.assertEqual(physical_tokens(7680, 128, 128, 1024, [16])['physical_tokens'], 24576)
        self.assertEqual(physical_tokens(36864, 128, 128, 1024, [16])['physical_tokens'], 112128)

    def test_exact_recorded_pool(self):
        rows = compare_budgets(LOG, [7680, 36864], 128, 128, 1024)['calculated_budget_scenarios']
        self.assertEqual(rows[0]['calculated_total_kv_mib'], 1109.625)
        self.assertEqual(rows[0]['delta_vs_recorded_kv_mib'], 0)
        self.assertEqual(rows[1]['calculated_attention_mib'], 3723)
        self.assertEqual(rows[1]['delta_vs_recorded_kv_mib'], 2907)
        self.assertFalse(rows[1]['is_capacity_test'])

    def test_recycled_geometry_is_versioned_not_applied_to_old_logs(self):
        self.assertEqual(physical_tokens(7680, 128, 128, 1024, [16], True)['physical_tokens'], 16896)
        self.assertEqual(physical_tokens(36864, 128, 128, 1024, [16], True)['physical_tokens'], 75264)
        new_log = 'KVMEM_POOL_MODE native_recycle_v1 budget=7680 output_reserve=1024\n' + LOG
        row = compare_budgets(new_log, [7680], 128, 128, 1024)['calculated_budget_scenarios'][0]
        self.assertEqual(row['delta_vs_recorded_kv_mib'], -255)

    def test_incomplete_log_rejected(self):
        with self.assertRaises(ValueError):
            native_geometry('KVMEM_POOL_LIMIT resident_tokens=24576 logical_context=262144')

    def test_pressure_prefix_demote_geometry(self):
        self.assertEqual(physical_tokens(7680, 128, 128, 1024, [16], True, True)['physical_tokens'], 9216)
        self.assertEqual(physical_tokens(36864, 128, 128, 1024, [16], True, True)['physical_tokens'], 38400)
        with self.assertRaises(ValueError):
            physical_tokens(7680, 128, 128, 1024, [16], False, True)

    def test_inconsistent_capacity_rejected(self):
        with self.assertRaises(ValueError):
            native_geometry(LOG.replace('1536 blocks', '1535 blocks'))

    def test_invalid_geometry_rejected(self):
        for budget in (0, -1, 127, 255, 2**31):
            with self.assertRaises(ValueError):
                physical_tokens(budget, 128, 128, 1024, [16])

    def test_draft_block_is_not_target(self):
        result = weight_groups([{'name': 'blk.63.ffn_down.weight', 'bytes': 1048576},
                                {'name': 'blk.64.ffn_down.weight', 'bytes': 2097152},
                                {'name': 'mtp.fc.weight', 'bytes': 1048576}], 64)
        self.assertEqual(result['container_payload_mib_by_group'],
                         {'target_blocks': 1, 'auxiliary_blocks': 2, 'mtp_named_weights': 1})


if __name__ == '__main__':
    unittest.main()
