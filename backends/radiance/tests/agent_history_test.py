import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from agent_history import SourceHistory, HISTORY, TOOLS, cached_tokens, final_code_check, verify_teacher_fixture, decode_below_floor, draft_step_ms


class AgentFixtureTest(unittest.TestCase):
    def test_draft_cycle_proxy_requires_fixed_depth_and_enough_cycles(self):
        self.assertIsNone(draft_step_ms({}, 0))
        self.assertIsNone(draft_step_ms({'draft_n': 7, 'predicted_ms': 66}, 7))
        self.assertIsNone(draft_step_ms({'draft_n': 118, 'predicted_ms': 1000}, 7))
        self.assertAlmostEqual(draft_step_ms({'draft_n': 1127, 'predicted_ms': 10658.601013}, 7), 66.20249076)

    def test_decode_guard_requires_an_enabled_floor_and_enough_tokens(self):
        self.assertFalse(decode_below_floor({}, 18))
        self.assertFalse(decode_below_floor({'predicted_n': 2, 'predicted_per_second': 1}, 18))
        self.assertFalse(decode_below_floor({'predicted_n': 64, 'predicted_per_second': 1}, 0))
        self.assertFalse(decode_below_floor({'predicted_n': 64, 'predicted_per_second': 19}, 18))
        self.assertTrue(decode_below_floor({'predicted_n': 64, 'predicted_per_second': 17}, 18))

    def test_teacher_messages_and_tools_have_not_drifted(self):
        self.assertTrue(verify_teacher_fixture()['messages_and_tools_match'])

    def test_history_ownership_and_pending_guard(self):
        f = SourceHistory(32768, 8192, 7)
        original = f.history.copy()
        batch = f.append(200)
        self.assertEqual(batch['batch_id'], '0000')
        self.assertEqual(batch['checksum'], '007391')
        self.assertEqual(f.history[:2], original)
        self.assertEqual(HISTORY, original)
        self.assertEqual(len(TOOLS), 1)
        with self.assertRaises(RuntimeError):
            f.append(400)
        with self.assertRaises(RuntimeError):
            f.observe(200, 201)

    def test_256k_produces_32_tool_batches_after_base(self):
        f = SourceHistory(262144, 8192, 7)
        count, prompt = 1, 200
        while True:
            batch = f.append(prompt)
            self.assertIsNotNone(batch)
            next_prompt = prompt + batch['lines'] * 7 + 113
            f.observe(prompt, next_prompt)
            prompt = next_prompt
            count += 1
            if batch['final']:
                break
            self.assertLess(count, 34)
        self.assertEqual(count, 33)
        self.assertGreaterEqual(prompt, 262144 - 1024)
        self.assertLess(prompt, 262144 - 512)

    def test_usage_spelling_and_syntax_are_not_semantic_verdicts(self):
        self.assertEqual(cached_tokens({'prompt_cache_hit_tokens': 7}), 7)
        self.assertEqual(cached_tokens({'prompt_tokens_details': {'cached_tokens': 9}}), 9)
        self.assertFalse(final_code_check('def incomplete(')['python_syntax'])
        result = final_code_check('```python\ndef f():\n    return 1\n```')
        self.assertTrue(result['python_syntax'])
        self.assertIn('not established', result['semantic_correctness'])
        truncated = final_code_check('Here is code:\n```python\ndef f(')
        self.assertTrue(truncated['unclosed_code_fence'])
        self.assertFalse(truncated['python_syntax'])


if __name__ == '__main__':
    unittest.main()
