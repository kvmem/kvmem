import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from summarize_task2 import cached_followup


class CacheSummaryTest(unittest.TestCase):
    def test_replay_native_counts(self):
        self.assertTrue(cached_followup({'timings': {'cache_n': 128}}))
        self.assertFalse(cached_followup({'timings': {'cache_n': 0}}))

    def test_usage_fallback(self):
        self.assertTrue(cached_followup({'usage': {'prompt_tokens_details': {'cached_tokens': 32}}}))
        self.assertFalse(cached_followup({}))

    def test_explicit_agent_result_is_authoritative(self):
        self.assertFalse(cached_followup({'retained_prefix': False, 'timings': {'cache_n': 128}}))
        self.assertTrue(cached_followup({'retained_prefix': True}))


if __name__ == '__main__':
    unittest.main()
