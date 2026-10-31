import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from agent_history import review_performance_guard


class PerformancePauseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.guard = {'round': 4, 'threshold_tps': 40, 'timings': {'predicted_per_second': 38}}

    def decision(self, action='continue', wrong_hash=False):
        state = json.loads((self.path / 'performance-pause.json').read_text())
        data = {'action': action, 'reason': 'inspected saved evidence',
                'guard_sha256': 'wrong' if wrong_hash else state['guard_sha256']}
        Path(state['decision_file']).write_text(json.dumps(data))

    def status(self):
        return json.loads((self.path / 'performance-pause.json').read_text())['status']

    def test_default_stops_and_preserves_guard(self):
        with self.assertRaises(RuntimeError):
            review_performance_guard(self.path, self.guard)
        self.assertEqual(json.loads((self.path / 'performance-guard-r004.json').read_text()), self.guard)

    def test_explicit_matching_review_continues_without_changing_guard(self):
        with patch('agent_history.time.sleep', side_effect=lambda _: self.decision()):
            review_performance_guard(self.path, self.guard, 5, lambda: True)
        self.assertEqual(self.status(), 'continued')
        self.assertEqual(json.loads((self.path / 'performance-guard.json').read_text()), self.guard)

    def test_stale_decision_cannot_continue(self):
        with patch('agent_history.time.sleep', side_effect=lambda _: self.decision(wrong_hash=True)):
            with self.assertRaises(RuntimeError):
                review_performance_guard(self.path, self.guard, 5, lambda: True)
        self.assertEqual(self.status(), 'invalid_decision')

    def test_abort_stops(self):
        with patch('agent_history.time.sleep', side_effect=lambda _: self.decision('abort')):
            with self.assertRaises(RuntimeError):
                review_performance_guard(self.path, self.guard, 5, lambda: True)
        self.assertEqual(self.status(), 'aborted')

    def test_resource_guard_or_process_exit_ends_wait(self):
        with self.assertRaises(RuntimeError):
            review_performance_guard(self.path, self.guard, 5, lambda: False)
        self.assertEqual(self.status(), 'service_stopped')

    def test_wait_is_bounded(self):
        with patch('agent_history.time.monotonic', side_effect=[0, 6, 6]):
            with self.assertRaises(TimeoutError):
                review_performance_guard(self.path, self.guard, 5, lambda: True)
        self.assertEqual(self.status(), 'timed_out')


if __name__ == '__main__':
    unittest.main()
