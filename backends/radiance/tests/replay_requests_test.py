import sys
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from replay_requests import read_plan, make_runner
import replay_requests


class ReplayTest(unittest.TestCase):
    def test_driver_exit_code_is_propagated(self):
        with patch.object(sys, 'argv', ['replay_requests.py', '--plan', 'unused.json']), \
             patch.object(replay_requests, 'read_plan', return_value=[({}, {}, '')]), \
             patch.object(replay_requests.agent_history, 'run'), \
             patch.object(replay_requests.smoke_server, 'main', return_value=1):
            self.assertEqual(replay_requests.main(), 1)

    def test_payload_parameters_are_preserved_and_quality_is_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload = {'model': 'old', 'messages': [{'role': 'user', 'content': 'hi'}],
                       'stream': False, 'temperature': 1, 'seed': 42, 'max_tokens': 512,
                       'tools': [], 'tool_choice': 'none', 'reasoning_budget_tokens': 128}
            (root / 'request.json').write_text(json.dumps(payload))
            (root / 'plan.json').write_text(json.dumps([{'name': 'test', 'request': 'request.json',
                                                       'expected_terms': ['missing']}]))
            entries = read_plan(root / 'plan.json')
            expected = dict(payload, model='bringup')
            self.assertEqual(entries[0][1], expected)
            def request(url, body, **kw):
                self.assertEqual(body, expected)
                return {'choices': [{'message': {'content': 'other'}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 3, 'completion_tokens': 2}}
            rows = []
            report = make_runner(entries)(request, root, 100, 1, 10, rows)
            self.assertTrue(report['pass'])
            self.assertEqual(report['quality_checks']['expected_terms_present'], [False])
            self.assertTrue(report['diagnostic_request_replay'])


if __name__ == '__main__':
    unittest.main()
