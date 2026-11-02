"""Performance evidence failure injection without a server or GPU."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from benchmark_evidence import (ERROR_BODY_LIMIT, input_provenance, record_http_turn,
                                validate_performance_input, verify_question_query)


class EvidenceTest(unittest.TestCase):
    def record(self, action):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        native = Path(directory.name) / 'native.log'
        native.write_bytes(b'previous turn\n')
        row = {'turns': []}
        snapshots = []
        body = {'messages': [{'role': 'user', 'content': 'question'}]}
        class Opener:
            def open(self, request, timeout):
                with native.open('ab') as stream:
                    stream.write(b'question native evidence\n')
                return action(request)
        def invoke():
            return record_http_turn(Opener(), 'http://fixture.invalid/v1/chat/completions', body,
                                    'question', native, row, lambda: snapshots.append(copy.deepcopy(row)))
        return invoke, row, snapshots, body

    def test_success_records_intent_before_send_and_question_log_only(self):
        invoke, row, snapshots, body = self.record(lambda _: io.BytesIO(b'{"ok":true}'))
        turn = invoke()
        self.assertEqual(snapshots[0]['turns'][0]['status'], 'running')
        self.assertEqual(turn['response'], {'ok': True})
        self.assertEqual(turn['native'], 'question native evidence\n')
        body['messages'].clear()
        self.assertEqual(len(turn['request']['messages']), 1)

    def test_http_error_keeps_failed_request_status_and_body(self):
        for code in (400, 500):
            with self.subTest(code=code):
                def fail(request):
                    raise HTTPError(request.full_url, code, 'fixture failure', {}, io.BytesIO(b'{"error":"specific cause"}'))
                invoke, row, snapshots, _ = self.record(fail)
                with self.assertRaises(HTTPError): invoke()
                turn = row['turns'][0]
                self.assertEqual(turn['status'], 'fail')
                self.assertEqual(turn['http_status'], code)
                self.assertEqual(json.loads(turn['error_body']), {'error': 'specific cause'})
                self.assertEqual(turn['stage'], 'question')
                self.assertIn('question', turn['request']['messages'][0]['content'])
                self.assertGreaterEqual(turn['wall_s'], 0)
                self.assertEqual(snapshots[-1], row)

    def test_error_body_has_a_bounded_capture(self):
        def fail(request):
            raise HTTPError(request.full_url, 500, 'failure', {}, io.BytesIO(b'x' * (ERROR_BODY_LIMIT + 10)))
        invoke, row, _, _ = self.record(fail)
        with self.assertRaises(HTTPError): invoke()
        self.assertEqual(len(row['turns'][0]['error_body']), ERROR_BODY_LIMIT)
        self.assertTrue(row['turns'][0]['error_body_truncated'])

    def test_timeout_keeps_request_and_native_evidence(self):
        def fail(_): raise TimeoutError('fixture timeout')
        invoke, row, _, _ = self.record(fail)
        with self.assertRaises(TimeoutError): invoke()
        self.assertIn('fixture timeout', row['turns'][0]['error'])
        self.assertEqual(row['turns'][0]['status'], 'fail')
        self.assertEqual(row['turns'][0]['native'], 'question native evidence\n')

    def test_log_error_does_not_mask_http_error(self):
        def fail(request):
            raise HTTPError(request.full_url, 400, 'failure', {}, io.BytesIO(b'original cause'))
        invoke, row, _, _ = self.record(fail)
        with patch.object(Path, 'read_bytes', side_effect=OSError('unreadable log')):
            with self.assertRaises(HTTPError): invoke()
        self.assertIn('unreadable log', row['turns'][0]['native_log_error'])
        self.assertEqual(row['turns'][0]['error_body'], 'original cause')

    def test_query_replay_must_belong_to_question_and_match_bounds(self):
        with self.assertRaises(ValueError):
            verify_question_query('KVMEM_QUERY source=tool_or_continuation recall=0\n', 12536, 12501, 512, 2048)
        question = ('KVMEM_QUERY query_begin=12511 query_end=12527 prefix=12511\n'
                    'KVMEM_QUERY replay=1 reason=bounded_replay begin=12511 end=12536\n')
        self.assertEqual(verify_question_query(question, 12536, 12501, 512, 2048)['scope'], 'question_turn_only')
        with self.assertRaises(ValueError):
            verify_question_query(question.replace('prefix=12511', 'prefix=12510'), 12536, 12501, 512, 2048)

    def test_smoke_and_incomplete_data_are_rejected(self):
        for value in ({'status': 'pass', 'smoke_only': True}, {'status': 'pass', 'measurement_kind': 'smoke'}, {'status': 'running'}):
            with self.subTest(value=value), self.assertRaises(ValueError): validate_performance_input(value)
        raw = {'status': 'pass', 'started': 123, 'workers': {'candidate': {'sha256': 'test'}}, 'smoke_only': False}
        validate_performance_input(raw)
        self.assertEqual(input_provenance(raw)['workers'], raw['workers'])
        self.assertIsNone(input_provenance({})['smoke_only'])


if __name__ == '__main__': unittest.main()
