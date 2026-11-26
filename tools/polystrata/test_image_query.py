"""The actual frontend must preserve no-Query and real Query spans across images."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest

SOURCE = Path(os.environ.get("POLYSTRATA_SOURCE", Path(__file__).resolve().parents[2]/"backends/polystrata/terminal"))
sys.path.insert(0, str(SOURCE))
from serve.frontend import ChatTemplate
from serve.kvmem import Prompt
from serve.server import ByteTokenizer, MockEngine, Service
from serve.kvmem_monitor import snapshot


class FakeVision:
    def __init__(self, root):
        self.dir = Path(root)
        self.rows = self.dir / "rows.sve"
        self.rows.write_bytes(b"fixture")
    def encode(self, source):
        return self.rows, 5


class ImageQuery(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="poly image query ")
        self.addCleanup(directory.cleanup)
        self.tok = ByteTokenizer()
        engine = MockEngine(self.tok, "ok", max_context=4096)
        engine.info = {"kvmem": 1}
        self.service = Service(engine, self.tok, ChatTemplate(SOURCE/"serve/chat_template.jinja"),
                               vision=FakeVision(directory.name))
        self.addCleanup(self.service.drop_embeddings)

    def image_message(self):
        return {"role": "user", "content": [{"type": "text", "text": "read this "},
                {"type": "image", "source": "fixture.png"}, {"type": "text", "text": " return CODE-7319"}]}

    def test_assistant_and_tool_continuations_have_no_query(self):
        for role in ("assistant", "tool"):
            with self.subTest(role=role):
                messages = [self.image_message()]
                if role == "tool":
                    messages.append({"role": "assistant", "content": "", "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": {}}}]})
                messages.append({"role": role, "content": "continue", "tool_call_id": "call_1"})
                ids, _, _ = self.service.prepare(messages, None, {}, max_new=8)
                self.assertIsInstance(ids, Prompt)
                self.assertEqual(ids.kvmem_query, (0, 0))
                self.assertEqual(ids.count(self.tok.encode("<|image_pad|>", parse_special=True)[0]), 5)
                self.service.drop_embeddings()

    def test_user_query_span_survives_image_expansion(self):
        ids, _, _ = self.service.prepare([self.image_message()], None, {}, max_new=8)
        first, last = ids.kvmem_query
        self.assertLess(first, last)
        text = self.tok.decode(ids[first:last])
        self.assertIn("read this", text)
        self.assertIn("CODE-7319", text)
        self.assertEqual(ids[first:last].count(self.tok.encode("<|image_pad|>", parse_special=True)[0]), 5)

    def test_image_hint_visible_during_prepare_and_cleared_on_failure(self):
        self.service.kvmem_monitor_gpu_uuid = 'GPU-test'
        # A previous request's engine statistics must not appear on an image
        # request that is still preparing and has not entered the engine.
        self.service.engine.progress_ms = 125
        self.service.engine.prefill_tok_s_mean = 250
        original = self.service.vision.encode
        observed = []
        def encode(source):
            rows = snapshot(self.service)['active']
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]['has_media'])
            self.assertIsNone(rows[0]['prefill_ms'])
            self.assertIsNone(rows[0]['prefill_tps'])
            self.assertNotIn('fixture.png', str(rows))
            observed.append(rows[0]['id'])
            return original(source)
        self.service.vision.encode = encode
        ids, thinking, limit = self.service.prepare([self.image_message()], None, {}, max_new=8)
        self.assertEqual(len(observed), 1)
        self.assertFalse(snapshot(self.service)['active'])
        for kind, value in self.service.run(ids, thinking, None, limit, {}, threading.Event()):
            rows = snapshot(self.service)['active']
            if rows:
                self.assertTrue(rows[0]['has_media'])
                self.assertEqual(rows[0]['id'], observed[0])
        self.assertFalse(snapshot(self.service)['active'])
        def fail(source):
            encode(source)
            raise RuntimeError('image preparation failed')
        self.service.vision.encode = fail
        with self.assertRaisesRegex(RuntimeError, 'image preparation failed'):
            self.service.prepare([self.image_message()], None, {}, max_new=8)
        self.assertFalse(snapshot(self.service)['active'])
        self.service.prepare([{'role': 'user', 'content': 'hello'}], None, {}, max_new=8)
        self.assertIsNone(self.service.embeddings.kvmem_monitor_id)

    def test_new_user_after_image_continuation_gets_only_new_query(self):
        messages = [self.image_message(), {"role": "assistant", "content": "noted"},
                    {"role": "user", "content": "What is the code?"}]
        ids, _, _ = self.service.prepare(messages, None, {}, max_new=8)
        first, last = ids.kvmem_query
        self.assertEqual(self.tok.decode(ids[first:last]), "What is the code?")
        self.assertNotIn(self.tok.encode("<|image_pad|>", parse_special=True)[0], ids[first:last])


if __name__ == "__main__":
    unittest.main()
