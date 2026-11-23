"""KVMem forbids middle truncation before changing history, settings or workers."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

SOURCE = Path(os.environ.get("POLYSTRATA_SOURCE", Path(__file__).resolve().parents[2] / "backends/polystrata"))
sys.path.insert(0, str(SOURCE))
from serve import runconfig, server
from serve.frontend import ChatTemplate
from serve.kvmem import Prompt


class PromptPolicy(unittest.TestCase):
    def setUp(self):
        self.tok = server.ByteTokenizer()
        self.engine = server.MockEngine(self.tok, "ok", max_context=512)
        self.svc = server.Service(self.engine, self.tok, ChatTemplate(SOURCE / "serve/chat_template.jinja"))
        self.addCleanup(self.svc.drop_embeddings)

    def test_startup_settings_reject_even_lazy_kvmem(self):
        for lazy in (False, True):
            cfg = {"args": ["--kvmem-budget", "128"], "truncate_middle": True, "lazy_load": lazy}
            with self.subTest(lazy=lazy), self.assertRaisesRegex(ValueError, "truncate_middle.*KVMem"):
                server.model_settings(cfg)
        for value in (None, False):
            cfg = {"args": ["--kvmem-budget", "128"]}
            if value is not None:
                cfg["truncate_middle"] = value
            self.assertFalse(server.model_settings(cfg)["truncate_middle"])
        for args in ([], ["--kvmem-budget", "0"]):
            self.assertTrue(server.model_settings({"args": args, "truncate_middle": True})["truncate_middle"])

    def test_switch_rejects_before_unloading_current_worker(self):
        cfg = {"args": ["--kvmem-budget", "128"], "truncate_middle": True}
        before = self.svc.engine
        with mock.patch.object(server, "run_config", return_value=cfg), \
                mock.patch.object(server, "load_tokenizer") as tokenizer, \
                mock.patch.object(server, "start_model") as start, \
                mock.patch.object(self.engine, "unload", create=True) as unload:
            with self.assertRaisesRegex(ValueError, "truncate_middle.*KVMem"):
                self.svc._switch_to({"config": "unused.json", "id": "bad-config"})
            tokenizer.assert_not_called()
            start.assert_not_called()
            unload.assert_not_called()
        self.assertIs(self.svc.engine, before)

    def test_settings_application_is_atomic_and_can_disable(self):
        self.engine.info = {"kvmem": 1}
        before = self.svc.fit_max_tokens
        with self.assertRaisesRegex(ValueError, "truncate_middle.*KVMem"):
            self.svc.apply_settings({"fit_max_tokens": not before, "truncate_middle": True})
        self.assertEqual(self.svc.fit_max_tokens, before)
        self.assertFalse(self.svc.truncate_middle)
        self.svc.truncate_middle = True
        self.svc.apply_settings({"truncate_middle": False})
        self.assertFalse(self.svc.truncate_middle)

    def test_runtime_rejects_short_long_image_and_tool_requests_before_encoding(self):
        requests = [
            [{"role": "user", "content": "hi"}],
            [{"role": "user", "content": "x" * 800}],
            [{"role": "user", "content": [{"type": "text", "text": "read"},
                {"type": "image", "source": "https://example.invalid/image.png"}]}],
            [{"role": "tool", "content": "tool result", "tool_call_id": "call_1"}],
        ]
        self.svc.truncate_middle = True
        for info, spawn in (({"kvmem": 1}, None), ({"kvmem": "1"}, None),
                            ({}, ("worker", ["--kvmem-budget", "128"]))):
            self.engine.info, self.engine.spawn = info, spawn
            for messages in requests:
                with self.subTest(info=info, spawn=spawn, messages=messages), \
                        mock.patch.object(self.svc, "encode_prompt") as encode, \
                        mock.patch.object(self.svc, "_note_unreadable_tool_images") as fetch:
                    with self.assertRaisesRegex(ValueError, "truncate_middle.*KVMem"):
                        self.svc.prepare(messages, None, {}, 32)
                    encode.assert_not_called()
                    fetch.assert_not_called()
            with self.assertRaisesRegex(ValueError, "truncate_middle.*KVMem"):
                self.svc.truncate_prompt_middle([1, 2], 32)

    def test_disabled_preserves_query_and_rejects_overflow(self):
        self.engine.info = {"kvmem": 1}
        messages = [{"role": "user", "content": "question: CODE-7319?"}]
        for settings in ({}, {"truncate_middle": False}):
            self.svc.apply_settings(server.model_settings({"args": ["--kvmem-budget", "128"], **settings}))
            ids, _, _ = self.svc.prepare(messages, None, {}, 32)
            self.assertIsInstance(ids, Prompt)
            begin, end = ids.kvmem_query
            self.assertLess(begin, end)
            self.assertIn("CODE-7319", self.tok.decode(ids[begin:end]))
            with self.assertRaises(ValueError):
                self.svc.prepare([{"role": "user", "content": "x" * 800}], None, {}, 32)

    def test_native_truncation_keeps_exact_prefix_and_suffix(self):
        self.svc.truncate_middle = True
        messages = [{"role": "user", "content": "x" * 800}]
        original = self.svc.encode_prompt(messages, None, {})
        ids, _, _ = self.svc.prepare(messages, None, {}, 32)
        room = self.engine.max_context - server.CTX_SLACK - 32
        first, last = (room + 1) // 2, room // 2
        self.assertEqual(ids, original[:first] + original[-last:])

    def test_config_edit_rejects_and_allows_recovery(self):
        cfg = {"args": ["--kvmem-budget", "128"], "truncate_middle": False}
        original = json.dumps(cfg)
        with self.assertRaisesRegex(ValueError, "truncate_middle.*KVMem"):
            runconfig.apply(cfg, {"truncate_middle": True, "fit_max_tokens": True})
        self.assertEqual(json.dumps(cfg), original)
        for value in (False, None):
            new, _ = runconfig.apply({**cfg, "truncate_middle": True}, {"truncate_middle": value})
            self.assertFalse(new.get("truncate_middle", False))
        self.assertTrue(runconfig.apply({}, {"truncate_middle": True})[0]["truncate_middle"])

    def test_joined_config_args_follow_startup_interpretation(self):
        for arg in ("--kvmem-budget 128", "--kvmem-budget=128"):
            cfg = {"args": [arg], "truncate_middle": False}
            with self.subTest(arg=arg), self.assertRaisesRegex(ValueError, "truncate_middle.*KVMem"):
                runconfig.apply(cfg, {"truncate_middle": True})
            self.assertEqual(cfg, {"args": [arg], "truncate_middle": False})
            with tempfile.TemporaryDirectory(prefix="poly-config-") as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps({**cfg, "truncate_middle": True}))
                with self.assertRaisesRegex(ValueError, "truncate_middle.*KVMem"):
                    server.model_settings(server.run_config(path))


class PromptPolicyHTTP(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="poly-prompt-policy-")
        self.addCleanup(directory.cleanup)
        self.config = Path(directory.name) / "strata-test.json"
        tok = server.ByteTokenizer()
        self.engine = server.MockEngine(tok, "ok", max_context=256)
        self.engine.info = {"kvmem": 1}
        self.svc = server.Service(self.engine, tok, ChatTemplate(SOURCE / "serve/chat_template.jinja"))
        self.svc.config_path = str(self.config)
        self.httpd = server.serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = "http://127.0.0.1:" + str(self.httpd.server_address[1])

    def request(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.code, json.load(response)

    def test_config_rejection_does_not_write_config_or_backup(self):
        for args in ([], ["--kvmem-budget", "128"]):
            self.config.write_text(json.dumps({"args": args}), encoding="utf-8")
            before = self.config.read_bytes()
            status, body = self.request("/config", {"set": {"fit_max_tokens": True, "truncate_middle": True}})
            self.assertEqual(status, 400, body)
            self.assertIn("truncate_middle", body["error"]["message"])
            self.assertEqual(self.config.read_bytes(), before)
            self.assertEqual(list(self.config.parent.iterdir()), [self.config])
        for value in (False, None):
            self.config.write_text(json.dumps({"args": ["--kvmem-budget", "128"], "truncate_middle": True}))
            status, body = self.request("/config", {"set": {"truncate_middle": value}})
            self.assertEqual(status, 200, body)
            self.assertFalse(json.loads(self.config.read_text()).get("truncate_middle", False))

    def test_three_api_routes_return_400_for_stale_enabled_setting(self):
        self.svc.truncate_middle = True
        for path, body in (
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
            ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}),
            ("/v1/responses", {"input": "hi", "store": False}),
        ):
            with self.subTest(path=path), mock.patch.object(self.engine, "generate") as generate:
                status, body = self.request(path, body)
                self.assertEqual(status, 400, body)
                self.assertIn("truncate_middle", body["error"]["message"])
                generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
