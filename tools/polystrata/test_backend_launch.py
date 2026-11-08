"""Backend selection without a GPU: exercise dry runs and the real launch handoff."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


class BackendLaunch(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="poly backend ")
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        source = root / "source"
        (source / "serve").mkdir(parents=True)
        tokenizer = root / "tokenizer"
        tokenizer.mkdir()
        for path in (root/"worker.exe", root/"model.gguf", source/"serve/server.py", source/"serve/kvmem.py",
                     *(tokenizer/name for name in ("vocab.json", "merges.txt", "tokenizer.json", "token_type.json", "chat_template.jinja"))):
            path.touch()
        spec = importlib.util.spec_from_file_location("poly_launcher", Path(__file__).with_name("server.py"))
        self.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.launcher)
        self.args = ["server.py", "--source", str(source), "--worker", str(root/"worker.exe"),
                     "--model", str(root/"model.gguf"), "--tokenizer", str(tokenizer)]

    def invoke(self, extra, dry=True, devices=None):
        captured = {}
        def frontend():
            self.frontend_argv = list(sys.argv)
            captured.update(json.loads(Path(sys.argv[sys.argv.index("--config")+1]).read_text(encoding="utf-8")))
            return 0
        query = mock.Mock(return_value=devices) if devices else mock.Mock(side_effect=AssertionError("unexpected NVIDIA query"))
        with mock.patch.object(sys, "argv", self.args + extra + (["--dry-run"] if dry else [])), \
             mock.patch.object(sys, "path", list(sys.path)), \
             mock.patch.dict(sys.modules, {"serve": SimpleNamespace(server=SimpleNamespace(main=frontend))}), \
             mock.patch.object(self.launcher.subprocess, "check_output", query), \
             contextlib.redirect_stdout(io.StringIO()) as out, \
             contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(self.launcher.main(), 0)
        self.last_stderr = err.getvalue()
        return json.loads(out.getvalue()) if dry else captured

    def test_mcp_file_reaches_frontend_only(self):
        path = Path(self.directory.name) / 'mcp config.json'
        path.write_text('{"mcpServers": {}}', encoding='utf-8')
        cfg = self.invoke(["--backend", "hip", "--gpu", "0", "--", "--elastic-experts-headroom",
                           "--mcp-config", str(path), "--threads", "4"], dry=False)
        self.assertEqual(self.frontend_argv[-2:], ["--mcp-config", str(path.resolve())])
        self.assertNotIn("--mcp-config", cfg["args"])
        self.assertNotIn(str(path), cfg["args"])
        index = cfg["args"].index("--elastic-experts-headroom")
        self.assertEqual(cfg["args"][index:index + 3], ["--elastic-experts-headroom", "--threads", "4"])

    def test_hip_handoff_skips_nvidia_and_keeps_ordinal(self):
        for dry in (True, False):
            for gpu in ("0", "1"):
                with self.subTest(dry=dry, gpu=gpu):
                    cfg = self.invoke(["--backend", "hip", "--gpu", gpu], dry)
                    self.assertEqual((cfg["backend"], cfg["gpu"]), ("hip", int(gpu)))
                    self.assertIn("KVMEM_EXPERIMENTAL", self.last_stderr)
                    self.assertNotIn("CUDA_DEVICE_ORDER", cfg["env"])
                    self.assertNotIn("kvmem_monitor_gpu_uuid", cfg)
                    argv = cfg["args"]
                    self.assertEqual(argv[argv.index("--kvmem-block-tokens")+1], "32")

    def test_default_cuda_and_uuid_mapping_are_preserved(self):
        uuid = "GPU-12345678"
        for backend in ([], ["--backend", "cuda"]):
            cfg = self.invoke(backend + ["--gpu", uuid], False, "7, " + uuid + "\n")
            self.assertEqual((cfg["backend"], cfg["gpu"]), ("cuda", 7))
            self.assertNotIn("KVMEM_EXPERIMENTAL", self.last_stderr)
            self.assertEqual(cfg["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            self.assertEqual(cfg["kvmem_monitor_gpu_uuid"], uuid)

    def test_invalid_device_syntax_is_an_admission_error(self):
        for backend, gpu in (("hip", "-1"), ("hip", "0,1"), ("hip", "GPU-12345678"),
                             ("hip", "2147483648"), ("hip", "9"*100), ("cuda", "0")):
            with self.subTest(backend=backend, gpu=gpu), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.invoke(["--backend", backend, "--gpu", gpu])
                self.assertEqual(error.exception.code, 2)

    def test_gemma_formats_and_external_assistant_reach_worker(self):
        assistant=Path(self.directory.name)/'assistant.gguf'
        assistant.write_bytes(b'GGUF fixture')
        for kv in ('f16','int8','int8k'):
            cfg=self.invoke(['--gpu','GPU-12345678','--kv-type',kv,'--mtp-drafts','3','--draft-model',str(assistant)])
            argv=cfg['args']
            self.assertEqual(argv[argv.index('--kv')+1], 'fp16' if kv=='f16' else kv)
            self.assertEqual(argv[argv.index('--draft-model')+1],str(assistant.resolve()))
            self.assertIn('--kvmem-mtp',argv)


if __name__ == "__main__":
    unittest.main()
