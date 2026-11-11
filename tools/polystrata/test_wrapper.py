"""Model-free admission checks for the direct Python launcher."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class WrapperTest(unittest.TestCase):
    def test_admission_and_argument_boundaries(self):
        with tempfile.TemporaryDirectory(prefix="polystrata wrapper ") as directory:
            root = Path(directory)
            source = root / "source with spaces"
            (source / "serve").mkdir(parents=True)
            tokenizer = root / "tokenizer with spaces"
            tokenizer.mkdir()
            for path in (root/"worker.exe", root/"vision worker.exe", root/"mmproj with spaces.gguf",
                         root/"model with spaces.gguf", source/"serve/server.py", source/"serve/kvmem.py",
                         *(tokenizer/name for name in ("vocab.json","merges.txt","tokenizer.json","token_type.json","chat_template.jinja"))):
                path.touch()
            command = [sys.executable, str(Path(__file__).with_name("server.py")), "--dry-run",
                       "--source",str(source),"--worker",str(root/"worker.exe"),"--model",str(root/"model with spaces.gguf"),
                       "--tokenizer",str(tokenizer),"--gpu","GPU-12345678", "--context","16384","--budget","4096","--reserve","8192"]
            env = dict(os.environ)
            env.pop("STRATA_API_KEY", None)
            env.pop("STRATA_FATTN", None)
            env.pop("STRATA_FDEC", None)
            def run(extra):
                return subprocess.run(command+extra, capture_output=True, text=True, env=env,
                                      creationflags=0x08000000 if os.name == "nt" else 0)
            valid = run(["--", "--stage-mib", "512"])
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertEqual(json.loads(valid.stdout)['env']['STRATA_FATTN'], '1')
            self.assertEqual(json.loads(valid.stdout)['env']['STRATA_FDEC'], '1')
            native_args = json.loads(valid.stdout)['args']
            self.assertEqual(native_args[native_args.index('--ram-experts-mib') + 1], '-1')
            for value in ('-2', '-1', '0', '4096'):
                explicit = run(['--', '--ram-experts-mib', value])
                self.assertEqual(explicit.returncode, 0, explicit.stderr)
                argv = json.loads(explicit.stdout)['args']
                self.assertEqual(argv.count('--ram-experts-mib'), 1)
                self.assertEqual(argv[argv.index('--ram-experts-mib') + 1], value)
            for option in ('--ram-experts-mib=4096', '--ram-experts-mib 4096', '--ram-experts-mib\t4096'):
                joined = run(['--', option])
                self.assertEqual(joined.returncode, 0, joined.stderr)
                self.assertIn(option, json.loads(joined.stdout)['args'])
                self.assertNotIn('--ram-experts-mib', json.loads(joined.stdout)['args'])
            env.update(STRATA_FATTN='0', STRATA_FDEC='0')
            disabled = run([])
            self.assertEqual(disabled.returncode, 0, disabled.stderr)
            self.assertEqual(json.loads(disabled.stdout)['env']['STRATA_FATTN'], '0')
            self.assertEqual(json.loads(disabled.stdout)['env']['STRATA_FDEC'], '0')
            env.pop('STRATA_FATTN')
            env.pop('STRATA_FDEC')
            for scorer in ('page-softmax', 'meanq-cosine'):
                for alpha in ('0', '0.2'):
                    for history in ('--lrfu', '--no-lrfu'):
                        combined = run(['--query-scorer', scorer, '--graph-alpha', alpha, history,
                                        '--query-max-tokens', '128', '--replay-max-tokens', '256'])
                        self.assertEqual(combined.returncode, 0, combined.stderr)
                        argv = json.loads(combined.stdout)['args']
                        self.assertEqual(argv[argv.index('--kvmem-query-scorer')+1], scorer)
                        self.assertEqual(float(argv[argv.index('--kvmem-graph-alpha')+1]), float(alpha))
                        self.assertEqual('--kvmem-lrfu' in argv, history == '--lrfu')
                        self.assertEqual('--no-kvmem-lrfu' in argv, history == '--no-lrfu',
                                         'Explicit off must override the native cosine default')
                        self.assertEqual(argv[argv.index('--kvmem-query-replay-max-tokens')+1], '256')
            for size in (32,64):
                configured=run(['--kvmem-block-tokens',str(size),'--budget','128','--reserve',str(size),
                                '--prefill',str(size),'--sink','0','--recent',str(size)])
                self.assertEqual(configured.returncode,0,configured.stderr)
                argv=json.loads(configured.stdout)['args']
                self.assertEqual(argv[argv.index('--kvmem-block-tokens')+1],str(size))
            for invalid in (['--kvmem-block-tokens','16'],['--kvmem-block-tokens','32','--budget','96'],
                            ['--kvmem-block-tokens','64','--reserve','32'],
                            ['--kvmem-block-tokens','32','--recent','16'],
                            ['--','--kvmem-block-tokens','32']):
                self.assertEqual(run(invalid).returncode,2,invalid)
            rounded = run(['--budget','10001','--reserve','1001','--prefill','999','--sink','101','--recent','101'])
            self.assertEqual(rounded.returncode,0,rounded.stderr)
            plan = json.loads(rounded.stdout)
            expected = {'--kvmem-block-tokens':'32','--kvmem-budget':'9984','--kvmem-gen-reserve':'992',
                        '--prompt-batch':'992','--kvmem-sink-tokens':'96','--kvmem-recent-tokens':'96'}
            for flag, value in expected.items():
                self.assertEqual(plan['args'][plan['args'].index(flag)+1],value)
            self.assertEqual(len(plan['token_adjustments']),5)
            self.assertEqual(rounded.stderr.count('KVMEM_CONFIG'),5)
            if os.name == 'nt':
                launcher = Path(__file__).resolve().parents[2] / 'scripts/windows/start-backend.ps1'
                powershell = str(Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe')
                launched = subprocess.run([powershell,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(launcher),
                    '-Backend','polystrata','-Worker',str(root/'worker.exe'),'-Model',str(root/'model with spaces.gguf'),
                    '-PolystrataSource',str(source),'-Tokenizer',str(tokenizer),'-Python',sys.executable,
                    '-Gpu','GPU-12345678','-Budget','10001','-Reserve','1001','-Prefill','999',
                    '-SinkTokens','101','-RecentTokens','101','-DryRun'],
                    capture_output=True,encoding='oem',env=env,creationflags=0x08000000)
                self.assertEqual(launched.returncode,0,launched.stderr)
                shared = json.loads(launched.stdout[launched.stdout.index('{'):])
                self.assertEqual(shared['token_settings']['block_tokens'],32)
                self.assertEqual(shared['token_settings']['effective'],dict(Budget=9984,Reserve=992,Prefill=992,SinkTokens=96,RecentTokens=96,Context=262144))
                self.assertEqual(len(shared['token_settings']['adjustments']),5)
                for flag,value in expected.items():
                    self.assertEqual(shared['native_argv'][shared['native_argv'].index(flag)+1],value)
            aligned = run(['--budget','9984','--reserve','992','--prefill','992','--sink','96','--recent','96'])
            self.assertEqual(aligned.returncode,0,aligned.stderr)
            self.assertEqual(json.loads(aligned.stdout)['token_adjustments'],[])
            # The original sum exceeds Context by two; the effective capacities fit.
            boundary = run(['--context','256','--budget','129','--reserve','129','--prefill','128','--max-tokens','0'])
            self.assertEqual(boundary.returncode,0,boundary.stderr)
            cfg = json.loads(valid.stdout)
            self.assertEqual(cfg['args'][cfg['args'].index('--kvmem-block-tokens')+1],'32')
            self.assertEqual(cfg["args"][1], str(root/"model with spaces.gguf"))
            self.assertEqual(cfg["gpu"], 0)
            self.assertEqual(cfg["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", cfg["env"])
            thinking = run(["--thinking-mode","on","--thinking-budget","128","--reasoning-effort","high","--preserve-thinking"])
            self.assertEqual(thinking.returncode,0,thinking.stderr)
            thinking_cfg=json.loads(thinking.stdout)
            self.assertEqual(thinking_cfg["reasoning_effort"],"high")
            self.assertEqual(thinking_cfg["reasoning_budget_tokens"],128)
            self.assertTrue(thinking_cfg["preserve_thinking"])
            self.assertNotIn("--thinking-budget",thinking_cfg["args"])
            self.assertEqual(json.loads(run(["--thinking-mode","off"]).stdout)["reasoning_effort"],"none")
            for bad in (["--thinking-mode","on","--reasoning-effort","none"],
                        ["--thinking-mode","off","--reasoning-effort","high"],
                        ["--thinking-budget","-1"],["--reasoning-effort","xhigh"]):
                self.assertEqual(run(bad).returncode,2)
            self.assertIn("--kvmem-history-reuse", cfg["args"])
            reuse = run(["--history-reuse"])
            self.assertEqual(reuse.returncode, 0, reuse.stderr)
            self.assertIn("--kvmem-history-reuse", json.loads(reuse.stdout)["args"])
            retained = run(["--history-reuse","--retained-sessions","4"])
            self.assertEqual(retained.returncode, 0, retained.stderr)
            retained_args = json.loads(retained.stdout)["args"]
            self.assertEqual(retained_args[retained_args.index("--kvmem-sessions")+1],"4")
            mtp = run(["--mtp-drafts","1","--history-reuse","--retained-sessions","2"])
            self.assertEqual(mtp.returncode,0,mtp.stderr)
            mtp_args = json.loads(mtp.stdout)["args"]
            self.assertIn("--kvmem-mtp",mtp_args)
            self.assertNotIn("--no-draft",mtp_args)
            self.assertEqual(mtp_args[mtp_args.index("--window")+1],"2")
            wide = run(["--mtp-drafts","3"])
            self.assertEqual(wide.returncode,0,wide.stderr)
            wide_args = json.loads(wide.stdout)["args"]
            self.assertIn("--kvmem-mtp",wide_args)
            self.assertEqual(wide_args[wide_args.index("--window")+1],"4")
            concurrent = run(["--concurrency","2","--history-reuse",
                              "--retained-sessions","2","--mtp-drafts","2"])
            self.assertEqual(concurrent.returncode,0,concurrent.stderr)
            concurrent_cfg = json.loads(concurrent.stdout)
            self.assertEqual(concurrent_cfg["gpu"],0)
            self.assertNotIn("replicas",concurrent_cfg)
            self.assertEqual(concurrent_cfg["args"][concurrent_cfg["args"].index("--batch")+1],"2")
            vision = run(["--vision-worker",str(root/"vision worker.exe"),"--mmproj",str(root/"mmproj with spaces.gguf"),
                          "--vision-device","cpu","--vision-tokens","768","--vision-threads","6"])
            self.assertEqual(vision.returncode,0,vision.stderr)
            vision_cfg=json.loads(vision.stdout)["vision"]
            self.assertFalse(vision_cfg["gpu"])
            self.assertEqual(vision_cfg["max_tokens"],768)
            self.assertEqual(vision_cfg["threads"],6)
            concurrent_vision = run(["--concurrency","2",
                                     "--vision-worker",str(root/"vision worker.exe"),
                                     "--mmproj",str(root/"mmproj with spaces.gguf"),"--vision-device","cpu"])
            self.assertEqual(concurrent_vision.returncode,0,concurrent_vision.stderr)
            disk = run(["--disk-path",str(root/"disk with spaces"),"--disk-mib","512","--mtp-drafts","1"])
            self.assertEqual(disk.returncode,0,disk.stderr)
            disk_args=json.loads(disk.stdout)["args"]
            self.assertEqual(disk_args[disk_args.index("--kvmem-disk-mib")+1],"512")
            unlimited = run(["--max-tokens","0"])
            self.assertEqual(unlimited.returncode,0,unlimited.stderr)
            self.assertIsNone(json.loads(unlimited.stdout)["kvmem_default_max_tokens"])
            for extra in (["--budget","0"], ["--budget","127"], ["--reserve","31"], ["--context","2147483648"],
                          ["--prefill","0"], ["--max-tokens","-1"], ["--recent","0"], ["--sink","4096"],
                          ["--host-mib","0"], ["--port","65536"], ["--lrfu-lambda","nan"], ["--lrfu-top-frac","0"],
                          ["--host","0.0.0.0"], ["--gpu","GPU-a,GPU-b"], ["--concurrency","9"],
                          ["--gpu","GPU-a,GPU-a","--concurrency","2"], ["--concurrency","9"],
                          ["--","--model","other.gguf"],
                          ["--","--window","4"], ["--","--kvmem-budget=128"], ["--","--conversation-cache-mib","1"],
                          ["--","--batch","2"],
                          ["--no-history-reuse","--retained-sessions","2"], ["--history-reuse","--retained-sessions","0"],
                          ["--history-reuse","--retained-sessions","17"], ["--mtp-drafts","4"],
                          ["--mtp-drafts","1","--","--no-draft"], ["--disk-mib","64"],
                          ["--vision-worker",str(root/"vision worker.exe")], ["--mmproj",str(root/"mmproj with spaces.gguf")],
                          ["--vision-worker",str(root/"missing.exe"),"--mmproj",str(root/"mmproj with spaces.gguf")],
                          ["--vision-worker",str(root/"vision worker.exe"),"--mmproj",str(root/"missing.gguf")],
                          ["--vision-worker",str(root/"vision worker.exe"),"--mmproj",str(root/"mmproj with spaces.gguf"),
                           "--vision-tokens","0"],
                          ["--disk-path",str(root/"disk")], ["--disk-path",str(root/"disk"),"--disk-mib","0"]):
                with self.subTest(extra=extra):
                    self.assertEqual(run(extra).returncode, 2)
            # Incomplete exports must not fall back to another model's template.
            (tokenizer/"chat_template.jinja").unlink()
            self.assertEqual(run([]).returncode, 2)


if __name__ == "__main__":
    unittest.main()
