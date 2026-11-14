"""Exercise PowerShell binding and the argv received by a real child process."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHELLS = [path for name in ('powershell.exe', 'pwsh.exe') if (path := shutil.which(name))]


@unittest.skipUnless(os.name == 'nt' and SHELLS, 'Windows PowerShell launchers')
class WorkerArgsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary_dir = tempfile.TemporaryDirectory(prefix='kvmem-contract-worker-')
        cls.addClassCleanup(cls.binary_dir.cleanup)
        directory = Path(cls.binary_dir.name)
        source = directory / 'worker.cs'
        source.write_text(r'''
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;
using System.Web.Script.Serialization;
class Worker {
    static void Main(string[] args) {
        Console.OutputEncoding = new UTF8Encoding(false);
        var executable = Assembly.GetExecutingAssembly().Location;
        var json = new JavaScriptSerializer();
        var gpu = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES");
        if (args.Length == 1 && args[0] == "--kvmem-capabilities") {
            File.WriteAllText(executable + ".probe.json", json.Serialize(new { gpu = gpu }));
            Console.WriteLine(File.ReadAllText(executable + ".capabilities.json"));
            return;
        }
        var received = new string[Math.Max(0, args.Length - 1)];
        if (received.Length > 0) Array.Copy(args, 1, received, 0, received.Length);
        var result = json.Serialize(new { received = received, gpu = gpu });
        File.WriteAllText(executable + ".started.json", result);
        Console.WriteLine(result);
    }
}
''', encoding='utf-8')
        compiler = Path(os.environ['SystemRoot']) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
        if not compiler.is_file():
            compiler = Path(os.environ['SystemRoot']) / 'Microsoft.NET/Framework/v4.0.30319/csc.exe'
        cls.binary = directory / 'worker.exe'
        # A tiny model-free argv/contract fixture, not an inference-engine build.
        subprocess.run([str(compiler), '/nologo', '/reference:System.Web.Extensions.dll',
                        '/out:' + str(cls.binary), str(source)], check=True, capture_output=True, timeout=30)

    def test_canonical_dflash2_maps_to_each_native_backend(self):
        draft = self.folder / "draft & ' spaces.gguf"; draft.touch()
        target = self.folder / 'target.gguf'; target.touch()
        for shell in SHELLS:
            with self.subTest(shell=shell, backend='ninfer'):
                plan = self.plan(self.run_launcher(shell, 'start-backend.ps1', [
                    '-SpeculativeMode', 'dflash2', '-DraftTokens', '5', '-DraftPolicy', 'fixed', '-Concurrency', '8']))
                args = plan['argv']; self.assertEqual(args[args.index('--spec') + 1], 'dflash2')
                self.assertEqual(args[args.index('--max-concurrency') + 1], '8')
                self.assertEqual(args.count('--spec'), 1); self.assertNotIn('--adaptive-mtp', args)
            def llama(extra):
                return subprocess.run([shell, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File',
                    str(ROOT / 'scripts/windows/start-backend.ps1'), '-Backend', 'llamacpp', '-Worker', sys.executable,
                    '-Model', str(target), '-Gpu', 'GPU-test', '-SpeculativeMode', 'dflash2', '-DraftTokens', '5',
                    '-DryRun', *extra], capture_output=True, encoding='utf-8', errors='replace', timeout=45)
            with self.subTest(shell=shell, backend='llamacpp'):
                args = self.plan(llama(['-DraftModel', str(draft)]))['argv']
                self.assertEqual(args[args.index('--spec-type') + 1], 'draft-dflash')
                self.assertEqual(args[args.index('--spec-draft-model') + 1], str(draft))
                self.assertEqual(args[args.index('--spec-draft-n-max') + 1], '5')
                paired = self.plan(llama(['-DraftModel', str(draft), '-Concurrency', '8']))['argv']
                self.assertEqual(paired[paired.index('--parallel') + 1], '8')
                for invalid in ([], ['-DraftModel', str(draft), '-DraftPolicy', 'adaptive'],
                                ['-DraftModel', str(draft), '-Concurrency', '9']):
                    self.assertNotEqual(llama(invalid).returncode, 0)

    def test_cpu_vision_idle_timeout_reaches_worker(self):
        for shell in SHELLS:
            for seconds in ('0', '2', '300'):
                with self.subTest(shell=shell, seconds=seconds):
                    args = self.plan(self.run_launcher(shell, 'start-backend.ps1',
                        ['-Vision', '-VisionResidency', 'cpu', '-CpuVisionIdleSeconds', seconds]))['argv']
                    self.assertEqual(args[args.index('--cpu-vision-idle-seconds') + 1], seconds)
            args = self.plan(self.run_launcher(shell, 'start-backend.ps1', []))['argv']
            self.assertNotIn('--cpu-vision-idle-seconds', args)
            shortcut = subprocess.run([shell, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                '-File', str(ROOT / 'scripts/windows/start-ninfer.ps1'), '-Model', str(self.model),
                '-Worker', sys.executable, '-Gpu', 'GPU-test', '-CpuVisionIdleSeconds', '12', '-DryRun'],
                capture_output=True, encoding='utf-8', errors='replace', timeout=45)
            args = self.plan(shortcut)['argv']
            self.assertEqual(args[args.index('--cpu-vision-idle-seconds') + 1], '12')
            conflict = self.run_launcher(shell, 'start-backend.ps1', ['-Vision', '-VisionResidency', 'cpu',
                '-CpuVisionIdleSeconds', '12', '-WorkerArgs', '--cpu-vision-idle-seconds', '2'])
            self.assertNotEqual(conflict.returncode, 0)


    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='kvmem-argv-')
        self.folder = Path(self.tmp.name)
        self.model = self.folder / "model & ' spaces.ninfer"
        self.model.touch()
        self.worker = self.folder / "worker & ' spaces.exe"
        shutil.copyfile(self.binary, self.worker)
        self.contract_file = Path(str(self.worker) + '.capabilities.json')
        self.contract_file.write_text(json.dumps(dict(backend='ninfer', legacy_query_replay_revision=1)))
        self.probe_file = Path(str(self.worker) + '.probe.json')
        self.started_file = Path(str(self.worker) + '.started.json')
        self.file = self.folder / "args & ' spaces.json"

    def tearDown(self):
        self.tmp.cleanup()

    def run_launcher(self, shell, entry, extra, *, dry=True, gpu='GPU-test'):
        args = [shell, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File',
                str(ROOT / 'scripts/windows' / entry), '-Model', str(self.model),
                '-Worker', str(self.worker), '-Gpu', gpu]
        if entry == 'start-backend.ps1':
            args += ['-Backend', 'ninfer']
        elif entry == 'start-ninfer.ps1':
            args += ['-NoVision', '-MtpDrafts', '0']
        if dry:
            args += ['-DryRun']
        return subprocess.run(args + extra, capture_output=True, encoding='utf-8', errors='replace', timeout=45)

    def plan(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout[result.stdout.index('{'):])

    def test_old_worker_rejected_before_inference_for_zero_or_omitted_limits(self):
        self.contract_file.write_text('{"backend":"ninfer"}')
        cases = ([], ['-QueryMaxTokens', '512'], ['-QueryReplayMaxTokens', '2048'],
                 ['-QueryMaxTokens', '0', '-QueryReplayMaxTokens', '2048'],
                 ['-QueryMaxTokens', '512', '-QueryReplayMaxTokens', '0'],
                 ['-QueryMaxTokens', '0', '-QueryReplayMaxTokens', '0'])
        for shell in SHELLS:
            for extra in cases:
                with self.subTest(shell=shell, extra=extra):
                    result = self.run_launcher(shell, 'start-backend.ps1', extra, dry=False)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('full Query/replay', result.stderr)
                    self.assertIn('QueryReplayMaxTokens', result.stderr)
                    self.assertFalse(self.started_file.exists())
                    self.assertEqual(json.loads(self.probe_file.read_text())['gpu'], '-1')

    def test_query_capability_revision_must_be_a_positive_integer(self):
        for shell in SHELLS:
            for revision in (None, False, True, 0, -1, '1', 1.0):
                with self.subTest(shell=shell, revision=revision):
                    self.contract_file.write_text(json.dumps(dict(backend='ninfer', legacy_query_replay_revision=revision)))
                    result = self.run_launcher(shell, 'start-backend.ps1', [], dry=False)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('full Query/replay', result.stderr)
                    self.assertFalse(self.started_file.exists())
            self.contract_file.write_text('{"backend":"llamacpp","legacy_query_replay_revision":1}')
            self.assertNotEqual(self.run_launcher(shell, 'start-backend.ps1', [], dry=False).returncode, 0)
            self.assertFalse(self.started_file.exists())

    def test_compatible_worker_keeps_full_query_defaults_through_wrappers(self):
        for shell in SHELLS:
            for entry in ('start-backend.ps1', 'start-ninfer.ps1', 'start-dflash2.ps1'):
                with self.subTest(shell=shell, entry=entry):
                    result = self.plan(self.run_launcher(shell, entry, [], dry=False))
                    self.assertTrue(self.started_file.exists())
                    self.assertEqual(result['gpu'], 'GPU-test')
                    self.assertEqual(json.loads(self.probe_file.read_text())['gpu'], '-1')
                    for flag in ('--kvmem-query-max-tokens', '--kvmem-query-replay-max-tokens'):
                        self.assertEqual(result['received'][result['received'].index(flag) + 1], '0')

    def test_old_worker_keeps_explicit_positive_query_limits(self):
        self.contract_file.write_text('{"backend":"ninfer"}')
        for shell in SHELLS:
            result = self.plan(self.run_launcher(shell, 'start-backend.ps1',
                ['-QueryMaxTokens', '512', '-QueryReplayMaxTokens', '2048'], dry=False))
            self.assertFalse(self.probe_file.exists())
            for flag, value in (('--kvmem-query-max-tokens', '512'), ('--kvmem-query-replay-max-tokens', '2048')):
                self.assertEqual(result['received'][result['received'].index(flag) + 1], value)

    def test_worker_args_cannot_bypass_query_capability_check(self):
        self.contract_file.write_text('{"backend":"ninfer"}')
        for shell in SHELLS:
            for flag in ('--kvmem-query-max-tokens', '--kvmem-query-replay-max-tokens'):
                for native in ([flag, '0'], [flag + '=0']):
                    with self.subTest(shell=shell, native=native):
                        self.file.write_text(json.dumps(native))
                        result = self.run_launcher(shell, 'start-backend.ps1', [
                            '-QueryMaxTokens', '512', '-QueryReplayMaxTokens', '2048',
                            '-WorkerArgsFile', str(self.file)], dry=False)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('WorkerArgs cannot override', result.stderr)
                        self.assertFalse(self.probe_file.exists())
                        self.assertFalse(self.started_file.exists())

    def test_dflash2_pipeline_reaches_child(self):
        for shell in SHELLS:
            result = self.plan(self.run_launcher(shell, 'start-backend.ps1',
                ['-SpeculativeMode', 'dflash2', '-DraftTokens', '5', '-DraftPolicy', 'fixed'],
                dry=False, gpu='GPU-first,GPU-second'))
            self.assertEqual(result['gpu'], 'GPU-first,GPU-second')
            argv = result['received']
            self.assertEqual(argv[argv.index('--devices') + 1], '0,1')
            self.assertEqual(argv[argv.index('--spec') + 1], 'dflash2')
            self.assertEqual(argv[argv.index('--draft-tokens') + 1], '5')

    def test_file_invocation_collects_multiple_native_arguments(self):
        native = ['--default-reasoning-effort', 'medium', '--preserve-thinking']
        for shell in SHELLS:
            for entry in ('start-backend.ps1', 'start-ninfer.ps1'):
                with self.subTest(shell=shell, entry=entry):
                    plan = self.plan(self.run_launcher(shell, entry, ['-WorkerArgs', *native]))
                    self.assertEqual(plan['argv'][-len(native):], native)

    def test_file_preserves_exact_child_argv(self):
        native = ['--custom', 'two words', '', '"quoted"', 'C:\\folder with spaces\\',
                  'literal; & | $(Write-Output bad)', '中文']
        self.file.write_text(json.dumps(native, ensure_ascii=False), encoding='utf-8-sig')
        for shell in SHELLS:
            for entry in ('start-backend.ps1', 'start-ninfer.ps1'):
                with self.subTest(shell=shell, entry=entry):
                    result = self.plan(self.run_launcher(shell, entry, ['-WorkerArgsFile', str(self.file)], dry=False))
                    self.assertEqual(result['received'][-len(native):], native)
                    self.assertEqual(result['gpu'], 'GPU-test')

    def test_empty_and_singleton_arrays(self):
        for native in ([], ['one argument with spaces']):
            self.file.write_text(json.dumps(native), encoding='utf-8')
            for shell in SHELLS:
                for entry in ('start-backend.ps1', 'start-ninfer.ps1'):
                    with self.subTest(shell=shell, entry=entry, native=native):
                        plan = self.plan(self.run_launcher(shell, entry, ['-WorkerArgsFile', str(self.file)]))
                        if native:
                            self.assertEqual(plan['argv'][-1:], native)
                        self.assertNotIn(None, plan['argv'])

    def test_invalid_inputs_and_setting_conflicts_fail_before_worker_start(self):
        cases = [(text, ['-WorkerArgsFile', str(self.file)]) for text in
                 ('{}', 'null', '"string"', '[1]', '[null]', '["\\u0000"]', '[broken', '[],"extra":[]')]
        cases += [
            ('[]', ['-WorkerArgsFile', str(self.file), '-WorkerArgs', '--threads', '2']),
            ('[]', ['-UnknownLauncherSetting', 'value']),
            ('["--default-reasoning-effort","low"]', ['-ReasoningEffort', 'medium', '-WorkerArgsFile', str(self.file)]),
            ('["--preserve-thinking"]', ['-PreserveThinking', '-WorkerArgsFile', str(self.file)]),
            ('[]', ['-ThinkingMode', 'off', '-ReasoningEffort', 'medium']),
            ('[]', ['-ReasoningEffort', 'unsupported']),
        ]
        # Validation is shared by both entry points; run malformed-file cases on
        # the common launcher and cover wrapper conflicts separately above.
        for shell in SHELLS:
            for text, extra in cases:
                with self.subTest(shell=shell, text=text, extra=extra):
                    self.file.write_text(text, encoding='utf-8')
                    result = self.run_launcher(shell, 'start-backend.ps1', extra, dry=False)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn('"received"', result.stdout)

    def test_named_thinking_options_and_api_redaction(self):
        for shell in SHELLS:
            for entry in ('start-backend.ps1', 'start-ninfer.ps1'):
                with self.subTest(shell=shell, entry=entry):
                    plan = self.plan(self.run_launcher(shell, entry, [
                        '-ThinkingBudget', '256', '-ReasoningEffort', 'medium', '-PreserveThinking',
                        '-ApiKey', 'test-only-key', '-ModelAlias', 'two words']))
                    argv = plan['argv']
                    self.assertEqual(argv[argv.index('--default-thinking-budget') + 1], '256')
                    self.assertEqual(argv[argv.index('--default-reasoning-effort') + 1], 'medium')
                    self.assertIn('--preserve-thinking', argv)
                    self.assertEqual(argv[argv.index('--model-id') + 1], 'two words')
                    self.assertNotIn('test-only-key', json.dumps(plan))

    def test_dflash2_profile_reaches_the_child_without_mtp_adaptation(self):
        expected = {
            '--max-context': '262144', '--kvmem-budget': '36864',
            '--kvmem-gen-reserve': '10240', '--kvmem-host-mib': '12288',
            '--prefill-chunk': '256', '--draft-tokens': '5',
            '--ngram-draft-tokens': '0', '--cuda-graph-allowance-mib': '64',
            '--kv-dtype': 'int8', '--vision-residency': 'cpu',
            '--vision-max-merged': '1024', '--max-concurrency': '1',
        }
        for shell in SHELLS:
            with self.subTest(shell=shell):
                result = self.plan(self.run_launcher(shell, 'start-dflash2.ps1', [], dry=False))
                argv = result['received']
                for flag, value in expected.items():
                    self.assertEqual(argv[argv.index(flag) + 1], value, flag)
                spec = len(argv) - 1 - argv[::-1].index('--spec')
                self.assertEqual(argv[spec + 1], 'dflash2')
                self.assertIn('--gdn-state-fp16', argv)
                self.assertNotIn('--adaptive-mtp', argv)
                self.assertEqual(result['gpu'], 'GPU-test')
                override = self.plan(self.run_launcher(shell, 'start-dflash2.ps1', ['-Context', '131072', '-Port', '18201']))
                self.assertEqual(override['argv'][override['argv'].index('--max-context') + 1], '131072')
                self.assertEqual(override['argv'][override['argv'].index('--port') + 1], '18201')


if __name__ == '__main__':
    unittest.main()
