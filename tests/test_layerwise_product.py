"""Product-entry checks using disposable files, parameter previews and log reports only."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_launcher_plan as legacy
from launcher.plan import build_plan
from launcher.export import render_script
from launcher.recall import LAYERWISE_RESOURCES
from launcher.worker import strategy_profile, require_launch_features
from test_layerwise_launcher import contract
from console_memory import diagnose

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('layerwise_console', ROOT / 'scripts/local-console.py')
console = importlib.util.module_from_spec(spec)
spec.loader.exec_module(console)


class LayerwiseProductTest(unittest.TestCase):
    def setUp(self):
        self.fixture = legacy.PlanTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory = self.fixture.directory

    def config(self, backend='ninfer', **parameters):
        return self.fixture.config(backend, RecallStrategy='layerwise', **parameters)

    def test_selected_worker_profile_is_model_free_and_bound_to_selection(self):
        controller = console.Controller(ROOT, self.directory/'state',18199)
        caps = contract()
        caps.update(path=str(self.fixture.worker), executable_sha256='a'*64)
        request = {'schema_version':1,'parameters':{'Backend':'ninfer',
            'Worker':str(self.fixture.worker),'RecallStrategy':'layerwise'}}
        with patch.object(controller,'worker_capabilities',return_value=caps) as probe, \
             patch.object(controller,'inspect_model') as inspect:
            result = controller.selected_worker_capabilities(request)
            probe.assert_called_once_with(str(self.fixture.worker),'ninfer')
            inspect.assert_not_called()
        self.assertEqual(result['requested_worker'], str(self.fixture.worker))
        self.assertEqual(result['backend'],'ninfer')
        self.assertEqual(result['strategy'],'layerwise')
        self.assertEqual(result['worker']['executable_sha256'],'a'*64)
        result['profile']['speculative'].clear()
        self.assertTrue(caps['recall_strategies']['layerwise']['speculative'])
        self.assertIsNone(strategy_profile({'recall_strategies':caps['recall_strategies']},'layerwise'))
        self.assertIsNone(strategy_profile({},'legacy'))

    def test_all_backends_without_native_declaration_reject_layerwise(self):
        controller = console.Controller(ROOT, self.directory/'state', 18199)
        # A Legacy-capable executable is not evidence of layerwise support,
        # including when the selected backend is NInfer.
        old = dict(path=str(self.fixture.worker), executable_sha256='a'*64,
                   query_scoring_revision=2, independent_graph_lrfu=True, signed_lrfu_hits=True)
        for backend in ('ninfer', 'llamacpp', 'polystrata', 'radiance'):
            for capability in (old, dict(old, **contract(available=False))):
                with self.subTest(backend=backend, declared='recall_strategies' in capability):
                    parameters = dict(Backend=backend, Worker=str(self.fixture.worker), RecallStrategy='layerwise')
                    with patch.object(controller, 'worker_capabilities', return_value=capability) as probe:
                        result = controller.selected_worker_capabilities(dict(schema_version=1, parameters=parameters))
                    probe.assert_called_once_with(str(self.fixture.worker), backend)
                    self.assertEqual(result['backend'], backend)
                    self.assertFalse(result['profile'] and result['profile'].get('available'))
                    with self.assertRaisesRegex(ValueError, 'qualified layerwise'):
                        require_launch_features(capability, parameters)
                    require_launch_features(capability, dict(parameters, RecallStrategy='legacy'))

    def test_runtime_confirmation_requires_every_actual_account_and_policy(self):
        plan = build_plan(self.config('llamacpp', SpeculativeMode='mtp', DraftTokens=3,
                                     LayerwiseMtpHistory='first-layer',
                                     **{key: value for key, (_, value) in LAYERWISE_RESOURCES.items()}))
        controller = console.Controller(ROOT, self.directory / 'state', 18199)
        controller.active = self.config('llamacpp')
        controller.active_plan = plan
        base = 'KVMEM_RECALL strategy=layerwise current_layer_q=1 query_replay=0 mtp_history=first-layer '
        limits = ' '.join(f'{key}={value}' for key, value in plan['resolved_scoring'].items()
                          if key.endswith('_bytes'))
        controller.log_path.write_text(base + limits.rsplit(' ', 1)[0] + '\n', encoding='utf-8')
        self.assertFalse(controller.status()['scoring_confirmed'])
        controller.log_path.write_text(base + limits + '\n', encoding='utf-8')
        controller.observed_offset = 0
        self.assertTrue(controller.status()['scoring_confirmed'])
        controller.active_plan['resolved_scoring'] = dict(plan['resolved_scoring'], metadata_bytes=1)
        self.assertTrue(controller.status()['scoring_mismatch'])
        # The native Poly report uses aliases; it still reports configured limits.
        controller.observed_scoring.clear()
        report = 'INFO recall_strategy=layerwise current_layer_query=1 query_replay=0 mtp_history=first-layer '
        report += ' '.join('layerwise_' + key + '=' + str(value) for key, value in plan['resolved_scoring'].items()
                           if key.endswith('_bytes'))
        controller.parse_scoring([report])
        self.assertEqual(controller.observed_scoring, plan['resolved_scoring'])

    def test_layerwise_never_uses_legacy_auto_budget_recommendations(self):
        p = self.config()['parameters']
        result = diagnose('FATAL CUDA out of memory', p, {'effective': {}})
        self.assertEqual(result['reason'], 'layerwise_planning_unavailable')
        self.assertEqual(result['options'], [])
        controller = console.Controller(ROOT, self.directory / 'state', 18199)
        with patch.object(console, 'AUTO_BUDGET_SUPPORTED', True), \
             patch.object(controller, '_run_budget_plan') as run:
            with self.assertRaisesRegex(ValueError, 'Layerwise auto budget'):
                controller.adjust_budget(self.config())
            run.assert_not_called()

    def test_saved_configuration_and_shell_export_keep_explicit_policy_and_limits(self):
        data = self.config('llamacpp', SpeculativeMode='mtp', DraftTokens=3,
                           LayerwiseMtpHistory='upstream-global', LayerwiseActivationRamMiB=0,
                           LayerwiseActivationDiskMiB=128, LayerwiseStagingMiB=7,
                           DiskPath=str(self.directory/'activation'), DiskMiB=0,
                           SinkTokens=0, RecentTokens=0)
        controller = console.Controller(ROOT, self.directory / 'state', 18199)
        saved = controller.save(data)
        self.assertEqual(saved['parameters'], data['parameters'])
        self.assertEqual(controller.draft()['parameters'], data['parameters'])
        plan = build_plan(data)
        # Exercise rendering only; these Windows fixture paths are never executed.
        plan['target_platform'] = 'linux'
        script = render_script(plan, self.directory / 'test.lock')
        for flag in ('--kvmem-recall-strategy', '--kvmem-layerwise-mtp-history',
                     '--kvmem-layerwise-activation-disk-mib', '--kvmem-mtp-state'):
            self.assertIn(flag, script)
        self.assertNotIn('--kvmem-query-replay-max-tokens', script)
        self.assertEqual(plan['effective_parameters']['LayerwiseActivationRamMiB'], 0)
        self.assertEqual(plan['effective_parameters']['LayerwiseStagingMiB'], 7)

    @unittest.skipUnless(os.name == 'nt', 'Windows canonical parameter preview')
    def test_windows_layerwise_previews_match_portable_argv(self):
        for backend in ('ninfer', 'llamacpp'):
            for policy in ('first-layer', 'upstream-global'):
                data = self.config(backend, SpeculativeMode='mtp', DraftTokens=3,
                    DraftPolicy='fixed', LayerwiseMtpHistory=policy,
                    LayerwiseStagingMiB=7, SinkTokens=0, RecentTokens=0)
                config = self.directory / 'input.json'
                config.write_text(json.dumps(data['parameters']), encoding='utf-8')
                run = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive',
                    '-ExecutionPolicy', 'Bypass', '-File', str(ROOT / 'scripts/windows/describe-launch-json.ps1'),
                    '-BackendRoot', str(ROOT), '-ConfigFile', str(config)], capture_output=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr.decode('utf-8', 'replace'))
                result = json.loads(run.stdout.decode('utf-8-sig'))
                self.assertEqual(result['argv'], build_plan(data)['argv'])
                self.assertIs(type(result['effective_parameters']['Vision']), bool)
                self.assertIs(type(result['effective_parameters']['TensorParallel']), bool)
        for backend, block in (('llamacpp',32),('ninfer',64)):
            data = self.config(backend, Budget=block, Reserve=block, SinkTokens=0, RecentTokens=0)
            config = self.directory / 'input.json';config.write_text(json.dumps(data['parameters']),encoding='utf-8')
            run = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
                '-File',str(ROOT/'scripts/windows/describe-launch-json.ps1'),'-BackendRoot',str(ROOT),
                '-ConfigFile',str(config)],capture_output=True,timeout=20)
            self.assertEqual(run.returncode,0,run.stderr.decode('utf-8','replace'))
            self.assertEqual(json.loads(run.stdout.decode('utf-8-sig'))['argv'],build_plan(data)['argv'])

    @unittest.skipUnless(os.name == 'nt', 'Windows activation-only SSD preview')
    def test_windows_activation_only_ssd_uses_shared_directory_with_zero_canonical_quota(self):
        for explicit in ({}, {'DiskMiB':0}, {'DiskMiB':19}):
            data = self.config(LayerwiseActivationRamMiB=0, LayerwiseActivationDiskMiB=64,
                               DiskPath=str(self.directory/'activation'),**explicit)
            config=self.directory/'activation.json'
            config.write_text(json.dumps(data['parameters']),encoding='utf-8')
            run=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
                '-File',str(ROOT/'scripts/windows/describe-launch-json.ps1'),'-BackendRoot',str(ROOT),
                '-ConfigFile',str(config)],capture_output=True,timeout=30)
            self.assertEqual(run.returncode,0,run.stderr.decode('utf-8','replace'))
            result=json.loads(run.stdout.decode('utf-8-sig'))
            self.assertEqual(result['argv'],build_plan(data)['argv'])
            self.assertEqual(result['effective_parameters']['DiskMiB'],explicit.get('DiskMiB',0))

    def poly_args(self):
        tree = self.directory / 'poly'; (tree / 'serve').mkdir(exist_ok=True, parents=True)
        for name in ('server.py', 'kvmem.py'): (tree / 'serve' / name).touch()
        tokenizer = self.directory / 'tokenizer'; tokenizer.mkdir(exist_ok=True)
        for name in ('vocab.json', 'merges.txt', 'tokenizer.json', 'token_type.json', 'chat_template.jinja'):
            (tokenizer / name).touch()
        return ['--source', str(tree), '--worker', str(self.fixture.worker),
                '--model', str(self.fixture.models['llamacpp']), '--tokenizer', str(tokenizer),
                '--gpu', legacy.GPU, '--dry-run']

    def poly(self, *extra):
        return subprocess.run([sys.executable, '-B', str(ROOT / 'tools/polystrata/server.py'),
                               *self.poly_args(), *extra], capture_output=True, encoding='utf-8', timeout=15)

    @unittest.skipUnless(os.name == 'nt', 'Windows recipe parameter preview')
    def test_windows_recipe_vision_order_and_text_layerwise(self):
        build = self.directory / 'build'; (build / 'bin').mkdir(parents=True)
        (build / 'bin/llama-kvmem-server.exe').touch()
        projector = self.directory / 'mmproj.gguf'; projector.touch()
        command = ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
            '-File', str(ROOT / 'scripts/windows/start-server.ps1'), '-Recipe', 'iq3', '-BuildDir', str(build),
            '-Model', str(self.fixture.models['llamacpp']), '-Gpu', legacy.GPU, '-DryRun']
        run = subprocess.run(command + ['-Mmproj', str(projector)], capture_output=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stderr.decode('utf-8', 'replace'))
        old = json.loads(run.stdout.decode('utf-8-sig'))
        self.assertEqual(old['argv'][1:8], ['-m', str(self.fixture.models['llamacpp']), '--mmproj',
                                         str(projector), '--no-mmproj-offload', '--image-max-tokens', '512'])
        self.assertNotIn('--kvmem-recall-strategy', old['argv'])
        run = subprocess.run(command + ['-NoVision', '-RecallStrategy', 'layerwise',
            '-LayerwiseMtpHistory', 'first-layer'], capture_output=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stderr.decode('utf-8', 'replace'))
        new = json.loads(run.stdout.decode('utf-8-sig'))
        for flag in ('--mmproj', '--kvmem-sink-tokens', '--kvmem-recent-tokens'):
            self.assertNotIn(flag, new['argv'])
        self.assertIn('--kvmem-mtp-state', new['argv'])
        self.assertFalse(new['effective_parameters']['Vision'])

    @unittest.skipUnless(os.name == 'nt', 'Windows NInfer shortcut preview')
    def test_ninfer_shortcut_does_not_promote_legacy_defaults_to_explicit_new_bands(self):
        command = ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
            '-File', str(ROOT / 'scripts/windows/start-ninfer.ps1'), '-Model', str(self.fixture.models['ninfer']),
            '-Worker', str(self.fixture.worker), '-Gpu', legacy.GPU, '-RecallStrategy', 'layerwise',
            '-SpeculativeMode', 'none', '-NoVision', '-DryRun']
        run = subprocess.run(command, capture_output=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stderr.decode('utf-8', 'replace'))
        result = json.loads(run.stdout.decode('utf-8-sig'))
        for flag in ('--kvmem-sink-tokens', '--kvmem-recent-tokens'):
            self.assertNotIn(flag, result['argv'])
        for key, value in (('Budget', 32768), ('Reserve', 1024), ('HostMiB', 16384)):
            self.assertEqual(result['effective_parameters'][key], value)
        run = subprocess.run(command + ['-RecentTokens', '0', '-SinkTokens', '0', '-Budget', '1024',
                                       '-Reserve', '256', '-HostMiB', '1024'], capture_output=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stderr.decode('utf-8', 'replace'))
        explicit = json.loads(run.stdout.decode('utf-8-sig'))
        self.assertEqual(explicit['argv'][explicit['argv'].index('--kvmem-recent-tokens') + 1], '0')
        self.assertEqual(explicit['effective_parameters']['Budget'], 1024)

    @unittest.skipUnless(os.name == 'nt', 'Windows Poly frontend preview')
    def test_windows_poly_guess_translation_preserves_native_auto_bands(self):
        base = self.poly_args()
        tree, tokenizer = base[base.index('--source') + 1], base[base.index('--tokenizer') + 1]
        assistant = self.directory / 'assistant.gguf'; assistant.touch()
        data = self.config('llamacpp')['parameters']
        data.update(Backend='polystrata', PolystrataSource=tree, Tokenizer=tokenizer, Python=sys.executable,
                    SpeculativeMode='guess', DraftTokens=2, DraftModel=str(assistant), KvType='f16')
        config = self.directory / 'poly-input.json'; config.write_text(json.dumps(data), encoding='utf-8')
        run = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
            '-File', str(ROOT / 'scripts/windows/describe-launch-json.ps1'), '-BackendRoot', str(ROOT),
            '-ConfigFile', str(config)], capture_output=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr.decode('utf-8', 'replace'))
        output = json.loads(run.stdout.decode('utf-8-sig'))
        self.assertEqual(output['effective_parameters']['SpeculativeMode'], 'guess')
        self.assertIn('--draft-method', output['argv'])
        self.assertIn('--kvmem-mtp', output['native_argv'])
        for flag in ('--kvmem-sink-tokens', '--kvmem-recent-tokens', '--kvmem-layerwise-mtp-history'):
            self.assertNotIn(flag, output['native_argv'])

    def test_poly_frontend_preserves_legacy_and_layerwise_auto_provenance(self):
        legacy_run = self.poly()
        self.assertEqual(legacy_run.returncode, 0, legacy_run.stderr)
        old = json.loads(legacy_run.stdout)['args']
        self.assertEqual(old[old.index('--kvmem-recent-tokens') + 1], '64')
        self.assertEqual(old[old.index('--kvmem-host-mib') + 1], '4096')
        self.assertNotIn('--kvmem-recall-strategy', old)
        run = self.poly('--recall-strategy', 'layerwise', '--reserve', '32')
        self.assertEqual(run.returncode, 0, run.stderr)
        args = json.loads(run.stdout)['args']
        self.assertNotIn('--kvmem-sink-tokens', args)
        self.assertNotIn('--kvmem-recent-tokens', args)
        self.assertNotIn('--kvmem-query-replay-max-tokens', args)
        self.assertEqual(args[args.index('--kvmem-host-mib') + 1], '16384')
        self.assertEqual(args[args.index('--prompt-batch') + 1], '512')
        run = self.poly('--recall-strategy', 'layerwise', '--sink', '0', '--recent', '0', '--host-mib', '8192')
        args = json.loads(run.stdout)['args']
        self.assertEqual(args[args.index('--kvmem-recent-tokens') + 1], '0')
        self.assertEqual(args[args.index('--kvmem-host-mib') + 1], '8192')

    def test_poly_frontend_own_policy_guess_and_raw_override_boundaries(self):
        missing = self.poly('--recall-strategy', 'layerwise', '--mtp-drafts', '2')
        self.assertEqual(missing.returncode, 0, missing.stderr)  # Preview preserves omission.
        self.assertNotIn('--kvmem-layerwise-mtp-history', json.loads(missing.stdout)['args'])
        # The actual selected worker must declare a default; dry-run cannot invent one.
        from launcher.recall import require_recall_features
        with self.assertRaisesRegex(ValueError, 'history policy'):
            require_recall_features(contract(), {'Backend':'ninfer', 'RecallStrategy':'layerwise',
                'SpeculativeMode':'mtp', 'DraftTokens':2, 'Budget':1024, 'Reserve':128,
                'KvType':'int8', 'Concurrency':1})
        for policy in ('first-layer', 'upstream-global'):
            run = self.poly('--recall-strategy', 'layerwise', '--mtp-drafts', '2',
                            '--layerwise-mtp-history', policy)
            self.assertEqual(run.returncode, 0, run.stderr)
            args = json.loads(run.stdout)['args']
            self.assertIn('--kvmem-mtp', args)
            self.assertEqual(args[args.index('--kvmem-layerwise-mtp-history') + 1], policy)
        guess = self.poly('--recall-strategy', 'layerwise', '--mtp-drafts', '2', '--draft-method', 'guess')
        self.assertEqual(guess.returncode, 0, guess.stderr)  # Preview; actual architecture gate runs before startup.
        self.assertNotIn('--kvmem-layerwise-mtp-history', json.loads(guess.stdout)['args'])
        for extra in (['--layerwise-mtp-history', 'first-layer'], ['--', '--window', '1']):
            run = self.poly('--recall-strategy', 'layerwise', *extra)
            self.assertNotEqual(run.returncode, 0)


if __name__ == '__main__':
    unittest.main()
