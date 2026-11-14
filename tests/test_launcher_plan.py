"""Portable launch contract: independent invariants and legacy CLI comparisons."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from launcher.plan import build_plan, token_settings, capabilities
from launcher.config import normalize
from launcher.worker import probe_worker, require_launch_features
from launcher.export import render_script

GPU = 'GPU-00000000-0000-0000-0000-000000000001'


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Hosted Windows TEMP may use an 8.3 alias (RUNNER~1). Use one path
        # spelling for exact argv comparisons with PowerShell's FullName.
        self.directory = Path(self.tmp.name).resolve()
        self.worker = self.directory / 'worker with spaces.exe'
        self.worker.write_text('#!/bin/sh\nexit 0\n')
        self.worker.chmod(0o755)
        self.models = {}
        for backend, extension in [('ninfer', 'ninfer'), ('llamacpp', 'gguf')]:
            path = self.directory / ('model with spaces.' + extension)
            path.touch()
            self.models[backend] = path

    def config(self, backend='ninfer', **values):
        return {'schema_version': 1, 'parameters': dict(Backend=backend,
            Model=str(self.models[backend]), Worker=str(self.worker), Gpu=GPU, **values)}

    def plan(self, backend='ninfer', **values):
        return build_plan(self.config(backend, **values))

    def test_full_query_limits_require_new_worker_without_changing_requested_values(self):
        cases = ({}, {'QueryMaxTokens': 512}, {'QueryReplayMaxTokens': 2048},
                 {'QueryMaxTokens': 0, 'QueryReplayMaxTokens': 2048},
                 {'QueryMaxTokens': 512, 'QueryReplayMaxTokens': 0},
                 {'QueryMaxTokens': 0, 'QueryReplayMaxTokens': 0})
        for limits in cases:
            parameters = self.plan(**limits)['effective_parameters']
            for capabilities in ({}, *({'legacy_query_replay_revision': value}
                                      for value in (None, False, True, 0, -1, '1', 1.0))):
                with self.subTest(limits=limits, capabilities=capabilities), self.assertRaisesRegex(
                        ValueError, 'both QueryMaxTokens and QueryReplayMaxTokens'):
                    require_launch_features(capabilities, parameters)
            for revision in (1, 2):
                with self.subTest(limits=limits, revision=revision):
                    before = dict(parameters)
                    require_launch_features({'legacy_query_replay_revision': revision}, parameters)
                    self.assertEqual(parameters, before)
            argv = self.plan(**limits)['argv']
            for key, flag in (('QueryMaxTokens', '--kvmem-query-max-tokens'),
                              ('QueryReplayMaxTokens', '--kvmem-query-replay-max-tokens')):
                self.assertEqual(argv[argv.index(flag) + 1], str(limits.get(key, 0)))
        # Direct callers may supply an unexpanded parameter set.
        with self.assertRaisesRegex(ValueError, 'full Query/replay'):
            require_launch_features({}, {'Backend': 'ninfer'})

    def test_explicit_bounded_query_remains_compatible_with_old_workers(self):
        for backend in ('ninfer', 'llamacpp'):
            for limits in ((1, 1), (512, 2048)):
                p = self.plan(backend, QueryMaxTokens=limits[0], QueryReplayMaxTokens=limits[1])['effective_parameters']
                require_launch_features({}, p)
        for backend in ('llamacpp', 'polystrata', 'radiance'):
            require_launch_features({}, {'Backend': backend})
        for flag in ('--kvmem-query-max-tokens', '--kvmem-query-replay-max-tokens'):
            for args in ([flag, '0'], [flag + '=0']):
                with self.subTest(args=args), self.assertRaisesRegex(ValueError, 'native option'):
                    self.plan(QueryMaxTokens=512, QueryReplayMaxTokens=2048, WorkerArgs=args)

    def test_token_invariants_cover_page_boundaries(self):
        for backend in ('ninfer', 'llamacpp', 'polystrata'):
            for reserve in (128, 129, 255, 256, 1023, 1024):
                values = dict(Budget=1099, Reserve=reserve, SinkTokens=65,
                              RecentTokens=66, Prefill=517, Context=4096)
                plan = token_settings(backend, values)
                p, block = plan['effective'], plan['block_tokens']
                for key in ('Budget', 'Reserve', 'SinkTokens', 'RecentTokens'):
                    self.assertEqual(p[key] % block, 0)
                self.assertLessEqual(p['SinkTokens'] + p['RecentTokens'], p['Budget'])
                self.assertLessEqual(p['Budget'] + p['Reserve'], p['Context'])
                if backend != 'llamacpp':
                    self.assertLessEqual(p['Prefill'], p['Reserve'])
                if backend == 'ninfer':
                    self.assertEqual(p['Prefill'] % 128, 0)

    def test_capability_probe_and_digest_must_describe_the_same_worker(self):
        capability = dict(schema_version=1, backend='ninfer', query_scoring_revision=2,
                          independent_graph_lrfu=True, signed_lrfu_hits=True)
        def replace_during_probe(*args, **kwargs):
            replacement = self.worker.with_name('replacement.exe')
            replacement.write_text('a different worker without this contract\n')
            os.replace(replacement, self.worker)
            return subprocess.CompletedProcess(args[0], 0, stdout=json.dumps(capability), stderr='')
        with patch('launcher.worker.subprocess.run', side_effect=replace_during_probe):
            with self.assertRaisesRegex(ValueError, 'Worker changed'):
                probe_worker(self.worker, 'ninfer')

    def test_fixed_lora(self):
        adapter = self.directory / 'rank two.gguf'
        adapter.touch()
        argv = self.plan('llamacpp', Lora=str(adapter), LoraScale=-0.5)['argv']
        start = argv.index('--lora')
        self.assertEqual(argv[start:start+4], ['--lora', str(adapter), '--lora-scale', '-0.5'])
        self.assertNotIn('--lora', self.plan('llamacpp')['argv'])
        for scale in (0, -0.5, 1e-300):
            planned = self.plan('llamacpp', Lora=str(adapter), LoraScale=scale)['argv']
            self.assertEqual(float(planned[planned.index('--lora-scale')+1]), scale)
        for values in [dict(LoraScale=0.5), dict(Lora=str(adapter), LoraScale=float('nan')),
                       dict(Lora=str(adapter.with_suffix('.missing')))]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.plan('llamacpp', **values)
        with self.assertRaises(ValueError):
            self.plan('ninfer', Lora=str(adapter))
        with self.assertRaisesRegex(ValueError, 'worker'):
            require_launch_features({}, {'Lora': str(adapter)})

    def test_zero_native_prefill_sentinel_is_explicit(self):
        result = token_settings('ninfer', dict(Budget=1024, Reserve=512, SinkTokens=0,
                                              RecentTokens=64, Prefill=0, Context=2048))
        self.assertEqual(result['effective']['Prefill'], 512)
        self.assertEqual(result['effective']['SinkTokens'], 64)

    def test_invalid_token_capacity_and_small_reserve_fail(self):
        for values in (dict(Budget=128, Reserve=128, SinkTokens=128, RecentTokens=64),
                       dict(Budget=1024, Reserve=1024, Context=1024),
                       dict(Reserve=64), dict(RecentTokens=0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.plan(**values)

    def test_effective_token_values_are_the_only_emitted_values(self):
        result = self.plan(Budget=1296, Reserve=385, Prefill=257,
                           SinkTokens=65, RecentTokens=64, Context=2048)
        args = result['argv']
        for flag, expected in [('--kvmem-budget', '1280'), ('--kvmem-gen-reserve', '384'),
                               ('--prefill-chunk', '256'), ('--kvmem-sink-tokens', '64')]:
            self.assertEqual(args.count(flag), 1)
            self.assertEqual(args[args.index(flag) + 1], expected)
        self.assertEqual(result['requested_parameters']['Budget'], 1296)

    def test_unlimited_output_maps_to_native_semantics(self):
        self.assertNotIn('--default-max-tokens', self.plan(MaxTokens=0)['argv'])
        args = self.plan('llamacpp', MaxTokens=0)['argv']
        self.assertEqual(args[args.index('-n') + 1], '-1')

    def test_query_controls_are_independent_and_have_explicit_defaults(self):
        args = self.plan()['argv']
        self.assertIn('--no-kvmem-lrfu', args)
        self.assertEqual(args[args.index('--kvmem-graph-alpha') + 1], '0.2')
        args = self.plan(QueryScorer='meanq-cosine', GraphAlpha=.35, Lrfu=False,
                         ScoreAggregation='mean', QueryMaxTokens=128, QueryReplayMaxTokens=1024)['argv']
        for flag, value in [('--kvmem-graph-alpha', '0.35'), ('--kvmem-score-agg', 'mean'),
                            ('--kvmem-query-max-tokens', '128'), ('--kvmem-query-replay-max-tokens', '1024')]:
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn('--no-kvmem-lrfu', args)

    def test_ninfer_full_query_and_unlimited_replay_defaults(self):
        for values in ({}, {'QueryMaxTokens': 0, 'QueryReplayMaxTokens': 0}):
            result = self.plan(**values)
            for key in ('query_max_tokens', 'query_replay_max_tokens'):
                self.assertEqual(result['resolved_scoring'][key], 0)
        for key in ('QueryMaxTokens', 'QueryReplayMaxTokens'):
            with self.assertRaises(ValueError):
                self.plan(**{key: -1})
            with self.assertRaises(ValueError):
                self.plan('llamacpp', **{key: 0})

    def test_speculative_capability_boundaries(self):
        for backend, values in [('ninfer', dict(SpeculativeMode='dflash2', DraftPolicy='adaptive')),
                                ('llamacpp', dict(SpeculativeMode='mtp', DraftTokens=6)),
                                ('ninfer', dict(SpeculativeMode='none', NgramDrafts=5)),
                                ('llamacpp', dict(SpeculativeMode='dflash2')),
                                ('ninfer', dict(MtpDrafts=4, DraftTokens=3))]:
            with self.subTest(backend=backend, values=values), self.assertRaises(ValueError):
                self.plan(backend, **values)

    def test_concurrent_ngram_limit_preserves_other_settings(self):
        plan = self.plan(SpeculativeMode='mtp', DraftTokens=4, DraftPolicy='adaptive',
                         NgramDrafts=31, Concurrency=2)
        args = plan['argv']
        self.assertEqual(args[args.index('--ngram-draft-tokens') + 1], '15')
        self.assertEqual(plan['requested_parameters']['NgramDrafts'], 31)
        self.assertEqual(plan['effective_parameters']['NgramDrafts'], 15)
        self.assertIn('--adaptive-mtp', args)

    def test_multigpu_stage_order_and_identity(self):
        data = self.config(StageLayers=[10, 20])
        data['parameters']['Gpu'] = GPU + ',GPU-00000000-0000-0000-0000-000000000002'
        plan = build_plan(data)
        self.assertEqual(plan['environment']['CUDA_VISIBLE_DEVICES'], data['parameters']['Gpu'])
        self.assertEqual(plan['argv'][plan['argv'].index('--devices') + 1], '0,1')
        data['parameters']['Gpu'] = GPU + ',' + GPU
        with self.assertRaises(ValueError):
            build_plan(data)

    def test_dflash2_pipeline_preserves_devices_and_drafts(self):
        data = self.config(SpeculativeMode='dflash2', DraftTokens=5,
                           StageLayers=[40, 24], Concurrency=2)
        data['parameters']['Gpu'] = GPU + ',GPU-00000000-0000-0000-0000-000000000002'
        plan = build_plan(data)
        argv = plan['argv']
        self.assertEqual(plan['environment']['CUDA_VISIBLE_DEVICES'], data['parameters']['Gpu'])
        for flag, value in [('--devices', '0,1'), ('--stage-layers', '40,24'),
                            ('--spec', 'dflash2'), ('--draft-tokens', '5')]:
            self.assertEqual(argv[argv.index(flag) + 1], value)
        data['parameters'].update(Backend='llamacpp', Model=str(self.models['llamacpp']),
                                  DraftModel=str(self.models['llamacpp']))
        with self.assertRaises(ValueError):
            build_plan(data)

    def test_mixed_quantized_kv_and_literal_metadata(self):
        alias = 'model "quoted" $x `echo forbidden` ; 中文'
        plan = self.plan('llamacpp', KvTypeK='q8', KvTypeV='q4', ModelAlias=alias,
                         PreserveThinking=False)
        self.assertEqual(plan['argv'][plan['argv'].index('--alias') + 1], alias)
        self.assertEqual(plan['argv'][plan['argv'].index('--cache-type-k') + 1], 'q8_0')
        self.assertEqual(plan['argv'][plan['argv'].index('--cache-type-v') + 1], 'q4_0')
        with self.assertRaises(ValueError):
            self.plan('llamacpp', KvTypeK='q8', KvTypeV='f16')

    def test_heads_tensor_parallel_preserves_uuid_order_and_auto_weights(self):
        data = self.config()
        ids = [GPU, 'GPU-00000000-0000-0000-0000-000000000002']
        data['parameters'].update(Gpu=','.join(ids), TensorParallel=True)
        auto = build_plan(data)
        self.assertIn('--tensor-parallel', auto['argv'])
        self.assertEqual(auto['argv'][auto['argv'].index('--tensor-attention') + 1], 'heads')
        self.assertNotIn('--tensor-split', auto['argv'])
        self.assertEqual(auto['environment']['CUDA_VISIBLE_DEVICES'], ','.join(ids))
        data['parameters']['TensorSplit'] = [3, 1]
        manual = build_plan(data)
        self.assertEqual(manual['argv'][manual['argv'].index('--tensor-split') + 1], '3,1')
        data['parameters']['TensorParallel'] = False
        data['parameters']['TensorSplit'] = []
        self.assertNotIn('--tensor-parallel', build_plan(data)['argv'])
        for invalid in [dict(TensorParallel=True, Gpu=GPU),
                        dict(TensorParallel=True, Gpu=','.join(ids), StageLayers=[32, 32]),
                        dict(TensorParallel=True, Gpu=','.join(ids), TensorSplit=[3]),
                        dict(TensorParallel=True, Gpu=','.join(ids), TensorSplit=[float('nan'), 1]),
                        dict(TensorParallel=True, Gpu=','.join(ids), TensorSplit=[float('inf'), 1]),
                        dict(TensorParallel=True, Gpu=','.join(ids), TensorSplit=[True, 1]),
                        dict(TensorParallel=True, Gpu=','.join(ids), TensorSplit=[10**400, 1]),
                        dict(TensorParallel=True, Gpu=','.join(ids), TensorSplit=[1e308, 1e308]),
                        dict(TensorParallel=True, Gpu=','.join(ids), TensorSplit=[10**308, 10**308]),
                        dict(TensorParallel=True, Gpu=','.join(ids), TensorSplit=[0, 1]),
                        dict(TensorParallel=False, Gpu=','.join(ids), TensorSplit=[3, 1])]:
            bad = self.config(); bad['parameters'].update(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_plan(bad)

    def test_heads_tensor_parallel_requires_explicit_worker_capability(self):
        for contract in [{}, *[{'heads_tensor_parallel_revision': value}
                for value in (None, False, True, 0, -1, '1', 1.0)]]:
            with self.subTest(contract=contract):
                require_launch_features(contract, {})
                require_launch_features(contract, {'TensorParallel': False})
                with self.assertRaisesRegex(ValueError, 'heads tensor parallelism'):
                    require_launch_features(contract, {'TensorParallel': True})
        for revision in (1, 2):
            require_launch_features({'heads_tensor_parallel_revision': revision}, {'TensorParallel': True})

    def test_secret_redaction_and_private_plan(self):
        data = self.config(ApiKey='test-secret-value')
        public = build_plan(data)
        self.assertNotIn('test-secret-value', json.dumps(public))
        private = build_plan(data, reveal_secrets=True)
        self.assertEqual(private['argv'][private['argv'].index('--api-key') + 1], 'test-secret-value')
        self.assertNotIn('ApiKey', private['effective_parameters'])

    def test_checkpoint_recompute_setting_is_explicit_ninfer_environment(self):
        name = 'KVMEM_CHECKPOINT_EXTRA_RECOMPUTE_TOKENS'
        with patch.dict(os.environ, {name: '64', 'KVMEM_CHECKPOINT_MIN_GAP': '0'}):
            self.assertEqual(self.plan()['environment'][name], '0')
            for value in (0, 16, 2147483647):
                with self.subTest(value=value):
                    plan = self.plan(CheckpointExtraRecomputeTokens=value)
                    self.assertEqual(plan['environment'][name], str(value))
                    self.assertEqual(plan['effective_parameters']['CheckpointExtraRecomputeTokens'], value)
                    self.assertEqual(plan['argv'], self.plan()['argv'])
            linux = self.plan(CheckpointExtraRecomputeTokens=16)
            linux['target_platform'] = 'linux'
            self.assertIn('export ' + name + '=16', render_script(linux, '/tmp/worker.lock'))
        for value in (-1, 2147483648, True, 1.5, '16'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.plan(CheckpointExtraRecomputeTokens=value)
        with self.assertRaisesRegex(ValueError, 'NInfer only'):
            self.plan('llamacpp', CheckpointExtraRecomputeTokens=0)
        self.assertNotIn(name, self.plan('llamacpp')['environment'])
        with patch.dict(os.environ, {'KVMEM_CHECKPOINT_MIN_GAP': '64'}):
            self.plan(CheckpointExtraRecomputeTokens=0)
            with self.assertRaisesRegex(ValueError, 'MIN_GAP=0'):
                self.plan(CheckpointExtraRecomputeTokens=16)

    def test_cache_minimum_preserves_disabled_argv(self):
        self.assertEqual(self.plan()['argv'], self.plan(CacheMinTokens=0)['argv'])
        self.assertEqual(self.plan()['effective_parameters']['CacheMinTokens'], 0)
        for value in (1, 4096, 2147483647):
            plan = self.plan(CacheMinTokens=value)
            index = plan['argv'].index('--kvmem-cache-min-tokens')
            self.assertEqual(plan['argv'][index + 1], str(value))
            self.assertEqual(plan['effective_parameters']['CacheMinTokens'], value)
            plan['target_platform'] = 'linux'
            self.assertIn('--kvmem-cache-min-tokens', render_script(plan, '/tmp/worker.lock'))
        for value in (-1, 2147483648, True, 1.5, '4096'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.plan(CacheMinTokens=value)
        for backend in ('llamacpp', 'polystrata', 'radiance'):
            with self.subTest(backend=backend):
                normalize({'schema_version': 1, 'parameters': {'Backend': backend, 'CacheMinTokens': 4096}},
                          capabilities()['backends'])
        self.assertEqual(self.plan('llamacpp')['argv'], self.plan('llamacpp', CacheMinTokens=0)['argv'])
        argv=self.plan('llamacpp', CacheMinTokens=4096)['argv']
        self.assertEqual(argv[argv.index('--kvmem-cache-min-tokens')+1], '4096')
        with self.assertRaises(ValueError):
            self.plan(WorkerArgs=['--kvmem-cache-min-tokens', '4096'])

    def test_cache_minimum_requires_worker_support_only_when_enabled(self):
        for value in (None, False, True, 0, -1, '1', 1.0):
            contract = {'cache_admission_revision': value}
            require_launch_features(contract, {'CacheMinTokens': 0})
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'CacheMinTokens'):
                require_launch_features(contract, {'CacheMinTokens': 4096})
        require_launch_features({'cache_admission_revision': 1}, {'CacheMinTokens': 4096})

    def test_checkpoint_recompute_requires_worker_support_only_when_enabled(self):
        for value in (None, False, True, 0, -1, '1', 1.0):
            contract = {'checkpoint_recompute_revision': value}
            require_launch_features(contract, {'CheckpointExtraRecomputeTokens': 0})
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'checkpoint recompute'):
                require_launch_features(contract, {'CheckpointExtraRecomputeTokens': 16})
        for revision in (1, 2):
            require_launch_features({'checkpoint_recompute_revision': revision},
                                    {'CheckpointExtraRecomputeTokens': 16})

    def test_native_escape_hatch_is_bounded(self):
        for args in (['--threads', '4; echo bad'], ['--host', '0.0.0.0'],
                     ['--threads', '4', '--threads', '8'], ['--threads']):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.plan(WorkerArgs=args)
        args = self.plan(WorkerArgs=['--threads', '8'])['argv']
        self.assertEqual(args[-2:], ['--threads', '8'])

    def test_backend_native_mcp_file_options(self):
        path = self.directory / 'mcp config.json'
        path.write_text('{"mcpServers": {}}', encoding='utf-8')
        llama = ['--mcp-servers-config', str(path)]
        poly = ['--elastic-experts-headroom', '--mcp-config', str(path)]

        def checked(backend, args):
            return normalize({'schema_version': 1, 'parameters': {'Backend': backend, 'WorkerArgs': args}},
                             [backend])['parameters']['WorkerArgs']
        self.assertEqual(checked('polystrata', poly), poly)
        relative = self.config('llamacpp', WorkerArgs=[llama[0], path.name])
        resolved = normalize(relative, ['llamacpp'], path_root=self.directory)
        self.assertEqual(resolved['parameters']['WorkerArgs'], llama)
        self.assertEqual(self.plan('llamacpp', WorkerArgs=llama)['argv'][-2:], llama)
        params = {'Backend': 'llamacpp', 'WorkerArgs': llama}
        with self.assertRaisesRegex(ValueError, 'MCP'):
            require_launch_features({}, params)
        require_launch_features({'mcp_stdio': True}, params)
        for backend, args in [('llamacpp', poly[1:]), ('polystrata', llama),
                              ('ninfer', poly[1:]), ('radiance', poly[1:]), ('polystrata', poly[1:] * 2),
                              ('polystrata', ['--mcp-config']), ('polystrata', ['--mcp-config', str(path) + '.missing'])]:
            with self.subTest(backend=backend, args=args), self.assertRaises(ValueError):
                checked(backend, args)

    @unittest.skipUnless(os.name == 'nt', 'PowerShell regression comparison')
    def test_heads_capability_descriptions_agree_and_remain_runtime_gated(self):
        portable = capabilities()['backends']['ninfer']['supported_combinations']['multi_gpu']
        run = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive',
            '-ExecutionPolicy', 'Bypass', '-File', str(ROOT / 'scripts/windows/start-backend.ps1'),
            '-Backend', 'ninfer', '-Describe'], capture_output=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr.decode('utf-8', 'replace'))
        windows = json.loads(run.stdout.decode('utf-8-sig'))['supported_combinations']['multi_gpu']
        self.assertEqual(windows, portable)
        self.assertEqual(portable['mode'], 'layer')
        self.assertEqual(portable['modes'], ['layer', 'tensor'])
        self.assertEqual(portable['tensor_attention'], ['heads'])
        self.assertEqual(portable['heads_tensor_parallel_min_revision'], 1)
        with self.assertRaisesRegex(ValueError, 'heads tensor parallelism'):
            require_launch_features(portable, {'TensorParallel': True})

    @unittest.skipUnless(os.name == 'nt', 'PowerShell regression comparison')
    def test_windows_canonical_cli_equivalence(self):
        cases = [('ninfer', {}), ('llamacpp', {}),
                 ('llamacpp', dict(Lora=str(self.models['llamacpp']), LoraScale=-0.5)),
                 ('ninfer', dict(CheckpointExtraRecomputeTokens=16)),
                 ('ninfer', dict(CacheMinTokens=4096)),
                 ('ninfer', dict(TensorParallel=True, TensorSplit=[3.5, 1],
                     Gpu=GPU+',GPU-00000000-0000-0000-0000-000000000002')),
                 ('ninfer', dict(SpeculativeMode='mtp', DraftTokens=4, DraftPolicy='adaptive',
                                 NgramDrafts=31, Concurrency=2, MaxTokens=0)),
                 ('ninfer', dict(SpeculativeMode='dflash2', DraftTokens=5, Vision=True,
                                 VisionResidency='cpu', PreserveThinking=True)),
                 ('ninfer', dict(Budget=1296, Reserve=385, Prefill=257, SinkTokens=65,
                                 RecentTokens=64, Context=2048)),
                 ('llamacpp', dict(KvTypeK='q8_0', KvTypeV='q4_0', MaxTokens=0,
                                   ThinkingBudget=0, PreserveThinking=False, Concurrency=2)),
                 ('llamacpp', dict(SpeculativeMode='mtp', DraftTokens=3, ReasoningEffort='high'))]
        for backend, values in cases:
            with self.subTest(backend=backend, values=values):
                data = self.config(backend)
                data['parameters'].update(values)
                path = self.directory / 'config.json'
                path.write_text(json.dumps(data['parameters']), encoding='utf-8')
                run = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive',
                    '-ExecutionPolicy', 'Bypass', '-File', str(ROOT / 'scripts/windows/describe-launch-json.ps1'),
                    '-BackendRoot', str(ROOT), '-ConfigFile', str(path)], capture_output=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr.decode('utf-8', 'replace'))
                output = run.stdout.decode('utf-8-sig')
                old = json.loads(output[output.index('{'):])
                new = build_plan(data)
                self.assertEqual(new['argv'], old['argv'])
                self.assertEqual(new['token_settings'], old['token_settings'])
                self.assertEqual(new['environment'], old['environment'])


if __name__ == '__main__':
    unittest.main()
