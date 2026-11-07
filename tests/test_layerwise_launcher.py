"""Product recall choices, native argv and actual-worker qualification gates."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from launcher.config import normalize, ENGINE_DEFAULTS
from launcher.plan import build_plan, scoring_settings, token_settings
from launcher.recall import require_recall_features, resource_defaults, recall_args, LAYERWISE_RESOURCES
from launcher.worker import strategy_profile
from launcher.export import render_script
from launcher.worker import require_launch_features
import test_launcher_plan as legacy
import test_radiance_launcher as radiance
from test_model_health import gguf, ninfer


def contract(**changes):
    profile = dict(available=True, current_layer_q=True, query_replay=False,
        position_policy='original', max_devices=1, max_concurrency=2, minimum_reserve_blocks=1,
        parallel_modes=['single'],
        kv_formats=['q8_0', 'int8', 'f16'], vision_residency=[], speculative={
            'none': dict(available=True),
            'mtp': dict(available=True, history_policies=['first-layer'],
                        max_draft_tokens=3, adaptive=False, ngram=False, state_modes=['replay', 'native']),
            'dflash2': dict(available=True, max_draft_tokens=5, adaptive=False,
                           ngram=False, history='independent-native-window'),
            'guess': dict(available=True, max_draft_tokens=3, adaptive=True,
                         ngram=False, history='shared-target', model_architectures=['gemma4'])})
    profile.update(changes)
    return dict(recall_strategy_revision=1, recall_strategies={'layerwise': profile})


class LayerwiseLauncherTest(unittest.TestCase):
    def setUp(self):
        self.fixture = legacy.PlanTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def config(self, backend='ninfer', **values):
        return self.fixture.config(backend, RecallStrategy='layerwise', **values)

    def test_strategy_is_separate_from_legacy_scorer_and_replay(self):
        for backend in ('ninfer', 'llamacpp'):
            result = build_plan(self.config(backend, QueryScorer='meanq-cosine', GraphAlpha=.6,
                Lrfu=True, QueryMaxTokens=7, QueryReplayMaxTokens=9, SinkTokens=0, RecentTokens=0))
            argv = result['argv']
            self.assertEqual(argv[argv.index('--kvmem-recall-strategy') + 1], 'layerwise')
            for inactive in ('--kvmem-query-scorer', '--kvmem-score-agg', '--kvmem-graph-alpha',
                             '--kvmem-lrfu', '--kvmem-query-max-tokens', '--kvmem-query-replay-max-tokens'):
                self.assertNotIn(inactive, argv)
            self.assertEqual(result['requested_parameters']['QueryReplayMaxTokens'], 9)
            self.assertTrue(result['resolved_scoring']['current_layer_q'])
            self.assertFalse(result['resolved_scoring']['query_replay'])

    def test_missing_strategy_keeps_old_configuration_and_argv(self):
        old = self.fixture.config(QueryScorer='meanq-cosine', QueryReplayMaxTokens=2048)
        self.assertNotIn('RecallStrategy', normalize(old, ['ninfer'])['parameters'])
        argv = build_plan(old)['argv']
        self.assertNotIn('--kvmem-recall-strategy', argv)
        self.assertIn('--kvmem-query-replay-max-tokens', argv)
        require_launch_features({}, old['parameters'])

    def test_hot_checkpoint_target_is_a_count_and_requires_worker_support(self):
        flag = '--kvmem-layerwise-checkpoint-target'
        self.assertNotIn(flag, build_plan(self.config())['argv'])
        plan = build_plan(self.config(LayerwiseCheckpointTarget=10, LayerwiseMetadataMiB=1024))
        self.assertEqual(plan['argv'][plan['argv'].index(flag) + 1], '10')
        self.assertEqual(plan['resolved_scoring']['checkpoint_target'], 10)
        self.assertEqual(plan['resolved_scoring']['metadata_bytes'], 1024 << 20)
        p = plan['effective_parameters']
        require_launch_features(contract(checkpoint_retention={'max_target': 256}), p)
        for cap in (contract(), contract(checkpoint_retention={'max_target': 3})):
            with self.assertRaisesRegex(ValueError, 'checkpoint target'):
                require_launch_features(cap, p)
        for invalid in (0, False, 'auto'):
            with self.assertRaises(ValueError):
                build_plan(self.config(LayerwiseCheckpointTarget=invalid))

    def test_omitted_resources_preserve_backend_defaults_and_explicit_values(self):
        missing = self.config()
        normalized = normalize(missing, ['ninfer'])['parameters']
        self.assertFalse(set(LAYERWISE_RESOURCES) & normalized.keys())
        result = build_plan(missing)
        for _, (flag, _) in LAYERWISE_RESOURCES.items():
            self.assertNotIn(flag, result['argv'])
        self.assertNotIn('metadata_bytes', result['resolved_scoring'])
        for value in (256, 1024):
            original = self.config(LayerwiseMetadataMiB=value)
            saved = normalize(original, ['ninfer'])
            self.assertEqual(saved['parameters']['LayerwiseMetadataMiB'], value)
            plan = build_plan(saved)
            self.assertEqual(plan['argv'][plan['argv'].index('--kvmem-layerwise-metadata-mib')+1], str(value))
            self.assertEqual(plan['resolved_scoring']['metadata_bytes'], value << 20)
            self.assertEqual(normalize(saved, ['ninfer']), saved)
        self.assertEqual(recall_args(dict(RecallStrategy='layerwise',LayerwiseActivationDiskMiB=0)),
            ['--kvmem-recall-strategy','layerwise','--kvmem-layerwise-activation-disk-mib','0'])

    def test_backend_owned_default_resources_are_optional_typed_and_never_override(self):
        declared = dict(activation_ram_bytes=256<<20,activation_disk_bytes=0,staging_bytes=16<<20,
                        metadata_bytes=1024<<20,activation_metadata_bytes=128<<20)
        cap = contract(default_resources=declared)
        profile = strategy_profile(cap, 'layerwise')
        self.assertEqual(resource_defaults(profile)['LayerwiseMetadataMiB'], 1024<<20)
        self.assertEqual(resource_defaults(contract()['recall_strategies']['layerwise']), {})
        p = build_plan(self.config(LayerwiseMetadataMiB=256))['effective_parameters']
        require_launch_features(cap, p)
        self.assertEqual(p['LayerwiseMetadataMiB'],256)
        for invalid in (None, [], dict(metadata_bytes=True), dict(metadata_bytes=0),dict(metadata_bytes=-1)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError,'resource default'):
                strategy_profile(contract(default_resources=invalid),'layerwise')
        # Missing storage defaults are unknown; a complete zero pair is refused.
        require_launch_features(contract(), dict(p,LayerwiseActivationRamMiB=0))
        with self.assertRaisesRegex(ValueError,'RAM or SSD'):
            require_launch_features(cap,dict(p,LayerwiseActivationRamMiB=0))

    def test_linux_serializer_preserves_missing_and_explicit_resource_argv(self):
        # Serializer-only on Windows; real Linux script execution remains in
        # ExportTest on Linux. No shell/worker is launched by this check.
        for resource in ({}, {'LayerwiseMetadataMiB':256}, {'LayerwiseMetadataMiB':1024}):
            original=build_plan(self.config(**resource))
            portable=copy.deepcopy(original)
            portable.update(target_platform='linux',cwd='/test/package',environment={})
            portable['argv'][0]='/test/package/worker'
            portable['worker']={'executable_sha256':'a'*64}
            portable['effective_parameters']['Model']='/test/model.ninfer'
            script=render_script(portable,Path('/test/lock'))
            self.assertIn('a'*64,script)
            if resource:
                self.assertIn('  --kvmem-layerwise-metadata-mib\n  '+str(resource['LayerwiseMetadataMiB'])+'\n',script)
            else:
                self.assertNotIn('--kvmem-layerwise-metadata-mib',script)
            self.assertEqual(original['requested_parameters'].get('LayerwiseMetadataMiB'),resource.get('LayerwiseMetadataMiB'))

    def test_architecture_gate_reads_exact_native_container_metadata(self):
        model = self.fixture.directory / 'qwen-named-model.ninfer'
        p = build_plan(self.config())['effective_parameters']
        p['Model'] = str(model)
        cap = contract(model_architectures=['Qwen3_5ForCausalLM'])
        ninfer(model, mutate=lambda d: d['components']['text']['config'].update(
            architectures=['Qwen3_5ForCausalLM']))
        require_launch_features(cap, p)
        ninfer(model, mutate=lambda d: d['components']['text']['config'].update(
            architectures=['FlintForCausalLM']))
        with self.assertRaisesRegex(ValueError, 'FlintForCausalLM'):
            require_launch_features(cap, p)
        # A plausible model_type and filename cannot stand in for the exact ID.
        ninfer(model)
        with self.assertRaisesRegex(ValueError, 'unverified'):
            require_launch_features(cap, p)
        with self.assertRaisesRegex(ValueError, 'model architectures'):
            require_launch_features(contract(model_architectures=[]), p)
        # Older profiles that do not declare an architecture restriction retain
        # their existing contract. Legacy does not acquire a new model gate.
        require_launch_features(contract(), p)
        require_launch_features(cap, dict(p, RecallStrategy='legacy'))

    def test_layerwise_draft_width_has_no_implicit_backend_default(self):
        for mode in ('mtp', 'dflash2'):
            values = dict(SpeculativeMode=mode)
            if mode == 'mtp': values['LayerwiseMtpHistory'] = 'first-layer'
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'explicit draft width'):
                normalize(self.config(**values), ['ninfer'])

    def test_worker_owned_mtp_default_preserves_omission_and_explicit_override(self):
        plan = build_plan(self.config(SpeculativeMode='mtp', DraftTokens=3))
        parameters = plan['effective_parameters']
        cap = contract()
        spec = cap['recall_strategies']['layerwise']['speculative']['mtp']
        spec['default_history_policy'] = 'first-layer'
        spec['history_policies'] = ['first-layer', 'upstream-global']
        require_launch_features(cap, parameters)
        self.assertNotIn('LayerwiseMtpHistory', parameters)
        self.assertNotIn('--kvmem-layerwise-mtp-history', plan['argv'])
        for policy in ('first-layer', 'upstream-global'):
            require_launch_features(cap, dict(parameters, LayerwiseMtpHistory=policy))
        for bad in (None, '', 'bad', ['first-layer']):
            spec['default_history_policy'] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'MTP history default'):
                require_launch_features(cap, parameters)
        spec['default_history_policy'] = 'first-layer'
        spec['history_policy_caps'] = {'first-layer': dict(available=False)}
        with self.assertRaisesRegex(ValueError, 'selected MTP history policy'):
            require_launch_features(cap, parameters)
        del spec['history_policy_caps']
        del spec['default_history_policy']
        with self.assertRaisesRegex(ValueError, 'explicit history policy'):
            require_launch_features(cap, parameters)

    def test_mtp_requires_explicit_policy_and_dflash_forbids_it(self):
        omitted = build_plan(self.config(SpeculativeMode='mtp', DraftTokens=3))
        self.assertNotIn('--kvmem-layerwise-mtp-history', omitted['argv'])
        with self.assertRaisesRegex(ValueError, 'explicit history policy'):
            require_recall_features(contract(), omitted['effective_parameters'])
        for policy in ('first-layer', 'upstream-global'):
            result = build_plan(self.config('llamacpp', SpeculativeMode='mtp',
                DraftTokens=3, LayerwiseMtpHistory=policy))
            argv = result['argv']
            self.assertEqual(argv[argv.index('--kvmem-layerwise-mtp-history') + 1], policy)
            self.assertEqual(argv[argv.index('--kvmem-mtp-state') + 1], 'replay')
        for mode in ('none', 'dflash2'):
            with self.assertRaisesRegex(ValueError, 'native MTP'):
                normalize(self.config(SpeculativeMode=mode, DraftTokens=0 if mode == 'none' else 5,
                    LayerwiseMtpHistory='first-layer'), ['ninfer'])
        dflash = build_plan(self.config(SpeculativeMode='dflash2', DraftTokens=5))
        self.assertNotIn('--kvmem-layerwise-mtp-history', dflash['argv'])
        require_recall_features(contract(), dflash['effective_parameters'])

    def test_finite_accounts_are_independent_and_zero_bands_are_literal(self):
        result = build_plan(self.config(LayerwiseActivationRamMiB=0, LayerwiseActivationDiskMiB=64,
            DiskPath=str(self.fixture.directory/'activation'), DiskMiB=0,
            LayerwiseStagingMiB=3, LayerwiseMetadataMiB=19, LayerwiseActivationMetadataMiB=7,
            SinkTokens=0, RecentTokens=0))
        argv = result['argv']
        for flag, expected in (('--kvmem-layerwise-activation-ram-mib', '0'),
                               ('--kvmem-layerwise-activation-disk-mib', '64'),
                               ('--kvmem-layerwise-staging-mib', '3'),
                               ('--kvmem-layerwise-metadata-mib', '19'),
                               ('--kvmem-layerwise-activation-metadata-mib', '7'),
                               ('--kvmem-sink-tokens', '0'), ('--kvmem-recent-tokens', '0')):
            self.assertEqual(argv[argv.index(flag) + 1], expected)
        self.assertEqual(scoring_settings(argv)['activation_metadata_bytes'], 7 << 20)
        self.assertEqual(argv[argv.index('--kvmem-disk-mib') + 1], '0')
        for changes in (dict(LayerwiseActivationRamMiB=0, LayerwiseActivationDiskMiB=0),
                        dict(LayerwiseStagingMiB=0), dict(LayerwiseMetadataMiB=0),
                        dict(LayerwiseActivationMetadataMiB=True)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_plan(self.config(**changes))

    def test_activation_only_ssd_keeps_canonical_quota_disabled(self):
        values = dict(LayerwiseActivationRamMiB=0, LayerwiseActivationDiskMiB=64,
                      DiskPath=str(self.fixture.directory/'activation'))
        for explicit in ({}, {'DiskMiB':0}, {'DiskMiB':17}):
            result = build_plan(self.config(**values, **explicit))
            self.assertEqual(result['effective_parameters']['DiskMiB'], explicit.get('DiskMiB',0))
            self.assertEqual(result['argv'][result['argv'].index('--kvmem-disk-mib')+1],str(explicit.get('DiskMiB',0)))
        with self.assertRaisesRegex(ValueError,'DiskPath'):
            build_plan(self.config(LayerwiseActivationDiskMiB=64))
        with self.assertRaisesRegex(ValueError,'Omit DiskPath'):
            build_plan(self.config(DiskPath=values['DiskPath'],DiskMiB=0))
        with self.assertRaisesRegex(ValueError,'Omit DiskPath'):
            build_plan(self.fixture.config(DiskPath=values['DiskPath'],DiskMiB=0))

    def test_implicit_bands_stay_native_auto_and_reserve_floor_is_backend_declared(self):
        result = build_plan(self.config())
        self.assertNotIn('--kvmem-sink-tokens', result['argv'])
        self.assertNotIn('--kvmem-recent-tokens', result['argv'])
        self.assertIsNone(result['token_settings']['effective']['SinkTokens'])
        for backend, block in (('ninfer', 64), ('llamacpp', 32), ('polystrata', 32), ('radiance', 128)):
            for reserve in (block, 2 * block, 3 * block):
                p = dict(RecallStrategy='layerwise', Budget=1024, Reserve=reserve,
                         Prefill=256, Context=4096, SinkTokens=0, RecentTokens=0)
                self.assertEqual(token_settings(backend, p)['effective']['Reserve'], reserve)
                cap = contract(minimum_reserve_blocks=3)
                p.update(Backend=backend, Gpu=legacy.GPU, KvType='f16')
                if reserve < 3 * block:
                    with self.assertRaisesRegex(ValueError, 'reserve floor'): require_launch_features(cap, p)
                else: require_launch_features(cap, p)
        p = build_plan(self.config())['effective_parameters']
        cap = contract(); del cap['recall_strategies']['layerwise']['minimum_reserve_blocks']
        with self.assertRaisesRegex(ValueError, 'reserve floor'): require_launch_features(cap, p)

    def test_worker_contract_is_required_and_never_inherited_from_legacy(self):
        p = build_plan(self.config())['effective_parameters']
        for cap in ({}, {'query_scoring_revision': 999}, contract(available=False),
                    contract(current_layer_q=False), contract(query_replay=True)):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, 'qualified layerwise'):
                require_launch_features(cap, p)
        require_launch_features(contract(), p)
        for changes in (dict(max_devices=0), dict(max_concurrency=0), dict(kv_formats=[]),
                        dict(kv_formats=['f16'])):
            with self.assertRaises(ValueError): require_launch_features(contract(**changes), p)
        with self.assertRaises(ValueError): require_launch_features(contract(), dict(p, Vision=True))

    def test_policy_depth_state_and_native_window_are_checked_separately(self):
        p = build_plan(self.config('llamacpp', SpeculativeMode='mtp', DraftTokens=3,
                                  LayerwiseMtpHistory='first-layer'))['effective_parameters']
        require_launch_features(contract(), p)
        for key, value in (('history_policies', ['upstream-global']), ('max_draft_tokens', 2),
                           ('state_modes', ['native'])):
            cap = contract(); cap['recall_strategies']['layerwise']['speculative']['mtp'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): require_launch_features(cap, p)
        cap = contract(); del cap['recall_strategies']['layerwise']['speculative']['mtp']['state_modes']
        with self.assertRaises(ValueError): require_launch_features(cap, p)
        cap = contract(); cap['recall_strategies']['layerwise']['speculative']['mtp']['history_policy_caps'] = {
            'first-layer': {'max_draft_tokens': 2}}
        with self.assertRaises(ValueError): require_launch_features(cap, p)
        dflash = dict(p, Backend='ninfer', SpeculativeMode='dflash2', DraftTokens=5)
        dflash.pop('LayerwiseMtpHistory')
        cap = contract(); cap['recall_strategies']['layerwise']['speculative']['dflash2']['history'] = 'target-selected'
        with self.assertRaises(ValueError): require_launch_features(cap, dflash)

    def test_pipeline_capability_cannot_enable_tensor_parallel(self):
        p = build_plan(self.config())['effective_parameters']
        p.update(Gpu=legacy.GPU + ',GPU-00000000-0000-0000-0000-000000000002')
        cap = contract(max_devices=2, parallel_modes=['single', 'pipeline'])
        require_launch_features(cap, p)
        with self.assertRaisesRegex(ValueError, 'parallelism'):
            require_launch_features(cap, dict(p, TensorParallel=True))
        cap = contract(); del cap['recall_strategies']['layerwise']['parallel_modes']
        with self.assertRaisesRegex(ValueError, 'parallelism'):
            require_launch_features(cap, dict(p, Gpu=legacy.GPU))

    def test_guess_uses_real_gguf_architecture_and_no_own_policy(self):
        target = self.fixture.directory / 'qwen-named-gemma.gguf'
        assistant = self.fixture.directory / 'assistant.gguf'
        gguf(target, meta={'general.architecture': 'gemma4'})
        gguf(assistant, meta={'general.architecture': 'gemma4-assistant'})
        p = dict(Backend='polystrata', RecallStrategy='layerwise', Model=str(target),
                 DraftModel=str(assistant), Gpu=legacy.GPU, SpeculativeMode='guess',
                 DraftTokens=2, DraftPolicy='adaptive', KvType='f16')
        normalize({'schema_version': 1, 'parameters': p}, ['polystrata'])
        require_launch_features(contract(), p)
        with self.assertRaises(ValueError):
            normalize({'schema_version': 1, 'parameters': dict(p, LayerwiseMtpHistory='first-layer')}, ['polystrata'])
        gguf(target, meta={'general.architecture': 'qwen35moe'})
        with self.assertRaisesRegex(ValueError, 'trusted.*Gemma'):
            require_launch_features(contract(), p)
        gguf(target, meta={'general.architecture': 'gemma4'})
        with self.assertRaises(ValueError):
            require_launch_features(contract(), dict(p, SpeculativeMode='mtp', DraftPolicy='fixed',
                                                     LayerwiseMtpHistory='first-layer'))

    def test_optional_worker_window_is_per_owner_budget_plus_reserve(self):
        p = build_plan(self.config(Budget=64, Reserve=64, Context=4096,
            Prefill=64, SinkTokens=0, RecentTokens=0))['effective_parameters']
        require_launch_features(contract(), p)  # Older profile has no extra floor.
        with self.assertRaisesRegex(ValueError, r'Budget \+ Reserve >= 256'):
            require_launch_features(contract(minimum_window_tokens=256), p)
        require_launch_features(contract(minimum_window_tokens=256), dict(p, Budget=192))
        for invalid in (None, True, 0, -1, 1.5, '256'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'token constraint'):
                strategy_profile(contract(minimum_window_tokens=invalid), 'layerwise')

    def test_optional_worker_ngram_limits_preserve_explicit_values(self):
        for mode in ('mtp', 'dflash2'):
            values = dict(SpeculativeMode=mode, DraftTokens=15, NgramDrafts=63,
                Budget=192, Reserve=64, Prefill=64, SinkTokens=0, RecentTokens=0)
            if mode == 'mtp': values['LayerwiseMtpHistory'] = 'first-layer'
            plan = build_plan(self.config(**values))
            p = plan['effective_parameters']
            self.assertEqual(p['NgramDrafts'], 63)
            self.assertEqual(plan['argv'][plan['argv'].index('--ngram-draft-tokens')+1], '63')
            cap = contract(max_concurrency=8)
            spec = cap['recall_strategies']['layerwise']['speculative'][mode]
            spec.update(max_draft_tokens=15, ngram=True,
                        max_ngram_draft_tokens=63, max_ngram_draft_tokens_concurrent=15)
            require_launch_features(cap, p)
            require_launch_features(cap, dict(p, Concurrency=2, NgramDrafts=15))
            original = copy.deepcopy(p)
            with self.assertRaisesRegex(ValueError, 'NgramDrafts to 15'):
                require_launch_features(cap, dict(p, Concurrency=2))
            self.assertEqual(p, original)
            # The generic single-owner bound is still enforced independently.
            spec['max_ngram_draft_tokens'] = 31
            with self.assertRaisesRegex(ValueError, 'NgramDrafts to 31'):
                require_launch_features(cap, p)
            # Missing optional fields retain the original boolean contract.
            spec.pop('max_ngram_draft_tokens'); spec.pop('max_ngram_draft_tokens_concurrent')
            require_launch_features(cap, dict(p, Concurrency=2))
            for key in ('max_ngram_draft_tokens', 'max_ngram_draft_tokens_concurrent'):
                for invalid in (None, True, 0, -1, 1.5, '15'):
                    spec[key] = invalid
                    with self.subTest(mode=mode,key=key,invalid=invalid), self.assertRaisesRegex(ValueError, 'token constraint'):
                        strategy_profile(cap, 'layerwise')
                    spec.pop(key)

    def test_policy_ngram_limit_can_narrow_the_shared_entry(self):
        p = build_plan(self.config(SpeculativeMode='mtp', DraftTokens=3, NgramDrafts=15,
            LayerwiseMtpHistory='first-layer'))['effective_parameters']
        cap = contract()
        spec = cap['recall_strategies']['layerwise']['speculative']['mtp']
        spec.update(ngram=True, max_ngram_draft_tokens=63,
                    history_policy_caps={'first-layer': {'max_ngram_draft_tokens':7}})
        with self.assertRaisesRegex(ValueError, 'NgramDrafts to 7'):
            require_launch_features(cap, p)
        require_launch_features(cap, dict(p, NgramDrafts=7))

    def test_layerwise_ngram_physical_overlap_does_not_reuse_legacy_gate(self):
        values = dict(SpeculativeMode='mtp', DraftTokens=15, NgramDrafts=63,
            Budget=192, Reserve=64, Prefill=64, SinkTokens=0, RecentTokens=0)
        # Native new-mode owns physical scratch and validates actual credit.
        plan = build_plan(self.config(**values, LayerwiseMtpHistory='first-layer',Concurrency=2))
        self.assertEqual(plan['effective_parameters']['NgramDrafts'],63)
        self.assertEqual(plan['requested_parameters']['NgramDrafts'],63)
        # Preserve the original Legacy overlap check and concurrent normalization.
        values.update(SinkTokens=64, RecentTokens=64)
        with self.assertRaisesRegex(ValueError, 'Prefill and Reserve'):
            build_plan(self.fixture.config(**values))
        values.update(Reserve=128, Prefill=128)
        old = build_plan(self.fixture.config(**values, Concurrency=2))
        self.assertEqual(old['effective_parameters']['NgramDrafts'],15)

    def test_radiance_new_default_is_256(self):
        self.assertEqual(ENGINE_DEFAULTS['radiance']['Prefill'], 256)
        fixture = radiance.RadiancePlanTest(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.assertEqual(fixture.plan()['effective_parameters']['Prefill'], 256)
        self.assertEqual(fixture.plan(Prefill=512)['effective_parameters']['Prefill'], 512)
        current = fixture.plan(RecallStrategy='layerwise')
        self.assertEqual(current['effective_parameters']['Budget'], 32768)
        self.assertEqual(current['effective_parameters']['HostMiB'], 16384)
        self.assertNotIn('--kvmem-sink-tokens', current['argv'])
        small = fixture.plan(RecallStrategy='layerwise', Budget=128, Reserve=128, SinkTokens=0, RecentTokens=0)
        self.assertEqual(small['effective_parameters']['Budget'],128)


if __name__ == '__main__':
    unittest.main()
