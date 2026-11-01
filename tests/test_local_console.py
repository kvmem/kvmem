import importlib.util
import json
import mimetypes
import shutil
import os
from pathlib import Path
import tempfile
import socket
import sys
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('console', Path(__file__).resolve().parents[1] / 'scripts/local-console.py')
console = importlib.util.module_from_spec(spec)
spec.loader.exec_module(console)
EXE = '.exe' if os.name == 'nt' else ''


class ConsoleTests(unittest.TestCase):
    def test_ninfer_only_package_filters_backends_and_rejects_old_backend_drafts(self):
        package = Path(self.tmp.name) / 'ninfer-package'
        launcher = package / 'scripts/windows/start-backend.ps1'
        launcher.parent.mkdir(parents=True)
        launcher.write_bytes((console.ROOT / 'scripts/windows/start-backend.ps1').read_bytes())
        (package / 'scripts/launcher-capabilities.json').write_bytes((console.ROOT / 'scripts/launcher-capabilities.json').read_bytes())
        (package / 'console-build.json').write_text(json.dumps({'supported_backends': ['ninfer']}))
        state = package / 'state'
        controller = console.Controller(package, state, 18197)
        self.assertEqual(controller.backends, ['ninfer'])
        self.assertEqual(set(controller.worker_defaults()), {'ninfer'})
        registered = {name: str((Path(self.tmp.name) / (name + '-worker')).resolve())
                      for name in ('ninfer', 'llamacpp', 'polystrata')}
        (state / 'worker-paths.json').write_text(json.dumps(registered))
        self.assertEqual(controller.worker_defaults()['ninfer']['path'], registered['ninfer'])
        self.assertEqual(set(controller.worker_defaults()), {'ninfer'})
        (state / 'worker-paths.json').write_text(json.dumps({'unknown': registered['ninfer']}))
        with self.assertRaisesRegex(ValueError, 'Invalid worker-paths'):
            controller.worker_defaults()
        (state / 'worker-paths.json').unlink()
        with self.assertRaises(ValueError):
            console.normalize({'schema_version': 1, 'parameters': {'Backend': 'llamacpp'}}, controller.backends)
        controller.draft_path.write_text(json.dumps({'schema_version': 1, 'parameters': {'Backend': 'llamacpp', 'Model': 'old.gguf'}}))
        self.assertEqual(controller.draft()['parameters']['Backend'], 'ninfer')
        self.assertNotEqual(controller.draft()['parameters']['Model'], 'old.gguf')
        (package / 'console-build.json').write_text(json.dumps({'supported_backends': 'ninfer'}))
        with self.assertRaisesRegex(ValueError, 'backend list'):
            console.Controller(package, state, 18197)

    def test_polystrata_block_size_is_typed_backend_scoped_and_startup_fixed(self):
        def normalize(**values):
            return console.normalize({'schema_version':1,'parameters':{'Backend':'polystrata',**values}},
                                     ['polystrata','ninfer','llamacpp'])['parameters']
        self.assertEqual(normalize()['BlockTokens'],32)
        for size in (32,64):
            self.assertEqual(normalize(BlockTokens=size)['BlockTokens'],size)
        for value in (16,128,True,32.0,'32',None):
            with self.subTest(value=value), self.assertRaises(ValueError): normalize(BlockTokens=value)
        for backend in ('ninfer','llamacpp'):
            with self.assertRaises(ValueError): normalize(Backend=backend,BlockTokens=32)
        self.controller.active=self.config
        self.controller.active_plan={'resolved_scoring':{'block_tokens':32}}
        self.controller.log_path.write_text('KVMEM_BLOCKS block_tokens=32\n',encoding='utf-8')
        self.assertEqual(self.controller.status()['effective_scoring']['block_tokens'],32)
        self.assertTrue(self.controller.status()['scoring_confirmed'])
        self.controller.active_plan={'resolved_scoring':{'block_tokens':64}}
        self.assertTrue(self.controller.status()['scoring_mismatch'])

    @unittest.skipUnless(os.name == 'nt', 'Windows canonical launcher')
    def test_polystrata_plan_requires_matching_worker_block_capability(self):
        worker=Path(self.tmp.name)/f'worker{EXE}';worker.touch()
        config={'schema_version':1,'parameters':{'Backend':'polystrata','Worker':str(worker),'BlockTokens':32}}
        for supported,flag in (([32,64],32),([64],32),([],32),([32,64],64)):
            response={'argv':['python','server.py'],'native_argv':[str(worker),'--kvmem-block-tokens',str(flag)]}
            with patch.object(console.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(response))), \
                 patch.object(self.controller,'worker_capabilities',return_value={'block_tokens':{'supported':supported}}):
                if 32 in supported and flag==32:
                    self.assertEqual(self.controller.plan(config)['resolved_scoring']['block_tokens'],32)
                else:
                    with self.assertRaisesRegex(ValueError,'BlockTokens'):self.controller.plan(config)

    def test_elastic_experts_native_switch_is_opt_in_and_backend_scoped(self):
        def validate(args, backend='polystrata'):
            return console.normalize({'schema_version':1,'parameters':{'Backend':backend,'WorkerArgs':args}},
                                     ['polystrata','ninfer','llamacpp'])
        for args in ([], ['--elastic-experts'], ['--threads','8','--elastic-experts'],
                     ['--elastic-experts','--threads','8'], ['--elastic-experts-headroom'],
                     ['--elastic-experts','--elastic-experts-headroom'],
                     ['--elastic-experts-headroom','--elastic-headroom-mib','512','--elastic-physical-reserve-mib','256']):
            self.assertEqual(validate(args)['parameters']['WorkerArgs'],args)
        for args in (['--elastic-experts','--elastic-experts'], ['--elastic-experts','1'],
                     ['--elastic-experts-headroom','--elastic-experts-headroom'],
                     ['--elastic-headroom-mib','1025'], ['--elastic-physical-reserve-mib','255'],
                     ['--elastic-headroom-mib','-1'], ['--elastic-physical-reserve-mib','4097'],
                     ['--elastic-headroom-mib','512','--elastic-headroom-mib','256'],
                     ['--threads'], ['--threads','--elastic-experts'], ['--unknown']):
            with self.assertRaises(ValueError): validate(args)
        for backend in ('ninfer','llamacpp'):
            with self.assertRaises(ValueError): validate(['--elastic-experts'],backend)
            with self.assertRaises(ValueError): validate(['--elastic-headroom-mib','512'],backend)

    def test_reply_handles_abandoned_browser_during_headers_and_body(self):
        for stage in ('end_headers', 'body'):
            with self.subTest(stage=stage):
                handler=MagicMock()
                target=handler.end_headers if stage=='end_headers' else handler.wfile.write
                target.side_effect=ConnectionAbortedError('Windows browser abort')
                console.Handler.reply(handler,503,{'error':'No managed service is running'})
                self.assertTrue(handler.close_connection)

    def test_speculative_config_validates_backend_algorithm_combinations(self):
        def normalize(**overrides):
            return console.normalize({'schema_version': 1, 'parameters': {
                **console.DEFAULTS, 'SpeculativeMode': 'dflash2', 'DraftTokens': 5, **overrides}}, ['ninfer', 'llamacpp'])
        self.assertEqual(normalize()['parameters']['SpeculativeMode'], 'dflash2')
        for lanes in (4, 5, 8):
            self.assertEqual(normalize(Concurrency=lanes)['parameters']['Concurrency'], lanes)
        for invalid in ({'DraftPolicy': 'adaptive'}, {'Concurrency': 9}, {'DraftTokens': 16},
                        {'DraftTokens': 0}, {'DraftModel': 'external.gguf'},
                        {'SpeculativeMode': 'none'}, {'MtpDrafts': 3}, {'DraftPolicy': 'unknown'}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): normalize(**invalid)
        draft = Path(self.tmp.name) / 'draft with spaces.gguf'; draft.touch()
        value = normalize(Backend='llamacpp', DraftModel=str(draft))
        self.assertEqual(value['parameters']['DraftModel'], str(draft))
        self.assertEqual(normalize(Backend='llamacpp', DraftModel=str(draft), Concurrency=8)['parameters']['Concurrency'], 8)
        for invalid in ({'DraftModel': ''}, {'DraftTokens': 6}, {'NgramDrafts': 8}, {'Concurrency': 9}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize(Backend='llamacpp', **{'DraftModel': str(draft), **invalid})

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve())
        self.controller = console.Controller(console.ROOT, self.tmp.name, 18199)
        self.capability_stub = patch.object(self.controller, 'worker_capabilities', return_value={
            'schema_version': 1, 'query_scoring_revision': 2, 'independent_graph_lrfu': True,
            'signed_lrfu_hits': True, 'executable_sha256': 'a' * 64})
        self.capability_stub.start()
        self.addCleanup(self.capability_stub.stop)
        self.config = {'schema_version': 1, 'parameters': dict(console.DEFAULTS)}

    def test_scoring_controls_and_legacy_migration(self):
        for scorer in ('page-softmax', 'meanq-cosine'):
            for history in (False, True):
                for alpha in (0, .2, 1):
                    params = dict(console.DEFAULTS, QueryScorer=scorer, Lrfu=history,
                                  GraphAlpha=alpha, ScoreAggregation='max', QueryMaxTokens=128,
                                  QueryReplayMaxTokens=1024)
                    normalized = console.normalize({'schema_version': 1, 'parameters': params}, self.controller.backends)
                    self.assertEqual(normalized['parameters'], params)
        for no_history in (False, True):
            legacy = dict(console.DEFAULTS, NoLrfu=no_history, ScoreAggregation='max',
                          LrfuLambda=.4, LrfuTopFraction=.8)
            migrated = console.normalize({'schema_version': 1, 'parameters': legacy}, self.controller.backends)['parameters']
            self.assertEqual(migrated['QueryScorer'], 'meanq-cosine')
            self.assertEqual(migrated['Lrfu'], not no_history)
            self.assertNotIn('NoLrfu', migrated)
            if no_history: self.assertNotIn('LrfuLambda', migrated)
        for invalid in ({'QueryScorer': 'bad'}, {'GraphAlpha': float('nan')}, {'GraphAlpha': 1.1},
                        {'ScoreAggregation': 'median'}, {'QueryMaxTokens': 0},
                        {'QueryReplayMaxTokens': 0}, {'QueryScorer': 'page-softmax', 'Lrfu': False, 'LrfuLambda': .4}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                console.normalize({'schema_version': 1, 'parameters': dict(console.DEFAULTS, **invalid)}, self.controller.backends)

    def test_worker_contract_rejects_old_or_wrong_worker_and_caches_probe(self):
        worker = Path(self.tmp.name) / f'worker{EXE}'; worker.write_bytes(b'fixture')
        contract = dict(schema_version=1, backend='ninfer', query_scoring_revision=2,
                        independent_graph_lrfu=True, signed_lrfu_hits=True)
        with patch.object(console.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps(contract))) as run:
            first = console.Controller.worker_capabilities(self.controller, worker, 'ninfer')
            second = console.Controller.worker_capabilities(self.controller, worker, 'ninfer')
            self.assertEqual(first, second); self.assertEqual(run.call_count, 1)
            self.assertEqual(len(first['executable_sha256']), 64)
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '-1')
        for value in ([], dict(contract, backend='llamacpp'), dict(contract, query_scoring_revision=1)):
            self.controller.worker_cache.clear()
            with patch.object(console.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(value))):
                with self.assertRaisesRegex(ValueError, 'Worker does not support'):
                    console.Controller.worker_capabilities(self.controller, worker, 'ninfer')

    def test_checkpoint_recompute_plan_rejects_old_worker_before_start(self):
        config = json.loads(json.dumps(self.config))
        model = Path(self.tmp.name) / 'checkpoint.ninfer'; model.touch()
        worker = Path(self.tmp.name) / f'checkpoint{EXE}'; worker.touch(); worker.chmod(0o755)
        config['parameters'].update(Model=str(model), Worker=str(worker),
            Gpu='GPU-00000000-0000-0000-0000-000000000001', CheckpointExtraRecomputeTokens=16)
        with self.assertRaisesRegex(ValueError, 'checkpoint recompute'):
            self.controller.plan(config)
        self.controller.worker_capabilities.return_value['checkpoint_recompute_revision'] = 1
        plan = self.controller.plan(config)
        self.assertEqual(plan['environment']['KVMEM_CHECKPOINT_EXTRA_RECOMPUTE_TOKENS'], '16')

    def test_heads_tensor_parallel_plan_rejects_old_worker_before_start(self):
        model = Path(self.tmp.name) / 'tensor.ninfer'; model.touch()
        worker = Path(self.tmp.name) / f'tensor{EXE}'; worker.touch()
        worker.chmod(0o755)
        self.config['parameters'].update(Model=str(model), Worker=str(worker),
            Gpu='GPU-00000000-0000-0000-0000-000000000001,GPU-00000000-0000-0000-0000-000000000002',
            TensorParallel=True, TensorSplit=[3, 1])
        with self.assertRaisesRegex(ValueError, 'heads tensor parallelism'):
            self.controller.plan(self.config)
        self.assertIsNone(self.controller.process)
        self.controller.worker_capabilities.return_value['heads_tensor_parallel_revision'] = 1
        plan = self.controller.plan(self.config)
        self.assertEqual(plan['argv'][plan['argv'].index('--tensor-attention') + 1], 'heads')
        self.assertEqual(plan['argv'][plan['argv'].index('--tensor-split') + 1], '3,1')
        self.assertIsNone(self.controller.process)

    def test_actual_scoring_is_observed_independently_of_the_editing_draft(self):
        planned = self.controller.scoring_from_argv(['worker', '--kvmem-query-scorer', 'meanq-cosine',
            '--kvmem-score-agg', 'max', '--kvmem-graph-alpha', '.123456789', '--no-kvmem-lrfu'])
        self.controller.active = self.config
        self.controller.active_plan = {'resolved_scoring': planned}
        self.controller.log_path.write_text('KVMEM_SCORING scorer=meanq-cosine agg=max lrfu=0 lambda=0.5 top_frac=0.9 graph_alpha=0.123456789\nKVMEM_LIMITS query_max_tokens=512', encoding='utf-8')
        status = self.controller.status()
        self.assertFalse(status['scoring_confirmed'])
        with self.controller.log_path.open('a', encoding='utf-8') as stream:
            stream.write(' query_replay_max_tokens=2048\n')
        self.assertTrue(self.controller.status()['scoring_confirmed'])
        self.controller.save({'schema_version': 1, 'parameters': dict(console.DEFAULTS, QueryScorer='page-softmax')})
        self.assertEqual(self.controller.status()['effective_scoring']['scorer'], 'meanq-cosine')
        self.controller.active_plan['resolved_scoring'] = dict(planned, graph_alpha=.5)
        self.assertTrue(self.controller.status()['scoring_mismatch'])

    def test_mixed_package_components_are_rejected(self):
        package = Path(self.tmp.name) / 'mixed-package'; package.mkdir()
        asset = package / 'launcher.js'; asset.write_text('original', encoding='utf-8')
        digest = console.hashlib.sha256(asset.read_bytes()).hexdigest()
        (package / 'console-build.json').write_text(json.dumps({
            'launch_contract_version': 2 if os.name == 'nt' else 3, 'controller_files': {'launcher.js': digest}}), encoding='utf-8')
        backend_root = console.ROOT
        with patch.object(console, 'ROOT', package):
            control = console.Controller(backend_root, package / 'state', 18199)
            control.stop()
            asset.write_text('partial update', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Console file mismatch'):
                console.Controller(backend_root, package / 'state2', 18199)

    def test_stale_launcher_reports_version_mismatch_before_running(self):
        folder = Path(self.tmp.name) / 'old-launcher'
        script = folder / 'scripts/windows/start-backend.ps1'; script.parent.mkdir(parents=True)
        script.write_text("[ValidateSet('llamacpp', 'ninfer')][string]$Backend", encoding='utf-8')
        if os.name != 'nt':
            with self.assertRaises(FileNotFoundError):
                console.Controller(folder, folder / 'state', 18199)
            return
        control = console.Controller(folder, folder / 'state', 18199)
        with self.assertRaisesRegex(ValueError, 'Launcher version mismatch'):
            control.plan(self.config)
        self.assertIsNone(control.process)

    def tearDown(self):
        self.controller.stop()
        self.tmp.cleanup()

    def test_console_limits_external_launchers_to_mainline_backends(self):
        folder = Path(self.tmp.name) / 'external'
        script = folder / 'scripts/windows/start-backend.ps1'
        script.parent.mkdir(parents=True)
        script.write_text("[ValidateSet('llamacpp', 'ninfer', 'strata', 'polystrata')][string]$Backend", encoding='utf-8')
        if os.name != 'nt':
            manifest = json.loads((console.ROOT / 'scripts/launcher-capabilities.json').read_text())
            manifest['backends']['strata'] = {}
            (folder / 'scripts/launcher-capabilities.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'manifest'):
                console.Controller(folder, folder / 'state', 18199)
            return
        controller = console.Controller(folder, folder / 'state', 18199)
        self.assertEqual(controller.backends, ['llamacpp', 'ninfer', 'polystrata'])
        for backend in ('strata',):
            bad = {'schema_version': 1, 'parameters': {'Backend': backend}}
            with self.assertRaises(ValueError):
                console.normalize(bad, ['llamacpp', 'ninfer', 'strata', 'polystrata'])

    def test_polystrata_uses_single_gpu_and_embedded_solo_mtp(self):
        value={'schema_version':1,'parameters':{'Backend':'polystrata','KvType':'q8_0',
               'SpeculativeMode':'mtp','DraftTokens':3,'Concurrency':4}}
        self.assertEqual(console.normalize(value,['polystrata'])['parameters']['DraftTokens'],3)
        self.assertEqual(console.normalize(value,['polystrata'])['parameters']['DraftPolicy'],'adaptive')
        thinking={**value['parameters'],'ThinkingMode':'on','ThinkingBudget':128,'ReasoningEffort':'xhigh','PreserveThinking':True}
        self.assertEqual(console.normalize({'schema_version':1,'parameters':thinking},['polystrata'])['parameters']['ThinkingBudget'],128)
        for invalid in ({'DraftTokens':4},{'DraftModel':'draft.gguf'},{'DraftPolicy':'unknown'},
                        {'KvType':'int8'},{'Gpu':'GPU-12345678,GPU-87654321'},
                        {'ReasoningEffort':'max'},{'ThinkingMode':'on','ReasoningEffort':'none'}):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                console.normalize({'schema_version':1,'parameters':{**value['parameters'],**invalid}},['polystrata'])

    def test_draft_is_independent_and_survives_controller_restart(self):
        self.controller.save(self.config)
        self.assertIsNone(self.controller.status()['active_config'])
        second = console.Controller(console.ROOT, self.tmp.name, 18199)
        self.assertEqual(second.draft(), self.config)

    def test_credentials_are_never_saved_or_returned_in_runtime_status(self):
        key = 'test-only-credential-123456'
        self.config['parameters']['ApiKey'] = key
        saved = self.controller.save(self.config)
        self.assertNotIn('ApiKey', saved['parameters'])
        self.assertNotIn(key, self.controller.draft_path.read_text(encoding='utf-8'))
        self.controller.active = self.config
        status = self.controller.status()
        self.assertTrue(status['api_key_configured'])
        self.assertNotIn(key, json.dumps(status))
        self.assertEqual(self.controller.monitor_authorization(), 'Bearer ' + key)
        self.controller.redactions.add(key)
        self.controller.log_path.write_text('worker error: ' + key, encoding='utf-8')
        self.assertEqual(self.controller.log_tail(), 'worker error: <redacted>')

    @unittest.skipUnless(os.name == 'nt', 'Windows canonical launcher')
    def test_polystrata_lan_key_reaches_real_preflight_without_disk_or_argv_exposure(self):
        root = Path(self.tmp.name)
        frontend = root / 'frontend/serve'; frontend.mkdir(parents=True)
        tokenizer = root / 'tokenizer'; tokenizer.mkdir()
        for path in (root/'worker.exe', root/'model.gguf', frontend/'server.py', frontend/'kvmem.py',
                     *(tokenizer/name for name in ('vocab.json', 'merges.txt', 'tokenizer.json',
                                                  'token_type.json', 'chat_template.jinja'))):
            path.touch()
        key = 'test-only-polystrata-page-key'
        params = dict(Backend='polystrata', Model=str(root/'model.gguf'), Worker=str(root/'worker.exe'),
                      Tokenizer=str(tokenizer), PolystrataSource=str(frontend.parent), Python=sys.executable,
                      Gpu='GPU-12345678', Budget=4096, Reserve=1024, SinkTokens=64, RecentTokens=64)
        actual_run = console.subprocess.run
        seen = []
        def inspect_run(argv, **kwargs):
            config_path = Path(argv[argv.index('-ConfigFile') + 1])
            self.assertNotIn(key, config_path.read_text(encoding='utf-8'))
            self.assertNotIn(key, json.dumps(argv))
            seen.append(kwargs.get('env'))
            return actual_run(argv, **kwargs)
        with patch.dict(os.environ, {'STRATA_API_KEY': ''}), \
             patch.object(console.subprocess, 'run', side_effect=inspect_run), \
             patch.object(self.controller, 'worker_capabilities', return_value={'block_tokens': {'supported': [32, 64]}}):
            for host in ('0.0.0.0', '127.0.0.1'):
                for reveal in (False, True):
                    data = {'schema_version': 1, 'parameters': dict(params, ListenHost=host, ApiKey=key)}
                    result = self.controller.plan(data, reveal_secrets=reveal)
                    self.assertEqual(seen[-1]['STRATA_API_KEY'], key)
                    self.assertEqual(result['environment']['STRATA_API_KEY'], key if reveal else '<redacted>')
                    if not reveal: self.assertNotIn(key, json.dumps(result))
                    self.assertEqual(data['parameters']['ApiKey'], key)
                    self.assertEqual(os.environ['STRATA_API_KEY'], '')
            with self.assertRaisesRegex(ValueError, 'non-loopback listener requires STRATA_API_KEY'):
                self.controller.plan({'schema_version': 1, 'parameters': dict(params, ListenHost='0.0.0.0')})
            self.controller.plan({'schema_version': 1, 'parameters': dict(params, ListenHost='127.0.0.1')})
        self.assertEqual(list(root.glob('*.json')), [])

    @unittest.skipUnless(os.name == 'nt', 'Windows canonical launcher')
    def test_polystrata_preflight_error_redacts_key(self):
        worker = Path(self.tmp.name)/'worker.exe'; worker.touch()
        key = 'test-only-preflight-error-key'
        data = {'schema_version': 1, 'parameters': dict(Backend='polystrata', Worker=str(worker), ApiKey=key)}
        response = SimpleNamespace(returncode=1, stdout='', stderr='[KVMEM_CONFIG] rejected ' + key)
        with patch.object(console.subprocess, 'run', return_value=response):
            with self.assertRaisesRegex(ValueError, 'rejected <redacted>') as error:
                self.controller.plan(data)
            self.assertNotIn(key, str(error.exception))

    @unittest.skipUnless(os.name == 'nt', 'Windows canonical launcher')
    def test_native_service_mapping_and_private_launch_plan(self):
        key = 'test-only-credential-123456'
        worker = Path(self.tmp.name) / f'worker{EXE}'; worker.touch()
        template = Path(self.tmp.name) / "chat & ' $literal.jinja"; template.write_text('template', encoding='utf-8')
        for backend, suffix, alias_flag, template_flag, log_flag, log_value, off_flag in (
            ('llamacpp', '.gguf', '--alias', '--chat-template-file', '--verbosity', '5', '--no-think'),
            ('ninfer', '.ninfer', '--model-id', '--chat-template', '--log-level', 'debug', '--no-thinking')
        ):
            model = Path(self.tmp.name) / ('model' + suffix); model.touch()
            self.config['parameters'].update(Backend=backend, Model=str(model), Worker=str(worker),
                Gpu='GPU-00000000-0000-0000-0000-000000000001', ApiKey=key,
                ModelAlias="friendly & ' $literal", ChatTemplateFile=str(template), LogLevel='debug',
                ThinkingMode='off', Concurrency=2)
            plan = self.controller.plan(self.config)
            args = plan['argv']
            for flag, value in ((alias_flag, "friendly & ' $literal"), (template_flag, str(template)),
                                (log_flag, log_value), ('--api-key', '<redacted>'),
                                ('--parallel' if backend == 'llamacpp' else '--max-concurrency', '2')):
                self.assertEqual(args[args.index(flag)+1], value)
            self.assertIn(off_flag, args)
            self.assertNotIn(key, json.dumps(plan))
            private = self.controller.plan(self.config, reveal_secrets=True)['argv']
            self.assertEqual(private[private.index('--api-key')+1], key)
            self.assertNotIn('ApiKey', self.controller.draft()['parameters'])
            self.assertEqual(list(Path(self.tmp.name).glob('*.json')), [])

    def test_new_fields_validate_native_capabilities_and_single_key(self):
        for key, value in [('ApiKey', 'key,second'), ('ApiKey', 'key with space'), ('ApiKey', 'key\nheader'),
                           ('LogLevel', 'verbose'), ('ThinkingMode', 'on'), ('Concurrency', 9),
                           ('ChatTemplateFile', str(Path(self.tmp.name) / 'missing.jinja'))]:
            bad = json.loads(json.dumps(self.config)); bad['parameters'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                console.normalize(bad, self.controller.backends)
        self.config['parameters'].update(Backend='llamacpp', ThinkingMode='on', Concurrency=8)
        self.assertEqual(console.normalize(self.config, self.controller.backends), self.config)
        self.config['parameters']['Concurrency'] = 9
        with self.assertRaises(ValueError):
            console.normalize(self.config, self.controller.backends)

    @unittest.skipUnless(os.name == 'nt', 'Windows canonical launcher')
    def test_ninfer_thinking_and_native_array_survive_json_bridge(self):
        worker = Path(self.tmp.name) / f'worker{EXE}'; worker.touch()
        model = Path(self.tmp.name) / 'model with spaces.ninfer'; model.touch()
        self.config['parameters'].update(Model=str(model), Worker=str(worker),
            Gpu='GPU-00000000-0000-0000-0000-000000000001', ThinkingBudget=256,
            ReasoningEffort='medium', PreserveThinking=True, WorkerArgs=['--threads', '2'])
        argv = self.controller.plan(self.config)['argv']
        self.assertEqual(argv[-2:], ['--threads', '2'])
        self.assertEqual(argv[argv.index('--default-reasoning-effort') + 1], 'medium')
        self.assertEqual(argv[argv.index('--default-thinking-budget') + 1], '256')
        self.assertIn('--preserve-thinking', argv)
        self.assertEqual(list(Path(self.tmp.name).glob('*.json')), [])
        for overrides in ({'ReasoningEffort': 'bad'}, {'ThinkingMode': 'off'},
                          {'PreserveThinking': 'true'}):
            bad = json.loads(json.dumps(self.config))
            bad['parameters'].update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                console.normalize(bad, self.controller.backends)

    @unittest.skipUnless(os.name == 'nt', 'Windows canonical launcher')
    def test_registered_worker_default_and_explicit_override(self):
        model = Path(self.tmp.name) / 'test.ninfer'; model.touch()
        worker = Path(self.tmp.name) / f'installed worker{EXE}'; worker.touch()
        explicit = Path(self.tmp.name) / f'explicit worker{EXE}'; explicit.touch()
        registration = Path(self.tmp.name) / 'worker-paths.json'
        registration.write_text(json.dumps({'ninfer': str(worker)}), encoding='utf-8')
        self.config['parameters'].update(Model=str(model), Worker='',
                                         Gpu='GPU-00000000-0000-0000-0000-000000000001')
        self.assertEqual(self.controller.plan(self.config)['argv'][0], str(worker))
        self.assertEqual(self.config['parameters']['Worker'], '', 'do not mutate draft')
        self.config['parameters']['Worker'] = str(explicit)
        self.assertEqual(self.controller.plan(self.config)['argv'][0], str(explicit))
        explicit.unlink()
        with self.assertRaisesRegex(ValueError, '指定的后端程序不存在'):
            self.controller.plan(self.config)
        self.config['parameters']['Worker'] = ''
        worker.unlink()
        self.assertFalse(self.controller.worker_defaults()['ninfer']['available'])
        with self.assertRaisesRegex(ValueError, '未找到可用的后端程序'):
            self.controller.plan(self.config)

    def test_packaged_worker_default(self):
        old_root = self.controller.backend_root
        self.controller.backend_root = Path(self.tmp.name)
        worker = self.controller.backend_root / f'workers/ninfer/ninfer-kvmem-server{EXE}'
        worker.parent.mkdir(parents=True); worker.touch()
        try:
            self.assertEqual(self.controller.worker_defaults()['ninfer'],
                             {'path': str(worker), 'available': True})
        finally:
            self.controller.backend_root = old_root

    def test_worker_discovery_in_installation_root_bin_and_workers(self):
        installation = Path(self.tmp.name) / 'installation'
        self.controller.backend_root = installation
        cases = [('ninfer', f'ninfer-kvmem-server{EXE}', ''),
                 ('ninfer', f'ninfer-serve{EXE}', ''),
                 ('ninfer', f'ninfer-serve{EXE}', 'bin'),
                 ('ninfer', f'ninfer-kvmem-server{EXE}', 'workers/ninfer'),
                 ('llamacpp', f'llama-kvmem-server{EXE}', ''),
                 ('llamacpp', f'llama-kvmem-server{EXE}', 'bin'),
                 ('llamacpp', f'llama-kvmem-server{EXE}', 'workers/llamacpp')]
        with patch.object(console, 'ROOT', installation), patch.object(console.subprocess, 'run') as run:
            for backend, name, directory in cases:
                with self.subTest(backend=backend, name=name, directory=directory):
                    worker = installation / directory / name
                    worker.parent.mkdir(parents=True, exist_ok=True); worker.touch()
                    self.assertEqual(self.controller.worker_defaults()[backend],
                                     {'path': str(worker), 'available': True})
                    worker.unlink()
            run.assert_not_called()

    def test_worker_discovery_precedence_and_registration_failure(self):
        installation = Path(self.tmp.name) / 'installation'
        installation.mkdir()
        self.controller.backend_root = installation
        canonical = installation / f'ninfer-kvmem-server{EXE}'; canonical.touch()
        alias = installation / f'ninfer-serve{EXE}'; alias.touch()
        bin_worker = installation / f'bin/ninfer-kvmem-server{EXE}'
        bin_worker.parent.mkdir(); bin_worker.touch()
        with patch.object(console, 'ROOT', installation):
            self.assertEqual(self.controller.worker_defaults()['ninfer']['path'], str(canonical))
            canonical.unlink()
            self.assertEqual(self.controller.worker_defaults()['ninfer']['path'], str(alias))
            registered = Path(self.tmp.name) / f'registered{EXE}'; registered.touch()
            (Path(self.tmp.name) / 'worker-paths.json').write_text(json.dumps({'ninfer': str(registered)}))
            self.assertEqual(self.controller.worker_defaults()['ninfer']['path'], str(registered))
            registered.unlink()
            self.assertEqual(self.controller.worker_defaults()['ninfer'],
                             {'path': str(registered), 'available': False})

    def test_packaged_launch_prefers_bundled_worker_over_stale_paths(self):
        installation = Path(self.tmp.name) / 'installation'
        packaged = installation / f'workers/ninfer/ninfer-kvmem-server{EXE}'
        packaged.parent.mkdir(parents=True); packaged.touch()
        old = Path(self.tmp.name) / f'old-package/bin/ninfer-serve{EXE}'
        old.parent.mkdir(parents=True); old.touch()
        state = Path(self.tmp.name) / 'packaged-state'; state.mkdir()
        (state / 'worker-paths.json').write_text(json.dumps({'ninfer': str(old)}))
        controller = console.Controller(console.ROOT, state, 18197, prefer_bundled_worker=True)
        controller.backend_root = installation
        with patch.object(console, 'ROOT', installation):
            self.assertEqual(controller.worker_defaults()['ninfer']['path'], str(packaged))
            data = {'schema_version': 1, 'parameters': dict(console.DEFAULTS, Worker=str(old))}
            saved = controller.save(data)
            self.assertEqual(saved['parameters']['Worker'], str(packaged))
            self.assertEqual(controller.draft()['parameters']['Worker'], str(packaged))

    def test_worker_discovery_separate_console_root_and_bounded_scope(self):
        installation = Path(self.tmp.name) / 'console'
        installation.mkdir()
        self.controller.backend_root = Path(self.tmp.name) / 'backend'
        # Plain upstream llama-server and arbitrary/nested EXEs are not KVMem workers.
        (installation / f'llama-server{EXE}').touch()
        (installation / f'other{EXE}').touch()
        nested = installation / f'other-build/ninfer-serve{EXE}'
        nested.parent.mkdir(); nested.touch()
        with patch.object(console, 'ROOT', installation):
            self.assertFalse(self.controller.worker_defaults()['ninfer']['available'])
            self.assertFalse(self.controller.worker_defaults()['llamacpp']['available'])
            worker = installation / f'ninfer-serve{EXE}'; worker.touch()
            self.assertEqual(self.controller.worker_defaults()['ninfer'],
                             {'path': str(worker), 'available': True})
            selected = self.controller.backend_root / f'ninfer-kvmem-server{EXE}'
            selected.parent.mkdir(); selected.touch()
            self.assertEqual(self.controller.worker_defaults()['ninfer']['path'], str(selected))

    def test_rejects_command_injection_and_common_native_conflicts(self):
        for key, value in [('WorkerArgs', ['--port', '80']), ('WorkerArgs', ['--threads', '$(calc)']),
                           ('WorkerArgs', ['--threads', '8', '--threads', '4']),
                           ('WorkerArgs', '--threads 8'), ('DryRun', False), ('Port', True),
                           ('ListenHost', 'https://attacker.invalid'), ('Gpu', '0'), ('StageLayers', [False])]:
            bad = json.loads(json.dumps(self.config)); bad['parameters'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                console.normalize(bad, self.controller.backends)

    @unittest.skipUnless(os.name == 'nt', 'Windows native picker')
    def test_model_picker_cancel_validation_and_literal_path(self):
        model = Path(self.tmp.name) / "model & ' $literal.ninfer"
        model.touch()
        request = {'backend': 'ninfer', 'current_path': str(model)}
        with patch.object(console.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='{"path":null}', stderr='')):
            self.assertEqual(self.controller.browse_model(request), {'path': None})
        with patch.object(console.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps({'path':str(model)}), stderr='')) as run:
            self.assertEqual(self.controller.browse_model(request)['path'], str(model.resolve()))
            self.assertIn(str(model.parent.resolve()), run.call_args.args[0])
            self.assertNotIn('shell', run.call_args.kwargs)
        wrong = Path(self.tmp.name) / 'not-a-model.txt'; wrong.touch()
        with patch.object(console.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps({'path':str(wrong)}), stderr='')):
            with self.assertRaises(ValueError):
                self.controller.browse_model(request)
        self.assertFalse(self.controller.picker_lock.locked())
        with self.assertRaises(ValueError):
            self.controller.browse_model({'backend': 'powershell'})
        self.assertEqual(self.controller.status()['state'], 'stopped')

    @unittest.skipUnless(os.name == 'nt', 'Windows native picker')
    def test_template_picker_returns_literal_file_and_allows_cancel(self):
        template = Path(self.tmp.name) / "chat & ' $literal.jinja"; template.touch()
        request = {'backend': 'llamacpp', 'current_path': str(template)}
        with patch.object(console.subprocess, 'run', return_value=SimpleNamespace(returncode=0,
                stdout=json.dumps({'path': str(template)}), stderr='')) as run:
            self.assertEqual(self.controller.browse_model(request, template=True)['path'], str(template))
            self.assertIn('-Template', run.call_args.args[0])
            self.assertNotIn('shell', run.call_args.kwargs)
        with patch.object(console.subprocess, 'run', return_value=SimpleNamespace(returncode=0,
                stdout='{"path":null}', stderr='')):
            self.assertEqual(self.controller.browse_model(request, template=True), {'path': None})
        self.assertFalse(self.controller.picker_lock.locked())

    @unittest.skipUnless(os.name == 'nt', 'Windows native picker')
    def test_projector_picker_accepts_any_gguf_name_and_cancel(self):
        projector=Path(self.tmp.name)/"视觉 & ' $custom.GGUF";projector.touch()
        request={'backend':'polystrata','current_path':str(projector)}
        with patch.object(console.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps({'path':str(projector)}),stderr='')) as run:
            self.assertEqual(self.controller.browse_model(request,projector=True)['path'],str(projector.resolve()))
            self.assertIn('-Projector',run.call_args.args[0])
            self.assertIn(str(projector.parent.resolve()),run.call_args.args[0])
            self.assertNotIn('shell',run.call_args.kwargs)
        with patch.object(console.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='{"path":null}',stderr='')):
            self.assertEqual(self.controller.browse_model(request,projector=True),{'path':None})
        wrong=projector.with_suffix('.txt');wrong.touch()
        with patch.object(console.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps({'path':str(wrong)}),stderr='')):
            with self.assertRaises(ValueError):
                self.controller.browse_model(request,projector=True)
        with self.assertRaises(ValueError):
            self.controller.browse_model({'backend':'ninfer'},projector=True)
        self.assertFalse(self.controller.picker_lock.locked())

    @unittest.skipUnless(os.name == 'nt', 'Windows canonical launcher')
    def test_polystrata_projector_plan_matches_discovery_and_explicit_choice(self):
        import ctypes
        import model_health
        from test_model_health import gguf

        root=Path(self.tmp.name)
        frontend=root/'frontend/serve';frontend.mkdir(parents=True)
        tokenizer=root/'tokenizer';tokenizer.mkdir()
        for path in (root/'worker.exe',root/'model.gguf',root/'strata-vision.exe',frontend/'server.py',frontend/'kvmem.py',
                     *(tokenizer/name for name in ('vocab.json','merges.txt','tokenizer.json','token_type.json','chat_template.jinja'))):
            path.touch()
        params=dict(Backend='polystrata',Model=str(root/'model.gguf'),Worker=str(root/'worker.exe'),
                    Vision=True,VisionWorker=str(root/'strata-vision.exe'),Tokenizer=str(tokenizer),
                    PolystrataSource=str(frontend.parent),Python=sys.executable,Gpu='GPU-12345678')
        def plan(**kwargs):
            return self.controller.plan({'schema_version':1,'parameters':dict(params,**kwargs)})
        def chosen(expected,**kwargs):
            argv=plan(**kwargs)['argv']
            self.assertEqual(argv[argv.index('--mmproj')+1],str(expected.resolve()))
        with patch.object(self.controller,'worker_capabilities',return_value={'block_tokens':{'supported':[32,64]}}):
            with self.assertRaisesRegex(ValueError,'no unique adjacent projector'):
                plan()
            projector=root/'MMPROJ-Q8_0.GGUF';projector.touch()
            chosen(projector)
            # Windows enumeration must include hidden files just like static inspection.
            gguf(root/'model.gguf',meta={'general.architecture':'qwen35moe'})
            gguf(projector)
            (root/'strata-vision.exe').write_bytes(b'fixture')
            health_params={k:params[k] for k in ('Backend','Model','Vision','VisionWorker')}
            attrs=ctypes.windll.kernel32.GetFileAttributesW(str(projector))
            self.assertNotEqual(attrs,-1)
            self.assertTrue(ctypes.windll.kernel32.SetFileAttributesW(str(projector),attrs|2))
            try:
                self.assertTrue(model_health.Inspection(health_params,root).run()['can_start'])
                chosen(projector)
                visible=root/'visible-mmproj.gguf';gguf(visible)
                try:
                    self.assertFalse(model_health.Inspection(health_params,root).run()['can_start'])
                    with self.assertRaisesRegex(ValueError,'no unique adjacent projector'):
                        plan()
                    chosen(projector,Mmproj=str(projector))
                finally:
                    visible.unlink()
            finally:
                self.assertTrue(ctypes.windll.kernel32.SetFileAttributesW(str(projector),attrs))
            default=root/'mmproj-BF16.gguf';default.touch()
            with self.assertRaisesRegex(ValueError,'no unique adjacent projector'):
                plan()
            chosen(projector,Mmproj=str(projector))
            exact=root/'model-mmproj.gguf';exact.touch()
            chosen(exact)
            (root/'mmproj-model.gguf').touch()
            with self.assertRaisesRegex(ValueError,'no unique adjacent projector'):
                plan()
            custom=root/"视觉 & ' $custom.gguf";custom.touch()
            chosen(custom,Mmproj=str(custom))
            with self.assertRaisesRegex(ValueError,'Missing PolyStrata vision asset'):
                plan(Mmproj=str(root/'missing.gguf'))
            wrong=root/'projector.txt';wrong.touch()
            with self.assertRaisesRegex(ValueError,'Mmproj must be'):
                plan(Mmproj=str(wrong))

    def test_scan_filters_backend_auxiliary_and_split_files_without_loading(self):
        folder = Path(self.tmp.name)
        nested = folder / 'models'; nested.mkdir()
        for name in ('main.GGUF', 'mmproj-F16.gguf', 'vision-projector.gguf',
                     'split-00001-of-00002.gguf', 'split-00002-of-00002.gguf',
                     'native.ninfer', 'wrong.infer', 'download.gguf.part'):
            (nested / name).write_bytes(b'not parsed or loaded')
        result = self.controller.scan_models({'backend':'llamacpp','directory':str(folder)})
        self.assertEqual({x['name'] for x in result['models']}, {'main.GGUF','split-00001-of-00002.gguf'})
        self.assertFalse(result['limited'])
        result = self.controller.scan_models({'backend':'ninfer','directory':str(folder)})
        self.assertEqual([x['name'] for x in result['models']], ['native.ninfer'])
        self.assertEqual(self.controller.status()['state'], 'stopped')
        with self.assertRaises(ValueError):
            self.controller.scan_models({'backend':'ninfer','directory':folder.anchor})

    def test_scan_deduplicates_hardlinks_and_prefers_library_path(self):
        folder = Path(self.tmp.name)
        canonical = folder / 'ninfer' / 'iq3' / 'iq3.ninfer'
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b'model')
        old = folder / 'iq3.ninfer'
        os.link(canonical, old)
        # Another copy with the same filename and size must remain a separate result.
        other = folder / 'experiment' / 'iq3.ninfer'
        other.parent.mkdir(); other.write_bytes(b'other')
        result = self.controller.scan_models({'backend': 'ninfer', 'directory': str(folder)})
        self.assertEqual({x['path'] for x in result['models']}, {str(canonical), str(other)})
        self.assertFalse(result['limited'])

    def test_scan_is_bounded_and_reports_partial_results(self):
        folder = Path(self.tmp.name)
        for i in range(205):
            (folder / f'model-{i}.ninfer').touch()
        result = self.controller.scan_models({'backend':'ninfer','directory':str(folder)})
        self.assertEqual(len(result['models']), 200)
        self.assertTrue(result['limited'])

    @unittest.skipUnless(os.name == 'nt', 'Windows native picker')
    def test_missing_or_failed_picker_has_readable_error_and_releases_lock(self):
        request = {'backend': 'ninfer', 'current_path': ''}
        with patch.object(console, 'ROOT', Path(self.tmp.name)), patch.object(console.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, '选择窗口组件缺失'):
                self.controller.browse_model(request, folder=True)
            run.assert_not_called()
        with patch.object(console.subprocess, 'run', return_value=SimpleNamespace(
                returncode=1, stderr='bad encoding \ufffd\ufffd PowerShell stack', stdout='')):
            with self.assertRaisesRegex(ValueError, '无法打开系统选择窗口') as error:
                self.controller.browse_model(request, folder=True)
            self.assertNotIn('\ufffd', str(error.exception))
            self.assertNotIn('PowerShell', str(error.exception))
        with patch.object(console.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='not json')):
            with self.assertRaisesRegex(ValueError, '选择窗口返回异常'):
                self.controller.browse_model(request, folder=True)
        self.assertFalse(self.controller.picker_lock.locked())

    def test_model_directories_survive_restart_and_forgetting_keeps_files(self):
        folder = Path(self.tmp.name) / '模型 library & spaces'; folder.mkdir()
        model = folder / 'local.ninfer'; model.write_bytes(b'model')
        self.controller.update_model_directories({'action': 'add', 'path': '"' + str(folder) + '"'})
        second = console.Controller(console.ROOT, self.tmp.name, 18198)
        self.assertEqual(second.model_directories()['directories'], [{'path': str(folder), 'available': True}])
        second.update_model_directories({'action': 'add', 'path': str(folder)})
        self.assertEqual(len(second.model_directories()['directories']), 1)
        second.update_model_directories({'action': 'remove', 'path': str(folder)})
        self.assertTrue(model.is_file(), 'forgetting a folder must never remove model files')
        self.assertEqual(second.model_directories()['directories'], [])
        for path in ('relative-folder', str(folder / 'missing'), folder.anchor, str(model)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                second.update_model_directories({'action': 'add', 'path': path})

    def test_default_scan_finds_models_beside_exe_and_in_ancestor_library(self):
        base = Path(self.tmp.name) / 'fixture/KVMem'
        installation = base / 'monitor/build/bin'; installation.mkdir(parents=True)
        library = base / 'models/ninfer/local'; library.mkdir(parents=True)
        beside = installation / 'model.ninfer'; beside.write_bytes(b'model')
        canonical = library / 'local.ninfer'; canonical.write_bytes(b'model')
        unrelated = installation / 'other-build/unrelated.ninfer'
        unrelated.parent.mkdir(); unrelated.write_bytes(b'model')
        self.controller.backend_root = installation
        with patch.object(console, 'ROOT', installation), patch.object(console.subprocess, 'run') as run:
            result = self.controller.scan_models({'backend': 'ninfer'})
            self.assertEqual({item['path'] for item in result['models']}, {str(beside), str(canonical)})
            run.assert_not_called()

    def test_scan_remembers_folder_reports_other_formats_and_missing_location(self):
        library = Path(self.tmp.name) / 'library'; library.mkdir()
        (library / 'empty.ninfer').touch()
        (library / 'different.gguf').write_bytes(b'model')
        result = self.controller.scan_models({'backend': 'ninfer', 'directory': str(library)})
        self.assertEqual(result['models'][0]['warnings'], ['empty'])
        self.assertEqual(result['other_formats']['gguf'], 1)
        self.assertEqual(result['locations'][0]['status'], 'scanned')
        self.assertEqual(self.controller.model_directories()['directories'][0]['path'], str(library))
        # A saved library on a disconnected/removed volume must stay visible as unavailable.
        missing = Path(self.tmp.name) / 'unavailable library'
        missing.mkdir()
        self.controller.update_model_directories({'action': 'add', 'path': str(missing)})
        missing.rmdir()
        isolated = Path(self.tmp.name) / 'fixture/a/b/c/d/install'; isolated.mkdir(parents=True)
        self.controller.backend_root = isolated
        with patch.object(console, 'ROOT', isolated):
            result = self.controller.scan_models({'backend': 'ninfer'})
        self.assertTrue(any(item['path'] == str(missing) and item['status'] == 'unavailable'
                            for item in result['locations']))
        self.assertEqual(self.controller.status()['state'], 'stopped')

    @unittest.skipUnless(os.name == 'nt', 'Windows canonical launcher')
    def test_canonical_plan_preserves_literal_paths_and_gpu_identity(self):
        model = Path(self.tmp.name) / "model space & ' $literal.ninfer"
        worker = Path(self.tmp.name) / f'worker space{EXE}'
        model.touch(); worker.touch()
        self.config['parameters'].update(Model=str(model), Worker=str(worker),
                                         Gpu='GPU-00000000-0000-0000-0000-000000000001')
        plan = self.controller.plan(self.config)
        self.assertEqual(plan['argv'][:2], [str(worker), str(model)])
        self.assertEqual(plan['environment']['CUDA_VISIBLE_DEVICES'], self.config['parameters']['Gpu'])
        self.assertNotIn('DryRun', plan['argv'])
        self.assertEqual(list(Path(self.tmp.name).glob('*.json')), [])

    @unittest.skipUnless(os.name == 'nt', 'Windows canonical launcher')
    def test_multigpu_order_and_layers_reach_native_launcher(self):
        model = Path(self.tmp.name) / 'multi.ninfer'; model.touch()
        worker = Path(self.tmp.name) / f'worker{EXE}'; worker.touch()
        gpu_order = ['GPU-00000000-0000-0000-0000-000000000002',
                     'GPU-00000000-0000-0000-0000-000000000001']
        for devices, layers in [(gpu_order, [40, 24]), (gpu_order[::-1], [24, 40])]:
            self.config['parameters'].update(Model=str(model), Worker=str(worker),
                                             Gpu=','.join(devices), StageLayers=layers)
            self.controller.save(self.config)
            restored = console.Controller(console.ROOT, self.tmp.name, 18199).draft()
            plan = self.controller.plan(restored)
            self.assertEqual(plan['environment']['CUDA_VISIBLE_DEVICES'], ','.join(devices))
            args = plan['argv']
            self.assertEqual(args[args.index('--devices') + 1], '0,1')
            self.assertEqual(args[args.index('--stage-layers') + 1], ','.join(map(str, layers)))

    @unittest.skipUnless(os.name == 'nt', 'Windows canonical launcher')
    def test_independent_llama_kv_types_reach_native_flags(self):
        model = Path(self.tmp.name) / 'split kv.gguf'; model.touch()
        worker = Path(self.tmp.name) / f'worker{EXE}'; worker.touch()
        base = dict(console.DEFAULTS, Backend='llamacpp', Model=str(model), Worker=str(worker),
                    Gpu='GPU-00000000-0000-0000-0000-000000000001')
        quantized = ('q8_0', 'q5_0', 'q4_0')
        for key_type in (*quantized, 'f16', 'f32'):
            for value_type in (*quantized, 'f16', 'f32'):
                with self.subTest(k=key_type, v=value_type):
                    config = {'schema_version': 1, 'parameters': dict(base, KvTypeK=key_type, KvTypeV=value_type)}
                    if (key_type in quantized) != (value_type in quantized):
                        with self.assertRaisesRegex(ValueError, 'cannot mix floating-point and quantized'):
                            self.controller.plan(config)
                        continue
                    self.controller.save(config)
                    restored = console.Controller(console.ROOT, self.tmp.name, 18199).draft()
                    self.assertEqual(restored, config)
                    args = self.controller.plan(restored)['argv']
                    self.assertNotIn('--kv-dtype', args)
                    self.assertEqual(args[args.index('--cache-type-k') + 1], key_type)
                    self.assertEqual(args[args.index('--cache-type-v') + 1], value_type)
        for values, expected in [({'KvType': 'q5_0', 'KvTypeK': 'q8_0'}, ('q8_0', 'q5_0')),
                                 ({'KvTypeV': 'q4_0'}, ('q8_0', 'q4_0'))]:
            args = self.controller.plan({'schema_version': 1, 'parameters': dict(base, **values)})['argv']
            self.assertEqual(args[args.index('--cache-type-k') + 1], expected[0])
            self.assertEqual(args[args.index('--cache-type-v') + 1], expected[1])
        legacy = {'schema_version': 1, 'parameters': dict(base, KvType='f16')}
        args = self.controller.plan(legacy)['argv']
        self.assertEqual(args[args.index('--kv-dtype') + 1], 'f16')
        self.assertNotIn('--cache-type-k', args)
        for values in ({'KvTypeK': 'unsupported'}, {'KvTypeV': 'q3_0'}):
            with self.assertRaisesRegex(ValueError, 'K/V types must use'):
                self.controller.plan({'schema_version': 1, 'parameters': dict(base, **values)})
        self.assertIsNone(self.controller.process)

    def test_independent_kv_fields_are_llama_only_and_typed(self):
        for key in ('KvTypeK', 'KvTypeV'):
            for backend, value in [('ninfer', 'q4_0'), ('llamacpp', 4)]:
                with self.subTest(backend=backend, key=key, value=value):
                    with self.assertRaises(ValueError):
                        console.normalize({'schema_version': 1, 'parameters': {'Backend': backend, key: value}}, self.controller.backends)

    @unittest.skipUnless(os.name == 'nt', 'Windows canonical launcher')
    def test_preview_reports_port_occupancy_without_starting(self):
        model = Path(self.tmp.name) / 'port.ninfer'; model.touch()
        worker = Path(self.tmp.name) / f'port{EXE}'; worker.touch()
        self.config['parameters'].update(Model=str(model), Worker=str(worker),
                                         Gpu='GPU-00000000-0000-0000-0000-000000000001')
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0)); occupied.listen()
            port = occupied.getsockname()[1]
            self.config['parameters']['Port'] = port
            result = self.controller.plan(self.config)
            self.assertEqual(result['preflight'], {'port': port, 'listen_host': '127.0.0.1', 'port_available': False})
        result = self.controller.plan(self.config)
        self.assertTrue(result['preflight']['port_available'])
        self.config['parameters']['ListenHost'] = '0.0.0.0'
        result = self.controller.plan(self.config)
        self.assertEqual(result['preflight']['listen_host'], '0.0.0.0')
        self.assertTrue(result['preflight']['port_available'])
        self.assertEqual(result['argv'][result['argv'].index('--host') + 1], '0.0.0.0')
        self.assertIsNone(self.controller.process)

    @unittest.skipUnless(os.name == 'nt', 'Windows start path; Linux covered by generated revision tests')
    def test_listen_address_validation_and_wildcard_bind(self):
        for host in ('127.0.0.1', '0.0.0.0'):
            self.config['parameters']['ListenHost'] = host
            self.assertEqual(console.normalize(self.config, self.controller.backends), self.config)
        with patch.object(self.controller, 'inspect_model', return_value={'can_start': True}), patch.object(self.controller, 'plan', return_value={'argv': ['unused']}), patch.object(console.socket, 'socket') as factory:
            probe = factory.return_value.__enter__.return_value
            probe.bind.side_effect = OSError('occupied')
            with self.assertRaisesRegex(ValueError, '端口'):
                self.controller.start(self.config)
            probe.bind.assert_called_once_with(('0.0.0.0', 18200))
        self.assertIsNone(self.controller.process)

    def test_wildcard_health_and_local_url_use_loopback(self):
        self.config['parameters']['ListenHost'] = '0.0.0.0'
        self.controller.active = self.config
        self.controller.process = SimpleNamespace(poll=lambda: None, pid=123)
        response = MagicMock()
        response.__enter__.return_value.status = 200
        try:
            with patch.object(console.urllib.request, 'urlopen', return_value=response) as request:
                result = self.controller.status()
            self.assertEqual(result['state'], 'ready')
            self.assertEqual(result['url'], 'http://127.0.0.1:18200')
            request.assert_called_once_with('http://127.0.0.1:18200/health', timeout=0.5)
        finally:
            self.controller.process = None

    def test_same_origin_token_and_static_allowlist(self):
        server = console.Server(('127.0.0.1', 0), console.Handler)
        server.control, server.gpus = self.controller, []
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        origin = f'http://127.0.0.1:{server.server_port}'
        def request(path, headers=None, data=None):
            req = urllib.request.Request(origin + path, headers=headers or {},
                data=json.dumps(data).encode() if data is not None else None)
            try:
                with urllib.request.urlopen(req) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as error:
                with error:
                    return error.code, error.read()
        try:
            code, body = request('/api/launcher/info')
            self.assertEqual(code, 200)
            token = json.loads(body)['token']
            headers = {'Content-Type': 'application/json', 'X-KVMem-Token': token, 'Origin': origin}
            self.assertEqual(request('/api/launcher/save', headers, self.config)[0], 200)
            for bad in [dict(headers, Origin='https://attacker.invalid'), dict(headers, Host='attacker.invalid'),
                        dict(headers, **{'X-KVMem-Token': 'wrong'}),
                        dict(headers, **{'Sec-Fetch-Site': 'cross-site'})]:
                self.assertEqual(request('/api/launcher/start', bad, self.config)[0], 403)
            self.assertEqual(request('/monitor/../../scripts/local-console.py')[0], 404)
            self.assertEqual(request('/api/monitor/snapshot')[0], 503)
            self.assertEqual(request('/monitor/launcher.js')[0], 200)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_repeated_launch_reuses_only_same_installation_and_state(self):
        server = console.Server(('127.0.0.1', 0), console.Handler)
        server.control, server.gpus = self.controller, []
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        port = server.server_port
        try:
            self.assertTrue(console.matching_console(port, console.ROOT, self.tmp.name))
            self.assertFalse(console.matching_console(port, console.ROOT, Path(self.tmp.name) / 'other-state'))
            self.assertFalse(console.matching_console(port, Path(self.tmp.name) / 'other-install', self.tmp.name))
            self.assertFalse(console.matching_console(port, console.ROOT, self.tmp.name, True))
            self.assertFalse(console.matching_console(port, console.ROOT, self.tmp.name, host='0.0.0.0'))
            self.assertFalse(console.matching_console(port, console.ROOT, self.tmp.name,
                                                     profiles_root=Path(self.tmp.name) / 'other-profiles'))
            with patch.object(console.sys, 'argv', ['local-console.py', '--port', str(port),
                              '--state-dir', self.tmp.name, '--profiles-root', str(self.controller.profiles_root), '--open-browser']), \
                 patch.object(console, 'open_console_browser') as browser, \
                 patch.object(console, 'gpu_inventory') as inventory:
                console.main()
                browser.assert_called_once_with(f'http://127.0.0.1:{port}/monitor/')
                inventory.assert_not_called()
            self.assertIsNone(self.controller.process)
            with patch.object(console.sys, 'argv', ['local-console.py', '--port', str(port),
                              '--state-dir', self.tmp.name, '--profiles-root', str(Path(self.tmp.name) / 'other-profiles'),
                              '--open-browser']), patch.object(console, 'open_console_browser') as browser:
                with self.assertRaises(SystemExit) as error:
                    console.main()
                self.assertEqual(error.exception.code, 1)
                browser.assert_not_called()
            with patch.object(console.sys, 'argv', ['local-console.py', '--port', str(port),
                              '--state-dir', str(Path(self.tmp.name) / 'other-state'), '--open-browser']), \
                 patch.object(console, 'open_console_browser') as browser:
                with self.assertRaises(SystemExit) as error:
                    console.main()
                self.assertEqual(error.exception.code, 1)
                browser.assert_not_called()
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_existing_non_console_is_not_reused(self):
        opener = MagicMock()
        for body in (b'[]', b'{}', b'not JSON', b'x' * 262145,
                     json.dumps({'console_protocol': 1, 'console_root': str(console.ROOT),
                                 'backend_root': str(console.ROOT), 'state_dir': None}).encode()):
            opener.open.return_value.__enter__.return_value.read.return_value = body
            with patch.object(console.urllib.request, 'build_opener', return_value=opener):
                self.assertFalse(console.matching_console(18199, console.ROOT, self.tmp.name))

    def test_port_prompt_validates_values_and_supports_cancel(self):
        with patch('builtins.input', side_effect=['text', '0', '65536', '18199', '18198']), \
             patch('builtins.print') as output:
            self.assertEqual(console.choose_console_port(18199), 18198)
            self.assertIn('默认端口 18199 不可用', output.call_args_list[0].args[0])
        for value in ('', 'q', 'Q', 'exit', 'quit'):
            with self.subTest(value=value), patch('builtins.input', return_value=value), patch('builtins.print'):
                self.assertIsNone(console.choose_console_port(18199))
        for error in (EOFError, KeyboardInterrupt):
            with self.subTest(error=error), patch('builtins.input', side_effect=error), patch('builtins.print'):
                self.assertIsNone(console.choose_console_port(18199))

    def test_address_prompt_defaults_validation_and_cancel(self):
        for language in ('zh-CN', 'en'):
            with self.subTest(language=language), patch('builtins.input', side_effect=['', '']), patch('builtins.print'):
                self.assertEqual(console.choose_console_address('127.0.0.1', 18199, language), ('127.0.0.1', 18199))
            with patch('builtins.input', side_effect=['http://bad', '::1', '999.1.1.1', '224.0.0.1', '0.0.0.0', 'no', '0', '65536', '18201']), patch('builtins.print'):
                self.assertEqual(console.choose_console_address('127.0.0.1', 18199, language), ('0.0.0.0', 18201))
        with patch('builtins.input', side_effect=['localhost', '18198']), patch('builtins.print'):
            self.assertEqual(console.choose_console_address('127.0.0.1', 18199), ('127.0.0.1', 18198))
        for values in (['q'], ['', 'q'], [EOFError()], ['', KeyboardInterrupt()]):
            with self.subTest(values=repr(values)), patch('builtins.input', side_effect=values), patch('builtins.print'):
                self.assertIsNone(console.choose_console_address('127.0.0.1', 18199))

    def test_address_retry_changes_bind_and_browser_without_changing_inference_host(self):
        for host in ('0.0.0.0', '192.0.2.7'):
            server = MagicMock(server_port=18198)
            server.serve_forever.side_effect = KeyboardInterrupt
            with self.subTest(host=host), patch.object(console.sys, 'argv', ['local-console.py', '--state-dir', self.tmp.name,
                              '--open-browser', '--prompt-address']), \
                 patch.object(console, 'Server', side_effect=[OSError('address unavailable'), server]) as factory, \
                 patch.object(console, 'matching_console', return_value=False), \
                 patch.object(console, 'choose_console_address', side_effect=[('192.0.2.8', 18199), (host, 18198)]), \
                 patch.object(console, 'gpu_inventory', return_value=[]), \
                 patch.object(console.threading, 'Thread') as thread, patch('builtins.print'):
                console.main()
            self.assertEqual([call.args[0] for call in factory.call_args_list], [('192.0.2.8', 18199), (host, 18198)])
            expected = '127.0.0.1' if host == '0.0.0.0' else host
            thread.assert_called_once_with(target=console.open_console_browser,
                                           args=(f'http://{expected}:18198/monitor/',), daemon=True)
            self.assertEqual(server.control.draft()['parameters']['ListenHost'], '127.0.0.1')
            self.assertEqual(server.control.port, 18198)

    def test_address_prompt_cancel_does_not_create_controller(self):
        with patch.object(console.sys, 'argv', ['local-console.py', '--prompt-address']), \
             patch.object(console, 'choose_console_address', return_value=None), \
             patch.object(console, 'Controller') as controller, patch.object(console, 'Server') as server:
            console.main()
        controller.assert_not_called()
        server.assert_not_called()

    def test_lan_host_with_default_http_port_and_cross_site_rejection(self):
        handler = object.__new__(console.Handler)
        handler.connection = MagicMock()
        handler.connection.getsockname.return_value = ('192.0.2.7', 80)
        handler.server = SimpleNamespace(server_port=80, control=self.controller)
        handler.reply = MagicMock()
        for host in ('192.0.2.7', '192.0.2.7:80'):
            for origin in ('http://192.0.2.7', 'http://192.0.2.7:80'):
                handler.headers = {'Host': host, 'Origin': origin, 'X-KVMem-Token': self.controller.token}
                self.assertTrue(handler.trusted(mutate=True))
        for host, origin in [('192.0.2.7:81', 'http://192.0.2.7:81'),
                             ('attacker.invalid', 'http://attacker.invalid'),
                             ('192.0.2.7', 'http://attacker.invalid'),
                             ('localhost', 'http://localhost')]:
            handler.headers = {'Host': host, 'Origin': origin}
            self.assertFalse(handler.trusted())

    def test_nondefault_address_accepts_destination_host_but_rejects_other_origins(self):
        # A second loopback address exercises non-default binding without exposing
        # a test controller to the physical network.
        server = console.Server(('127.0.0.2', 0), console.Handler)
        server.control, server.gpus = self.controller, []
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        origin = f'http://127.0.0.2:{server.server_port}'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def request(path, headers=None, data=None):
            req = urllib.request.Request(origin + path, headers=headers or {},
                data=json.dumps(data).encode() if data is not None else None)
            try:
                with opener.open(req, timeout=2) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as error:
                with error:
                    return error.code, error.read()
        try:
            code, body = request('/api/launcher/info')
            self.assertEqual(code, 200)
            info = json.loads(body)
            self.assertEqual(info['listen_host'], '127.0.0.2')
            self.assertEqual(request('/monitor/')[0], 200)
            headers = {'Content-Type': 'application/json', 'Origin': origin, 'X-KVMem-Token': info['token']}
            self.assertEqual(request('/api/launcher/save', headers, self.config)[0], 200)
            for bad in (dict(headers, Origin='http://attacker.invalid'), dict(headers, Host='attacker.invalid'),
                        dict(headers, Host=f'192.0.2.7:{server.server_port}'),
                        dict(headers, **{'X-KVMem-Token': 'wrong'})):
                self.assertEqual(request('/api/launcher/save', bad, self.config)[0], 403)
            self.assertTrue(console.matching_console(server.server_port, console.ROOT, self.tmp.name, host='127.0.0.2'))
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_port_retry_uses_new_port_for_readiness_and_browser(self):
        server = MagicMock(server_port=18197)
        server.serve_forever.side_effect = KeyboardInterrupt
        with patch.object(console.sys, 'argv', ['local-console.py', '--state-dir', self.tmp.name,
                          '--open-browser', '--prompt-port']), \
             patch.object(console, 'Server', side_effect=[OSError('occupied'), OSError('occupied'), server]) as factory, \
             patch.object(console, 'matching_console', return_value=False), \
             patch.object(console, 'choose_console_port', side_effect=[18198, 18197]) as prompt, \
             patch.object(console, 'gpu_inventory', return_value=[]), \
             patch.object(console.threading, 'Thread') as thread:
            console.main()
        self.assertEqual([call.args[0] for call in factory.call_args_list],
                         [('127.0.0.1', 18199), ('127.0.0.1', 18198), ('127.0.0.1', 18197)])
        self.assertEqual([call.args for call in prompt.call_args_list], [(18199, 'zh-CN'), (18198, 'zh-CN')])
        self.assertEqual(server.control.port, 18197)
        thread.assert_called_once_with(target=console.open_console_browser,
                                       args=('http://127.0.0.1:18197/monitor/',), daemon=True)
        server.server_close.assert_called_once()

    def test_english_port_prompt_and_startup_instructions(self):
        with patch('builtins.input', side_effect=['text', '0', '65536', '18199', '18198']) as prompt, \
             patch('builtins.print') as output:
            self.assertEqual(console.choose_console_port(18199, 'en'), 18198)
        messages = '\n'.join(call.args[0] for call in output.call_args_list)
        self.assertIn('Default port 18199 is unavailable', messages)
        self.assertIn('Port must be an integer from 1 to 65535.', messages)
        self.assertIn('Choose a port other than 18199.', messages)
        self.assertTrue(messages.isascii())
        self.assertTrue(all(call.args[0].isascii() for call in prompt.call_args_list))
        for value in ('', 'q', EOFError(), KeyboardInterrupt()):
            options = {'side_effect': value} if isinstance(value, BaseException) else {'return_value': value}
            with self.subTest(value=repr(value)), patch('builtins.input', **options), patch('builtins.print') as output:
                self.assertIsNone(console.choose_console_port(18199, 'en'))
                self.assertIn('Startup cancelled.', output.call_args_list[-1].args[0])
        server = MagicMock(server_port=18198)
        server.serve_forever.side_effect = KeyboardInterrupt
        with patch.object(console.sys, 'argv', ['local-console.py', '--state-dir', self.tmp.name,
                          '--open-browser', '--prompt-port', '--language', 'en']), \
             patch.object(console, 'Server', side_effect=[OSError('occupied'), server]), \
             patch.object(console, 'matching_console', return_value=False), \
             patch.object(console, 'choose_console_port', return_value=18198) as prompt, \
             patch.object(console, 'gpu_inventory', return_value=[]), \
             patch.object(console.threading, 'Thread') as thread, patch('builtins.print') as output:
            console.main()
        prompt.assert_called_once_with(18199, 'en')
        thread.assert_called_once_with(target=console.open_console_browser,
                                       args=('http://127.0.0.1:18198/monitor/?lang=en',), daemon=True)
        messages = '\n'.join(call.args[0] for call in output.call_args_list)
        for expected in ('How to stop:', 'Ctrl+C', 'Stop service', 'Closing only the browser', 'Backends started separately'):
            self.assertIn(expected, messages)
        self.assertTrue(messages.isascii())
        server.server_close.assert_called_once()

    def test_english_entry_reuses_running_console_with_english_browser_hint(self):
        with patch.object(console.sys, 'argv', ['local-console.py', '--state-dir', self.tmp.name,
                          '--open-browser', '--language', 'en']), \
             patch.object(console, 'Server', side_effect=OSError('occupied')), \
             patch.object(console, 'matching_console', return_value=True), \
             patch.object(console, 'open_console_browser') as browser, \
             patch.object(console, 'gpu_inventory') as inventory:
            console.main()
        browser.assert_called_once_with('http://127.0.0.1:18199/monitor/?lang=en')
        inventory.assert_not_called()

    def test_port_prompt_cancel_does_not_start_server_or_browser(self):
        with patch.object(console.sys, 'argv', ['local-console.py', '--state-dir', self.tmp.name, '--prompt-port']), \
             patch.object(console, 'Server', side_effect=OSError('occupied')), \
             patch.object(console, 'matching_console', return_value=False), \
             patch.object(console, 'choose_console_port', return_value=None), \
             patch.object(console, 'gpu_inventory') as inventory, \
             patch.object(console, 'open_console_browser') as browser:
            console.main()
        inventory.assert_not_called()
        browser.assert_not_called()


class ConsoleAssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(console.ROOT / 'ui/monitor', self.root / 'ui/monitor')
        self.chat = self.root / 'ui/dist'
        self.chat.mkdir()
        self.expected = {
            '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
            '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png',
            '.ico': 'image/x-icon', '.woff': 'font/woff', '.woff2': 'font/woff2',
        }
        for extension in self.expected:
            (self.chat / ('index' + extension)).write_bytes(b'fixture')
        (self.chat / 'private.py').write_bytes(b'private')
        (self.root / 'ui/private.js').write_bytes(b'private')
        root_patch = patch.object(console, 'ROOT', self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.server = console.Server(('127.0.0.1', 0), console.Handler)
        self.server.control = SimpleNamespace(backend_root=self.root)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.origin = f'http://127.0.0.1:{self.server.server_port}'

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def assert_asset(self, route, content_type, expected_body):
        with urllib.request.urlopen(self.origin + route, timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get_content_type(), content_type)
            self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
            self.assertEqual(response.read(), expected_body)

    def test_monitor_assets_ignore_host_mime_configuration(self):
        # Simulates the value returned after Windows registry MIME overrides.
        with patch.object(mimetypes, 'guess_type', return_value=('text/plain', None)):
            for asset in (self.root / 'ui/monitor').iterdir():
                if not asset.is_file():
                    continue
                expected = {**self.expected, '.txt': 'text/plain'}[asset.suffix]
                with self.subTest(asset=asset.name):
                    self.assert_asset('/monitor/' + asset.name, expected, asset.read_bytes())
            for route in ('/monitor/', '/chat/'):
                self.assert_asset(route, 'text/html', (self.root / 'ui/monitor/index.html').read_bytes())

    def test_chat_assets_ignore_host_mime_configuration_in_all_locations(self):
        with patch.object(mimetypes, 'guess_type', return_value=('text/plain', None)):
            for directory in (self.chat, self.root / 'workers/share/kvmem/ui',
                              self.root / 'build/share/kvmem/ui'):
                if directory != self.chat:
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    self.chat.rename(directory)
                    self.chat = directory
                self.assert_asset('/chat/view/', 'text/html', b'fixture')
                for extension, expected in self.expected.items():
                    with self.subTest(directory=directory, extension=extension):
                        self.assert_asset('/chat/view/index' + extension, expected, b'fixture')

    def test_static_allowlists_and_traversal_protection_remain_enforced(self):
        for route in ('/monitor/../private.js', '/monitor/private.py',
                      '/chat/view/../private.js', '/chat/view/private.py'):
            with self.subTest(route=route):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(self.origin + route, timeout=5)
                self.assertEqual(error.exception.code, 404)
                error.exception.close()


if __name__ == '__main__':
    unittest.main()
