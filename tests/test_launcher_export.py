"""Execute generated scripts against a recording worker; no GPU or model load."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from launcher.export import export_profile, read_revision
from launcher.systemd import render_unit, unit_name

WORKER = '''#!/usr/bin/python3
import json, os, signal, sys, time
from pathlib import Path
if '--kvmem-capabilities' in sys.argv:
    print(json.dumps(dict(schema_version=1, backend='ninfer', query_scoring_revision=2,
                         independent_graph_lrfu=True, signed_lrfu_hits=True)))
    raise SystemExit(0)
output = Path(os.environ['KVMEM_TEST_OUTPUT'])
output.write_text(json.dumps(dict(argv=sys.argv, pid=os.getpid(), cwd=os.getcwd(),
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), libraries=os.environ.get('LD_LIBRARY_PATH'))))
if os.environ.get('KVMEM_TEST_HOLD'):
    def stop(sig, frame):
        output.with_suffix('.signal').write_text(str(sig))
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    while True: time.sleep(.02)
sys.exit(int(os.environ.get('KVMEM_TEST_EXIT', '0')))
'''


@unittest.skipIf(os.name == 'nt', 'Execute Linux shell scripts on Linux')
class ExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='kvmem-export-')
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.worker = self.directory / 'worker with "quotes" $dollar; 中文'
        self.worker.write_text(WORKER, encoding='utf-8')
        self.worker.chmod(0o755)
        self.model = self.directory / "model's data.ninfer"
        self.model.touch()
        self.output = self.directory / 'observed.json'
        self.profiles = self.directory / 'profiles with spaces'
        self.data = {'schema_version': 1, 'parameters': {'Backend': 'ninfer',
            'Model': str(self.model), 'Worker': str(self.worker),
            'Gpu': 'GPU-00000000-0000-0000-0000-000000000001'}}

    def export(self, **kwargs):
        return export_profile(self.data, ROOT, self.profiles, 'test-model', **kwargs)

    def execute(self, result, **environment):
        env = dict(os.environ, KVMEM_TEST_OUTPUT=str(self.output), **environment)
        return subprocess.run([result['script']], env=env, capture_output=True, text=True, timeout=10)

    def test_literal_argv_cwd_and_gpu_survive_independent_execution(self):
        alias = 'literal "quotes" $HOME `touch SHOULD_NOT_EXIST` ; 中文'
        self.data['parameters']['ModelAlias'] = alias
        result = self.export()
        _, _, plan = read_revision(self.profiles, 'test-model', result['revision'])
        run = self.execute(result)
        self.assertEqual(run.returncode, 0, run.stderr)
        observed = json.loads(self.output.read_text())
        self.assertEqual(observed['argv'], plan['argv'])
        self.assertEqual(observed['cwd'], plan['cwd'])
        self.assertEqual(observed['gpu'], self.data['parameters']['Gpu'])
        self.assertFalse((ROOT / 'SHOULD_NOT_EXIST').exists())

    def test_exit_code_is_preserved(self):
        self.assertEqual(self.execute(self.export(), KVMEM_TEST_EXIT='42').returncode, 42)

    def test_layerwise_export_executes_each_explicit_policy_with_exact_limits(self):
        # Independent recording producer; this executable never loads model data.
        capability = dict(schema_version=1, backend='ninfer', query_scoring_revision=2,
            independent_graph_lrfu=True, signed_lrfu_hits=True, recall_strategy_revision=1,
            recall_strategies={'layerwise': dict(available=True,current_layer_q=True,query_replay=False,
                position_policy='original',minimum_reserve_blocks=1,max_devices=1,max_concurrency=1,
                parallel_modes=['single'],kv_formats=['bf16'],vision_residency=[],speculative={
                    'mtp': dict(available=True,history_policies=['upstream-global','first-layer'],
                        state_modes=['native'],max_draft_tokens=3,adaptive=False,ngram=False)})})
        old = "print(json.dumps(dict(schema_version=1, backend='ninfer', query_scoring_revision=2,\n                         independent_graph_lrfu=True, signed_lrfu_hits=True)))"
        self.assertIn(old, WORKER)
        self.worker.write_text(WORKER.replace(old, 'print(' + repr(json.dumps(capability)) + ')'), encoding='utf-8')
        self.data['parameters'].update(RecallStrategy='layerwise',KvType='bf16',SpeculativeMode='mtp',
            DraftTokens=3,DraftPolicy='fixed',SinkTokens=0,RecentTokens=0,
            LayerwiseActivationRamMiB=0,LayerwiseActivationDiskMiB=128,LayerwiseStagingMiB=7,
            DiskPath=str(self.directory/'activation'),DiskMiB=0,
            LayerwiseMetadataMiB=11,LayerwiseActivationMetadataMiB=13)
        for policy in ('upstream-global','first-layer'):
            self.data['parameters']['LayerwiseMtpHistory'] = policy
            exported = self.export()
            directory, _, plan = read_revision(self.profiles,'test-model',exported['revision'])
            saved = json.loads((directory/'profile.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['parameters']['LayerwiseMtpHistory'],policy)
            run = self.execute(exported)
            self.assertEqual(run.returncode,0,run.stderr)
            argv = json.loads(self.output.read_text())['argv']
            self.assertEqual(argv,plan['argv'])
            for flag,value in (('--kvmem-recall-strategy','layerwise'),('--kvmem-layerwise-mtp-history',policy),
                ('--kvmem-layerwise-activation-ram-mib','0'),('--kvmem-layerwise-activation-disk-mib','128'),
                ('--kvmem-layerwise-staging-mib','7'),('--kvmem-layerwise-metadata-mib','11'),
                ('--kvmem-layerwise-activation-metadata-mib','13'),('--kvmem-recent-tokens','0')):
                self.assertEqual(argv.count(flag),1)
                self.assertEqual(argv[argv.index(flag)+1],value)
            self.assertNotIn('--kvmem-query-replay-max-tokens',argv)

    def test_no_ui_process_is_needed_and_lock_blocks_another_revision(self):
        first = self.export()
        second = self.export()
        env = dict(os.environ, KVMEM_TEST_OUTPUT=str(self.output), KVMEM_TEST_HOLD='1')
        proc = subprocess.Popen([first['script']], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not self.output.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(self.output.exists())
            self.assertEqual(json.loads(self.output.read_text())['pid'], proc.pid)
            self.assertEqual(self.execute(second).returncode, 75)
            proc.terminate()
            proc.communicate(timeout=5)
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(self.output.with_suffix('.signal').read_text(), str(int(signal.SIGTERM)))
            self.assertEqual(self.execute(second).returncode, 0)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.communicate(timeout=5)

    def test_manual_changes_are_preserved_and_reported(self):
        first = self.export()
        script = Path(first['script'])
        script.write_text(script.read_text() + '# user modification\n')
        with self.assertRaisesRegex(ValueError, 'modified'):
            read_revision(self.profiles, 'test-model', first['revision'])
        second = self.export()
        self.assertNotEqual(first['revision'], second['revision'])
        self.assertIn('# user modification', script.read_text())

    def test_exported_files_have_unix_permissions_and_line_endings(self):
        result = self.export()
        for path in Path(result['directory']).iterdir():
            self.assertNotIn(b'\r\n', path.read_bytes())
            self.assertEqual(path.stat().st_mode & 0o777, 0o700 if path.name == 'run.sh' else 0o600)
        self.assertEqual(subprocess.run(['bash', '-n', result['script']]).returncode, 0)

    def test_environment_secret_does_not_appear_in_export(self):
        key = 'private-test-credential'
        self.data['parameters']['ApiKey'] = key
        result = self.export()
        self.assertTrue(result['requires_api_key_environment'])
        for path in Path(result['directory']).iterdir():
            self.assertNotIn(key, path.read_text())
        self.assertEqual(self.execute(result).returncode, 78)
        self.assertEqual(self.execute(result, KVMEM_API_KEY=key).returncode, 0)
        observed = json.loads(self.output.read_text())['argv']
        self.assertEqual(observed[observed.index('--api-key') + 1], key)

    def test_stored_secret_is_private_and_outside_revision(self):
        self.data['parameters']['ApiKey'] = 'stored-test-credential'
        result = self.export(persist_secret=True)
        _, _, plan = read_revision(self.profiles, 'test-model', result['revision'])
        key = Path(plan['secret_ref']['path'])
        self.assertEqual(key.stat().st_mode & 0o777, 0o600)
        self.assertFalse(key.is_relative_to(Path(result['directory'])))
        for path in Path(result['directory']).iterdir():
            self.assertNotIn('stored-test-credential', path.read_text())
        self.assertEqual(self.execute(result).returncode, 0)

    def test_missing_runtime_assets_fail_without_invoking_worker(self):
        result = self.export()
        self.model.unlink()
        run = self.execute(result)
        self.assertEqual(run.returncode, 78)
        self.assertFalse(self.output.exists())

    def test_external_secret_with_shell_characters_is_data_only(self):
        key = self.directory / 'key "quotes" $HOME.txt'
        key.write_text('external-secret\n')
        result = self.export(secret_file=key)
        self.assertEqual(self.execute(result).returncode, 0)
        key.write_text('bad\nsecond-line\n')
        self.assertEqual(self.execute(result).returncode, 78)

    def test_invalid_identity_and_symlink_do_not_write_elsewhere(self):
        for value in ('../escape', 'UPPER', '', 'a/b', 'name.service'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                export_profile(self.data, ROOT, self.profiles, value)
        self.profiles.mkdir(exist_ok=True)
        external = self.directory / 'external'
        external.mkdir()
        (self.profiles / 'test-model').symlink_to(external, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.export()
        self.assertEqual(list(external.iterdir()), [])

    def test_failed_generation_does_not_publish_or_leave_secret(self):
        self.data['parameters']['ApiKey'] = 'store-only-if-complete'
        with self.assertRaises(ValueError):
            self.export(persist_secret=True, scope='system', user='root')
        profile = self.profiles / 'test-model'
        self.assertEqual(list((profile / 'revisions').iterdir()), [])
        self.assertEqual(list((profile / 'credentials').iterdir()), [])

    def test_systemd_unit_is_valid_on_target_systemd(self):
        result = self.export()
        run = subprocess.run(['systemd-analyze', '--user', 'verify', result['unit']],
                             capture_output=True, text=True, timeout=10)
        # The unit verifier can run without a user bus on systemd 249.
        self.assertEqual(run.returncode, 0, run.stderr)
        import getpass
        system = self.export(scope='system', user=getpass.getuser() if os.getuid() else 'nobody')
        run = subprocess.run(['systemd-analyze', 'verify', system['unit']],
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)


class UnitRenderingTest(unittest.TestCase):
    def test_unit_name_is_not_an_arbitrary_systemd_target(self):
        self.assertEqual(unit_name('my-model'), 'kvmem-my-model.service')
        with self.assertRaises(ValueError):
            unit_name('sshd.service')

    @unittest.skipIf(os.name == 'nt', 'Linux path semantics')
    def test_systemd_path_escaping_is_distinct_from_shell_quoting(self):
        plan = {'cwd': '/tmp/a%b', 'effective_parameters': {}}
        unit = render_unit(plan, '/tmp/a%b/script $variable.sh')
        self.assertIn('WorkingDirectory=/tmp/a%%b', unit)
        self.assertIn('ExecStart="/tmp/a%%b/script $$variable.sh"', unit)
        with self.assertRaises(ValueError):
            render_unit(plan, '/tmp/path\nExecStart=/bin/false')


if __name__ == '__main__':
    unittest.main()
