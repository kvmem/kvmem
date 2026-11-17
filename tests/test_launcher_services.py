"""Real Linux controller/systemd checks. No GPU, model loading or global env edits."""
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from launcher.services import Services, atomic_json
from launcher.export import read_revision
spec = importlib.util.spec_from_file_location('linux_console', ROOT / 'scripts/local-console.py')
console = importlib.util.module_from_spec(spec)
spec.loader.exec_module(console)

WORKER = '''#!/usr/bin/python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
if '--kvmem-capabilities' in sys.argv:
    print(json.dumps(dict(schema_version=1, backend='ninfer', query_scoring_revision=2,
                         independent_graph_lrfu=True, signed_lrfu_hits=True, legacy_query_replay_revision=1)))
    raise SystemExit(0)
print('KVMEM_SCORING scorer=page-softmax agg=mean lrfu=0 lambda=0.5 top_frac=0.9 graph_alpha=0.2', flush=True)
def limit(flag):
    return sys.argv[sys.argv.index(flag)+1] if flag in sys.argv else '0'
print('KVMEM_LIMITS query_max_tokens=' + limit('--kvmem-query-max-tokens') +
      ' query_replay_max_tokens=' + limit('--kvmem-query-replay-max-tokens'), flush=True)
if '--api-key' in sys.argv: print('credential-test ' + sys.argv[sys.argv.index('--api-key')+1], flush=True)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
    def log_message(self, *_): pass
HTTPServer(('127.0.0.1', int(sys.argv[sys.argv.index('--port')+1])), Handler).serve_forever()
'''


def wait_for(function, seconds=7):
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        last = function()
        if last: return last
        time.sleep(.1)
    raise AssertionError('Timed out waiting for real lifecycle state: ' + repr(last))


@unittest.skipIf(os.name == 'nt', 'Linux generated script controller')
class LinuxControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='kvmem-services-')
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.worker = self.directory / 'fake-worker'
        self.worker.write_text(WORKER); self.worker.chmod(0o700)
        model = self.directory / 'model.ninfer'; model.touch()
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0)); self.port = probe.getsockname()[1]
        self.controller = console.Controller(ROOT, self.directory / 'state', 18199)
        self.addCleanup(self.controller.close)
        inspection = patch.object(console.Controller, 'inspect_model', return_value={'can_start': True})
        inspection.start(); self.addCleanup(inspection.stop)
        self.data = dict(schema_version=1, parameters=dict(Backend='ninfer', Worker=str(self.worker),
            Model=str(model), Port=self.port, Gpu='GPU-00000000-0000-0000-0000-000000000001'))
        self.profile = 'test-' + uuid.uuid4().hex[:12]

    def export(self, **options):
        return self.controller.export_config(dict(config=self.data, profile_id=self.profile, **options))

    def test_generate_start_stop_and_worker_change_gate(self):
        result = self.export()
        identity = {key: result[key] for key in ('profile_id', 'revision')}
        self.controller.start_revision(identity)
        state = wait_for(lambda: self.controller.status() if self.controller.status()['state'] == 'ready' else None)
        self.assertEqual(state['owner'], 'temporary')
        self.assertEqual(state['generated_revision'], identity)
        self.assertTrue(state['scoring_confirmed'])
        self.controller.stop()
        self.assertEqual(self.controller.status()['state'], 'stopped')
        self.worker.write_text(WORKER + '\n# replacement\n')
        with self.assertRaisesRegex(ValueError, 'Worker changed'):
            self.controller.start_revision(identity)

    def test_occupied_port_is_not_adopted(self):
        result = self.export()
        with socket.socket() as occupied:
            occupied.bind(('0.0.0.0', self.port)); occupied.listen()
            with self.assertRaisesRegex(ValueError, 'occupied'):
                self.controller.start_revision({key: result[key] for key in ('profile_id', 'revision')})
        self.assertIsNone(self.controller.process)

    def test_worker_replacement_with_same_size_and_mtime_invalidates_probe(self):
        self.controller.worker_capabilities(self.worker,'ninfer')
        before=self.worker.stat()
        replaced=WORKER.replace('query_scoring_revision=2','query_scoring_revision=1')
        self.assertEqual(len(replaced),len(WORKER))
        self.worker.write_text(replaced)
        os.utime(self.worker,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'Worker does not support'):
            self.controller.worker_capabilities(self.worker,'ninfer')

    def test_no_powershell_files_are_needed(self):
        package = self.directory / 'package'
        (package / 'scripts').mkdir(parents=True)
        (package / 'scripts/launcher-capabilities.json').write_bytes((ROOT / 'scripts/launcher-capabilities.json').read_bytes())
        controller = console.Controller(package, self.directory / 'other-state', 18198)
        self.assertEqual(set(controller.backends), {'ninfer', 'llamacpp', 'radiance'})
        plan = controller.plan(self.data)
        self.assertEqual(plan['argv'][0], str(self.worker))
        controller.close()


@unittest.skipUnless(os.name != 'nt' and os.environ.get('KVMEM_TEST_SYSTEMD') == '1', 'Opt in to real user systemd services')
class SystemdTests(LinuxControllerTests):
    # Tests inherited above also ensure the common fixture works on this host.
    def setUp(self):
        super().setUp()
        self.manager = Services(self.controller.profiles_root)
        self.assertTrue(self.manager.available()['user'])
        self.addCleanup(self.cleanup_service)

    def cleanup_service(self):
        self.controller.service_selection = None
        target = self.manager._unit_path(self.profile, 'user')
        if target.is_file():
            # Only this test's unpredictable profile name is ever touched.
            self.manager._run(self.manager._command('user') + ['stop', target.name], check=False)
            self.manager._run(self.manager._command('user') + ['disable', target.name], check=False)
            target.unlink()
            self.manager._run(self.manager._command('user') + ['daemon-reload'])

    def install(self, **options):
        result = self.export(**options)
        self.manager.register(self.profile, result['revision'])
        return result

    def call(self, action):
        return self.controller.service_action(dict(action=action, profile_id=self.profile, scope='user'))

    def test_service_survives_console_close_and_reconnects_with_journal(self):
        self.data['parameters']['ApiKey'] = 'test-background-key'
        result = self.install(store_api_key=True)
        self.call('start')
        wait_for(lambda: self.controller.status()['state'] == 'ready')
        pid = self.controller.status()['pid']
        self.assertEqual(self.controller.status()['owner'], 'systemd')
        self.assertTrue(self.controller.status()['scoring_confirmed'])
        self.controller.close()
        second = console.Controller(ROOT, self.directory / 'state', 18198)
        self.addCleanup(second.close)
        first_log = second.log_snapshot()['text']
        self.assertNotIn('test-background-key', first_log)
        self.assertIn('credential-test <redacted>', first_log)
        state = second.status()
        self.assertEqual(state['state'], 'ready')
        self.assertEqual(state['pid'], pid)
        self.assertEqual(second.monitor_authorization(), 'Bearer test-background-key')
        self.assertNotIn('test-background-key', json.dumps(state))
        self.assertIn('KVMEM_SCORING', second.log_snapshot()['text'])
        self.call('enable')
        self.assertTrue(self.manager.status(self.profile, 'user')['enabled'])
        self.call('disable')
        self.assertFalse(self.manager.status(self.profile, 'user')['enabled'])
        second.stop()
        self.assertEqual(second.status()['state'], 'stopped')
        self.assertIn('KVMEM_SCORING', second.log_snapshot()['text'])
        third = console.Controller(ROOT, self.directory / 'state', 18198)
        self.addCleanup(third.close)
        self.assertIn('KVMEM_SCORING', third.log_snapshot()['text'])
        self.assertNotIn('test-background-key', third.log_snapshot()['text'])
        third.service_selection = None
        second.service_selection = None
        self.assertEqual(self.manager.status(self.profile, 'user')['active'], 'inactive')
        self.call('remove')
        self.assertFalse(Path(result['unit']).parent.parent.parent.joinpath('service-user.json').exists())

    def test_unit_changes_and_unknown_units_are_never_controlled(self):
        result = self.install()
        target = self.manager._unit_path(self.profile, 'user')
        original = target.read_bytes()
        target.write_bytes(original + b'# user change\n')
        try:
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.call('start')
        finally:
            target.write_bytes(original)
            self.manager._run(self.manager._command('user') + ['daemon-reload'])
        with self.assertRaises(ValueError):
            self.manager.action('../dbus', 'user', 'stop')
        self.call('start')
        wait_for(lambda: self.controller.status()['state'] == 'ready')
        another = self.export()
        with self.assertRaisesRegex(ValueError, 'Stop this service'):
            self.manager.register(self.profile, another['revision'])
        self.assertEqual(self.manager.status(self.profile, 'user')['revision'], result['revision'])

    def test_failure_restarts_but_explicit_stop_stays_stopped(self):
        self.install(); self.call('start')
        wait_for(lambda: self.controller.status()['state'] == 'ready')
        initial = self.manager.status(self.profile, 'user')
        os.kill(initial['pid'], signal.SIGKILL)
        wait_for(lambda: (state := self.manager.status(self.profile, 'user'))['worker_matches']
                 and state['pid'] != initial['pid'], seconds=18)
        current = self.manager.status(self.profile, 'user')
        self.assertNotEqual(current['invocation'], initial['invocation'])
        self.call('stop')
        self.assertEqual(self.manager.status(self.profile, 'user')['active'], 'inactive')

    def test_background_rejects_memory_only_key(self):
        self.data['parameters']['ApiKey'] = 'only-in-memory'
        result = self.export()
        with self.assertRaisesRegex(ValueError, 'key file'):
            self.manager.register(self.profile, result['revision'])

    def test_disconnect_never_stops_service_and_recovers_removed_unit(self):
        result = self.install()
        self.call('start')
        wait_for(lambda: self.controller.status()['state'] == 'ready')
        pid = self.manager.status(self.profile, 'user')['pid']
        self.call('disconnect')
        self.assertEqual(self.controller.status()['state'], 'stopped')
        self.assertEqual(self.manager.status(self.profile, 'user')['pid'], pid)
        self.call('select')
        self.manager._run(self.manager._command('user') + ['stop', 'kvmem-' + self.profile + '.service'])
        self.manager._unit_path(self.profile, 'user').unlink()
        self.manager._run(self.manager._command('user') + ['daemon-reload'])
        self.assertIn('missing', self.controller.status()['service_error'])
        self.call('disconnect')
        self.assertFalse(self.controller.service_selection_path.exists())
        self.controller.start_revision({key: result[key] for key in ('profile_id', 'revision')})
        wait_for(lambda: self.controller.status()['state'] == 'ready')
        self.assertEqual(self.controller.status()['owner'], 'temporary')
        self.controller.stop()

    def test_scoring_survives_busy_journal_and_reopen(self):
        noisy = WORKER.replace('        self.send_response(200);',
            "        if self.path == '/audit-noise':\n"
            "            for _ in range(1105): print('audit request noise', flush=True)\n"
            '        self.send_response(200);')
        self.worker.write_text(noisy)
        self.install(); self.call('start')
        wait_for(lambda: self.controller.status()['state'] == 'ready')
        self.assertTrue(self.controller.status()['scoring_confirmed'])
        with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/audit-noise', timeout=5) as response:
            self.assertEqual(response.status, 200)
        wait_for(lambda: 'KVMEM_SCORING' not in self.manager.journal(self.profile, 'user')['text'])
        self.assertTrue(self.controller.status()['scoring_confirmed'])
        self.controller.close()
        reopened = console.Controller(ROOT, self.directory / 'state', 18198)
        self.addCleanup(reopened.close)
        status = reopened.status()
        self.assertTrue(status['scoring_confirmed'], json.dumps(dict(status=status,
            report=self.manager.journal(self.profile, 'user', scoring=True))))
        previous = reopened.observed_service
        self.call('restart')
        wait_for(lambda: reopened.status()['state'] == 'ready')
        self.assertNotEqual(reopened.observed_service, previous)
        self.assertTrue(reopened.status()['scoring_confirmed'])
        reopened.service_selection = None

    def test_registration_failure_restores_previous_unit_or_removes_new_unit(self):
        first=self.export()
        original_run=self.manager._run
        def fail_reload(argv, **kwargs):
            if 'daemon-reload' in argv and kwargs.get('check',True):
                raise ValueError('injected daemon reload failure')
            return original_run(argv, **kwargs)
        with patch.object(self.manager,'_run',side_effect=fail_reload):
            with self.assertRaisesRegex(ValueError,'injected'):
                self.manager.register(self.profile,first['revision'])
        target=self.manager._unit_path(self.profile,'user')
        self.assertFalse(target.exists())
        self.manager.register(self.profile,first['revision'])
        before=target.read_bytes()
        second=self.export()
        with patch.object(self.manager,'_run',side_effect=fail_reload):
            with self.assertRaisesRegex(ValueError,'injected'):
                self.manager.register(self.profile,second['revision'])
        self.assertEqual(target.read_bytes(),before)
        self.assertEqual(self.manager.status(self.profile,'user')['revision'],first['revision'])


@unittest.skipIf(os.name == 'nt', 'Linux audit regressions')
class AuditRegressionTests(unittest.TestCase):
    setUp = LinuxControllerTests.setUp
    export = LinuxControllerTests.export

    def registration(self, secret=False):
        if secret:
            self.data['parameters']['ApiKey'] = 'audit-secret-0123456789abcdef0123456789abcdef'
        result = self.export(store_api_key=secret)
        _, manifest, plan = read_revision(self.controller.profiles_root, self.profile, result['revision'])
        manager = Services(self.controller.profiles_root)
        atomic_json(manager._record_path(self.profile, 'user'), dict(schema_version=1,
            profile_id=self.profile, revision=result['revision'], scope='user',
            unit_sha256=manifest['files']['kvmem-' + self.profile + '.service']))
        state = dict(profile_id=self.profile, revision=result['revision'], scope='user',
                     invocation='b' * 32, active='active')
        return manager, state, plan

    def test_disconnect_recovers_even_a_deleted_registration(self):
        self.controller.service_selection = dict(profile=self.profile, scope='system')
        atomic_json(self.controller.service_selection_path, self.controller.service_selection)
        with patch.object(Services, '_run', side_effect=AssertionError('Disconnect must not call systemctl')):
            self.controller.service_action(dict(action='disconnect', profile_id=self.profile, scope='system'))
        self.assertIsNone(self.controller.service_selection)
        self.assertFalse(self.controller.service_selection_path.exists())
        result = self.export()
        self.controller.start_revision({key: result[key] for key in ('profile_id', 'revision')})
        wait_for(lambda: self.controller.status()['state'] == 'ready')
        self.controller.stop()

    def test_private_state_and_logs_repair_existing_permissions(self):
        old_umask = os.umask(0o022)
        try:
            self.controller.state_dir.chmod(0o755)
            self.controller.log_path.write_text('existing private output\n')
            self.controller.log_path.chmod(0o644)
            reopened = console.Controller(ROOT, self.controller.state_dir, 18198)
            self.addCleanup(reopened.close)
            self.assertEqual(stat.S_IMODE(reopened.state_dir.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(reopened.log_path.stat().st_mode), 0o600)
            self.assertEqual(reopened.log_path.read_text(), 'existing private output\n')
            self.data['parameters']['ApiKey'] = 'audit-private-log-key'
            result = self.export()
            self.controller.start_revision({key: result[key] for key in ('profile_id', 'revision')})
            wait_for(lambda: self.controller.status()['state'] == 'ready')
            self.controller.stop()
            self.assertEqual(stat.S_IMODE(self.controller.log_path.stat().st_mode), 0o600)
            self.assertNotIn('audit-private-log-key', self.controller.log_snapshot()['text'])
        finally:
            os.umask(old_umask)

    def test_private_log_does_not_modify_link_targets(self):
        external = self.directory / 'unrelated'; external.write_text('keep me')
        external.chmod(0o644)
        for link in ('symbolic', 'hard'):
            with self.subTest(link=link):
                if link == 'symbolic': self.controller.log_path.symlink_to(external)
                else: os.link(external, self.controller.log_path)
                try:
                    with self.assertRaises((OSError, ValueError)):
                        console.private_log(self.controller.log_path, truncate=True)
                    self.assertEqual(external.read_text(), 'keep me')
                    self.assertEqual(stat.S_IMODE(external.stat().st_mode), 0o644)
                finally: self.controller.log_path.unlink()

    def test_journal_redacts_before_truncation_and_cli_uses_same_path(self):
        manager, state, _ = self.registration(secret=True)
        key = self.data['parameters']['ApiKey']
        tail = key[5:] + '\n'
        raw = 'prefix ' + key + '\n' + 'x' * (262144 - len(tail))
        output = subprocess.CompletedProcess([], 0, stdout=raw, stderr='')
        with patch.object(Services, 'status', return_value=state), patch.object(Services, '_run', return_value=output):
            result = manager.journal(self.profile, 'user')
            self.assertNotIn(key[5:], result['text'])
            self.assertIn('<redacted>', result['text'])
            self.assertLessEqual(len(result['text'].encode()), 262144)
            spec = importlib.util.spec_from_file_location('audit_cli', ROOT / 'scripts/launcher-cli.py')
            cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
            stdout = io.StringIO()
            with patch.object(sys, 'argv', ['launcher-cli.py', '--profiles-root', str(manager.root),
                 'service', '--action', 'logs', '--profile', self.profile]), contextlib.redirect_stdout(stdout):
                self.assertEqual(cli.main(), 0)
            self.assertNotIn(key[5:], stdout.getvalue())
            output.stdout = 'prefix ' + key + '\n' + '中文 complete line\n' * 30000
            result = manager.journal(self.profile, 'user')
            self.assertTrue(result['truncated'])
            self.assertLessEqual(len(result['text'].encode()), 262144)
            self.assertTrue(result['text'].startswith('中文 complete line\n'))

    def test_journal_withholds_output_when_credential_cannot_be_read(self):
        manager, state, plan = self.registration(secret=True)
        Path(plan['secret_ref']['path']).unlink()
        with patch.object(manager, 'status', return_value=state), patch.object(manager, '_run') as command:
            with self.assertRaisesRegex(ValueError, 'withheld'):
                manager.journal(self.profile, 'user')
            command.assert_not_called()

    def test_scoring_cache_is_bound_to_one_invocation(self):
        self.controller.service_selection = dict(profile=self.profile, scope='user')
        report = 'KVMEM_SCORING scorer=page-softmax agg=mean lrfu=0 lambda=0.5 top_frac=0.9 graph_alpha=0.2\n'
        with patch.object(self.controller, 'service_log', side_effect=[
            dict(text=report, session='a'), dict(text='', session='a'),
            dict(text='', session='b'), dict(text=report.replace('mean', 'max'), session='b')]) as logs:
            self.controller.observe_scoring(); before = dict(self.controller.observed_scoring)
            self.controller.observe_scoring(); self.assertEqual(self.controller.observed_scoring, before)
            self.controller.observe_scoring(); self.assertEqual(self.controller.observed_scoring, {})
            self.controller.observe_scoring(); self.assertEqual(self.controller.observed_scoring['agg'], 'max')
            self.assertTrue(all(call.kwargs == {'scoring': True} for call in logs.call_args_list))


@unittest.skipUnless(os.name != 'nt' and os.environ.get('KVMEM_TEST_SYSTEMD_SYSTEM') == '1',
                     'Opt in as root to a temporary system unit using an ordinary runtime account')
class SystemScopeTests(LinuxControllerTests):
    def test_ordinary_account_system_service_registration_and_reconnect(self):
        import pwd
        self.assertTrue(os.environ.get('KVMEM_TEST_SERVICE_USER'), 'Supply an explicit ordinary test account')
        account=pwd.getpwnam(os.environ['KVMEM_TEST_SERVICE_USER'])
        self.assertNotEqual(account.pw_uid,0)
        manager=Services(self.controller.profiles_root)
        exported=self.export(scope='system',user=account.pw_name)
        # Test-only staging by the administrator; production users generate as
        # the runtime account and install just the unit with sudo.
        for path in [self.directory,*self.directory.rglob('*')]:
            os.chown(path,account.pw_uid,account.pw_gid)
        target=manager._unit_path(self.profile,'system')
        self.assertFalse(target.exists())
        def cleanup():
            self.controller.service_selection=None
            manager._run(manager._command('system')+['stop',target.name],check=False)
            manager._run(manager._command('system')+['disable',target.name],check=False)
            target.unlink(missing_ok=True)
            manager._run(manager._command('system')+['daemon-reload'])
        self.addCleanup(cleanup)
        target.write_bytes(Path(exported['unit']).read_bytes()); target.chmod(0o644)
        manager._run(manager._command('system')+['daemon-reload'])
        manager.register(self.profile,exported['revision'],'system')
        self.controller.service_action(dict(action='start',profile_id=self.profile,scope='system'))
        wait_for(lambda:self.controller.status()['state']=='ready')
        first=manager.status(self.profile,'system')
        self.assertTrue(first['worker_matches'])
        self.assertEqual(Path(f"/proc/{first['pid']}").stat().st_uid,account.pw_uid)
        self.controller.close()
        second=console.Controller(ROOT,self.directory/'state',18198)
        self.assertEqual(second.status()['state'],'ready')
        self.assertEqual(second.status()['pid'],first['pid'])
        second.stop(); second.service_selection=None; second.close()
        self.assertEqual(manager.status(self.profile,'system')['active'],'inactive')


if __name__ == '__main__': unittest.main()
