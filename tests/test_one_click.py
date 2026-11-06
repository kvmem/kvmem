"""Windows entry point checks; optional offline runtime/package integration."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'


@unittest.skipUnless(os.name == 'nt', 'Windows entry point')
class OneClickTests(unittest.TestCase):
    def setUp(self):
        self.source_tmp = tempfile.TemporaryDirectory(prefix='kvmem-console-source-')
        self.addCleanup(self.source_tmp.cleanup)
        self.source = Path(self.source_tmp.name)
        for name in ('scripts', 'tools/polystrata', 'ui/monitor', 'backends/radiance/tools'):
            shutil.copytree(ROOT / name, self.source / name, ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('START-HERE-CN.bat', 'START-HERE-EN.bat'):
            shutil.copy2(ROOT / name, self.source / name)
        self.chat = self.source / 'ui/dist'
        self.chat.mkdir()
        (self.chat / 'index.html').write_text('<html>chat packaging fixture</html>', encoding='utf-8')
        (self.chat / 'ui-build.json').write_text('{"console_chat_protocol":1}', encoding='utf-8')

    def run_script(self, script, *args, **kwargs):
        return subprocess.run([str(POWERSHELL), '-NoProfile', '-ExecutionPolicy', 'Bypass',
                               '-File', str(script), *map(str, args)],
                              capture_output=True, text=True, timeout=60, **kwargs)

    def test_source_check_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_script(self.source / 'scripts/windows/start-ui.ps1', '-Check', cwd=folder)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Python runtime OK:', result.stdout)

    def test_source_check_rejects_missing_or_incompatible_chat(self):
        manifest = self.chat / 'ui-build.json'
        manifest.unlink()
        missing = self.run_script(self.source / 'scripts/windows/start-ui.ps1', '-Check')
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn('chat files are missing', missing.stdout)
        manifest.write_text('{"console_chat_protocol":0}', encoding='utf-8')
        old = self.run_script(self.source / 'scripts/windows/start-ui.ps1', '-Check')
        self.assertNotEqual(old.returncode, 0)
        self.assertIn('does not match', old.stdout)

    def test_batch_entries_prompt_for_ip_and_port_and_allow_cancel(self):
        cmd = Path(os.environ['SystemRoot']) / 'System32/cmd.exe'
        for entry, ip_label, port_label in (('START-HERE-CN.bat', '监听 IP', '控制台端口'),
                                           ('START-HERE-EN.bat', 'Listen IP', 'Console port')):
            with self.subTest(entry=entry):
                result = subprocess.run([str(cmd), '/d', '/c', str(self.source / entry),
                    '-ListenHost', '0.0.0.0', '-Port', '18198', '-NoBrowser'],
                    input='\nq\n', capture_output=True, text=True, encoding='utf-8', timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(ip_label + ' [0.0.0.0]', result.stdout)
                self.assertIn(port_label + ' [18198]', result.stdout)
                self.assertNotIn('KVMem console:', result.stdout)

    @unittest.skipUnless(os.environ.get('KVMEM_TEST_PYTHON_ARCHIVE') and os.environ.get('KVMEM_TEST_PSUTIL_WHEEL'), 'set KVMEM_TEST_PYTHON_ARCHIVE for portable integration')
    def test_offline_package_without_system_python(self):
        archive = Path(os.environ['KVMEM_TEST_PYTHON_ARCHIVE'])
        # Use the publisher-verified archive as a fixture; the mismatch case checks enforcement.
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        wheel = Path(os.environ['KVMEM_TEST_PSUTIL_WHEEL'])
        wheel_args = ('-PsutilWheel', wheel, '-PsutilSha256', hashlib.sha256(wheel.read_bytes()).hexdigest())
        with tempfile.TemporaryDirectory(prefix='kvmem-one-click-') as folder:
            package = Path(folder) / 'package with spaces'
            builder = ROOT / 'scripts/windows/package-console.ps1'
            result = self.run_script(builder, '-OutputDir', package, '-PythonArchive', archive,
                                     '-PythonSha256', '0' * 64, '-SourceDir', self.source, *wheel_args)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(package.exists(), 'mismatched archive must not modify the output')
            result = self.run_script(builder, '-OutputDir', package, '-PythonArchive', archive,
                                     '-PythonSha256', digest, '-SourceDir', self.source, *wheel_args)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((package / 'console-build.json').read_text())
            self.assertTrue(manifest['offline_console'])
            self.assertEqual(manifest['console_chat_protocol'], 1)
            self.assertTrue((package / 'ui/dist/index.html').is_file())
            self.assertTrue((package / 'scripts/console_chat.py').is_file())
            inspect = subprocess.run([str(package / 'runtime/python/python.exe'), '-I', '-S',
                                      str(package / 'scripts/model_health.py')],
                                     input=json.dumps({'schema_version': 1, 'parameters': {
                                         'Backend': 'ninfer', 'Model': str(package / 'missing.ninfer')}}),
                                     capture_output=True, text=True, timeout=15)
            self.assertEqual(inspect.returncode, 0, inspect.stderr)
            self.assertFalse(json.loads(inspect.stdout)['can_start'])
            self.assertEqual(manifest['python_archive_sha256'], digest)
            for name, expected in manifest['controller_files'].items():
                self.assertEqual(hashlib.sha256((package / name).read_bytes()).hexdigest(), expected)
            # Exercise the shipped helper with the bundled runtime, outside the
            # source checkout. A plan-only check must not need a native worker.
            fixture_spec = importlib.util.spec_from_file_location(
                'radiance_fixture', ROOT / 'backends/radiance/tests/draft_control_test.py')
            fixture = importlib.util.module_from_spec(fixture_spec)
            fixture_spec.loader.exec_module(fixture)
            model = Path(folder) / 'draft-model.RAD'
            fixture.container(model, ['dflash.fc.weight'], [
                ('draft.architectures', 'DFlash2Model'), ('draft.dflash_config.block_size', 8)])
            draft = subprocess.run([str(package / 'runtime/python/python.exe'), '-I', '-c',
                'import sys; sys.path.insert(0,sys.argv[1]); '
                'from launcher.radiance import draft_selection; '
                'print(draft_selection(sys.argv[2],"dflash2",None)["depth"])',
                str(package / 'scripts'), str(model)], cwd=folder, capture_output=True, text=True, timeout=15)
            self.assertEqual(draft.returncode, 0, draft.stdout + draft.stderr)
            self.assertEqual(draft.stdout.strip(), '7')
            self.assertTrue((package / 'licenses/Python-LICENSE.txt').is_file())
            self.assertTrue((package / 'scripts/windows/select-model.ps1').is_file(),
                            'native file/folder/template picker must ship with the console')
            # Discovery must identify a colocated worker without executing it.
            colocated_worker = package / 'ninfer-serve.exe'; colocated_worker.touch()
            env = dict(os.environ, PATH=str(POWERSHELL.parents[2]), PYTHONPATH='invalid-python-path')
            cmd = Path(os.environ['SystemRoot']) / 'System32/cmd.exe'
            # Test the actual double-click entry with spaces and a clean runtime PATH.
            for entry in ('START-HERE-CN.bat', 'START-HERE-EN.bat'):
                result = subprocess.run([str(cmd), '/d', '/c', str(package / entry), '-Check'],
                                        cwd=folder, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(str(package / 'runtime/python/python.exe'), result.stdout)
            log = Path(folder) / 'console.log'
            with log.open('w') as output, socket.socket() as first_busy, socket.socket() as next_busy:
                first_busy.bind(('127.0.0.1', 0)); first_busy.listen()
                next_busy.bind(('127.0.0.1', 0)); next_busy.listen()
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
                busy_port = first_busy.getsockname()[1]
                process = subprocess.Popen([str(POWERSHELL), '-NoProfile', '-ExecutionPolicy', 'Bypass',
                                            '-File', str(package / 'scripts/windows/start-ui.ps1'),
                                            '-Port', str(busy_port), '-NoBrowser'],
                                           cwd=folder, env=env, stdin=subprocess.PIPE, text=True, encoding='utf-8',
                                           stdout=output, stderr=subprocess.STDOUT)
                try:
                    process.stdin.write(f'bad\n0\n65536\n{busy_port}\n{next_busy.getsockname()[1]}\n{port}\n')
                    process.stdin.close()
                    origin = f'http://127.0.0.1:{port}'
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    deadline = time.monotonic() + 20
                    while True:
                        try:
                            with opener.open(origin + '/api/launcher/info', timeout=1) as response:
                                info = json.load(response)
                            break
                        except (OSError, urllib.error.URLError):
                            if process.poll() is not None or time.monotonic() >= deadline:
                                self.fail(log.read_text(encoding='utf-8', errors='replace'))
                            time.sleep(0.1)
                    self.assertEqual(info['console_root'], str(package.resolve()))
                    self.assertEqual(info['status']['state'], 'stopped')
                    self.assertTrue(info['prefer_bundled_worker'])
                    self.assertEqual(info['workers']['ninfer'],
                                     {'path': str(colocated_worker), 'available': True})
                    text = log.read_text(encoding='utf-8', errors='replace')
                    self.assertIn(f'端口 {busy_port} 不可用', text)
                    self.assertIn('端口必须是 1 到 65535 之间的整数', text)
                    self.assertIn(f'端口 {next_busy.getsockname()[1]} 不可用', text)
                    # The English batch entry reopens the same Chinese-started controller.
                    result = subprocess.run([str(cmd), '/d', '/c', str(package / 'START-HERE-EN.bat'),
                                             '-Port', str(port), '-NoBrowser'],
                                            input='\n\n', cwd=folder, env=env, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(f'already running: {origin}/monitor/?lang=en', result.stdout)
                    with opener.open(origin + '/api/launcher/info', timeout=2) as response:
                        self.assertEqual(json.load(response)['token'], info['token'], 'reuse must not restart the controller')
                    # Launch validation must use the selected console port, not the failed one.
                    request = urllib.request.Request(origin + '/api/launcher/preview',
                        data=json.dumps({'schema_version': 1, 'parameters': {'Backend': 'ninfer', 'Port': port}}).encode(),
                        headers={'Content-Type': 'application/json', 'Origin': origin, 'X-KVMem-Token': info['token']})
                    with self.assertRaises(urllib.error.HTTPError) as rejected:
                        opener.open(request, timeout=3)
                    with rejected.exception as response:
                        self.assertEqual(response.code, 400)
                        self.assertIn('must use different ports', json.load(response)['error'])
                    for name in ('', 'launcher.js', 'monitor.css', 'Strata-LICENSE.txt', 'outfit-latin-wght.woff2'):
                        with opener.open(origin + '/monitor/' + name, timeout=2) as response:
                            self.assertEqual(response.status, 200)
                finally:
                    # Kill only this test's process tree, including the owned Python host.
                    subprocess.run([str(Path(os.environ['SystemRoot']) / 'System32/taskkill.exe'),
                                    '/PID', str(process.pid), '/T', '/F'], capture_output=True)
                    process.wait(timeout=10)
            # The embedded runtime must also support the controller's worker gate.
            python = package / 'runtime/python/python.exe'
            result = subprocess.run([str(python), '-I', '-S', str(package / 'scripts/local-console.py'), '--worker-gate'],
                                    input=json.dumps([str(python), '-I', '-S', '-c', 'print(123)']) + '\n',
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.strip(), '123')


if __name__ == '__main__':
    unittest.main()
