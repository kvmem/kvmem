"""Exercise the real pre-checkout APT bootstrap against isolated fake transports."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AptBootstrapTest(unittest.TestCase):
    def run_bootstrap(self, failures, hosted=False):
        workflow = (ROOT / '.github/workflows/server-compat.yml').read_text()
        tail = workflow.split('run: &prepare-ubuntu-apt |\n', 1)[1]
        lines = []
        for line in tail.splitlines():
            if line and not line.startswith('          '):
                break
            lines.append(line[10:])
        script = '\n'.join(lines).replace('/etc/apt/', './apt/') + '\n'
        bash = shutil.which('bash')
        if os.name == 'nt':
            candidate = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe'
            if candidate.is_file():
                bash = str(candidate)
        if not bash:
            self.skipTest('bash is required for the Linux CI bootstrap')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'bin').mkdir()
            (root / 'apt/apt.conf.d').mkdir(parents=True)
            (root / 'apt/sources.list.d').mkdir()
            source = ('deb [signed-by=/usr/share/keyrings/ubuntu.gpg] http://azure.archive.ubuntu.com/ubuntu/ jammy main\n'
                      'deb http://security.ubuntu.com/ubuntu jammy-security main\n')
            vendor = 'deb [signed-by=/keys/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.4 noble main\n'
            deb822 = ('Types: deb\nURIs: http://archive.ubuntu.com/ubuntu\n'
                      'Suites: noble noble-updates\nComponents: main universe\n'
                      'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n')
            if hosted:
                source = source.replace('http://azure.archive.ubuntu.com/ubuntu/', 'mirror+file:./apt/apt-mirrors.txt')
                deb822 = deb822.replace('http://archive.ubuntu.com/ubuntu', 'mirror+file:./apt/apt-mirrors.txt')
                (root / 'apt/apt-mirrors.txt').write_text('http://azure.archive.ubuntu.com/ubuntu/\nhttps://archive.ubuntu.com/ubuntu\n')
            (root / 'apt/sources.list').write_text(source)
            (root / 'apt/sources.list.d/rocm.list').write_text(vendor)
            (root / 'apt/sources.list.d/ubuntu.sources').write_text(deb822)
            commands = {
                'id': 'echo 0\n',
                'timeout': 'test "$1" = 180 || exit 9\nshift\nexec "$@"\n',
                'apt-get': '''test "$*" = update || exit 8
test -f apt/apt.conf.d/80kvmem-downloads || exit 7
grep -q 'APT::Update::Error-Mode "any";' apt/apt.conf.d/80kvmem-downloads || exit 6
count=0
if [ -f count ]; then count=$(cat count); fi
count=$((count+1))
echo "$count" > count
if [ -f "failure-$count" ]; then cat "failure-$count"; exit 100; fi
echo 'Indexes authenticated successfully'
''',
            }
            for name, body in commands.items():
                path = root / 'bin' / name
                path.write_text('#!/usr/bin/env bash\nset -e\n' + body, newline='\n')
                path.chmod(0o755)
            for index, failure in enumerate(failures, 1):
                (root / f'failure-{index}').write_text(failure)
            (root / 'bootstrap.sh').write_text('export PATH="$PWD/bin:$PATH"\n' + script, newline='\n')
            result = subprocess.run([bash, '--noprofile', '--norc', 'bootstrap.sh'], cwd=root,
                                    capture_output=True, text=True, timeout=20)
            self.assertTrue((root / 'count').exists(), result.stdout + result.stderr)
            after = {p.name: p.read_text() for p in [root / 'apt/sources.list',
                     root / 'apt/sources.list.d/rocm.list', root / 'apt/sources.list.d/ubuntu.sources']}
            self.assertEqual(after['rocm.list'], vendor, result.stderr)
            self.assertIn('Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg', after['ubuntu.sources'])
            self.assertIn('Suites: noble noble-updates', after['ubuntu.sources'])
            self.assertIn('jammy-security main', after['sources.list'])
            return result, after, int((root / 'count').read_text())

    def test_original_source_succeeds_without_fallback(self):
        result, after, calls = self.run_bootstrap([])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, 1)
        self.assertIn('https://azure.archive.ubuntu.com/ubuntu/', after['sources.list'])
        self.assertIn('https://security.ubuntu.com/ubuntu ', after['sources.list'])
        self.assertIn('https://archive.ubuntu.com/ubuntu\n', after['ubuntu.sources'])

    def test_dns_failure_falls_back_and_preserves_suites_and_keys(self):
        result, after, calls = self.run_bootstrap(["Could not resolve 'archive.ubuntu.com'"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, 2)
        self.assertIn('https://mirrors.kernel.org/ubuntu/ jammy main', after['sources.list'])
        self.assertIn('https://mirrors.kernel.org/ubuntu jammy-security', after['sources.list'])
        self.assertIn('URIs: https://mirrors.kernel.org/ubuntu\n', after['ubuntu.sources'])

    def test_signature_failure_is_not_retried(self):
        result, after, calls = self.run_bootstrap(['NO_PUBKEY: repository signature invalid'])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, 1)
        self.assertNotIn('mirrors.kernel.org', after['ubuntu.sources'])

    def test_hosted_mirror_indirection_falls_back_to_direct_authenticated_archive(self):
        result, after, calls = self.run_bootstrap(['Temporary failure resolving azure.archive.ubuntu.com'], hosted=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, 2)
        for name in ['sources.list', 'ubuntu.sources']:
            self.assertNotIn('mirror+file:', after[name])
            self.assertIn('https://mirrors.kernel.org/ubuntu', after[name])

    def test_missing_release_is_not_retried(self):
        result, _, calls = self.run_bootstrap(['E: repository has no Release file'])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, 1)

    def test_signature_error_with_transport_error_is_not_retried(self):
        result, _, calls = self.run_bootstrap(['Could not resolve archive.ubuntu.com\nNO_PUBKEY'])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, 1)

    def test_failed_mirror_still_blocks_the_job(self):
        result, _, calls = self.run_bootstrap(['Temporary failure resolving archive.ubuntu.com',
                                              'Temporary failure resolving mirrors.kernel.org'])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, 2)


if __name__ == '__main__':
    unittest.main()
