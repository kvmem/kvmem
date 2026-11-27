import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    'prepare_backends', Path(__file__).resolve().parents[1] / 'scripts/prepare-backends.py')
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PatchSeriesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous_root = prepare.ROOT
        prepare.ROOT = self.root
        self.repo = self.root / 'backends/test'
        self.repo.mkdir(parents=True)
        self.run_git(self.root, 'init', '-q')
        self.run_git(self.repo, 'init', '-q')
        self.run_git(self.repo, 'config', 'core.autocrlf', 'false')
        self.source = self.repo / 'source.txt'
        self.source.write_text('base\n', encoding='utf-8')
        self.run_git(self.repo, 'add', '.')
        self.run_git(self.repo, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                     '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture')
        self.base = self.run_git(self.repo, 'rev-parse', 'HEAD').decode().strip()
        previous_tree = self.run_git(self.repo, 'rev-parse', 'HEAD^{tree}').decode().strip()
        patches = []
        for identity, content in [('format', 'format\n'), ('kernel', 'kernel\n')]:
            self.source.write_text(content, encoding='utf-8')
            tree = prepare.source_tree(self.repo)
            patch = self.root / f'{identity}.patch'
            patch.write_bytes(self.run_git(self.repo, 'diff', '--binary', previous_tree, tree))
            patches.append({'id': identity, 'path': patch.name,
                            'sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
                            'requires': [patches[-1]['id']] if patches else [],
                            'after_tree': tree})
            previous_tree = tree
        self.entry = {'path': 'backends/test', 'base_commit': self.base,
                      'patches': patches, 'patched_tree': previous_tree}
        self.source.write_text('base\n', encoding='utf-8')

    def tearDown(self):
        prepare.ROOT = self.previous_root
        self.temp.cleanup()

    @staticmethod
    def run_git(directory, *args):
        return subprocess.check_output(['git', '-C', str(directory), *args],
                                       stderr=subprocess.STDOUT)

    def test_apply_overlapping_patches_and_repeat_without_changing_index(self):
        index = (self.repo / '.git/index').read_bytes()
        prepare.prepare('test', self.entry, False)
        self.assertEqual(self.source.read_text(), 'kernel\n')
        prepare.prepare('test', self.entry, False)
        prepare.prepare('test', self.entry, True)
        self.assertEqual((self.repo / '.git/index').read_bytes(), index)

    def test_resume_known_prefix(self):
        self.source.write_text('format\n', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            prepare.prepare('test', self.entry, True)
        self.assertEqual(self.source.read_text(), 'format\n')
        prepare.prepare('test', self.entry, False)
        self.assertEqual(self.source.read_text(), 'kernel\n')

    def test_unknown_edits_are_preserved(self):
        extra = self.repo / 'local.txt'
        extra.write_text('developer work\n', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'outside the recorded'):
            prepare.prepare('test', self.entry, False)
        self.assertEqual(extra.read_text(), 'developer work\n')
        self.assertEqual(self.source.read_text(), 'base\n')

    def test_wrong_hash_or_dependency_stops_before_editing(self):
        self.entry['patches'][1]['requires'] = ['missing']
        with self.assertRaisesRegex(RuntimeError, 'dependencies'):
            prepare.prepare('test', self.entry, False)
        self.entry['patches'][1]['requires'] = ['format']
        self.entry['patches'][1]['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'patch hash'):
            prepare.prepare('test', self.entry, False)
        self.assertEqual(self.source.read_text(), 'base\n')

    def test_invalid_later_patch_does_not_apply_first_patch(self):
        patch = self.root / 'kernel.patch'
        patch.write_text('not a patch\n', encoding='utf-8')
        self.entry['patches'][1]['sha256'] = hashlib.sha256(patch.read_bytes()).hexdigest()
        with self.assertRaises(subprocess.CalledProcessError):
            prepare.prepare('test', self.entry, False)
        self.assertEqual(self.source.read_text(), 'base\n')

    def test_incorrect_intermediate_tree_fails_preflight(self):
        self.entry['patches'][0]['after_tree'] = '0' * 40
        with self.assertRaisesRegex(RuntimeError, 'patch produces'):
            prepare.prepare('test', self.entry, False)
        self.assertEqual(self.source.read_text(), 'base\n')

    def test_remove_optimization_and_rebuild_from_base(self):
        self.entry['patches'].pop()
        self.entry['patched_tree'] = self.entry['patches'][0]['after_tree']
        prepare.prepare('test', self.entry, False)
        self.assertEqual(self.source.read_text(), 'format\n')
        prepare.prepare('test', self.entry, True)

    def test_ready_sources_still_verify_the_recorded_patch_chain(self):
        prepare.prepare('test', self.entry, False)
        patch = self.root / 'kernel.patch'
        patch.write_text('not a patch\n', encoding='utf-8')
        self.entry['patches'][1]['sha256'] = hashlib.sha256(patch.read_bytes()).hexdigest()
        with self.assertRaises(subprocess.CalledProcessError):
            prepare.prepare('test', self.entry, True)
        self.assertEqual(self.source.read_text(), 'kernel\n')

    def test_polystrata_export_updates_series_and_preserves_real_index(self):
        script = self.root/'tools/polystrata/export_patch.py'
        script.parent.mkdir(parents=True)
        script.write_bytes((Path(__file__).resolve().parents[1]/'tools/polystrata/export_patch.py').read_bytes())
        self.entry['patches'] = [dict(self.entry['patches'][0], id='kvmem')]
        self.entry['patched_tree'] = self.entry['patches'][0]['after_tree']
        manifest = self.root/'backends/versions.json'
        manifest.write_text(json.dumps({'polystrata':self.entry}), encoding='utf-8')
        self.source.write_text('latest backend development\n', encoding='utf-8')
        (self.repo/'added.txt').write_text('new content\n', encoding='utf-8')
        index = (self.repo/'.git/index').read_bytes()
        run = subprocess.run([sys.executable,str(script)],capture_output=True,text=True,check=True)
        self.assertTrue(json.loads(run.stdout)['roundtrip'])
        exported = json.loads(manifest.read_text(encoding='utf-8'))['polystrata']
        self.assertNotIn('patch',exported)
        self.assertNotIn('patch_sha256',exported)
        self.assertEqual(exported['patched_tree'],exported['patches'][0]['after_tree'])
        prepare.prepare('polystrata',exported,True)
        self.assertEqual((self.repo/'.git/index').read_bytes(),index)

    def test_source_bundle_verifies_all_patches_and_files_without_git(self):
        # Renaming only this test fixture's marker selects exported-source verification.
        (self.root / '.git').rename(self.root / 'fixture-git')
        manifest = {'backends': {'test': self.entry}, 'files': {
            'backends/test/source.txt': hashlib.sha256(self.source.read_bytes()).hexdigest()}}
        (self.root / 'SOURCE-MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')
        prepare.prepare('test', self.entry, True)
        self.source.write_text('modified bundle\n', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'differs from manifest'):
            prepare.prepare('test', self.entry, True)


class HipfireIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous_root = prepare.ROOT
        prepare.ROOT = self.root
        self.manifest_dir = self.root / 'backends/hipfire'
        self.manifest_dir.mkdir(parents=True)
        self.manifest = self.manifest_dir / 'upstream.json'
        self.entry = {
            'layout': 'external-checkout-with-patches',
            'repository': 'https://github.com/example/hipfire.git',
            'base_commit': 'a' * 40,
            'patched_tree': 'b' * 40,
            'patches': [{'id': 'one'}],
        }
        self.manifest.write_text(json.dumps(self.entry), encoding='utf-8')
        self.versions = {'hipfire': {
            'layout': 'external-checkout-with-patches',
            'manifest': 'backends/hipfire/upstream.json',
            'checkout': 'checkout',
        }}
        (self.root / 'backends/versions.json').write_text(
            json.dumps(self.versions), encoding='utf-8')

    def tearDown(self):
        prepare.ROOT = self.previous_root
        self.temp.cleanup()

    @staticmethod
    def run_git(directory, *args):
        return subprocess.check_output(['git', '-C', str(directory), *args],
                                       stderr=subprocess.STDOUT)

    def test_repository_index_resolves_the_manifest(self):
        previous = prepare.ROOT
        prepare.ROOT = Path(__file__).resolve().parents[1]
        try:
            versions = json.loads((prepare.ROOT / 'backends/versions.json').read_text(
                encoding='utf-8'))
            entry = prepare.backend_entry(versions, 'hipfire')
            self.assertEqual(entry['layout'], 'external-checkout-with-patches')
            self.assertEqual(len(entry['patches']), 4)
            self.assertEqual(entry['patches'][-1]['id'], 'layerwise-no-refresh')
            self.assertEqual(entry['patches'][-1]['after_tree'], entry['patched_tree'])
        finally:
            prepare.ROOT = previous

    def test_manifest_layout_mismatch_and_escape_are_rejected(self):
        self.versions['hipfire']['layout'] = 'git-submodules-with-patches'
        with self.assertRaisesRegex(RuntimeError, 'layout'):
            prepare.backend_entry(self.versions, 'hipfire')
        self.versions['hipfire']['layout'] = 'external-checkout-with-patches'
        self.versions['hipfire']['manifest'] = '../outside.json'
        with self.assertRaisesRegex(RuntimeError, 'escapes the repository'):
            prepare.backend_entry(self.versions, 'hipfire')

    def test_ready_requires_base_and_recorded_patched_tree(self):
        repo = self.root / 'checkout'
        repo.mkdir()
        self.run_git(repo, 'init', '-q')
        self.run_git(repo, 'config', 'core.autocrlf', 'false')
        (repo / 'source.txt').write_text('base\n', encoding='utf-8')
        self.run_git(repo, 'add', '.')
        self.run_git(repo, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                     '-c', 'commit.gpgsign=false', 'commit', '-qm', 'base')
        base = self.run_git(repo, 'rev-parse', 'HEAD').decode().strip()
        tree = self.run_git(repo, 'rev-parse', 'HEAD^{tree}').decode().strip()
        self.entry['base_commit'] = base
        self.entry['patched_tree'] = tree
        ready, reason = prepare.hipfire_ready(self.entry, repo)
        self.assertTrue(ready, reason)
        (repo / 'dirty.txt').write_text('local\n', encoding='utf-8')
        ready, reason = prepare.hipfire_ready(self.entry, repo)
        self.assertFalse(ready)
        self.assertIn('patched tree', reason)
        (repo / 'dirty.txt').unlink()
        self.run_git(repo, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                     '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-qm', 'extra')
        ready, reason = prepare.hipfire_ready(self.entry, repo)
        self.assertFalse(ready)
        self.assertIn('HEAD', reason)

    def test_all_skips_unprepared_hipfire_without_failing(self):
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['prepare-backends.py', '--backend', 'all']), \
                redirect_stdout(output):
            prepare.main()
        self.assertIn('hipfire: skipped', output.getvalue())


if __name__ == '__main__':
    unittest.main()
