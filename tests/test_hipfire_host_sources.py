import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    'hipfire_host_sources',
    Path(__file__).resolve().parents[1] / 'scripts/hipfire_host_sources.py')
host_sources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(host_sources)


class HostSourcesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.host = self.root / 'kvmem/src/host'
        self.host.mkdir(parents=True)
        self.manifest = self.root / 'kvmem/host-sources.txt'
        for name in ('a.cpp', 'b.cpp', 'c.cpp'):
            (self.host / name).write_text(f'// {name}\n', encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, text):
        self.manifest.write_text(text, encoding='utf-8')

    def test_check_requires_every_source_exactly_once(self):
        self.write_manifest('src/host/a.cpp\nsrc/host/b.cpp\nsrc/host/c.cpp\n')
        entries, discovered = host_sources.check(self.root)
        self.assertEqual(entries, discovered)
        self.write_manifest('src/host/a.cpp\nsrc/host/b.cpp\n')
        with self.assertRaisesRegex(host_sources.ManifestError, 'missing'):
            host_sources.check(self.root)

    def test_check_rejects_duplicate_and_escaping_paths(self):
        self.write_manifest('src/host/a.cpp\nsrc/host/a.cpp\n')
        with self.assertRaisesRegex(host_sources.ManifestError, 'duplicate'):
            host_sources.check(self.root)
        self.write_manifest('../outside.cpp\n')
        with self.assertRaisesRegex(host_sources.ManifestError, 'under kvmem'):
            host_sources.check(self.root)
        self.write_manifest('src/host/a.cpp\nsrc/other.cpp\n')
        with self.assertRaisesRegex(host_sources.ManifestError, 'expected src/host'):
            host_sources.check(self.root)

    def test_check_rejects_missing_source_and_extra_entry(self):
        self.write_manifest('src/host/a.cpp\nsrc/host/b.cpp\nsrc/host/missing.cpp\n')
        with self.assertRaisesRegex(host_sources.ManifestError, 'does not exist'):
            host_sources.check(self.root)
        self.write_manifest('src/host/a.cpp\nsrc/host/b.cpp\nsrc/host/c.cpp\n')
        (self.host / 'd.cpp').write_text('// d\n', encoding='utf-8')
        with self.assertRaisesRegex(host_sources.ManifestError, 'missing'):
            host_sources.check(self.root)

    def test_write_preserves_order_appends_sorted_and_drops_stale(self):
        self.write_manifest('src/host/b.cpp\nsrc/host/a.cpp\nsrc/host/stale.cpp\n')
        entries = host_sources.write(self.root)
        self.assertEqual(entries, [
            'src/host/b.cpp', 'src/host/a.cpp', 'src/host/c.cpp'])
        self.assertEqual(self.manifest.read_text(encoding='utf-8'),
                         'src/host/b.cpp\nsrc/host/a.cpp\nsrc/host/c.cpp\n')

    def test_repository_manifest_matches_host_sources(self):
        entries, discovered = host_sources.check(
            Path(__file__).resolve().parents[1])
        self.assertEqual(set(entries), set(discovered))
        self.assertIn('src/host/raw_kv_store.cpp', entries)


if __name__ == '__main__':
    unittest.main()
