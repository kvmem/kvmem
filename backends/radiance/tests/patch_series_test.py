"""Patch retirement must preserve ordered source identity and package provenance."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from patch_series import patch_specs, verified_patches


class PatchSeriesTest(unittest.TestCase):
    def test_repository_series_is_complete(self):
        meta, specs, paths = verified_patches()
        self.assertEqual(len(paths), 3)
        self.assertEqual([s.get('id') for s in specs[1:]],
                         ['offload-budget', 'generated-history'])
        self.assertNotIn('diff --git a/libr3diqr/', paths[0].read_text(encoding='utf8'))
        for spec in specs[1:]:
            self.assertTrue(spec['retire_when'])

    def test_generated_history_directory_matches_the_portable_header(self):
        _, specs, paths = verified_patches()
        patch = next(path for spec, path in zip(specs, paths) if spec.get('id') == 'generated-history')
        marker = '+++ b/core/kvmem/generated_history.hpp\n'
        hunk = patch.read_text(encoding='utf8').split(marker, 1)[1].split('diff --git', 1)[0]
        embedded = ''.join(line[1:] for line in hunk.splitlines(keepends=True) if line.startswith('+'))
        portable = Path(__file__).resolve().parents[3] / 'kvmem/include/kvmem/generated_history.hpp'
        self.assertEqual(embedded, portable.read_text(encoding='utf8'))

    def test_missing_damaged_or_duplicate_patch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'base.patch').write_bytes(b'base')
            (root/'extension.patch').write_bytes(b'extension')
            meta = {'native_patch': 'base.patch',
                    'native_patch_sha256': hashlib.sha256(b'base').hexdigest(),
                    'additional_native_patches': [{'path':'extension.patch',
                        'sha256':hashlib.sha256(b'extension').hexdigest()}]}
            def save():
                (root/'upstream.json').write_text(json.dumps(meta), encoding='utf8')
            save()
            self.assertEqual(len(verified_patches(root)[2]), 2)
            (root/'extension.patch').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verified_patches(root)
            (root/'extension.patch').unlink()
            with self.assertRaises(FileNotFoundError):
                verified_patches(root)
            meta['additional_native_patches'] = [patch_specs(meta)[0]]
            save()
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                verified_patches(root)
            # Removing a superseded patch is an explicit metadata edit.
            meta['additional_native_patches'] = []
            save()
            self.assertEqual(len(verified_patches(root)[2]), 1)


if __name__ == '__main__':
    unittest.main()
