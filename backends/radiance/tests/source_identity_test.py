import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
TOOLS = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
from source_identity import source_revision

class SourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.meta=json.loads((TOOLS.parent/'upstream.json').read_text(encoding='utf-8'))
        (self.root/'kernel.cpp').write_bytes(b'fixed source\n')
        self.manifest={'revision':self.meta['revision'],'patches':[self.meta['native_patch_sha256'],*[p['sha256'] for p in self.meta.get('additional_native_patches',[])]],
                       'files':{'kernel.cpp':hashlib.sha256(b'fixed source\n').hexdigest()}}
        self.save()
    def save(self):
        (self.root/'.radiance-source-manifest.json').write_text(json.dumps(self.manifest),encoding='utf-8')
    def test_complete_source_matches(self):
        self.assertEqual(source_revision(self.root),self.meta['revision'])
    def test_changed_file_refused(self):
        (self.root/'kernel.cpp').write_bytes(b'changed source\n')
        with self.assertRaises(ValueError):source_revision(self.root)
    def test_unlisted_compilation_unit_refused(self):
        (self.root/'extra.cpp').write_bytes(b'new source\n')
        with self.assertRaises(ValueError):source_revision(self.root)
    def test_old_patch_chain_refused(self):
        self.manifest['patches']=['0'*64];self.save()
        with self.assertRaises(ValueError):source_revision(self.root)
    def test_path_escape_refused(self):
        self.manifest['files']={'../outside.cpp':'0'*64};self.save()
        with self.assertRaises(ValueError):source_revision(self.root)

if __name__=='__main__':unittest.main()
