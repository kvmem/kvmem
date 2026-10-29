"""Console package integrity and portable archive metadata; no native build."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('linux_package', ROOT/'scripts/linux/package-console.py')
package=importlib.util.module_from_spec(spec); spec.loader.exec_module(package)
spec=importlib.util.spec_from_file_location('chat_update', ROOT/'scripts/launcher/chat_update.py')
chat_update=importlib.util.module_from_spec(spec); spec.loader.exec_module(chat_update)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.directory=Path(self.tmp.name)
        self.chat=self.directory/'chat'; self.chat.mkdir()
        (self.chat/'index.html').write_text('<html>fixture</html>')
        (self.chat/'ui-build.json').write_text(json.dumps(dict(console_chat_protocol=1)))
        self.wheel=self.directory/'psutil-7.2.2-cp36-abi3-manylinux_2_28_x86_64.whl'
        with zipfile.ZipFile(self.wheel,'w') as archive:
            archive.writestr('psutil/__init__.py',"__version__='7.2.2'\n")
            archive.writestr('psutil-7.2.2.dist-info/licenses/LICENSE','fixture license')
        self.sha=hashlib.sha256(self.wheel.read_bytes()).hexdigest()

    def build(self, output='package', sha=None):
        return package.package(ROOT,self.directory/output,self.chat,self.wheel,sha or self.sha)

    def test_complete_manifest_lf_and_executable_tar_entries(self):
        result=self.build(); directory=Path(result['directory'])
        info=json.loads((directory/'console-build.json').read_text())
        self.assertEqual(info['supported_backends'],['ninfer'])
        self.assertFalse(info['native_worker_included'])
        self.assertNotIn(b'\r',(directory/'SHA256SUMS').read_bytes())
        for line in (directory/'SHA256SUMS').read_text().splitlines():
            sha,name=line.split('  ',1)
            self.assertEqual(hashlib.sha256((directory/name).read_bytes()).hexdigest(),sha)
        for name,expected in info['controller_files'].items():
            self.assertEqual(hashlib.sha256((directory/name).read_bytes()).hexdigest(),expected)
        with tarfile.open(result['archive']) as archive:
            for name in ('START-HERE-CN.sh','START-HERE-EN.sh','scripts/linux/start-ui.sh'):
                member=archive.getmember('package/'+name)
                self.assertEqual(member.mode,0o755)
                self.assertNotIn(b'\r',archive.extractfile(member).read())
            self.assertEqual(archive.getmember('package/scripts/local-console.py').mode,0o644)
        with self.assertRaisesRegex(ValueError,'already exists'): self.build()

    def test_bad_wheel_or_chat_cannot_create_a_package(self):
        with self.assertRaisesRegex(ValueError,'SHA-256'): self.build(sha='0'*64)
        self.assertFalse((self.directory/'package').exists())
        (self.chat/'ui-build.json').write_text('{"console_chat_protocol":0}')
        with self.assertRaisesRegex(ValueError,'chat'): self.build()

    def test_wheel_traversal_is_rejected_and_partial_folder_removed(self):
        with zipfile.ZipFile(self.wheel,'a') as archive:
            archive.writestr('../outside.py','malicious fixture')
        self.sha=hashlib.sha256(self.wheel.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'Unexpected wheel'): self.build()
        self.assertFalse((self.directory/'package').exists())
        self.assertFalse(list(self.directory.glob('.console-package-*')))
        self.assertFalse((self.directory/'outside.py').exists())


class ChatUpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.directory=Path(self.tmp.name)
        self.chat=self.directory/'chat'; self.chat.mkdir()
        self.base=self.directory/'release-rtx50.zip'
        self.output=self.directory/'update.zip'
        self.api='a'*64
        self.old_chat=chat_update.encode({'console_chat_protocol':1,'entry_files':{'api.ts':self.api}})
        self.new_chat=chat_update.encode({'console_chat_protocol':1,'source_entry_files':{'api.ts':self.api}})
        self.files={
            'scripts/local-console.py':b'original controller',
            'scripts/windows/start-backend.ps1':b'original launcher',
            'workers/ninfer/server.exe':b'original native engine',
            'runtime/python/python.exe':b'original runtime',
            'ui/monitor/index.html':b'original monitor',
            chat_update.CHAT+'index.html':b'old chat',
            chat_update.CHAT+'ui-build.json':self.old_chat,
        }
        self.controller={n:chat_update.digest(self.files[n]) for n in self.files if n.startswith(('scripts/','ui/monitor/'))}
        self.files['console-build.json']=chat_update.encode({
            'console_chat_protocol':1,'controller_files':self.controller,
            'chat_manifest_sha256':chat_update.digest(self.old_chat),
            'supported_backends':['ninfer'],'package_revision':'rc1'})
        for name,data in {
            'index.html':b'<script src="_app/current.js"></script>',
            'ui-build.json':self.new_chat,
            '_app/current.js':b'import("./chunk.js"); const css="current.css";',
            '_app/chunk.js':b'current chunk',
            '_app/current.css':b'current styles',
            '_app/old.js':b'obsolete build',
            '_app/old.css':b'obsolete styles',
            '_app/version.json':b'{"version":"new"}',
            'icon.png':b'icon',
            'monitor/index.html':b'must not replace monitor',
            'monitor-build.json':b'{}',
        }.items():
            path=self.chat/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
        self.write_base()

    def write_base(self,extra=None,wrong_checksum=False):
        sums={n:chat_update.digest(data) for n,data in self.files.items()}
        if wrong_checksum: sums['scripts/local-console.py']='0'*64
        with zipfile.ZipFile(self.base,'w') as archive:
            for name,data in self.files.items(): archive.writestr('release/'+name,data)
            archive.writestr('release/SHA256SUMS',''.join(h+'  '+n+'\n' for n,h in sums.items()))
            if extra: archive.writestr(extra,b'bad path')
        self.sha=chat_update.digest(self.base.read_bytes())

    def build(self,sha=None,output=None):
        return chat_update.package(self.base,sha or self.sha,self.chat,output or self.output)

    def test_overlay_keeps_runtime_contract_and_rebuilds_matching_checksums(self):
        result=self.build()
        self.assertEqual(result['controller_files_unchanged'],len(self.controller))
        with zipfile.ZipFile(self.output) as archive:
            overlay={n:archive.read(n) for n in archive.namelist()}
        self.assertTrue(all(n.startswith(chat_update.CHAT) or n in
            ('console-build.json','SHA256SUMS','CHAT-UPDATE.txt') for n in overlay))
        self.assertIn(chat_update.CHAT+'_app/chunk.js',overlay)
        self.assertIn(chat_update.CHAT+'_app/current.css',overlay)
        self.assertNotIn(chat_update.CHAT+'_app/old.js',overlay)
        self.assertNotIn(chat_update.CHAT+'_app/old.css',overlay)
        self.assertNotIn(chat_update.CHAT+'monitor/index.html',overlay)
        combined={**self.files,**overlay}
        manifest=json.loads(combined['console-build.json'])
        self.assertEqual(manifest['controller_files'],self.controller)
        self.assertEqual(manifest['supported_backends'],['ninfer'])
        self.assertEqual(manifest['package_revision'],'rc1')
        self.assertEqual(manifest['chat_assets_update']['base_archive_sha256'],self.sha)
        for line in combined['SHA256SUMS'].decode().splitlines():
            expected,name=line.split('  ',1)
            self.assertEqual(chat_update.digest(combined[name]),expected,name)
        self.build(output=self.directory/'repeat.zip')
        self.assertEqual(self.output.read_bytes(),(self.directory/'repeat.zip').read_bytes())
        self.assertTrue(self.output.with_suffix('.zip.sha256').read_text().startswith(result['sha256']))
        with self.assertRaisesRegex(ValueError,'already exists'): self.build()

    def test_wrong_archive_or_internal_checksum_cannot_publish_an_update(self):
        with self.assertRaisesRegex(ValueError,'SHA-256'): self.build(sha='0'*64)
        self.write_base(wrong_checksum=True)
        with self.assertRaisesRegex(ValueError,'checksum mismatch'): self.build()
        self.assertFalse(self.output.exists())

    def test_changed_api_or_protocol_requires_separate_validation(self):
        for value,error in [({'console_chat_protocol':2,'source_entry_files':{'api.ts':self.api}},'protocol'),
                            ({'console_chat_protocol':1,'source_entry_files':{'api.ts':'b'*64}},'API source'),
                            ({'console_chat_protocol':1},'API source')]:
            (self.chat/'ui-build.json').write_bytes(chat_update.encode(value))
            with self.assertRaisesRegex(ValueError,error): self.build()
            self.assertFalse(self.output.exists())

    def test_unsafe_or_case_colliding_base_paths_are_rejected(self):
        for name in ('release/../outside.txt','release/C:/outside.txt','release/scripts/LOCAL-console.py'):
            self.write_base(extra=name)
            with self.assertRaisesRegex(ValueError,'Unsafe|Duplicate'): self.build()
            self.assertFalse(self.output.exists())


if __name__=='__main__': unittest.main()
