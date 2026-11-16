"""The preview packager must reject incomplete or damaged GPU code bundles."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from package_windows_rdna3 import anonymize_source_paths, check_gpu_bundles

TARGETS = ('gfx1100', 'gfx1101', 'gfx1102', 'gfx1103')


def bundle(targets):
    triples = [('hip-amdgcn-amd-amdhsa--'+t).encode() for t in targets]
    start = 32 + sum(24+len(t) for t in triples)
    payload = b'\x7fELF' + b'\0'*12
    entries = b''.join(struct.pack('<QQQ', start+i*len(payload), len(payload), len(t))+t
                       for i,t in enumerate(triples))
    return b'__CLANG_OFFLOAD_BUNDLE__'+struct.pack('<Q',len(triples))+entries+payload*len(triples)


class PackageBundleTest(unittest.TestCase):
    def test_multiple_complete_bundles(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'kernels.dll'
            p.write_bytes(b'prefix'+bundle(TARGETS)+b'padding'+bundle(TARGETS))
            self.assertEqual(check_gpu_bundles(p,TARGETS),2)

    def test_missing_architecture_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'kernels.dll';p.write_bytes(bundle(TARGETS[:-1]))
            with self.assertRaisesRegex(ValueError,'bundle targets'):
                check_gpu_bundles(p,TARGETS)

    def test_damaged_code_object_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'kernels.dll';p.write_bytes(bundle(TARGETS)[:-8])
            with self.assertRaisesRegex(ValueError,'invalid .* code object'):
                check_gpu_bundles(p,TARGETS)

    def test_source_path_anonymization_preserves_other_sections(self):
        data=bytearray(1024);data[:2]=b'MZ';struct.pack_into('<I',data,0x3c,128)
        data[128:132]=b'PE\0\0';struct.pack_into('<H',data,134,2)
        first=152
        data[first:first+8]=b'.rdata\0\0';struct.pack_into('<II',data,first+16,256,512)
        data[first+40:first+48]=b'.text\0\0\0';struct.pack_into('<II',data,first+56,128,768)
        path=b'C:/Users/example/project/example.cpp\0';data[512:512+len(path)]=path
        data[768:772]=b'CODE'
        out,count=anonymize_source_paths(bytes(data),'C:/Users/example')
        self.assertEqual(count,1)
        self.assertIn((b'C:/Users/'+b'x'*7+b'/project/example.cpp\0'),out)
        self.assertEqual(data[768:896],out[768:896])
        self.assertEqual(len(data),len(out))

    def test_path_outside_data_section_is_rejected(self):
        data=bytearray(1024);data[:2]=b'MZ';struct.pack_into('<I',data,0x3c,128)
        data[128:132]=b'PE\0\0';struct.pack_into('<H',data,134,1)
        data[152:160]=b'.text\0\0\0';struct.pack_into('<II',data,168,256,512)
        path=b'C:/Users/example/project/example.cpp\0';data[512:512+len(path)]=path
        with self.assertRaisesRegex(ValueError,'outside the read-only data section'):
            anonymize_source_paths(bytes(data),'C:/Users/example')


if __name__ == '__main__':
    unittest.main()
