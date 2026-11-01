"""Small synthetic containers exercise corruption without loading any model weights."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import model_health as health


def gguf(path, *, meta=None, tensor='weight', kind=0, offset=0, payload=b'\0'*4, columns=1):
    def string(s):
        raw = s.encode('utf-8'); return struct.pack('<Q', len(raw)) + raw
    metadata = {'general.architecture':'qwen35', **(meta or {})}
    raw = struct.pack('<4sIQQ', b'GGUF', 3, 1, len(metadata))
    for key, value in metadata.items():
        raw += string(key)
        raw += struct.pack('<I', 8) + string(value) if isinstance(value, str) else struct.pack('<IQ', 10, value)
    raw += string(tensor) + struct.pack('<IQIQ', 1, columns, kind, offset)
    raw += b'\0' * (-len(raw) % 32) + payload
    path.write_bytes(raw)


def ninfer(path, *, parts=1, vision=None, mutate=None):
    resources = {'tokenizer.json':b'{}', 'tokenizer_config.json':b'{}', 'chat_template.jinja':b'hello'}
    payload = bytearray(b'\0'*4)
    objects = [dict(id='weight', kind='tensor', shape=[1], format='fp32', layout='contiguous_le_v1', offset=0, bytes=4)]
    components = {'text':{'config':{'model_type':'qwen3_5_text'}, 'resources':{}}}
    if vision is not None:
        components['vision'] = {'config':{}, 'resources':{}}
        resources['preprocessor_config.json'] = vision
    for role, data in resources.items():
        oid = 'resource/' + role
        objects.append(dict(id=oid, kind='resource', encoding='raw_bytes_v1', offset=len(payload), bytes=len(data)))
        payload.extend(data)
        components['vision' if role.startswith('preprocessor') else 'text']['resources'][role] = oid
    payload.extend(b'\0' * (512 * parts - len(payload)))
    directory = dict(components=components, objects=objects, bindings={'text.weight':{'object':'weight'}}, uses=[],
                     files=[{'path':None if i==0 else 'part.ninfer','payload_bytes':512} for i in range(parts)])
    if mutate:
        mutate(directory)
    encoded = json.dumps(directory).encode('utf-8')
    header = struct.pack('<8sQ16s', b'NINFER\0\3', len(encoded), b'a'*16)
    prefix = header + encoded
    path.write_bytes(prefix + b'\0' * (-len(prefix) % 4096) + payload[:512])
    if parts == 2:
        header = struct.pack('<8sQ16s', b'NINPRT\0\3', 1, b'a'*16)
        path.with_name('part.ninfer').write_bytes(header + b'\0'*(4096-len(header)) + payload[512:])


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def inspect(self, path, backend='llamacpp', **kwargs):
        return health.Inspection(dict(Backend=backend, Model=str(path), **kwargs), self.root).run()

    def test_gguf_text_and_polystrata_inspect_same_format(self):
        model = self.root/'moe.gguf'; gguf(model, meta={'general.architecture':'qwen35moe'})
        for backend in ('llamacpp','polystrata'):
            result = self.inspect(model, backend)
            self.assertEqual(result['status'], 'ready')
            self.assertTrue(result['can_start'])
            self.assertEqual(result['model']['tensors'], 1)
            self.assertEqual(result['checks'][-1]['status'], 'unknown')

    def test_extension_cannot_disguise_wrong_container(self):
        path=self.root/'fake.gguf'; path.write_bytes(b'not a model'*10)
        self.assertFalse(self.inspect(path)['can_start'])
        path=self.root/'wrong.ninfer'; gguf(path)
        self.assertFalse(self.inspect(path, 'ninfer')['can_start'])

    def test_llama_projector_selection_and_corruption(self):
        model=self.root/'model.gguf';gguf(model)
        self.assertFalse(self.inspect(model,Vision=True)['can_start'])
        projector=self.root/'mmproj.gguf';gguf(projector)
        self.assertTrue(self.inspect(model,Vision=True)['can_start'])
        second=self.root/'other-mmproj.gguf';gguf(second)
        self.assertFalse(self.inspect(model,Vision=True)['can_start'])
        self.assertTrue(self.inspect(model,Vision=True,Mmproj=str(projector))['can_start'])
        projector.write_bytes(b'corrupt')
        self.assertFalse(self.inspect(model,Vision=True,Mmproj=str(projector))['can_start'])

    def test_polystrata_projector_discovery_and_explicit_override(self):
        model=self.root/'model.gguf';gguf(model,meta={'general.architecture':'qwen35moe'})
        worker=self.root/'strata-vision.exe';worker.write_bytes(b'fixture')
        def inspect(**kwargs):
            return self.inspect(model,'polystrata',Vision=True,VisionWorker=str(worker),**kwargs)
        self.assertFalse(inspect()['can_start'])
        projector=self.root/'MMPROJ-Q8_0.GGUF';gguf(projector)
        self.assertTrue(inspect()['can_start'])
        default=self.root/'mmproj-BF16.gguf';gguf(default)
        self.assertFalse(inspect()['can_start'],'multiple projectors require a choice, even with BF16 present')
        self.assertTrue(inspect(Mmproj=str(projector))['can_start'])
        exact=self.root/'model-mmproj.gguf';gguf(exact)
        self.assertTrue(inspect()['can_start'],'model-named projector takes priority')
        gguf(self.root/'mmproj-model.gguf')
        self.assertFalse(inspect()['can_start'],'two model-named projectors are ambiguous')
        custom=self.root/"视觉 & ' $custom.gguf";gguf(custom)
        self.assertTrue(inspect(Mmproj=str(custom))['can_start'])
        self.assertFalse(inspect(Mmproj=str(self.root/'missing.gguf'))['can_start'])
        wrong=self.root/'projector.txt';gguf(wrong)
        self.assertFalse(inspect(Mmproj=str(wrong))['can_start'])
        custom.write_bytes(b'corrupt')
        self.assertFalse(inspect(Mmproj=str(custom))['can_start'])

    def test_bonsai_gguf_block_extents(self):
        path = self.root/'bonsai.gguf'
        for kind, size in [(142,34),(143,28)]:
            gguf(path, kind=kind, columns=128, payload=b'\0'*size)
            report = self.inspect(path)
            self.assertTrue(report['can_start'])
            self.assertNotIn('gguf_types', [c['code'] for c in report['checks']])
            gguf(path, kind=kind, columns=128, payload=b'\0'*(size-1))
            self.assertFalse(self.inspect(path)['can_start'])

    def test_bonsai_native_plane_extents(self):
        path = self.root/'bonsai.ninfer'
        # One row: compact base at 0, optional high plane at 256,
        # FP16 scale at 512 (PTQ1) or 256 (T5); the final two bytes need no tail padding.
        for format_name, size in [('ptq1_g128_fp16',514),('t5_g128_fp16',258)]:
            directory = dict(components={'text':{'config':{'model_type':'qwen3_5_text'},'resources':{}}},
                objects=[dict(id='weight',kind='tensor',shape=[1,128],format=format_name,
                              layout='row_split_k128_v1',offset=0,bytes=size)],
                bindings={'text.weight':{'object':'weight'}},uses=[],files=[dict(path=None,payload_bytes=1024)])
            def write():
                encoded=json.dumps(directory).encode()
                prefix=struct.pack('<8sQ16s',b'NINFER\0\3',len(encoded),b'a'*16)+encoded
                path.write_bytes(prefix+b'\0'*(-len(prefix)%4096)+b'\0'*1024)
            write()
            report=self.inspect(path,'ninfer')
            self.assertTrue(report['can_start'],report)
            self.assertNotIn('ninfer_encoding',[c['code'] for c in report['checks']])
            directory['objects'][0]['bytes']=size-1
            write()
            self.assertFalse(self.inspect(path,'ninfer')['can_start'])

    def test_missing_empty_and_truncated_files_block(self):
        path=self.root/'missing.gguf'
        self.assertFalse(self.inspect(path)['can_start'])
        path.touch(); self.assertFalse(self.inspect(path)['can_start'])
        gguf(path); path.write_bytes(path.read_bytes()[:-1])
        self.assertFalse(self.inspect(path)['can_start'])

    def test_future_version_is_unavailable_not_corrupt(self):
        path=self.root/'future.gguf'; path.write_bytes(b'GGUF'+struct.pack('<I',99))
        result=self.inspect(path)
        self.assertEqual(result['status'],'unavailable'); self.assertTrue(result['can_start'])

    def test_unknown_quantization_is_not_rejected(self):
        path=self.root/'future.gguf'; gguf(path,kind=9999)
        result=self.inspect(path)
        self.assertEqual(result['status'],'warning'); self.assertTrue(result['can_start'])

    def test_gguf_huge_metadata_has_bounded_read(self):
        path=self.root/'big.gguf'
        path.write_bytes(struct.pack('<4sIQQQ',b'GGUF',3,1,1,1<<60))
        result=self.inspect(path)
        self.assertEqual(result['status'],'unavailable')

    def test_gguf_shards_missing_wrong_index_and_tensor_overlap(self):
        path=self.root/'model-00001-of-00002.gguf'; second=self.root/'model-00002-of-00002.gguf'
        meta={'split.count':2,'split.no':0,'split.tensors.count':2}
        gguf(path,meta=meta,tensor='first')
        self.assertFalse(self.inspect(path)['can_start'])
        gguf(second,meta={**meta,'split.no':1},tensor='second')
        self.assertEqual(self.inspect(path)['model']['files'],2)
        self.assertFalse(self.inspect(second)['can_start'])
        gguf(second,meta={**meta,'split.no':1},tensor='first')
        self.assertFalse(self.inspect(path)['can_start'])

    def test_ninfer_native_schema_and_all_continuations(self):
        path=self.root/'text.ninfer'; ninfer(path,parts=2)
        result=self.inspect(path,'ninfer')
        self.assertEqual(result['status'],'ready'); self.assertEqual(result['model']['files'],2)
        second=self.root/'part.ninfer'; second.unlink()
        self.assertFalse(self.inspect(path,'ninfer')['can_start'])

    def test_isolated_inspector_reads_ninfer(self):
        path = self.root/'isolated.ninfer'
        ninfer(path, mutate=lambda d: d['components']['text']['config'].update(
            num_hidden_layers=4, num_key_value_heads=4, head_dim=256,
            layer_types=['linear_attention']*3 + ['full_attention']))
        # Bundled Python uses isolated paths; script-local imports must happen
        # after the command-line entry point adds its scripts directory.
        result = subprocess.run(
            [sys.executable, '-I', str(ROOT/'scripts'/'model_health.py')],
            input=json.dumps({'schema_version': 1, 'parameters': {
                'Backend': 'ninfer', 'Model': str(path)}}),
            cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report['can_start'], report)
        self.assertEqual(report['model']['layers'], 4)

    def test_ninfer_wrong_uuid_and_truncated_payload(self):
        path=self.root/'text.ninfer'; ninfer(path,parts=2)
        second=self.root/'part.ninfer'; data=bytearray(second.read_bytes());data[16]=0;second.write_bytes(data)
        self.assertFalse(self.inspect(path,'ninfer')['can_start'])
        ninfer(path);path.write_bytes(path.read_bytes()[:-1])
        self.assertFalse(self.inspect(path,'ninfer')['can_start'])

    def test_ninfer_traversal_is_rejected_before_open(self):
        path=self.root/'text.ninfer'
        for name in ('../escape.ninfer','C:escape.ninfer'):
            ninfer(path,parts=2,mutate=lambda d: d['files'][1].update(path=name))
            self.assertFalse(self.inspect(path,'ninfer')['can_start'])

    def test_ninfer_object_range_and_encoded_extent(self):
        path=self.root/'text.ninfer'
        for mutation in (lambda d:d['objects'][0].update(offset=900),lambda d:d['objects'][0].update(shape=[2])):
            ninfer(path,mutate=mutation)
            self.assertFalse(self.inspect(path,'ninfer')['can_start'])

    def test_ninfer_future_layout_is_unverified(self):
        path=self.root/'text.ninfer';ninfer(path,mutate=lambda d:d['objects'][0].update(layout='future_v9'))
        result=self.inspect(path,'ninfer')
        self.assertTrue(result['can_start']);self.assertEqual(result['status'],'warning')

    def test_ninfer_missing_directory_field_is_corrupt(self):
        path=self.root/'text.ninfer';ninfer(path,mutate=lambda d:d.pop('uses'))
        self.assertFalse(self.inspect(path,'ninfer')['can_start'])

    def test_ninfer_feature_failure_is_scoped_to_enabled_option(self):
        path=self.root/'text.ninfer';ninfer(path)
        for options in ({'Vision':True},{'SpeculativeMode':'mtp'},{'SpeculativeMode':'dflash2'}):
            result=self.inspect(path,'ninfer',**options)
            self.assertFalse(result['can_start'])
            self.assertTrue(any(c.get('fix') for c in result['checks']))
        self.assertTrue(self.inspect(path,'ninfer',Vision=False,SpeculativeMode='none')['can_start'])

    def test_invalid_vision_resource_does_not_disable_text(self):
        path=self.root/'vision.ninfer';ninfer(path,vision=b'{broken')
        self.assertTrue(self.inspect(path,'ninfer',Vision=False)['can_start'])
        self.assertFalse(self.inspect(path,'ninfer',Vision=True)['can_start'])

    def test_reference_contract_does_not_certify_unknown_worker(self):
        path=self.root/'vision.ninfer';ninfer(path,vision=b'{"image_mean":[0.1,0.2,0.3]}')
        result=self.inspect(path,'ninfer',Vision=True)
        self.assertTrue(result['can_start'])
        self.assertTrue(any(c['code']=='vision_reference' and 'actual=' in c['detail'] for c in result['checks']))

    def test_poly_vision_requires_both_external_files(self):
        path=self.root/'moe.gguf';gguf(path,meta={'general.architecture':'qwen35moe'})
        self.assertFalse(self.inspect(path,'polystrata',Vision=True)['can_start'])
        self.assertTrue(self.inspect(path,'polystrata',Vision=False)['can_start'])

    def test_no_tensor_payload_is_read(self):
        path=self.root/'text.ninfer';ninfer(path)
        from model_health_ninfer.reader import Artifact
        original=Artifact.read_object
        read_ids=[]
        def read(artifact,oid):
            self.assertTrue(oid.startswith('resource/'));read_ids.append(oid)
            return original(artifact,oid)
        with patch.object(Artifact,'read_object',read):
            self.assertTrue(self.inspect(path,'ninfer')['can_start'])
        self.assertNotIn('resource/tokenizer.json',read_ids)

    def test_inspector_ignores_credentials_and_rejects_bad_parameters(self):
        p=health.request_parameters({'schema_version':1,'parameters':{'Backend':'ninfer','ApiKey':'secret','Vision':True}})
        self.assertNotIn('ApiKey',p)
        with self.assertRaises(ValueError):
            health.request_parameters({'schema_version':1,'parameters':{'Backend':'ninfer','Vision':'true'}})

    def test_controller_timeout_and_fresh_start_gate(self):
        spec=importlib.util.spec_from_file_location('health_console',ROOT/'scripts/local-console.py')
        console=importlib.util.module_from_spec(spec);spec.loader.exec_module(console)
        controller=console.Controller(ROOT,self.root/'state',18299)
        path=self.root/'text.ninfer';ninfer(path)
        worker=self.root/'worker.exe';worker.write_bytes(b'fixture')
        data={'schema_version':1,'parameters':{'Backend':'ninfer','Model':str(path),'Worker':str(worker)}}
        with patch.object(console.subprocess,'run',side_effect=console.subprocess.TimeoutExpired('inspector',10)):
            self.assertTrue(controller.inspect_model(data)['can_start'])
        with patch.object(controller.inspection_slots, 'acquire', return_value=False):
            self.assertEqual(controller.inspect_model(data)['status'], 'unavailable')
            with self.assertRaisesRegex(ValueError, 'inspection busy'):
                controller.start(data)
        path.write_bytes(b'broken')
        with patch.object(controller,'plan') as plan:
            with self.assertRaisesRegex(ValueError,'inspection blocked'):
                controller.start(data)
            plan.assert_not_called()


if __name__ == '__main__':
    unittest.main()
