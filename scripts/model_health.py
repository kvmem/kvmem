"""Bounded CPU-only model inspection. Never imports a worker or reads weight payloads.

The report distinguishes container facts from runtime compatibility. This module is
also a subprocess entry point so the controller can enforce a wall-clock timeout.
"""
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import time

MAX_DIRECTORY = 16 * 1024 * 1024
MAX_RESOURCE = 1024 * 1024
MAX_TENSORS = 200000
MAX_PARTS = 128
FIELDS = {'Backend', 'Model', 'Worker', 'Vision', 'Mmproj', 'VisionWorker',
          'Tokenizer', 'ChatTemplateFile', 'SpeculativeMode', 'MtpDrafts', 'DraftModel'}


class Incomplete(Exception):
    """Not enough information; never grounds for rejecting a model."""


class Invalid(ValueError):
    pass


def request_parameters(data):
    if not isinstance(data, dict) or data.get('schema_version') != 1 or not isinstance(data.get('parameters'), dict):
        raise ValueError('Expected configuration schema 1')
    p = {k: v for k, v in data['parameters'].items() if k in FIELDS}
    if p.get('Backend') not in {'llamacpp', 'ninfer', 'polystrata', 'radiance'}:
        raise ValueError('Unsupported inspection backend')
    for k, v in p.items():
        if k == 'Vision':
            valid = type(v) is bool
        elif k == 'MtpDrafts':
            valid = type(v) is int and 0 <= v <= 2147483647
        else:
            valid = isinstance(v, str) and '\0' not in v and len(v) <= 8192
        if not valid:
            raise ValueError('Invalid inspection parameter: ' + k)
    return p


def unchecked(reason):
    return {'schema_version': 1, 'status': 'unavailable', 'can_start': True,
            'model': {}, 'features': {}, 'checks': [
                {'code': 'inspection_unavailable', 'layer': 'file', 'status': 'unknown',
                 'zh': '体检未完成，可尝试启动', 'en': 'Inspection incomplete; launch may be attempted',
                 'detail': str(reason)[:600]}]}


class Inspection:
    def __init__(self, parameters, base):
        self.p, self.base = parameters, Path(base)
        self.deadline = time.monotonic() + 5
        self.files = {}
        self.report = {'schema_version': 1, 'status': 'ready', 'can_start': True,
                       'model': {}, 'features': {}, 'checks': []}

    def tick(self):
        if time.monotonic() > self.deadline:
            raise Incomplete('Static inspection time limit reached')

    def check(self, code, status, zh, en, detail='', layer='file', fix=None):
        item = dict(code=code, status=status, zh=zh, en=en, detail=str(detail)[:600], layer=layer)
        if fix:
            item['fix'] = fix
        self.report['checks'].append(item)

    def path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.base / path

    def track(self, path):
        self.tick()
        value = path.stat()
        if not stat.S_ISREG(value.st_mode):
            raise Invalid('Expected a regular file: ' + path.name)
        self.files[path] = (value.st_size, value.st_mtime_ns, value.st_ino)
        return value.st_size

    def required_file(self, path, code, zh, en, fix=None):
        try:
            if self.track(path) == 0:
                raise Invalid('Empty file: ' + path.name)
            self.check(code, 'pass', zh, en, path.name, 'config')
            return True
        except (FileNotFoundError, Invalid) as e:
            self.check(code, 'error', zh, en, str(e), 'config', fix)
            return False

    def gguf(self, path):
        size = self.track(path)
        with path.open('rb') as f:
            used = 0

            def read(n):
                nonlocal used
                used += n
                if used > 64 * 1024 * 1024:
                    raise Incomplete('GGUF metadata read limit reached')
                self.tick()
                raw = f.read(n)
                if len(raw) != n:
                    raise Invalid('Truncated GGUF directory: ' + path.name)
                return raw

            def number(fmt):
                return struct.unpack('<' + fmt, read(struct.calcsize(fmt)))[0]

            def skip(n):
                self.tick()
                if n < 0 or f.tell() + n > size:
                    raise Invalid('GGUF metadata exceeds file: ' + path.name)
                f.seek(n, 1)

            def string(keep=True):
                n = number('Q')
                if keep and n > MAX_RESOURCE:
                    raise Incomplete('GGUF string exceeds inspection limit')
                if not keep:
                    skip(n)
                    return None
                return read(n).decode('utf-8')

            types = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'B', 10:'Q', 11:'q', 12:'d'}

            def value(kind, keep=True):
                if kind in types:
                    result = number(types[kind])
                    if kind == 7 and result not in (0, 1):
                        raise Invalid('Invalid GGUF boolean')
                    if isinstance(result, float) and not math.isfinite(result):
                        raise Invalid('Nonfinite GGUF value')
                    return result
                if kind == 8:
                    return string(keep)
                if kind != 9:
                    raise Incomplete('Unknown GGUF metadata type: ' + str(kind))
                subtype, count = number('I'), number('Q')
                if count > 2000000:
                    raise Incomplete('GGUF array exceeds inspection limit')
                if subtype in types:
                    skip(count * struct.calcsize(types[subtype]))
                elif subtype == 8:
                    for _ in range(count):
                        string(False)
                else:
                    raise Incomplete('Unsupported GGUF array type')
                return {'count': count}

            if read(4) != b'GGUF':
                raise Invalid('Expected GGUF magic: ' + path.name)
            version = number('I')
            if version not in (2, 3):
                raise Incomplete('Unsupported GGUF version: ' + str(version))
            tensors, kvs = number('Q'), number('Q')
            if tensors > MAX_TENSORS or kvs > 10000:
                raise Incomplete('GGUF directory exceeds inspection limit')
            metadata, keys = {}, set()
            for _ in range(kvs):
                key = string()
                if key in keys:
                    raise Invalid('Duplicate GGUF metadata key: ' + key[:100])
                keys.add(key)
                kind = number('I')
                keep = key == 'tokenizer.chat_template' or key.startswith(('general.', 'split.')) or key.endswith(('.block_count', '.context_length', '.expert_count', '.nextn_predict_layers'))
                result = value(kind, keep)
                if keep or key in {'tokenizer.ggml.tokens', 'tokenizer.chat_template'}:
                    metadata[key] = result
            from model_health_ninfer.formats import GGUF_FORMATS_BY_TYPE
            blocks = {k: (v.block_elements, v.block_bytes) for k, v in GGUF_FORMATS_BY_TYPE.items()}
            blocks.update({0:(1,4), 1:(1,2), 3:(32,20), 7:(32,24), 9:(32,40), 15:(256,292),
                           24:(1,1), 25:(1,2), 26:(1,4), 27:(1,8), 28:(1,8), 30:(1,2),
                           142:(128,34), 143:(128,28)})
            names, spans, unknown = set(), [], set()
            alignment = metadata.get('general.alignment', 32)
            if type(alignment) is not int or not 1 <= alignment <= 1048576 or alignment & (alignment - 1):
                raise Invalid('Invalid GGUF alignment')
            for _ in range(tensors):
                name, rank = string(), number('I')
                if name in names or not 1 <= rank <= 4:
                    raise Invalid('Duplicate tensor or invalid GGUF tensor rank')
                names.add(name)
                dims = [number('Q') for _ in range(rank)]
                if any(d == 0 for d in dims) or math.prod(dims) > (1 << 63):
                    raise Invalid('Invalid GGUF tensor dimensions')
                kind, offset = number('I'), number('Q')
                if offset % alignment:
                    raise Invalid('Misaligned GGUF tensor: ' + name)
                extent = None
                if kind in blocks:
                    block, nbytes = blocks[kind]
                    if dims[0] % block:
                        raise Invalid('Invalid GGUF quantization block shape: ' + name)
                    extent = math.prod(dims) // block * nbytes
                else:
                    unknown.add(kind)
                spans.append((offset, extent, name))
            start = (f.tell() + alignment - 1) // alignment * alignment
            end = 0
            for offset, extent, name in sorted(spans):
                if offset < end or start + offset >= size or (extent is not None and start + offset + extent > size):
                    raise Invalid('Overlapping or truncated GGUF tensor: ' + name)
                end = offset + (extent or 1)
        return dict(metadata=metadata, names=names, unknown=unknown, bytes=size, version=version)

    def inspect_gguf(self, path):
        first = self.gguf(path)
        meta = first['metadata']
        count = meta.get('split.count', 1)
        if type(count) is not int or not 1 <= count <= MAX_PARTS:
            raise Incomplete('GGUF split count exceeds inspection limit')
        names, total = set(first['names']), first['bytes']
        unknown = set(first['unknown'])
        if count > 1:
            match = re.fullmatch(r'(.+)-(\d{5})-of-(\d{5})\.gguf', path.name, re.I)
            if not match or int(match[3]) != count:
                raise Invalid('GGUF split filename does not match split.count')
            if int(match[2]) != 1 or meta.get('split.no') != 0:
                raise Invalid('Select the first GGUF shard (-00001-of-...)')
            for index in range(1, count):
                part = self.gguf(path.with_name(f'{match[1]}-{index+1:05}-of-{count:05}.gguf'))
                pm = part['metadata']
                if (pm.get('split.no') != index or pm.get('split.count') != count
                        or pm.get('split.tensors.count') != meta.get('split.tensors.count')
                        or pm.get('general.architecture') != meta.get('general.architecture')):
                    raise Invalid('Inconsistent GGUF shard metadata')
                if names & part['names']:
                    raise Invalid('Duplicate tensors across GGUF shards')
                names.update(part['names']); total += part['bytes']; unknown.update(part['unknown'])
            if len(names) != meta.get('split.tensors.count'):
                raise Invalid('GGUF split tensor count does not match')
        if not names:
            raise Invalid('GGUF contains no tensors')
        arch = meta.get('general.architecture', '')
        if not isinstance(arch, str):
            raise Invalid('Invalid GGUF architecture metadata')
        tokenizer = meta.get('tokenizer.ggml.tokens', {})
        if not isinstance(tokenizer, dict):
            raise Invalid('Expected GGUF tokenizer token array')
        self.report['model'] = dict(name=str(meta.get('general.name', path.name))[:160],
            format='GGUF', version=first['version'], architecture=arch or 'unknown',
            tensors=len(names), files=count, bytes=total,
            context=meta.get(arch + '.context_length'), layers=meta.get(arch + '.block_count'))
        mtp = meta.get(arch + '.nextn_predict_layers')
        self.report['features'] = dict(text='present', vision='external',
            mtp='present' if isinstance(mtp, int) and mtp > 0 or any('nextn' in n for n in names) else 'unknown',
            tokenizer='present' if tokenizer.get('count', 0) else 'unknown',
            template='present' if meta.get('tokenizer.chat_template') else 'unknown')
        self.check('gguf_directory', 'pass', 'GGUF 目录与分片检查通过', 'GGUF directory and shards checked', f'{len(names)} tensors / {count} files')
        if unknown:
            self.check('gguf_types', 'unknown', '部分量化格式尚未验证尺寸', 'Some quantization sizes are unverified', ', '.join(map(str, sorted(unknown))))
        if self.p['Backend'] == 'polystrata' and arch != 'qwen35moe':
            self.check('poly_arch', 'warning', '架构不在当前 PolyStrata 启动器声明范围', 'Architecture outside the current PolyStrata launcher profile', f'actual={arch}, profile=qwen35moe', 'worker')
        if self.p.get('Vision'):
            if self.p['Backend'] == 'polystrata':
                explicit = self.p.get('Mmproj')
                if explicit:
                    projector = self.path(explicit)
                else:
                    exact = [p for p in (path.with_name(path.stem + '-mmproj.gguf'), path.with_name('mmproj-' + path.stem + '.gguf')) if p.is_file()]
                    candidates = exact or sorted(p for p in path.parent.iterdir() if p != path and p.is_file() and 'mmproj' in p.name.lower() and p.suffix.lower() == '.gguf')
                    projector = candidates[0] if len(candidates) == 1 else None
                if projector is None:
                    self.check('projector_missing', 'error', '请在“模板与图像 → 图像配套文件”选择匹配的视觉投影文件', 'Select the matching projector in Templates & vision > Vision assets', 'Mmproj: no unique adjacent GGUF', 'config', 'disable_vision')
                elif projector.suffix.lower() != '.gguf':
                    self.check('projector_invalid', 'error', '视觉投影文件必须是 GGUF', 'Vision projector must be a GGUF file', str(projector), 'config', 'disable_vision')
                elif self.required_file(projector, 'projector', '视觉投影文件', 'Vision projector', 'disable_vision'):
                    try:
                        self.gguf(projector)
                        self.report['features']['vision'] = 'present'
                    except (Invalid, UnicodeError) as e:
                        self.check('projector_invalid', 'error', '视觉投影文件损坏', 'Invalid vision projector', str(e), 'config', 'disable_vision')
                worker = self.path(self.p.get('Worker', ''))
                vision_worker = self.path(self.p['VisionWorker']) if self.p.get('VisionWorker') else worker.parent / 'strata-vision.exe'
                self.required_file(vision_worker, 'vision_worker', '视觉 worker', 'Vision worker', 'disable_vision')
                self.check('projector_pair', 'unknown', '视觉投影与文本模型的配对兼容性未验证', 'Projector/model pairing is unverified', layer='worker')
            else:
                explicit = self.p.get('Mmproj')
                exact = [p for p in (path.with_name(path.stem + '-mmproj.gguf'), path.with_name('mmproj-' + path.stem + '.gguf')) if p.is_file()]
                candidates = exact or sorted(p for p in path.parent.iterdir() if p.is_file() and 'mmproj' in p.name.lower() and p.suffix.lower() == '.gguf')
                if not explicit and len(candidates) != 1:
                    self.check('projector_missing', 'error', '请选择匹配的视觉投影文件', 'Select the matching vision projector', 'Mmproj: no unique adjacent GGUF', 'config', 'disable_vision')
                else:
                    projector = self.path(explicit) if explicit else candidates[0]
                    if self.required_file(projector, 'projector', '视觉投影文件', 'Vision projector', 'disable_vision'):
                        try:
                            self.gguf(projector)
                            self.report['features']['vision'] = 'present'
                            self.check('projector_pair', 'unknown', '视觉文件已找到，配对兼容性由加载检查', 'Projector found; pairing is checked when loading', str(projector), 'worker')
                        except (Invalid, UnicodeError) as e:
                            self.check('projector_invalid', 'error', '视觉投影文件损坏', 'Invalid vision projector', str(e), 'config', 'disable_vision')

    def inspect_ninfer(self, path):
        from model_health_ninfer.reader import Artifact
        from model_health_ninfer.schema import ArtifactError, TensorObject
        from model_health_ninfer.framing import HEADER, MAGIC
        size = self.track(path)
        with path.open('rb') as f:
            raw = f.read(HEADER.size)
        if len(raw) < HEADER.size:
            raise Invalid('Truncated NInfer entry header')
        magic, length, _ = HEADER.unpack(raw)
        if magic != MAGIC:
            if magic.startswith(b'NINFER'):
                raise Incomplete('Unsupported NInfer artifact version')
            raise Invalid('Expected NInfer v3 entry magic')
        if length > size - HEADER.size:
            raise Invalid('NInfer directory exceeds entry file')
        if length > MAX_DIRECTORY:
            raise Incomplete('NInfer directory exceeds 16 MiB inspection limit')
        try:
            with Artifact(path) as artifact:
                directory = artifact.directory
                if len(directory.files) > MAX_PARTS or len(directory.objects) > MAX_TENSORS:
                    raise Incomplete('NInfer directory exceeds inspection limit')
                for index, part in enumerate(directory.files[1:], 1):
                    self.tick()
                    name = part.path
                    if ':' in name or '\\' in name or '/' in name or Path(name).name != name:
                        raise Invalid('Unsafe NInfer continuation filename')
                    sibling = path.parent / name
                    if sibling.resolve().parent != path.parent.resolve():
                        raise Invalid('Continuation must remain in the model directory')
                    self.track(sibling)
                    artifact._file(index)  # Native reader checks UUID, index and exact file size.
                unknown = set()
                for obj in artifact.objects:
                    self.tick()
                    try:
                        artifact.object(obj.id)
                    except ArtifactError as e:
                        if any(s in str(e) for s in ('unknown numeric format', 'unknown tensor layout', 'unsupported resource encoding')):
                            unknown.add(str(e)[:160])
                        else:
                            raise
                components = directory.components
                text = components.get('text', {})
                config = text.get('config', {})
                self.report['model'] = dict(name=str(directory.metadata.get('name', path.name))[:160],
                    format='NInfer', version=3, architecture=str(config.get('model_type', 'unknown'))[:160],
                    tensors=sum(isinstance(o, TensorObject) for o in artifact.objects),
                    files=len(directory.files), bytes=artifact.file_bytes,
                    context=config.get('max_position_embeddings'), layers=config.get('num_hidden_layers'))
                if not self.report['model']['tensors']:
                    raise Invalid('NInfer model contains no tensors')
                features = self.report['features']
                features.update({key: 'present' if key in components else 'absent' for key in ('text','vision','mtp','dflash2')})
                resources = text.get('resources', {})
                features['tokenizer'] = 'present' if 'tokenizer.json' in resources else 'absent'
                features['template'] = 'present' if 'chat_template.jinja' in resources else 'absent'
                self.check('ninfer_directory', 'pass', 'NInfer 目录、对象范围与分片检查通过', 'NInfer directory, object ranges and shards checked', f'{len(directory.objects)} objects / {len(directory.files)} files')
                if unknown:
                    self.check('ninfer_encoding', 'unknown', '部分对象使用尚未支持的编码', 'Some object encodings are unverified', '; '.join(sorted(unknown)))
                if 'text' not in components:
                    self.check('no_text', 'error', '缺少文本组件', 'Text component missing', layer='config')
                for role in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
                    if role not in resources and not (role == 'chat_template.jinja' and self.p.get('ChatTemplateFile')):
                        self.check('missing_' + role, 'warning', '缺少文本资源', 'Text resource missing', role, 'config')
                if self.p.get('Vision') and 'vision' not in components:
                    self.check('no_vision', 'error', '已启用图像，但模型没有视觉组件', 'Vision enabled but the model has no vision component', layer='config', fix='disable_vision')
                mode = self.p.get('SpeculativeMode') or ('mtp' if self.p.get('MtpDrafts', 0) else 'none')
                if mode in ('mtp','dflash2') and mode not in components:
                    self.check('no_draft', 'error', '模型缺少所选推测解码组件', 'Selected speculative component missing', mode, 'config', 'disable_speculative')
                # Read only small resources; tokenizer vocabulary and all tensor payloads stay unread.
                resource_bytes = 0
                for component, body in components.items():
                    if component not in ('text', 'vision'):
                        continue
                    active = component == 'text' or self.p.get('Vision', False)
                    if len(body.get('resources', {})) > 128:
                        raise Incomplete('Component resource count exceeds inspection limit')
                    for role, oid in body.get('resources', {}).items():
                        self.tick()
                        if role not in {'tokenizer_config.json', 'generation_config.json', 'preprocessor_config.json', 'video_preprocessor_config.json', 'chat_template.jinja'}:
                            continue
                        obj = artifact.by_id[oid]
                        if obj.bytes > MAX_RESOURCE or obj.encoding != 'raw_bytes_v1':
                            self.check('resource_limit', 'unknown', '资源内容未检查', 'Resource content not checked', component + '/' + role)
                            continue
                        resource_bytes += obj.bytes
                        if resource_bytes > 8 * MAX_RESOURCE:
                            raise Incomplete('Embedded resource read limit reached')
                        try:
                            content = artifact.read_object(oid).decode('utf-8')
                            if not content.strip():
                                raise ValueError('Empty resource')
                            if role.endswith('.json'):
                                decoded = json.loads(content)
                                if not isinstance(decoded, dict):
                                    raise ValueError('Expected a JSON object')
                                if component == 'vision' and role in ('preprocessor_config.json', 'video_preprocessor_config.json') and config.get('model_type') == 'qwen3_5_text':
                                    # Reference frontend contract, not a claim about an arbitrary selected executable.
                                    expected = {'patch_size':16, 'temporal_patch_size':2, 'merge_size':2,
                                                'image_mean':[0.5]*3, 'image_std':[0.5]*3}
                                    for key, want in expected.items():
                                        if decoded.get(key) != want:
                                            self.check('vision_reference', 'warning', '视觉资源与参考前端要求不一致', 'Vision resource differs from the reference frontend contract',
                                                       f'{role}.{key}: actual={decoded.get(key)!r}; reference={want!r}',
                                                       'worker', 'disable_vision' if active else None)
                        except (ValueError, UnicodeError) as e:
                            self.check('resource_invalid', 'error' if active else 'warning', '内嵌资源无法解析', 'Embedded resource cannot be parsed', component + '/' + role + ': ' + str(e), 'config', 'disable_vision' if component == 'vision' and active else None)
        except ArtifactError as e:
            if re.search(r"unknown \[\x27", str(e)):
                raise Incomplete(str(e)) from e
            raise Invalid(str(e)) from e

    def inspect_radiance(self, path):
        import importlib.util
        source=Path(__file__).resolve().parents[1]/'backends/radiance/tools/inspect_rad.py'
        if not source.is_file():raise Incomplete('Radiance container inspector is not installed')
        size=self.track(path)
        spec=importlib.util.spec_from_file_location('model_health_rad_reader',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        try:info=module.inspect(path,max_directory_bytes=MAX_DIRECTORY)
        except ValueError as error:raise Invalid(str(error)) from error
        self.tick()
        self.report['model']=dict(format='rad',architecture=info['arch_id'],files=1,bytes=size,
                                  name=info['model'],quantization=info['quant'])
        self.report['features']=dict(text='present',vision='unknown',mtp='unknown',dflash2='unknown',tokenizer='unknown',template='unknown')
        vision=bool(info['entries_by_name_prefix'].get('v') and info.get('vision_metadata'))
        self.report['features']['vision']='present' if vision else 'absent'
        if self.p.get('Vision') and not vision:
            raise Invalid('Image input is enabled but this RAD container has no native vision tower/geometry')
        self.check('rad_directory','pass','RAD 目录与文件范围检查通过','RAD directory and file ranges checked',
                   str(info['entries'])+' entries; not a weight checksum or runtime compatibility check')

    def run(self):
        try:
            if not self.p.get('Model', '').strip():
                raise Invalid('Choose a model file')
            path = self.path(self.p['Model'])
            if self.p['Backend'] == 'ninfer':
                self.inspect_ninfer(path)
            elif self.p['Backend']=='radiance':
                self.inspect_radiance(path)
            else:
                self.inspect_gguf(path)
            if self.p.get('ChatTemplateFile'):
                self.required_file(self.path(self.p['ChatTemplateFile']), 'template_override', '自定义聊天模板文件', 'Custom chat template file')
            if self.p['Backend'] == 'polystrata' and self.p.get('Tokenizer'):
                directory = self.path(self.p['Tokenizer'])
                for name in ('vocab.json', 'merges.txt', 'tokenizer.json', 'token_type.json', 'chat_template.jinja'):
                    self.required_file(directory / name, 'poly_tokenizer', '外部 tokenizer 资源', 'External tokenizer resource')
                self.check('tokenizer_pair', 'unknown', '外部分词器与模型的语义一致性未验证', 'External tokenizer/model semantic pairing is unverified', layer='worker')
            if self.p.get('DraftModel'):
                self.required_file(self.path(self.p['DraftModel']), 'draft_model', '外部草稿模型文件', 'External draft model file')
            if self.p.get('Worker'):
                self.required_file(self.path(self.p['Worker']), 'worker_file', 'Worker 文件', 'Worker file')
            self.check('worker_compatibility', 'unknown', 'Worker 加载兼容性未验证', 'Worker loading compatibility is unverified',
                       'CPU metadata only; no weights loaded, GPU allocation or inference.', 'worker')
            for file, identity in self.files.items():
                info = file.stat()
                if (info.st_size, info.st_mtime_ns, info.st_ino) != identity:
                    raise Incomplete('A model/resource file changed during inspection; retry')
        except Incomplete as e:
            self.check('incomplete', 'unknown', '部分检查未完成，可尝试启动', 'Some checks incomplete; launch may be attempted', str(e))
            self.report['status'] = 'unavailable'
        except (Invalid, FileNotFoundError, UnicodeError) as e:
            self.check('invalid_file', 'error', '模型文件检查失败', 'Model file check failed', str(e))
        except (OSError, RecursionError, MemoryError) as e:
            self.check('unreadable', 'unknown', '无法完成静态检查', 'Static inspection could not complete', str(e))
            self.report['status'] = 'unavailable'
        checks = self.report['checks']
        if any(c['status'] == 'error' for c in checks):
            self.report.update(status='blocked', can_start=False)
        elif self.report['status'] != 'unavailable' and any(c['status'] in ('warning', 'unknown') and c['code'] != 'worker_compatibility' for c in checks):
            self.report['status'] = 'warning'
        return self.report


if __name__ == '__main__':
    # The packaged console uses isolated Python; import only the reader shipped
    # beside this entry point, independently of cwd and global site packages.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        parameters = request_parameters(json.loads(sys.stdin.buffer.read(65537)))
        result = Inspection(parameters, Path.cwd()).run()
    except Exception as error:
        # A reader failure is not evidence that the model is invalid.
        result = unchecked(type(error).__name__ + ': ' + str(error))
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
