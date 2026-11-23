"""Data-only launch plans. No model loading and no shell command evaluation."""
import copy
import json
import os
from pathlib import Path, PureWindowsPath

from .config import DEFAULTS, SPECULATIVE_PROFILES, ROOT, normalize
from .recall import LAYERWISE_RESOURCES, recall_args, strategy

MANIFEST = ROOT / 'scripts/launcher-capabilities.json'
NINFER_KV = ('bf16', 'int8', 'fp8', 'nvfp4', 'k8v4', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8')
LAUNCH_DEFAULTS = dict(DEFAULTS, MtpDrafts=0, AdaptiveMtp=False, FastPrefill=False,
    NgramDrafts=0, NgramMinMatch=12, Concurrency=1, RetainedSessions=4, Vision=False,
    VisionResidency='resident', CpuVisionIdleSeconds=300, VisionTokens=1024,
    DeviceProfile='off', ThinkingBudget=0, ThinkingMode='default', DiskMiB=4096,
    VramReserveMiB=256, BlockTokens=32)
RANGES = {'Context': (256, 2147483647), 'Budget': (128, 2147483647),
    'Reserve': (32, 2147483647), 'HostMiB': (1, 1048576), 'Prefill': (1, 2147483647),
    'MtpDrafts': (0, 15), 'DraftTokens': (0, 15), 'NgramDrafts': (0, 63),
    'NgramMinMatch': (4, 64), 'Concurrency': (1, 8), 'RetainedSessions': (1, 16),
    'VisionTokens': (1, 16384), 'DiskMiB': (0, 1048576), 'VramReserveMiB': (256, 65536),
    'ReplayMaxTokens': (1, 65536)}


def capabilities(root=ROOT):
    path = Path(root) / 'scripts/launcher-capabilities.json'
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if (data.get('schema_version') != 1 or data.get('launch_contract_version') != 3
            or data.get('query_scoring_revision', 0) < 2
            or not isinstance(data.get('backends'), dict)
            or not data['backends'] or set(data['backends']) - set(SPECULATIVE_PROFILES) - {'schema_version'}):
        raise ValueError('Incompatible launcher capability manifest')
    return data


def scoring_settings(argv):
    """Scoring metadata for UI and CLI consumers of the same native argv."""
    def value(flag, default):
        return argv[argv.index(flag) + 1] if flag in argv else default
    if value('--kvmem-recall-strategy', 'legacy') == 'layerwise':
        result = {'strategy': 'layerwise', 'current_layer_q': True, 'query_replay': False}
        if '--kvmem-layerwise-checkpoint-target' in argv:
            result['checkpoint_target'] = int(value('--kvmem-layerwise-checkpoint-target', None))
        if '--kvmem-layerwise-mtp-history' in argv:
            result['mtp_history'] = value('--kvmem-layerwise-mtp-history', None)
        for _, (flag, _) in LAYERWISE_RESOURCES.items():
            if flag in argv:
                result[flag.removeprefix('--kvmem-layerwise-').replace('-', '_').replace('_mib', '_bytes')] = int(value(flag, None)) << 20
        return result
    return {'scorer': value('--kvmem-query-scorer', 'page-softmax'),
            'agg': value('--kvmem-score-agg', 'mean'), 'lrfu': '--kvmem-lrfu' in argv,
            'lambda': float(value('--kvmem-lrfu-lambda', .5)),
            'top_frac': float(value('--kvmem-lrfu-top-frac', .9)),
            'graph_alpha': float(value('--kvmem-graph-alpha', .2)),
            'query_max_tokens': int(value('--kvmem-query-max-tokens', 512)),
            'query_replay_max_tokens': int(value('--kvmem-query-replay-max-tokens', 2048))}


def token_settings(backend, values, block=32, supplied=None):
    """Resolve token invariants; all consumers use the resulting effective values."""
    block = 128 if backend == 'radiance' else 64 if backend == 'ninfer' else block
    if block not in ((128,) if backend=='radiance' else (32,64)):
        raise ValueError('BlockTokens must be 32 or 64')
    if strategy(values) == 'layerwise':
        explicit = values if supplied is None else supplied
        requested = {key: values[key] for key in ('Budget', 'Reserve', 'Prefill', 'Context')}
        for key in ('SinkTokens', 'RecentTokens'):
            requested[key] = explicit.get(key)
        effective = dict(requested)
        if requested['Budget'] <= 0 or requested['Budget'] % block or requested['Reserve'] % block or requested['Reserve'] < block:
            raise ValueError('Layerwise Budget/Reserve require block alignment and Reserve >= one block')
        for key in ('SinkTokens', 'RecentTokens'):
            if requested[key] is not None and (requested[key] < 0 or requested[key] % block):
                raise ValueError(f'Layerwise {key} requires block alignment; explicit zero is preserved')
        if all(requested[key] is not None for key in ('SinkTokens', 'RecentTokens')) and requested['SinkTokens'] + requested['RecentTokens'] > requested['Budget']:
            raise ValueError('SinkTokens + RecentTokens must fit layerwise startup Budget')
        if requested['Budget'] + requested['Reserve'] > requested['Context']:
            raise ValueError('Budget + Reserve must fit Context')
        if requested['Prefill'] <= 0:
            raise ValueError('Layerwise Prefill must be positive')
        return dict(requested=requested, effective=effective, block_tokens=block, adjustments=[])
    requested = {k: values[k] for k in ('Budget', 'Reserve', 'SinkTokens', 'RecentTokens', 'Prefill', 'Context')}
    effective = dict(requested)
    minima = {'Budget': 2*block if backend=='radiance' else block if backend == 'llamacpp' else 128, 'Reserve': block,
              'SinkTokens': 0 if backend == 'polystrata' else block,
              'RecentTokens': 0 if backend == 'llamacpp' else block}
    for key, minimum in minima.items():
        value = requested[key] // block * block
        if (backend == 'llamacpp' and key in ('Budget', 'Reserve')) or (backend != 'polystrata' and key == 'SinkTokens'):
            value = max(minimum, value)
        if value < minimum:
            raise ValueError(f'{key} must retain at least {minimum} tokens after alignment')
        effective[key] = value
    if backend in ('ninfer', 'polystrata', 'radiance'):
        prefill = requested['Prefill'] or (1024 if backend == 'ninfer' else 0)
        alignment = 128 if backend == 'ninfer' else 1
        effective['Prefill'] = min(prefill, effective['Reserve']) // alignment * alignment
        if effective['Prefill'] < alignment:
            raise ValueError(f'Prefill and Reserve must allow at least {alignment} prefill tokens')
    if backend=='radiance' and effective['Reserve'] <= effective['Prefill']+block:
        raise ValueError('Radiance Reserve must exceed Prefill plus one 128-token block')
    if effective['SinkTokens'] + effective['RecentTokens'] > effective['Budget']:
        raise ValueError('SinkTokens + RecentTokens must fit Budget after alignment')
    if effective['Budget'] + effective['Reserve'] > effective['Context']:
        raise ValueError('Budget + Reserve must fit Context after alignment')
    changes = [dict(parameter=k, requested=v, effective=effective[k],
                    reason='prefill alignment and Reserve limit' if k == 'Prefill' else f'alignment to {block} tokens')
               for k, v in requested.items() if effective[k] != v]
    return dict(requested=requested, effective=effective, block_tokens=block, adjustments=changes)


def scoring_args(p):
    if strategy(p) == 'layerwise':
        return recall_args(p)
    cosine = p.get('QueryScorer', 'page-softmax') == 'meanq-cosine'
    scorer = p.get('QueryScorer', 'page-softmax')
    aggregation = p.get('ScoreAggregation', 'max' if cosine else 'mean')
    alpha = p.get('GraphAlpha', 0 if cosine else 0.2)
    history = p.get('Lrfu', cosine)
    args = ['--kvmem-query-scorer', scorer, '--kvmem-score-agg', aggregation,
            '--kvmem-graph-alpha', format(alpha, '.15g')]
    if history:
        args += ['--kvmem-lrfu', '--kvmem-lrfu-lambda', format(p.get('LrfuLambda', .5), '.15g'),
                 '--kvmem-lrfu-top-frac', format(p.get('LrfuTopFraction', .9), '.15g')]
    else:
        args += ['--no-kvmem-lrfu']
    for key, flag in [('QueryMaxTokens', '--kvmem-query-max-tokens'),
                      ('QueryReplayMaxTokens', '--kvmem-query-replay-max-tokens')]:
        if key in p or p.get('Backend', 'ninfer') == 'ninfer':
            args += [flag, str(p.get(key, 0))]
    return recall_args(p) + args


def _existing_file(value, label):
    path = Path(value)
    if not value or not path.is_file():
        raise ValueError(f'Missing {label} file: {value}')
    return str(path.resolve())


def _speculative(p, supplied):
    backend = p['Backend']
    if backend == 'ninfer':
        p.setdefault('CheckpointExtraRecomputeTokens', 0)
    p.setdefault('CacheMinTokens', 0)
    mode = p.get('SpeculativeMode') or ('mtp' if p['MtpDrafts'] else 'none')
    spec = SPECULATIVE_PROFILES[backend].get(mode)
    if spec is None:
        raise ValueError('Unsupported SpeculativeMode')
    count = p.get('DraftTokens', p['MtpDrafts'])
    if (strategy(p) == 'layerwise' and mode != 'none'
            and 'DraftTokens' not in supplied and 'MtpDrafts' not in supplied):
        raise ValueError('Layerwise speculative decoding requires an explicit draft width')
    if 'DraftTokens' not in supplied and mode != 'none' and not count:
        count = 5 if mode == 'dflash2' else 4
    adaptive = p.get('DraftPolicy', 'adaptive' if p['AdaptiveMtp'] else 'fixed') == 'adaptive'
    if mode == 'none':
        if count or adaptive or p['NgramDrafts'] or p.get('DraftModel'):
            raise ValueError('Disabled speculative decoding cannot have active draft options')
    else:
        if not 1 <= count <= spec['max_draft_tokens']:
            raise ValueError('Draft count exceeds this backend/algorithm capability')
        if adaptive and not spec['adaptive']:
            raise ValueError('Adaptive drafts are unavailable for this backend/algorithm')
        if p['NgramDrafts'] and not spec['ngram']:
            raise ValueError('Ngram drafts are unavailable for this backend/algorithm')
        if p['Concurrency'] > spec['max_concurrency'] or len(p['Gpu'].split(',')) > spec['max_devices']:
            raise ValueError('Speculative mode does not support this concurrency or GPU count')
        draft = p.get('DraftModel')
        if spec['draft_asset'] == 'embedded' and draft:
            raise ValueError('This backend uses the embedded draft component')
        if spec['draft_asset'] == 'required_gguf' and not draft:
            raise ValueError('DFlash2 requires DraftModel with a compatible GGUF')
    if p.get('DraftModel'):
        p['DraftModel'] = _existing_file(p['DraftModel'], 'DraftModel')
        if Path(p['DraftModel']).suffix.lower() != '.gguf':
            raise ValueError('DraftModel must be a GGUF file')
    p.update(SpeculativeMode=mode, MtpDrafts=count, AdaptiveMtp=adaptive)


def build_plan(data, root=ROOT, *, platform=None, reveal_secrets=False):
    """Resolve the existing structured UI contract on either supported platform.

    The standalone PowerShell CLI keeps its wider native escape hatch. This API
    deliberately shares the console's bounded WorkerArgs contract.
    """
    root = Path(root).resolve()
    platform = platform or ('windows' if os.name == 'nt' else 'linux')
    if platform not in ('windows', 'linux'):
        raise ValueError('Unsupported target platform')
    data = copy.deepcopy(data)
    params = data.get('parameters') if isinstance(data, dict) else None
    if isinstance(params, dict):
        for key in ('Worker', 'Model', 'DraftModel', 'ChatTemplateFile', 'Mmproj', 'Lora', 'DiskPath',
                    'Tokenizer', 'PolystrataSource', 'Python', 'VisionWorker', 'RadianceHome', 'RocmPath'):
            value = params.get(key)
            if isinstance(value, str) and value:
                if platform == 'linux' and PureWindowsPath(value).drive:
                    raise ValueError(f'{key} requires a Linux path')
                if '\x00' in value or '\n' in value or '\r' in value:
                    raise ValueError(f'{key} contains an unsupported control character')
                params[key] = str((root / value).resolve())
    caps = capabilities(root)['backends']
    supplied = normalize(data, tuple(caps), path_root=root)['parameters']
    p = dict(LAUNCH_DEFAULTS, **supplied)
    backend = p['Backend']
    if backend == 'radiance':
        from .radiance import build_radiance_plan
        return build_radiance_plan(supplied, root, platform, reveal_secrets, caps[backend])
    if backend == 'polystrata':
        raise ValueError('Portable PolyStrata plan generation is not available yet; use the Windows launcher')
    for key, (low, high) in RANGES.items():
        if strategy(p) == 'layerwise' and key == 'Budget':
            low = 1  # The strategy-specific token check enforces native block alignment.
        if key in p and not low <= p[key] <= high:
            raise ValueError(f'{key} must be in [{low}, {high}]')
    if p['VisionResidency'] not in ('resident', 'cpu') or p['DeviceProfile'] not in ('off', 'auto'):
        raise ValueError('Unsupported VisionResidency or DeviceProfile')
    gpu_ids = p['Gpu'].split(',') if p['Gpu'] else []
    if not 1 <= len(gpu_ids) <= 8 or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('Select 1..8 distinct GPU UUIDs, in stage order')
    suffix = '.exe' if platform == 'windows' else ''
    name = ('ninfer-kvmem-server' if backend == 'ninfer' else 'llama-kvmem-server') + suffix
    p['Worker'] = _existing_file(p['Worker'] or str(root / 'workers' / backend / name), 'Worker')
    p['Model'] = _existing_file(p['Model'], 'Model')
    if Path(p['Model']).suffix.lower() != caps[backend]['model_format']:
        raise ValueError(f'{backend} requires a {caps[backend]["model_format"]} model')
    if os.name != 'nt' and platform == 'linux' and not os.access(p['Worker'], os.X_OK):
        raise ValueError('Worker is not executable')
    activation_disk = strategy(p) == 'layerwise' and p.get('LayerwiseActivationDiskMiB', 0) > 0
    if activation_disk and 'DiskMiB' not in supplied:
        p['DiskMiB'] = 0
    if p.get('DiskPath') and not p['DiskMiB'] and not activation_disk:
        raise ValueError('Omit DiskPath when DiskMiB is 0')
    if ('DiskPath' in supplied or supplied.get('DiskMiB', 0) > 0 or activation_disk) and not p.get('DiskPath', '').strip():
        raise ValueError('SSD storage requires a nonempty DiskPath')
    if p.get('ChatTemplateFile'):
        p['ChatTemplateFile'] = _existing_file(p['ChatTemplateFile'], 'ChatTemplateFile')
    physical_prefill = p['Prefill']
    tokens = token_settings(backend, p, supplied=supplied)
    p.update(tokens['effective'])
    _speculative(p, supplied)
    p['KvType'] = p.get('KvType') or ('int8' if backend == 'ninfer' else 'q8_0')
    argv = [p['Worker']]

    def pair(flag, value):
        if value is not None:
            argv.extend((flag, str(value)))

    if backend == 'ninfer':
        layers = p.get('StageLayers', [])
        if layers and (len(gpu_ids) < 2 or len(layers) != len(gpu_ids)):
            raise ValueError('StageLayers requires one positive count per GPU and at least two GPUs')
        if p['KvType'] not in NINFER_KV:
            raise ValueError('Unsupported NInfer KV format')
        if p['FastPrefill'] and p['KvType'] not in ('int8', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8'):
            raise ValueError('Fast prefill requires INT8 or supported rotated KV')
        if p['VisionResidency'] == 'cpu' and not p['Vision']:
            raise ValueError('CPU Vision requires Vision')
        if strategy(p) != 'layerwise' and p['Concurrency'] > 1:
            p['NgramDrafts'] = min(p['NgramDrafts'], 15)
        if p['NgramDrafts'] and (not p['MtpDrafts'] or (strategy(p) != 'layerwise' and
                max(p['MtpDrafts'], p['NgramDrafts']) + p['MtpDrafts'] > p['Reserve'])):
            raise ValueError('Ngram requires speculative decoding and enough Reserve')
        argv.append(p['Model'])
        for flag, key in [('--host', 'ListenHost'), ('--port', 'Port'), ('--max-context', 'Context'),
            ('--max-concurrency', 'Concurrency'), ('--prefill-chunk', 'Prefill'),
            ('--kvmem-budget', 'Budget'), ('--kvmem-gen-reserve', 'Reserve'), ('--kvmem-host-mib', 'HostMiB'),
            ('--kvmem-sink-tokens', 'SinkTokens'), ('--kvmem-recent-tokens', 'RecentTokens'),
            ('--kvmem-sessions', 'RetainedSessions'), ('--kv-dtype', 'KvType'), ('--device-profile', 'DeviceProfile')]:
            pair(flag, p[key])
        if p['MaxTokens']:
            pair('--default-max-tokens', p['MaxTokens'])
        pair('--devices' if len(gpu_ids) > 1 else '--device', ','.join(str(i) for i in range(len(gpu_ids))))
        if layers:
            pair('--stage-layers', ','.join(map(str, layers)))
        if p.get('TensorParallel'):
            argv.extend(('--tensor-parallel', '--tensor-attention', 'heads'))
            if p.get('TensorSplit'):
                pair('--tensor-split', ','.join(format(x, '.17g') for x in p['TensorSplit']))
        if p['MtpDrafts']:
            for flag, key in [('--spec', 'SpeculativeMode'), ('--draft-tokens', 'MtpDrafts'),
                              ('--ngram-draft-tokens', 'NgramDrafts'), ('--ngram-min-match', 'NgramMinMatch')]:
                pair(flag, p[key])
        if p['AdaptiveMtp']:
            argv.append('--adaptive-mtp')
        if p['FastPrefill']:
            argv.append('--fast-prefill-kernel')
        if p['Vision']:
            argv.append('--vision')
            pair('--vision-residency', p['VisionResidency'])
            pair('--vision-max-merged', p['VisionTokens'])
            if p['VisionResidency'] == 'cpu':
                pair('--cpu-vision-idle-seconds', p['CpuVisionIdleSeconds'])
        if p['ThinkingBudget']:
            pair('--default-thinking-budget', p['ThinkingBudget'])
        if p.get('ReasoningEffort'):
            pair('--default-reasoning-effort', p['ReasoningEffort'])
        if p.get('PreserveThinking'):
            argv.append('--preserve-thinking')
    else:
        forbidden = {'StageLayers', 'FastPrefill', 'NgramDrafts', 'NgramMinMatch', 'DeviceProfile', 'CpuVisionIdleSeconds'} & supplied.keys()
        if forbidden:
            raise ValueError('NInfer-only settings: ' + ', '.join(sorted(forbidden)))
        aliases = dict(fp16='f16', fp32='f32', q8='q8_0', q5='q5_0', q4='q4_0')
        k = p.get('KvTypeK') or p['KvType']
        v = p.get('KvTypeV') or p['KvType']
        k, v = aliases.get(k, k), aliases.get(v, v)
        quantized = {'q8_0', 'q5_0', 'q4_0'}
        if k not in quantized | {'f16', 'f32'} or v not in quantized | {'f16', 'f32'}:
            raise ValueError('Unsupported llama.cpp K/V format')
        if (k in quantized) != (v in quantized):
            raise ValueError('Cannot mix floating-point and quantized K/V')
        for flag, value in [('-m', p['Model']), ('--host', p['ListenHost']), ('--port', p['Port']),
            ('-c', p['Context']), ('-n', p['MaxTokens'] or -1), ('--kvmem-budget', p['Budget']),
            ('--kvmem-gen-reserve', p['Reserve']), ('-b', p['Prefill']), ('-ub', physical_prefill),
            ('--kvmem-sink-tokens', p['SinkTokens']), ('--kvmem-recent-tokens', p['RecentTokens'])]:
            pair(flag, value)
        if {'KvTypeK', 'KvTypeV'} & supplied.keys():
            pair('--cache-type-k', k)
            pair('--cache-type-v', v)
        else:
            pair('--kv-dtype', p['KvType'])
        pair('--kvmem-host-mib', p['HostMiB'])
        pair('--kvmem-sessions', p['RetainedSessions'])
        if 'Concurrency' in supplied:
            pair('--parallel', p['Concurrency'])
        if 'ThinkingBudget' in supplied:
            pair('--reasoning-budget', p['ThinkingBudget'] or -1)
        if p.get('ReasoningEffort'):
            pair('--reasoning-effort', p['ReasoningEffort'])
        if 'PreserveThinking' in supplied:
            pair('--chat-template-kwargs', json.dumps({'preserve_thinking': p['PreserveThinking']}, separators=(',', ':')))
        if p.get('Lora'):
            adapter = _existing_file(p['Lora'], 'Lora')
            if Path(adapter).suffix.lower() != '.gguf':
                raise ValueError('Lora must be a GGUF file')
            pair('--lora', adapter)
            pair('--lora-scale', p.get('LoraScale', 1))
        if p['Vision']:
            model = Path(p['Model'])
            if not p.get('Mmproj'):
                exact = [model.parent / (model.stem + '-mmproj.gguf'), model.parent / ('mmproj-' + model.stem + '.gguf')]
                candidates = [path for path in exact if path.is_file()] or list(model.parent.glob('*mmproj*.gguf'))
                if len(candidates) != 1:
                    raise ValueError('Vision requires Mmproj: no unique adjacent projector was found')
                p['Mmproj'] = str(candidates[0])
            projector = _existing_file(p['Mmproj'], 'Mmproj')
            if Path(projector).suffix.lower() != '.gguf':
                raise ValueError('Mmproj must be a GGUF file')
            pair('--mmproj', projector)
            argv.append('--no-mmproj-offload' if p['VisionResidency'] == 'cpu' else '--mmproj-offload')
            pair('--image-max-tokens', p['VisionTokens'])
    if p.get('ApiKey'):
        pair('--api-key', p['ApiKey'] if reveal_secrets else '<redacted>')
    for key, flag in [('ModelAlias', '--model-id' if backend == 'ninfer' else '--alias'),
                      ('ChatTemplateFile', '--chat-template' if backend == 'ninfer' else '--chat-template-file')]:
        if p.get(key):
            pair(flag, p[key])
    if p.get('LogLevel'):
        pair('--log-level' if backend == 'ninfer' else '--verbosity', p['LogLevel'] if backend == 'ninfer'
             else ('off', 'error', 'warning', 'info', 'trace', 'debug').index(p['LogLevel']))
    if p['ThinkingMode'] == 'off':
        argv.append('--no-thinking' if backend == 'ninfer' else '--no-think')
    if p['ThinkingMode'] == 'on':
        argv.append('--enable-thinking')
    argv.append('--kvmem')
    if p['CacheMinTokens']:
        pair('--kvmem-cache-min-tokens', p['CacheMinTokens'])
    if backend == 'llamacpp':
        pair('--kvmem-block-tokens', tokens['block_tokens'])
        if p['MtpDrafts']:
            pair('--spec-type', 'draft-mtp' if p['SpeculativeMode'] == 'mtp' else 'draft-dflash')
            pair('--spec-draft-n-max', p['MtpDrafts'])
            if p.get('DraftModel'):
                pair('--spec-draft-model', p['DraftModel'])
    if p.get('DiskPath'):
        pair('--kvmem-disk-path', p['DiskPath'])
        pair('--kvmem-disk-mib', p['DiskMiB'])
    argv += scoring_args(supplied)
    if backend == 'llamacpp' and strategy(p) == 'layerwise' and p['SpeculativeMode'] == 'mtp':
        pair('--kvmem-mtp-state', 'replay')
    argv += p.get('WorkerArgs', [])
    environment = {'CUDA_VISIBLE_DEVICES': p['Gpu'], 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
    if backend == 'ninfer':
        # Explicit zero keeps an inherited environment from enabling an unsaved policy.
        environment['KVMEM_CHECKPOINT_EXTRA_RECOMPUTE_TOKENS'] = str(p['CheckpointExtraRecomputeTokens'])
        if p['CheckpointExtraRecomputeTokens'] and os.environ.get('KVMEM_CHECKPOINT_MIN_GAP', '0').strip() not in ('', '0'):
            raise ValueError('CheckpointExtraRecomputeTokens requires KVMEM_CHECKPOINT_MIN_GAP=0')
    if platform == 'linux':
        libraries = [Path(p['Worker']).parent, root / 'lib']
        environment['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys(str(x) for x in libraries if x.is_dir()))
    safe_supplied, safe_effective = copy.deepcopy(supplied), copy.deepcopy(p)
    for safe in (safe_supplied, safe_effective):
        safe.pop('ApiKey', None)
    return dict(schema_version=1, launch_contract_version=3, target_platform=platform,
                backend=backend, argv=argv, cwd=str(root), environment=environment,
                token_settings=tokens, requested_parameters=safe_supplied, effective_parameters=safe_effective,
                api_key_configured=bool(p.get('ApiKey')), capabilities=copy.deepcopy(caps[backend]),
                resolved_scoring=scoring_settings(argv),
                warnings='\n'.join(f"KVMEM_CONFIG {x['parameter']}: {x['requested']} -> {x['effective']} ({x['reason']})"
                                   for x in tokens['adjustments']))
