"""Shared configuration contract extracted from the existing console."""
import copy
import json
import math
import re
from pathlib import Path
from .recall import LAYERWISE_RESOURCES, validate_recall

ROOT = Path(__file__).resolve().parents[2]
SPECULATIVE_PROFILES = json.loads((ROOT / 'scripts/speculative-capabilities.json').read_text(encoding='utf-8-sig'))
DEFAULTS = dict(Backend='ninfer', Model='', Worker='', Gpu='', Context=262144,
                Budget=32768, Reserve=1024, HostMiB=16384, Prefill=128,
                MaxTokens=1024, Port=18200, ListenHost='127.0.0.1',
                SinkTokens=12288, RecentTokens=1024, CacheMinTokens=0)
# Initial UI recipes; never merge engine-specific flags into saved configurations.
ENGINE_DEFAULTS = {'ninfer': {'Context': 262144,
            'Budget': 47104,
            'Reserve': 2048,
            'MaxTokens': 0,
            'ThinkingBudget': 0,
            'ReasoningEffort': 'xhigh',
            'PreserveThinking': True,
            'HostMiB': 12288,
            'SinkTokens': 12288,
            'RecentTokens': 1024,
            'Concurrency': 1,
            'QueryScorer': 'page-softmax',
            'ScoreAggregation': 'mean',
            'GraphAlpha': 0.2,
            'Lrfu': False,
            'QueryMaxTokens': 0,
            'QueryReplayMaxTokens': 0,
            'CheckpointExtraRecomputeTokens': 0,
            'CacheMinTokens': 0,
            'ThinkingMode': 'default',
            'KvType': 'int8',
            'Prefill': 256,
            'DraftTokens': 4,
            'DraftPolicy': 'adaptive',
            'NgramDrafts': 31,
            'NgramMinMatch': 12,
            'FastPrefill': True,
            'DeviceProfile': 'auto',
            'Vision': True,
            'VisionResidency': 'cpu',
            'VisionTokens': 1024,
            'Backend': 'ninfer',
            'SpeculativeMode': 'mtp',
            'RetainedSessions': 1},
 'llamacpp': {'Context': 262144,
              'Budget': 47104,
              'Reserve': 2048,
              'MaxTokens': 0,
              'ThinkingBudget': 16384,
              'ReasoningEffort': 'xhigh',
              'PreserveThinking': True,
              'HostMiB': 12288,
              'SinkTokens': 12288,
              'RecentTokens': 1024,
              'Concurrency': 1,
              'QueryScorer': 'page-softmax',
              'ScoreAggregation': 'mean',
              'GraphAlpha': 0.2,
              'Lrfu': False,
              'QueryMaxTokens': 512,
              'QueryReplayMaxTokens': 2048,
              'ThinkingMode': 'on',
              'KvTypeK': 'q8_0',
              'KvTypeV': 'q8_0',
              'Prefill': 256,
              'DraftTokens': 3,
              'DraftPolicy': 'fixed',
              'Vision': True,
              'VisionResidency': 'cpu',
              'VisionTokens': 1024,
              'Backend': 'llamacpp',
              'SpeculativeMode': 'mtp',
              'RetainedSessions': 1},
 'polystrata': {'Context': 262144,
                'Budget': 36864,
                'Reserve': 4096,
                'MaxTokens': 0,
                'ThinkingBudget': 0,
                'ReasoningEffort': 'xhigh',
                'PreserveThinking': True,
                'HostMiB': 12288,
                'SinkTokens': 12288,
                'RecentTokens': 1024,
                'Concurrency': 1,
                'QueryScorer': 'page-softmax',
                'ScoreAggregation': 'mean',
                'GraphAlpha': 0.2,
                'Lrfu': False,
                'QueryMaxTokens': 512,
                'QueryReplayMaxTokens': 2048,
                'ThinkingMode': 'on',
                'KvType': 'q8_0',
                'Prefill': 512,
                'DraftTokens': 2,
                'DraftPolicy': 'adaptive',
                'Vision': False,
                'VramReserveMiB': 256,
                'Backend': 'polystrata',
                'SpeculativeMode': 'mtp',
                'RetainedSessions': 2,
                'WorkerArgs': ['--elastic-experts-headroom',
                               '--elastic-headroom-mib',
                               '128',
                               '--elastic-physical-reserve-mib',
                               '512']}}
# Backward-compatible fallback for partial requests; the UI sends its full recipe.
RADIANCE_LAUNCH_DEFAULTS = dict(Backend='radiance', Context=65536, Budget=4096, Reserve=1024,
    HostMiB=8192, SinkTokens=128, RecentTokens=512, Concurrency=1, RetainedSessions=8,
    QueryScorer='page-softmax', ScoreAggregation='mean', GraphAlpha=.2, Lrfu=False,
    QueryMaxTokens=512, QueryReplayMaxTokens=2048, KvType='fp8', StateCacheType='f32', Prefill=256,
    SpeculativeMode='none', DraftTokens=0, DraftPolicy='fixed', Vision=False, MaxTokens=0,
    ThinkingMode='default', ThinkingBudget=0, VramReserveMiB=1024,
    VisionTokens=1024, VisionResidency='resident', EmbeddingPlacement='vram', CpuThreads=2, VisionCacheMiB=256)
ENGINE_DEFAULTS['radiance'] = {'Backend': 'radiance', 'Context': 262144, 'Budget': 36864, 'Reserve': 4096, 'HostMiB': 12288, 'SinkTokens': 12288, 'RecentTokens': 1024, 'Concurrency': 1, 'RetainedSessions': 1, 'QueryScorer': 'page-softmax', 'ScoreAggregation': 'mean', 'GraphAlpha': 0.2, 'Lrfu': False, 'QueryMaxTokens': 512, 'QueryReplayMaxTokens': 2048, 'KvType': 'fp8', 'StateCacheType': 'f32', 'Prefill': 256, 'SpeculativeMode': 'dflash2', 'DraftTokens': 7, 'DraftPolicy': 'fixed', 'Vision': True, 'MaxTokens': 0, 'ThinkingMode': 'default', 'ThinkingBudget': 0, 'VramReserveMiB': 256, 'EmbeddingPlacement': 'ram', 'CpuThreads': 8, 'VisionResidency': 'cpu', 'VisionTokens': 256, 'VisionCacheMiB': 64, 'ReasoningEffort': 'xhigh', 'PreserveThinking': True, 'NgramDrafts': 0, 'NgramMinMatch': 12}

STRINGS = {'StateCacheType', 'EmbeddingPlacement', 'RadianceHome', 'RocmPath', 'Tokenizer','PolystrataSource','Python','VisionWorker','Mmproj','Lora','QueryScorer','Backend', 'Model', 'Worker', 'Gpu', 'ListenHost', 'KvType', 'KvTypeK', 'KvTypeV', 'DiskPath',
           'VisionResidency',
           'DeviceProfile', 'ScoreAggregation', 'QueryScorer', 'ApiKey', 'ModelAlias', 'LogLevel',
           'ChatTemplateFile', 'ThinkingMode', 'ReasoningEffort', 'SpeculativeMode', 'DraftPolicy', 'DraftModel'}
INTEGERS = {'CpuThreads', 'VisionCacheMiB', 'BlockTokens', 'VramReserveMiB', 'Context', 'Budget', 'Reserve', 'HostMiB', 'Prefill', 'MaxTokens', 'Port',
            'MtpDrafts', 'NgramDrafts', 'NgramMinMatch', 'Concurrency', 'RetainedSessions',
            'VisionTokens', 'ThinkingBudget', 'SinkTokens', 'RecentTokens', 'DiskMiB', 'DraftTokens',
            'QueryMaxTokens', 'QueryReplayMaxTokens', 'ReplayMaxTokens', 'CheckpointExtraRecomputeTokens', 'CacheMinTokens'}
BOOLEANS = {'AdaptiveMtp', 'FastPrefill', 'Vision', 'NoLrfu', 'Lrfu', 'PreserveThinking', 'TensorParallel'}
DECIMALS = {'LrfuLambda', 'LrfuTopFraction', 'GraphAlpha'}
STRINGS.update(('RecallStrategy', 'LayerwiseMtpHistory'))
INTEGERS.add("LayerwiseCheckpointTarget")
INTEGERS.update(LAYERWISE_RESOURCES)
# Deliberately limited native escape hatch. Common options must not be duplicated.
MCP_OPTIONS = {'llamacpp': '--mcp-servers-config', 'polystrata': '--mcp-config'}
NATIVE = {'llamacpp': {'--threads', '--threads-batch', '--n-gpu-layers', MCP_OPTIONS['llamacpp']},
          'radiance': {'--http-threads'}, 'ninfer': {'--threads'},
          'polystrata': {'--threads', '--elastic-headroom-mib', '--elastic-physical-reserve-mib', MCP_OPTIONS['polystrata']}}
NATIVE_SWITCHES = {'polystrata': {'--elastic-experts', '--elastic-experts-headroom'}}


def without_secrets(data):
    clean = copy.deepcopy(data)
    clean.get('parameters', {}).pop('ApiKey', None)
    return clean


def normalize(data, backends, *, path_root=None):
    if (not isinstance(data, dict) or type(data.get('schema_version')) is not int
            or data.get('schema_version') != 1 or set(data) != {'schema_version', 'parameters'}):
        raise ValueError('Unsupported configuration schema')
    params = data.get('parameters')
    if not isinstance(params, dict):
        raise ValueError('parameters must be an object')
    allowed = STRINGS | INTEGERS | BOOLEANS | DECIMALS | {'LoraScale', 'StageLayers', 'TensorSplit', 'WorkerArgs'}
    if set(params) - allowed:
        raise ValueError('Unknown parameters: ' + ', '.join(sorted(set(params) - allowed)))
    p = copy.deepcopy(params)
    if p.get('Backend') not in NATIVE or p.get('Backend') not in backends:
        raise ValueError('This backend is not installed in the selected backend checkout')
    if {'Lora', 'LoraScale'} & p.keys() and p['Backend'] != 'llamacpp':
        raise ValueError('Lora configures llama.cpp only')
    if 'LoraScale' in p:
        scale = p['LoraScale']
        if not p.get('Lora') or type(scale) not in (int, float) or not -3.4028234663852886e38 <= scale <= 3.4028234663852886e38:
            raise ValueError('LoraScale requires Lora and a finite float scale')
    if p['Backend'] == 'polystrata':
        p.setdefault('BlockTokens', 32)
        if type(p['BlockTokens']) is not int or p['BlockTokens'] not in (32, 64):
            raise ValueError('PolyStrata BlockTokens must be 32 or 64')
    elif p['Backend']=='radiance' and p.get('BlockTokens',128)!=128:
        raise ValueError('Radiance BlockTokens must be 128')
    elif p['Backend']!='radiance' and 'BlockTokens' in p:
        raise ValueError('BlockTokens configures PolyStrata in this console')
    for key, value in p.items():
        if key in STRINGS and (not isinstance(value, str) or '\x00' in value or len(value) > 8192):
            raise ValueError(f'{key} must be a string without NUL characters')
        if key in INTEGERS and (type(value) is not int or not 0 <= value <= 2147483647):
            raise ValueError(f'{key} must be a nonnegative integer')
        if key in BOOLEANS and type(value) is not bool:
            raise ValueError(f'{key} must be boolean')
        if key in DECIMALS and (type(value) not in (int, float) or not 0 <= value <= 1):
            raise ValueError(f'{key} must be between 0 and 1')
    # Old signed-cosine templates predate QueryScorer. Resolve their intent once
    # and submit canonical independent controls, including explicit false.
    legacy_scoring = 'QueryScorer' not in p and bool({'NoLrfu', 'LrfuLambda', 'LrfuTopFraction'} & p.keys())
    if legacy_scoring:
        p['QueryScorer'] = 'meanq-cosine' if (
            not p.get('NoLrfu', False) or p.get('ScoreAggregation') == 'max' or
            'LrfuLambda' in p or 'LrfuTopFraction' in p) else 'page-softmax'
    if 'NoLrfu' in p:
        if 'Lrfu' in p and p['Lrfu'] == p['NoLrfu']:
            raise ValueError('Lrfu conflicts with NoLrfu')
        p['Lrfu'] = not p.pop('NoLrfu')
        if legacy_scoring and not p['Lrfu']:
            p.pop('LrfuLambda', None); p.pop('LrfuTopFraction', None)
    if p.get('QueryScorer', 'page-softmax') not in {'page-softmax', 'meanq-cosine'}:
        raise ValueError('Unsupported QueryScorer')
    if p.get('ScoreAggregation', 'mean') not in {'mean', 'max'}:
        raise ValueError('Unsupported ScoreAggregation')
    for key in ('QueryMaxTokens', 'QueryReplayMaxTokens'):
        minimum = 0 if p['Backend'] == 'ninfer' else 1
        if key in p and p[key] < minimum:
            raise ValueError(f'{key} must be >= {minimum}; zero means unlimited for NInfer')
    if p['Backend'] != 'radiance' and {'CpuThreads','VisionCacheMiB'} & p.keys():
        raise ValueError('CpuThreads and VisionCacheMiB currently configure Radiance only')
    if 'CheckpointExtraRecomputeTokens' in p and p['Backend'] != 'ninfer':
        raise ValueError('CheckpointExtraRecomputeTokens configures NInfer only')
    history = p.get('Lrfu', p.get('QueryScorer') == 'meanq-cosine')
    if not history and {'LrfuLambda', 'LrfuTopFraction'} & p.keys():
        raise ValueError('LRFU tuning requires enabled LRFU')
    if 'LrfuTopFraction' in p and p['LrfuTopFraction'] <= 0:
        raise ValueError('LrfuTopFraction must be in (0,1]')
    if p.get('ListenHost', '127.0.0.1') not in {'127.0.0.1', '0.0.0.0'}:
        raise ValueError('ListenHost must be 127.0.0.1 or 0.0.0.0')
    if not 1 <= p.get('Port', 18200) <= 65535:
        raise ValueError('Port must be between 1 and 65535')
    if p.get('ApiKey') and (len(p['ApiKey']) > 512 or not re.fullmatch(r'[A-Za-z0-9._~+/-]+=*', p['ApiKey'])):
        raise ValueError('ApiKey must be a single ASCII token (letters, digits, . _ ~ + / - and optional trailing =)')
    if p.get('LogLevel', '') not in {'', 'off', 'error', 'warning', 'info', 'trace', 'debug'}:
        raise ValueError('Unsupported LogLevel')
    if p.get('ThinkingMode', 'default') not in {'default', 'on', 'off'}:
        raise ValueError('ThinkingMode must be default, on or off')
    if p['Backend'] == 'ninfer' and p.get('ThinkingMode') == 'on':
        raise ValueError('NInfer supports default/off at startup; thinking-on is selected by requests')
    if p.get('ReasoningEffort') not in (None, 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
        raise ValueError('Unsupported ReasoningEffort')
    if p.get('ThinkingMode') == 'off' and p.get('ReasoningEffort'):
        raise ValueError('ReasoningEffort conflicts with ThinkingMode off')
    if p['Backend'] != 'llamacpp' and {'KvTypeK', 'KvTypeV'} & p.keys():
        raise ValueError('K/V 独立格式仅适用于 llama.cpp；NInfer 请使用 KV 格式选项。')
    if p.get('Concurrency') is not None and not 1 <= p['Concurrency'] <= 8:
        raise ValueError('Concurrency must be 1..8')
    if p.get('ChatTemplateFile') and not Path(p['ChatTemplateFile']).is_file():
        raise ValueError('ChatTemplateFile must be an existing template file')
    if 'SpeculativeMode' in p or 'DraftTokens' in p or 'DraftPolicy' in p or 'DraftModel' in p:
        mode = p.get('SpeculativeMode', 'mtp' if p.get('MtpDrafts', 0) else 'none')
        profile = SPECULATIVE_PROFILES[p['Backend']].get(mode)
        if profile is None:
            raise ValueError('Unsupported SpeculativeMode')
        count = p.get('DraftTokens', p.get('MtpDrafts', 2 if mode == 'guess' else 5 if mode == 'dflash2' else (3 if p['Backend']=='radiance' else 4) if mode == 'mtp' else 0))
        if (p.get('RecallStrategy') == 'layerwise' and mode != 'none'
                and 'DraftTokens' not in p and 'MtpDrafts' not in p):
            raise ValueError('Layerwise speculative decoding requires an explicit draft width')
        policy = p.get('DraftPolicy', 'adaptive' if p.get('AdaptiveMtp') else 'fixed')
        if policy not in {'fixed', 'adaptive'}:
            raise ValueError('DraftPolicy must be fixed or adaptive')
        if 'DraftTokens' in p and 'MtpDrafts' in p and count != p['MtpDrafts']:
            raise ValueError('DraftTokens conflicts with MtpDrafts')
        if 'DraftPolicy' in p and 'AdaptiveMtp' in p and (policy == 'adaptive') != p['AdaptiveMtp']:
            raise ValueError('DraftPolicy conflicts with AdaptiveMtp')
        if p['Backend'] == 'polystrata' and mode in ('mtp', 'guess'):
            policy = 'adaptive'
            p['DraftPolicy'] = policy
            p.pop('AdaptiveMtp', None)
        if mode == 'none':
            if count or policy == 'adaptive' or p.get('NgramDrafts') or p.get('DraftModel'):
                raise ValueError('Disabled speculative decoding cannot have active draft options')
        else:
            if not 1 <= count <= profile['max_draft_tokens']:
                raise ValueError('DraftTokens exceeds this backend/algorithm capability')
            if policy == 'adaptive' and not profile['adaptive']:
                raise ValueError('Adaptive drafts are unavailable for this backend/algorithm')
            if p.get('NgramDrafts') and not profile['ngram']:
                raise ValueError('Ngram drafts are unavailable for this backend/algorithm')
            if p.get('Concurrency', 1) > profile['max_concurrency']:
                raise ValueError('This speculative algorithm does not support the selected concurrency')
            if len(p.get('Gpu', '').split(',')) > profile['max_devices']:
                raise ValueError('This speculative algorithm does not support the selected GPU count')
            draft = p.get('DraftModel', '')
            if profile['draft_asset'] == 'embedded' and draft:
                raise ValueError('NInfer uses an embedded draft component')
            if profile['draft_asset'] == 'required_gguf' and not draft:
                raise ValueError(f'{mode} requires DraftModel with a compatible GGUF')
            if draft and (Path(draft).suffix.lower() != '.gguf' or not Path(draft).is_file()):
                raise ValueError('DraftModel must be an existing GGUF file')
    if p['Backend']=='polystrata':
        if p.get('KvType','f16') not in {'f16','q8_0','int8','int8k'}: raise ValueError('PolyStrata KV: Qwen supports f16/q8_0; Gemma supports f16/int8/int8k')
        if ',' in p.get('Gpu',''): raise ValueError('PolyStrata currently supports one GPU')
        if p.get('ReasoningEffort') not in (None,'none','low','medium','high','xhigh'): raise ValueError('PolyStrata reasoning effort must be none, low, medium or high')
        if p.get('ThinkingMode')=='on' and p.get('ReasoningEffort')=='none': raise ValueError('ThinkingMode conflicts with ReasoningEffort none')
    gpu = p.get('Gpu', '')
    if gpu and not all(re.fullmatch(r'GPU-[0-9a-fA-F-]{8,64}', x) for x in gpu.split(',')):
        raise ValueError('Select GPUs by UUID, separated by commas')
    layers = p.get('StageLayers', [])
    if not isinstance(layers, list) or len(layers) > 8 or any(type(x) is not int or x < 1 for x in layers):
        raise ValueError('StageLayers must contain positive integers')
    split = p.get('TensorSplit', [])
    if not isinstance(split, list) or len(split) > 8 or any(
            type(x) not in (int, float) or not 0 < x <= float('1.7976931348623157e308') for x in split):
        raise ValueError('TensorSplit must contain finite positive weights')
    if not math.isfinite(sum(float(x) for x in split)):
        raise ValueError('TensorSplit total weight must be finite')
    if p['Backend'] != 'ninfer' and ({'TensorParallel', 'TensorSplit'} & p.keys()):
        raise ValueError('TensorParallel and TensorSplit configure NInfer only')
    if p.get('TensorParallel'):
        ids = gpu.split(',') if gpu else []
        if not 2 <= len(ids) <= 8 or len(set(ids)) != len(ids):
            raise ValueError('TensorParallel requires 2..8 distinct GPU UUIDs')
        if layers:
            raise ValueError('TensorParallel cannot be combined with StageLayers')
        if split and len(split) != len(ids):
            raise ValueError('TensorSplit requires one weight per GPU, in UUID order')
    elif split:
        raise ValueError('TensorSplit requires TensorParallel')
    args = p.get('WorkerArgs', [])
    if not isinstance(args, list) or len(args) > 20 or any(not isinstance(x, str) for x in args):
        raise ValueError('Native arguments must be an array of strings')
    seen = set()
    index = 0
    while index < len(args):
        flag = args[index]
        if flag in seen:
            raise ValueError(f'Unsupported or duplicate native option: {flag}')
        seen.add(flag)
        index += 1
        if flag in NATIVE_SWITCHES.get(p['Backend'], set()):
            continue
        if flag == MCP_OPTIONS.get(p['Backend']):
            if (index == len(args) or not args[index].strip() or args[index].startswith('--')
                    or any(c in args[index] for c in '\x00\r\n')):
                raise ValueError(f'{flag} requires an existing MCP configuration file')
            path = Path(path_root or '.') / args[index]
            if not path.is_file():
                raise ValueError(f'{flag} requires an existing MCP configuration file')
            args[index] = str(path.resolve())
            index += 1
            continue
        if (flag not in NATIVE[p['Backend']] or index == len(args)
                or not re.fullmatch(r'\d{1,6}', args[index])):
            raise ValueError(f'Unsupported or incomplete native option: {flag}')
        if flag == '--elastic-headroom-mib' and not 0 <= int(args[index]) <= 1024:
            raise ValueError('Elastic headroom must be 0..1024 MiB')
        if flag == '--elastic-physical-reserve-mib' and not 256 <= int(args[index]) <= 4096:
            raise ValueError('Elastic physical reserve must be 256..4096 MiB')
        index += 1
    validate_recall(p)
    return {'schema_version': 1, 'parameters': p}
