"""hipfire-specific launch plans built on the public KVMem contract."""
import copy
import os
import stat
from pathlib import Path


def _existing_file(value, label):
    path = Path(value)
    if not value or not path.is_file():
        raise ValueError(f'Missing {label} file: {value}')
    return str(path.resolve())


def _safe(parameters):
    clean = copy.deepcopy(parameters)
    clean.pop('ApiKey', None)
    return clean


def build_hipfire_plan(p, supplied, root, platform, reveal_secrets, capabilities, tokens):
    root = Path(root).resolve()
    if platform not in ('windows', 'linux'):
        raise ValueError('Unsupported target platform')
    strategy = p.get('RecallStrategy', 'legacy')
    if strategy not in ('legacy', 'layerwise'):
        raise ValueError('hipfire RecallStrategy must be legacy or layerwise')
    if p.get('KvType') != 'q8':
        raise ValueError('hipfire KVMem currently requires q8 KV')
    if p.get('GraphAlpha', 0) != 0:
        raise ValueError('hipfire graph alpha is unsupported; use 0')
    if p.get('CacheMinTokens', 0) != 0:
        raise ValueError('hipfire CacheMinTokens is unsupported; use 0')
    if strategy == 'legacy':
        for key in ('QueryMaxTokens', 'QueryReplayMaxTokens'):
            if p.get(key, 0) <= 0:
                raise ValueError(f'hipfire Legacy {key} must be positive')
    gpu_ids = p['Gpu'].split(',') if p['Gpu'] else []
    if len(gpu_ids) != 1 or len(set(gpu_ids)) != 1:
        raise ValueError('hipfire currently supports one GPU')
    if p.get('TensorParallel') or p.get('StageLayers') or p.get('TensorSplit'):
        raise ValueError('hipfire multi-GPU planning is not supported')
    if p.get('ModelAlias') or p.get('ChatTemplateFile'):
        raise ValueError('hipfire does not expose model alias or external chat-template overrides')
    if strategy == 'layerwise' and p.get('LayerwiseCheckpointTarget'):
        raise ValueError('hipfire Layerwise checkpoint target is not exposed by this launcher')
    mode = p.get('SpeculativeMode') or 'none'
    idle_timeout = p.get('IdleTimeout', 0)
    if type(idle_timeout) is not int or not 0 <= idle_timeout <= 86400:
        raise ValueError('hipfire IdleTimeout must be between 0 and 86400 seconds')
    p['IdleTimeout'] = idle_timeout
    scorer = p.get('QueryScorer', 'page-softmax')
    aggregation = p.get('ScoreAggregation', 'mean')
    if mode not in ('none', 'mtp', 'dflash2'):
        raise ValueError('hipfire SpeculativeMode must be none, mtp or dflash2')
    if p.get('Vision') and mode != 'none':
        raise ValueError('hipfire vision requests require speculative decoding off')
    if mode == 'dflash2' and not p.get('DraftModel'):
        raise ValueError('hipfire DFlash2 requires an explicit HFQ DraftModel')
    if mode != 'dflash2' and p.get('DraftModel'):
        raise ValueError('hipfire DraftModel is only valid for DFlash2')
    if p.get('Vision') and not p.get('VisionSidecar'):
        raise ValueError('hipfire Vision requires a VisionSidecar HFQ file')
    if not p.get('Vision') and p.get('VisionSidecar'):
        raise ValueError('hipfire VisionSidecar requires Vision=true')
    if p.get('ApiKey'):
        raise ValueError('hipfire uses ApiKeyFile, not a raw ApiKey argument')
    if p.get('ListenHost') != '127.0.0.1' and not p.get('ApiKeyFile'):
        raise ValueError('hipfire non-loopback listening requires ApiKeyFile')

    worker = _existing_file(p['Worker'], 'Worker')
    model = _existing_file(p['Model'], 'Model')
    if p.get('VisionSidecar'):
        p['VisionSidecar'] = _existing_file(p['VisionSidecar'], 'VisionSidecar')
    if p.get('DraftModel'):
        p['DraftModel'] = _existing_file(p['DraftModel'], 'DraftModel')
    api_key_file = None
    if p.get('ApiKeyFile'):
        api_key_file = Path(p['ApiKeyFile'])
        if not api_key_file.is_absolute() or not api_key_file.is_file():
            raise ValueError('ApiKeyFile must be an absolute path to an existing file')
        if os.name != 'nt':
            key_mode = stat.S_IMODE(api_key_file.stat().st_mode)
            if key_mode & 0o077:
                raise ValueError('ApiKeyFile must not be accessible by group or other users')
        api_key_file = str(api_key_file.resolve())
        p['ApiKeyFile'] = api_key_file
    if os.name != 'nt' and platform == 'linux' and not os.access(worker, os.X_OK):
        raise ValueError('Worker is not executable')
    slots = p['Concurrency']
    if strategy == 'legacy':
        pages_per_slot = (p['Budget'] + p['Reserve'] + 127) // 128 + 5
    else:
        # budget + one prefill chunk + four selection intervals + partial tail
        # + four safety pages. This matches the validated 45-page 4K recipe.
        pages_per_slot = p['Budget'] // 128 + (p['Prefill'] + 127) // 128 + 9
    pages = pages_per_slot * slots
    argv = [worker, 'serve', model, p['ListenHost'], str(p['Port']),
            '--max-seq', str(p['Context']), '--kv-mode', 'q8',
            '--idle-timeout', str(idle_timeout),
            '--spec', 'off' if mode == 'none' else 'dflash' if mode == 'dflash2' else mode]
    if p.get('Vision'):
        argv += ['--vision', p['VisionSidecar']]
    if api_key_file:
        argv += ['--api-key-file', api_key_file]
    if strategy == 'legacy':
        argv += ['--kvmem', '--kvmem-block-tokens', '128',
                 '--kvmem-budget', str(p['Budget']),
                 '--kvmem-gen-reserve', str(p['Reserve']),
                 '--kvmem-host-mib', str(p['HostMiB']),
                 '--kvmem-sink-tokens', str(p['SinkTokens']),
                 '--kvmem-recent-tokens', str(p['RecentTokens']),
                 '--kvmem-sessions', str(p['RetainedSessions']),
                 '--kvmem-recall-strategy', 'legacy',
                 '--kvmem-query-scorer', scorer,
                 '--kvmem-score-agg', aggregation,
                 '--kvmem-graph-alpha', '0',
                 '--kvmem-query-max-tokens', str(p['QueryMaxTokens']),
                 '--kvmem-query-replay-max-tokens', str(p['QueryReplayMaxTokens']),
                 '--kvmem-cache-min-tokens', '0']
        argv.append('--kvmem-lrfu' if p.get('Lrfu') else '--no-kvmem-lrfu')
        if p.get('Lrfu'):
            argv += ['--kvmem-lrfu-lambda', str(p.get('LrfuLambda', .5)),
                     '--kvmem-lrfu-top-frac', str(p.get('LrfuTopFraction', .9))]
        if p.get('DiskPath'):
            argv += ['--kvmem-disk-path', p['DiskPath'], '--kvmem-disk-mib', str(p['DiskMiB'])]
    argv += p.get('WorkerArgs', [])

    environment = {
        'HIP_VISIBLE_DEVICES': p['Gpu'],
        'ROCR_VISIBLE_DEVICES': p['Gpu'],
        'HIPFIRE_SERVE_MULTI_SLOT': '1',
        'HIPFIRE_SERVE_MULTI_SLOT_SLOTS': str(slots),
        'HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK': str(p['Prefill']),
        'HIPFIRE_SERVE_MAX_BATCH_TOKENS': str(p['Prefill'] * slots),
        'HIPFIRE_SLOTS_PAGED_PAGES': str(pages),
        'HIPFIRE_GRAPH': '0',
        'HIPFIRE_VISION_MODE': 'off' if not p.get('Vision') else 'resident' if p.get('VisionResidency') == 'resident' else 'auto',
        'HIPFIRE_KVMEM_PREFIX_PIN': '0',
        'HIPFIRE_KVMEM_QUERY_SPAN': '1',
        'HIPFIRE_KVMEM_SELECT_METHOD': 'retrieval',
        'HIPFIRE_KVMEM_LRFU': '0',
        'HIPFIRE_KVMEM_HEAD_LIVE': '1' if strategy == 'legacy' and mode == 'mtp' else '0',
        'HIPFIRE_KVMEM_DN_IDLE': '1',
        'HIPFIRE_DFLASH_DRAFT': p.get('DraftModel') or '',
    }
    if strategy == 'layerwise':
        environment.update({
            'HIPFIRE_KVMEM': '1',
            'HIPFIRE_KVMEM_RECALL_STRATEGY': 'layerwise',
            'HIPFIRE_KVMEM_BUDGET_TOKENS': str(p['Budget']),
            'HIPFIRE_KVMEM_GEN_RESERVE': str(p['Reserve']),
            'HIPFIRE_KVMEM_HOST_BYTES': str(p['HostMiB'] << 20),
            'HIPFIRE_KVMEM_SINK_BLOCKS': str(p['SinkTokens'] // 128),
            'HIPFIRE_KVMEM_RECENT_BLOCKS': str(p['RecentTokens'] // 128),
            'HIPFIRE_KVMEM_SELECT_INTERVAL': '4',
        })
        if p.get('LayerwiseMtpHistory'):
            environment['HIPFIRE_KVMEM_LAYERWISE_MTP_HISTORY'] = p['LayerwiseMtpHistory']
        for key, native in (
            ('LayerwiseActivationRamMiB', 'HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_RAM_MIB'),
            ('LayerwiseActivationDiskMiB', 'HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_DISK_MIB'),
            ('LayerwiseStagingMiB', 'HIPFIRE_KVMEM_LAYERWISE_STAGING_MIB'),
            ('LayerwiseMetadataMiB', 'HIPFIRE_KVMEM_LAYERWISE_METADATA_MIB'),
            ('LayerwiseActivationMetadataMiB', 'HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_METADATA_MIB'),
        ):
            if key in p:
                environment[native] = str(p[key])
        if p.get('DiskPath'):
            environment['HIPFIRE_KVMEM_SPILL_DIR'] = p['DiskPath']
            environment['HIPFIRE_KVMEM_SPILL_BYTES'] = str(p['DiskMiB'] << 20)
    rocm_value = (p.get('RocmPath') or os.environ.get('HIPFIRE_ROCM_PATH')
                  or os.environ.get('ROCM_PATH') or os.environ.get('HIP_PATH'))
    if rocm_value:
        rocm = str(Path(rocm_value).resolve())
        environment['HIPFIRE_ROCM_PATH'] = rocm
        environment['ROCM_PATH'] = rocm
        environment['HIP_PATH'] = rocm
    if p.get('LogLevel'):
        environment['RUST_LOG'] = {'warning': 'warn'}.get(p['LogLevel'], p['LogLevel'])
    if platform == 'linux':
        libraries = [Path(worker).parent, root / 'lib']
        if rocm_value:
            libraries += [Path(rocm_value) / 'lib', Path(rocm_value) / 'lib64']
        libraries += [Path(item) for item in os.environ.get('LD_LIBRARY_PATH', '').split(':') if item]
        environment['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys(str(x) for x in libraries if x.is_dir()))

    resolved = dict(strategy=strategy, scorer=scorer, agg=aggregation,
                    lrfu=bool(p.get('Lrfu')),
                    **{'lambda': float(p.get('LrfuLambda', .5))},
                    top_frac=float(p.get('LrfuTopFraction', .9)), graph_alpha=0.0,
                    query_max_tokens=p.get('QueryMaxTokens', 0),
                    query_replay_max_tokens=p.get('QueryReplayMaxTokens', 0), block_tokens=128)
    if strategy == 'layerwise':
        resolved.update(current_layer_q=True, query_replay=False,
                        mtp_history=p.get('LayerwiseMtpHistory') or 'first-layer')
    effective = dict(p, MtpDrafts=0, AdaptiveMtp=False)
    return dict(schema_version=1, launch_contract_version=3, target_platform=platform,
                backend='hipfire', argv=argv, cwd=str(root), environment=environment,
                token_settings=tokens, requested_parameters=_safe(supplied),
                effective_parameters=_safe(effective), api_key_configured=bool(api_key_file),
                credential_mode='api-key-file' if api_key_file else None,
                capabilities=copy.deepcopy(capabilities), resolved_scoring=resolved,
                warnings='\n'.join(f"KVMEM_CONFIG {x['parameter']}: {x['requested']} -> {x['effective']} ({x['reason']})"
                                   for x in tokens['adjustments']))
