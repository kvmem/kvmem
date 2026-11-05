"""Radiance launch contract; backend-native formats remain explicit."""
import copy
import os
from pathlib import Path
import json


def draft_selection(model, mode, depth):
    import importlib.util
    tools=Path(__file__).resolve().parents[2]/'backends/radiance/tools'
    # Use the same container checks as the standalone Radiance launch tools.
    spec=importlib.util.spec_from_file_location('radiance_draft_control',tools/'draft_control.py')
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    return reader.resolve(Path(model), 'dflash' if mode=='dflash2' else mode, depth)


def vision_geometry(model):
    import importlib.util
    source=Path(__file__).resolve().parents[2]/'backends/radiance/tools/inspect_rad.py'
    spec=importlib.util.spec_from_file_location('radiance_vision_inspection',source)
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    info=reader.inspect(Path(model))
    if info['arch_id']!='qwen35' or not info['entries_by_name_prefix'].get('v'):
        raise ValueError('Radiance image input currently requires a qwen35 .rad with its native vision tower')
    meta=info.get('vision_metadata',{})
    try:
        patch=int(meta['vision_config.patch_size']);merge=int(meta['vision_config.spatial_merge_size'])
    except (KeyError,TypeError,ValueError) as error:
        raise ValueError('RAD model is missing native vision patch/merge geometry') from error
    if not 1<=patch<=128 or not 1<=merge<=16:raise ValueError('Invalid native vision geometry')
    return patch,merge


def build_radiance_plan(supplied, root, platform, reveal_secrets, capability):
    from .config import DEFAULTS, ENGINE_DEFAULTS
    from .plan import token_settings, scoring_args, scoring_settings, _existing_file, _speculative
    p = dict(DEFAULTS, **ENGINE_DEFAULTS['radiance'])
    p.update(MtpDrafts=0, AdaptiveMtp=False, NgramDrafts=0, NgramMinMatch=12,
             DraftTokens=0, DraftPolicy='fixed')
    # The common legacy aliases retain their meaning when explicitly supplied.
    if 'DraftTokens' not in supplied and 'MtpDrafts' in supplied:
        p['DraftTokens'] = supplied['MtpDrafts']
    if 'SpeculativeMode' not in supplied and supplied.get('MtpDrafts'):
        p['SpeculativeMode'] = 'mtp'
    p.update(supplied)
    cosine=p['QueryScorer']=='meanq-cosine'
    for key,default in [('ScoreAggregation','max' if cosine else 'mean'),('GraphAlpha',0 if cosine else .2),('Lrfu',cosine)]:
        if key not in supplied:p[key]=default
    if p['SpeculativeMode']=='mtp' and not {'DraftTokens','MtpDrafts'} & supplied.keys():
        p['DraftTokens']=3
    gpu_ids=p['Gpu'].split(',')
    if not 1<=len(gpu_ids)<=2 or not all(gpu_ids) or len(set(gpu_ids))!=len(gpu_ids):
        raise ValueError('Radiance requires one or two distinct AMD GPUs by ROCr UUID')
    if p.get('StageLayers') or p.get('FastPrefill') or p.get('DeviceProfile','off')!='off':
        raise ValueError('Radiance does not implement NInfer layer splitting or CUDA fast-prefill/device profiles')
    if p['EmbeddingPlacement'] not in ('ram','vram'):
        raise ValueError('Radiance EmbeddingPlacement must be ram or vram')
    if p.get('Vision'):
        if p.get('VisionResidency') not in ('resident','cpu'):
            raise ValueError('Radiance VisionResidency must be resident or cpu')
        if not 64<=p['VisionTokens']<=16384:raise ValueError('VisionTokens must be in [64,16384]')
    p.setdefault('DiskMiB',4096)
    if not 0<=p['DiskMiB']<=1048576:
        raise ValueError('DiskMiB must be in [0,1048576]')
    if p.get('DiskPath') and not p['DiskMiB']:
        raise ValueError('Omit DiskPath when DiskMiB is 0')
    if ('DiskPath' in supplied or supplied.get('DiskMiB',0)>0) and not p.get('DiskPath','').strip():
        raise ValueError('SSD storage requires a nonempty DiskPath')
    if p.get('NgramDrafts'):
        raise ValueError('Radiance KVMem n-gram drafts are not implemented yet')
    for key,low,high in [('CpuThreads',1,256),('VisionCacheMiB',0,65536),('Context',256,2147483647),('Budget',256,2147483647),('Reserve',256,2147483647),
        ('HostMiB',1,1048576),('Prefill',1,2147483647),('Concurrency',1,8),('RetainedSessions',0,1024),
        ('VramReserveMiB',256,65536)]:
        if not low<=p[key]<=high:raise ValueError(f'{key} must be in [{low},{high}]')
    if p.get('KvType') not in ('fp8','bf16','fp8_v4','k6_v4'):
        raise ValueError('Radiance KV format must be fp8, bf16, fp8_v4 or k6_v4')
    if p.get('StateCacheType','f32') not in ('f32','f16'):
        raise ValueError('Radiance recurrent state format must be f32 or f16')
    tokens=token_settings('radiance',p);p.update(tokens['effective'])
    suffix='.exe' if platform=='windows' else ''
    p['Worker']=_existing_file(p['Worker'] or root/'workers/radiance'/('radiance-kvmem-server'+suffix),'Worker')
    p['Model']=_existing_file(p['Model'],'Model')
    if Path(p['Model']).suffix.lower()!='.rad':raise ValueError('Radiance requires a .rad model')
    mode=p['SpeculativeMode']
    if mode in ('mtp','dflash2'):
        explicit_depth=p['DraftTokens'] if {'DraftTokens','MtpDrafts'} & supplied.keys() else None
        selection=draft_selection(p['Model'],mode,explicit_depth)
        p['DraftTokens']=selection['depth']
    _speculative(p,supplied)
    if platform=='linux' and os.name!='nt' and not os.access(p['Worker'],os.X_OK):
        raise ValueError('Worker is not executable')
    home=Path(p.get('RadianceHome') or Path(p['Worker']).parent/'radiance_home').resolve()
    _existing_file(home/'kernels.rkb','Radiance kernel database')
    if not (home/'architectures').is_dir() or not (home/'kernels').is_dir():
        raise ValueError('RadianceHome requires architectures/ and kernels/')
    p['RadianceHome']=str(home)
    argv=[p['Worker']]
    def pair(flag,value):argv.extend((flag,str(value)))
    for flag,key in [('--model','Model'),('--radiance-home','RadianceHome'),('--host','ListenHost'),('--port','Port'),
        ('--max-model-len','Context'),('--max-num-seqs','Concurrency'),('--max-num-batched-tokens','Prefill'),
        ('--kvmem-budget','Budget'),('--kvmem-gen-reserve','Reserve'),('--kvmem-host-mib','HostMiB'),
        ('--kvmem-sink-tokens','SinkTokens'),('--kvmem-recent-tokens','RecentTokens'),('--kvmem-sessions','RetainedSessions'),
        ('--kv-cache-dtype','KvType'),('--state-cache-dtype','StateCacheType'),('--gpu-headroom-mib','VramReserveMiB'),('--num-speculative-tokens','MtpDrafts')]:pair(flag,p[key])
    argv+=['--tp',str(len(gpu_ids)),'--tp-wire','exact','--no-prefix-cache','--host-pool-mib','0','--prefix-cache-mib','0',
           '--placement','all_vram','--embedding-placement',p['EmbeddingPlacement'],
           '--draft-window',p['DraftPolicy'],'--kvmem','--kvmem-block-tokens','128']
    pair('--kvmem-drafter','off' if mode=='none' else mode)
    if p.get('Vision'):
        patch,merge=vision_geometry(p['Model'])
        pair('--vision-residency',p['VisionResidency'])
        pair('--mm-max-patches',p['VisionTokens']*merge*merge)
        pair('--image-max-pixels',p['VisionTokens']*(patch*merge)**2)
    else:pair('--mm-max-patches',0)
    if p.get('DiskPath'):
        pair('--kvmem-disk-path',p['DiskPath']);pair('--kvmem-disk-mib',p['DiskMiB'])
    if p['MaxTokens']:pair('--default-max-tokens',p['MaxTokens'])
    if p['ThinkingBudget']:pair('--thinking-token-budget',p['ThinkingBudget'])
    kwargs={}
    if 'PreserveThinking' in supplied:kwargs['preserve_thinking']=p['PreserveThinking']
    if p['ThinkingMode']=='off':pair('--reasoning-effort','none')
    elif p.get('ReasoningEffort'):pair('--reasoning-effort',p['ReasoningEffort'])
    elif p['ThinkingMode']=='on':kwargs['enable_thinking']=True
    if kwargs:pair('--chat-template-kwargs',json.dumps(kwargs,separators=(',',':')))
    if p.get('ModelAlias'):pair('--served-model-name',p['ModelAlias'])
    if p.get('ChatTemplateFile'):pair('--override-chat-template',_existing_file(p['ChatTemplateFile'],'ChatTemplateFile'))
    if p.get('ApiKey'):pair('--api-key',p['ApiKey'] if reveal_secrets else '<redacted>')
    level=p.get('LogLevel','info')
    if level in ('debug','trace'):argv.append('-v' if level=='debug' else '-vv')
    elif level!='info':raise ValueError('Radiance LogLevel currently supports info, debug or trace')
    argv+=scoring_args(supplied)+p.get('WorkerArgs',[])
    env={'ROCR_VISIBLE_DEVICES':p['Gpu'],'HIP_VISIBLE_DEVICES':','.join(str(i) for i in range(len(gpu_ids))),
         'KVMEM_RAD_MTP_HEAD':'shared','OMP_NUM_THREADS':str(p['CpuThreads']),'KVMEM_RAD_VISION_CACHE_MIB':str(p['VisionCacheMiB'])}
    if p.get('RocmPath'):
        rocm=Path(p['RocmPath']).resolve()
        runtime=rocm/('bin/amdhip64_7.dll' if platform=='windows' else 'lib/libamdhip64.so')
        _existing_file(runtime,'ROCm runtime');p['RocmPath']=str(rocm)
        env.update(ROCM_PATH=str(rocm),HIP_PATH=str(rocm))
        if platform=='windows':env['PATH']=str(rocm/'bin')+os.pathsep+os.environ.get('PATH','')
        else:env['LD_LIBRARY_PATH']=':'.join(str(x) for x in (rocm/'lib',rocm/'lib/llvm/lib',rocm/'llvm/lib') if x.is_dir())
    safe_supplied=copy.deepcopy(supplied);safe_effective=copy.deepcopy(p)
    for safe in (safe_supplied,safe_effective):safe.pop('ApiKey',None)
    return dict(schema_version=1,launch_contract_version=3,target_platform=platform,backend='radiance',
        argv=argv,cwd=str(root),environment=env,token_settings=tokens,requested_parameters=safe_supplied,
        effective_parameters=safe_effective,api_key_configured=bool(p.get('ApiKey')),capabilities=copy.deepcopy(capability),
        resolved_scoring=scoring_settings(argv),warnings='\n'.join(f"KVMEM_CONFIG {x['parameter']}: {x['requested']} -> {x['effective']} ({x['reason']})" for x in tokens['adjustments']))
