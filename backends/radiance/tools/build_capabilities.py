"""Construct a worker capability payload for the selected native kernel profile."""
import argparse
import copy
import json
from pathlib import Path


def for_profile(manifest,profile):
    capability=copy.deepcopy(manifest['backends']['radiance'])
    if profile=='all':return capability
    if profile in ('qwen35-iqr-gfx1101','qwen35-iqr-rdna3'):
        capability['build_profile']=profile
        capability['qualification']={'status':'text-query-pressure-passed','engine':'2.0.0',
            'platform':'windows','gpu':'gfx1101','model':'qwen3.8-27b-iq2r-l-v4',
            'tensor_parallel':[1],'max_concurrency':1,'kv':'bf16','state_cache':'f32',
            'prefill':256,'budget':1024,'report':'docs/radiance-v2-migration-20261116.md'}
        capability['multimodal']=False
        capability['speculative_decoding']=True
        capability['speculative']['modes']={
            'none':{'max_draft_tokens':0},
            'dflash2':{**capability['speculative']['modes']['dflash2'],'max_devices':1}}
        capability['qualification']['dflash2']={
            'status':'text-query-pressure-passed','draft_tokens':7,'budget':3072,
            'recent_tokens':2176,'kv':'bf16','state_cache':'f32',
            'report':'docs/radiance-v2-dflash2-rx7800m-20261116.md'}
        capability.pop('packed_kv_geometry',None)
        combinations=capability['supported_combinations']
        combinations['images']=False
        combinations.pop('image_input',None)
        combinations.pop('adaptive_mtp',None)
        combinations['text'].update(max_devices=1,max_concurrency=1,tensor_parallel=[1],kv=['bf16','fp8'],state_cache=['f32'])
        combinations['active_disk']['max_concurrency']=1
        capability['concurrent_requests']=False
        if profile=='qwen35-iqr-rdna3':
            # Architecture compatibility is not evidence that every card/model fits
            # memory or passed runtime qualification on the available test device.
            observed=capability.pop('qualification')
            capability['qualification']={
                'status':'per-device-qualification-required','platform':'windows',
                'target_family':['gfx1100','gfx1101','gfx1102','gfx1103'],
                'runtime_evidence':observed}
        return capability
    if profile!='qwen35-dense-tp1':raise ValueError('unsupported Radiance kernel profile: '+profile)
    combinations=capability['supported_combinations']
    for name in ('text','image_input'):
        combinations[name]['max_devices']=min(combinations[name]['max_devices'],1)
    combinations['text']['tensor_parallel']=[1]
    for mode in capability['speculative']['modes'].values():
        if 'max_devices' in mode:mode['max_devices']=min(mode['max_devices'],1)
    capability['build_profile']=profile
    return capability


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--kernel-profile',choices=('all','qwen35-dense-tp1','qwen35-iqr-gfx1101','qwen35-iqr-rdna3'),required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    print(json.dumps(for_profile(manifest,args.kernel_profile),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
