"""Construct a worker capability payload for the selected native kernel profile."""
import argparse
import copy
import json
from pathlib import Path


def for_profile(manifest,profile):
    capability=copy.deepcopy(manifest['backends']['radiance'])
    if profile=='all':return capability
    if profile!='qwen35-dense-tp1':raise ValueError('unsupported Radiance kernel profile: '+profile)
    combinations=capability['supported_combinations']
    for name in ('text','image_input'):
        combinations[name]['max_devices']=min(combinations[name]['max_devices'],1)
    combinations['text']['tensor_parallel']=[1]
    for mode in capability['speculative']['modes'].values():
        if 'max_devices' in mode:mode['max_devices']=min(mode['max_devices'],1)
    qualification=capability.get('qualification',{})
    if 'tensor_parallel' in qualification:
        qualification['tensor_parallel']=[count for count in qualification['tensor_parallel'] if count==1]
    qualification.pop('tp2_mtp_depths',None)
    capability['build_profile']=profile
    return capability


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--kernel-profile',choices=('all','qwen35-dense-tp1'),required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    print(json.dumps(for_profile(manifest,args.kernel_profile),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
