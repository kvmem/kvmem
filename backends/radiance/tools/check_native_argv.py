"""Validate normal shared/standalone launch argv with the native parser-only test binary.

The probe compiles observer_main.cpp with KVMEM_RADIANCE_ARGV_PARSE_TEST. It
removes only a trailing --help, runs the actual observer/native parsers, then
returns before resolving weights or starting an engine. No GPU work is issued.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from launcher.plan import build_plan
from draft_control import resolve
import serve


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--rocm',type=Path,required=True)
    parser.add_argument('--gpu',default='GPU-0000000000000000')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    parameters=dict(Backend='radiance',Worker=str(args.probe.resolve()),Model=str(args.model.resolve()),
                    RadianceHome=str(args.home.resolve()),Gpu=args.gpu,RocmPath=str(args.rocm.resolve()))
    cases=[]
    for name,settings in [('shared-off',{}),('shared-rich',dict(ThinkingBudget=32,MaxTokens=64,
            ReasoningEffort='medium',PreserveThinking=True,ModelAlias='native parser test',
            Vision=True,VisionResidency='cpu',VisionTokens=64,LogLevel='debug',
            QueryScorer='meanq-cosine',Lrfu=True,QueryMaxTokens=128,QueryReplayMaxTokens=768))]:
        plan=build_plan({'schema_version':1,'parameters':dict(parameters,**settings)},root=ROOT)
        cases.append((name,plan['argv'][1:],True,dict(os.environ,**plan['environment'])))
    model=serve.inspect(args.model)
    mode='dflash2' if 'DFlash2' in str(model['draft_metadata'].get('draft.architectures','')) else 'mtp'
    selection=resolve(args.model,'dflash' if mode=='dflash2' else mode)
    plan=build_plan({'schema_version':1,'parameters':dict(parameters,SpeculativeMode=mode,
                                                        DraftTokens=selection['depth'],DraftPolicy='fixed')},root=ROOT)
    cases.append(('shared-embedded-'+mode,plan['argv'][1:],True,dict(os.environ,**plan['environment'])))
    standalone=argparse.Namespace(exe=args.probe,model=args.model,home=args.home,rocm=args.rocm,
                                   port=18380,drafter='auto',speculative_tokens=None)
    for draft in ('auto','off'):
        standalone.drafter=draft
        config,_=serve.model_profile(standalone,serve.profile())
        env=serve.child_environment(standalone,'windows' if os.name=='nt' else 'linux',False,config)
        cases.append(('standalone-'+draft,serve.command_for(standalone,config)[1:],True,env))
    valid=cases[0][1]
    cases.append(('reject-removed-prefix-flag',
                  ['--prefix-cache-host-mib' if value=='--prefix-cache-mib' else value for value in valid],False,cases[0][3]))
    cases.append(('reject-unknown-native-flag',[*valid,'--invalid-native-argv-regression'],False,cases[0][3]))
    results=[]
    for name,argv,expected,env in cases:
        command=[str(args.probe.resolve()),*argv,'--help']
        result=subprocess.run(command,capture_output=True,text=True,env=env)
        passed=(result.returncode==0 and 'KVMEM_NATIVE_ARGV_VALID' in result.stdout) if expected else result.returncode!=0
        results.append({'name':name,'command':command,'expected_valid':expected,'returncode':result.returncode,
                        'pass':passed,'stdout':result.stdout,'stderr':result.stderr})
    args.output.write_text(json.dumps({'host_only':True,'cases':results},indent=2)+'\n')
    print(json.dumps({'cases':len(results),'passed':sum(result['pass'] for result in results),'evidence':str(args.output)},indent=2))
    if not all(result['pass'] for result in results):raise SystemExit(1)


if __name__=='__main__':
    main()
