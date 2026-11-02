"""Separate document/question turns, native Query evidence and block-size negotiation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','worker','model','tokenizer','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--gpu',required=True)
    ap.add_argument('--block-tokens',type=int,choices=(32,64),required=True)
    ap.add_argument('--kv-type',choices=('f16','q8_0'),default='q8_0')
    ap.add_argument('--draft-model',type=Path)
    ap.add_argument('--mtp-drafts',type=int,choices=range(4),default=2)
    ap.add_argument('--host-mib',type=int,default=512)
    ap.add_argument('--context',type=int,default=32768)
    ap.add_argument('--port',type=int,default=18540)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    native=a.out/'native.log'
    cmd=[sys.executable,str(Path(__file__).with_name('server.py')),
         '--source',str(a.source),'--worker',str(a.worker),'--model',str(a.model),'--tokenizer',str(a.tokenizer),
         '--gpu',a.gpu,'--port',str(a.port),'--kvmem-block-tokens',str(a.block_tokens),'--kv-type',a.kv_type,
         '--context',str(a.context),'--budget','2048','--reserve','512','--prefill','128','--host-mib',str(a.host_mib),
         '--sink','64','--recent','64','--mtp-drafts',str(a.mtp_drafts),'--retained-sessions','4',
         '--log',str(native),'--','--threads','8','--stage-mib','512','--tail-skip','0','--expert-swaps','0']
    if a.draft_model:
        at=cmd.index('--');cmd[at:at]=['--draft-model',str(a.draft_model)]
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def req(path,payload=None,timeout=300):
        data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode()
        with opener.open(urllib.request.Request(f'http://127.0.0.1:{a.port}'+path,data=data,
                         headers={'Content-Type':'application/json'}),timeout=timeout) as r:return json.load(r)
    result={'command':cmd,'gpu_uuid':a.gpu,'block_tokens':a.block_tokens,'kv_type':a.kv_type,'turns':[]}
    with (a.out/'server.log').open('w',encoding='utf-8') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000 if os.name=='nt' else 0)
        try:
            deadline=time.monotonic()+240
            while True:
                if p.poll() is not None:raise RuntimeError('frontend exited '+str(p.returncode))
                try:req('/health',timeout=1);break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError('frontend health')
                    time.sleep(.25)
            result['props']=req('/props')
            assert result['props']['kvmem']['block_tokens']==a.block_tokens,result['props']
            document='Reference material:\n'+''.join(
                'Project Atlas access key: MARBLE-4827.\n' if i==70 else
                f'Record {i}: Project Cedar-{i} inspection date 2031-04-19; serial REF-{i:05d}.\n'
                for i in range(353))
            (a.out/'document.txt').write_text(document,encoding='utf-8')
            messages=[{'role':'system','content':'When I send reference material, acknowledge only with Received. Answer my subsequent question using that material.'},
                      {'role':'user','content':document}]
            def turn():
                body={'model':'kvmem-polystrata-qwen35','messages':messages,'temperature':0,'max_tokens':32,
                      'chat_template_kwargs':{'enable_thinking':False}}
                start=time.monotonic();response=req('/v1/chat/completions',body)
                result['turns'].append({'request':json.loads(json.dumps(body)),'response':response,'seconds':time.monotonic()-start})
                (a.out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
                return response
            first=turn();ack=first['choices'][0]['message']['content'].strip()
            assert ack=='Received.',first
            before=native.read_text(encoding='utf-8',errors='replace')
            messages.extend([{'role':'assistant','content':ack},
                             {'role':'user','content':'What is the exact access key for Project Atlas? Reply only with the key.'}])
            second=turn()
            assert second['choices'][0]['message']['content'].strip()=='MARBLE-4827',second
            cached=second['usage']['prompt_tokens_details']['cached_tokens']
            assert cached>first['usage']['prompt_tokens']-256,second
            text=native.read_text(encoding='utf-8',errors='replace')[len(before):]
            spans=[tuple(map(int,m)) for m in re.findall(r'KVMEM_QUERY query_begin=(\d+) query_end=(\d+) prefix=(\d+)',text)]
            assert spans and all(0<end-begin<=512 and begin==prefix for begin,end,prefix in spans),text
            assert 'replay=1' in text,text
            forwarded=(a.out/'server.log').read_text(encoding='utf-8',errors='replace')
            assert f'KVMEM_BLOCKS block_tokens={a.block_tokens}' in forwarded,forwarded
            result.update(status='pass',cached_tokens=cached,question_native_spans=spans,startup_block_confirmed=True)
        except Exception as e:
            result.update(status='fail',error=str(e));raise
        finally:
            if p.poll() is None:
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,check=True)
                else:p.terminate()
                p.wait(timeout=30)
            (a.out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','block_tokens','cached_tokens','question_native_spans')}))


if __name__=='__main__':main()
