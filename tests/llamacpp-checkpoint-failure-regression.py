"""Native hybrid commit-budget failure: JSON/SSE errors and bounded rollback.

Requires a dedicated GPU and a hybrid model whose native state exercises the
specified checkpoint budget. The default 128 MiB fixture is qualified with
Qwen3.5-4B IQ3_XXS; it cannot fit even a short generation's required checkpoints.
Normal generation after a rejected Query is separately covered by the existing
checkpoint coverage regression with --host-mib 192 --expect-capacity-error.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--exe',type=Path,required=True)
    cli.add_argument('--model',type=Path,required=True)
    cli.add_argument('--gpu',required=True)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--host-mib',type=int)
    cli.add_argument('--cancel',action='store_true',help='disconnect a continuation after real output, then restore its old parent')
    cli.add_argument('--mtp',action='store_true',help='enable the model native MTP draft for cancellation qualification')
    args=cli.parse_args()
    if args.host_mib is None:args.host_mib=1024 if args.cancel else 128
    args.out.mkdir(parents=True,exist_ok=False)
    processes=subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if processes:raise RuntimeError('GPU occupied: '+processes)
    inventory=subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    memory,util=map(int,inventory.split(','))
    if memory>256 or util>5:raise RuntimeError('GPU not idle: '+inventory)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    command=[str(args.exe.resolve()),'-m',str(args.model.resolve()),'-ngl','99','-c','8192','-b','128','-ub','128',
             '--no-ui','--no-think','--kv-dtype','q8_0','--spec-type','draft-mtp' if args.mtp else 'none','--host','127.0.0.1','--port',str(port),
             '--kvmem-budget','1024','--kvmem-gen-reserve','256','--kvmem-host-mib','512','--kvmem-sessions','2','--kvmem-trace']
    if args.mtp:command+=['--spec-draft-n-max','3']
    env={k:v for k,v in os.environ.items() if not k.startswith(('KVMEM_','GGML_','LLAMA_ARG_'))}
    env.update(CUDA_VISIBLE_DEVICES=args.gpu,KVMEM_CHECKPOINT_POLICY='spread',KVMEM_CHECKPOINT_INTERVAL='256',
               KVMEM_CHECKPOINT_COUNT='6',KVMEM_CHECKPOINT_HOT='1',KVMEM_CHECKPOINT_HOST_MIB=str(args.host_mib))
    (args.out/'manifest.json').write_text(json.dumps({'command':command,'gpu':inventory,'env':{k:v for k,v in env.items() if k.startswith('KVMEM_')}},indent=2))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base='http://127.0.0.1:'+str(port)
    def request(name,messages,stream=False,reset=False):
        body={'model':'checkpoint-test','messages':messages,'max_tokens':8,'temperature':0,'seed':42,
              'enable_thinking':False,'stream':stream,'cache_reset':reset,'kvmem':{'conversation_id':'failure-fixture'}}
        (args.out/(name+'-request.json')).write_text(json.dumps(body))
        req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        try:
            with opener.open(req,timeout=120) as response:status,raw=response.status,response.read()
        except urllib.error.HTTPError as error:status,raw=error.code,error.read()
        (args.out/(name+'-response.txt')).write_bytes(raw)
        return status,raw
    system={'role':'system','content':'Read the reference notes. Acknowledge notes with only Noted. Answer later questions briefly from the notes.'}
    notes='Reference notes:\n'+'\n'.join(f'Record {i:04d}: the coastal survey team checked the tide gauge and logged a calm evening.' for i in range(128))
    messages=[system,{'role':'user','content':notes}]
    log_path=args.out/'server.log'
    with log_path.open('wb') as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=log,creationflags=0x08000000 if os.name=='nt' else 0)
        try:
            deadline=time.monotonic()+120
            while True:
                if child.poll() is not None:raise RuntimeError('server exited')
                try:
                    with opener.open(base+'/health',timeout=1):break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError('server startup')
                    time.sleep(.2)
            if args.cancel:
                status,raw=request('parent',messages)
                assert status==200,raw
                parent=json.loads(raw)
                def usage():
                    with opener.open(base+'/api/monitor/snapshot',timeout=5) as response:
                        return json.load(response)['kvmem']
                before=usage()
                continuation=messages+[parent['choices'][0]['message'],{'role':'user','content':'List the integers from 1 to 1000, one per line. Continue without a summary.'}]
                body={'model':'checkpoint-test','messages':continuation,'max_tokens':1024,'temperature':0,'seed':42,
                      'enable_thinking':False,'stream':True,'kvmem':{'conversation_id':'failure-fixture'}}
                (args.out/'cancel-request.json').write_text(json.dumps(body))
                req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                offset=log_path.stat().st_size
                received=[]
                response=opener.open(req,timeout=120)
                try:
                    for line in response:
                        received.append(line.decode())
                        if not line.startswith(b'data: '):continue
                        assert line.strip()!=b'data: [DONE]', 'request completed before cancellation'
                        event=json.loads(line[6:])
                        assert 'error' not in event,event
                        if any(c.get('delta',{}).get('content') for c in event.get('choices',[])):
                            # Close the actual connection after real generation,
                            # rather than merely abandoning the Python reader.
                            response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                            break
                    else:raise AssertionError('no generated output before disconnect')
                finally:response.close()
                (args.out/'cancel-received.txt').write_text(''.join(received))
                deadline=time.monotonic()+15
                while True:
                    trace=log_path.read_bytes()[offset:].decode(errors='replace')
                    after=usage()
                    if 'KVMEM_CHECKPOINT_ROLLBACK' in trace and not after['active_sessions']:break
                    if time.monotonic()>deadline:raise AssertionError('disconnect did not finish native rollback')
                    time.sleep(.1)
                assert 'multimodal_rollback_failed' not in trace
                assert 'KVMEM_CHECKPOINT_COMMIT' not in trace,'cancelled continuation was published'
                assert after['physical_checkpoint_bytes']<=before['physical_checkpoint_bytes']+16384,(before,after)
                status,raw=request('parent-restored',messages)
                assert status==200,raw
                restored=json.loads(raw)
                cached=restored['usage'].get('prompt_tokens_details',{}).get('cached_tokens',restored['usage'].get('prompt_cache_hit_tokens',0))
                assert cached>=parent['usage']['prompt_tokens']-32,restored['usage']
                assert restored['choices'][0]['message']==parent['choices'][0]['message']
                (args.out/'result.json').write_text(json.dumps({'passed':True,'cancelled_after_output':True,
                    'parent_cached_tokens':cached,'before':before,'after':after},indent=2))
                print('PASS disconnect, native rollback, checkpoint release, old parent reuse',flush=True)
                return
            for stream in (False,True):
                name='stream' if stream else 'json'
                offset=log_path.stat().st_size
                status,raw=request(name,messages,stream,True)
                trace=log_path.read_bytes()[offset:].decode(errors='replace')
                assert 'checkpoint_commit_failed' in trace, 'fixture did not reach late commit failure'
                assert 'KVMEM_CHECKPOINT_ROLLBACK' in trace and 'multimodal_rollback_failed' not in trace
                if stream:
                    assert status==200
                    events=[json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith('data: ') and line!='data: [DONE]']
                    assert any('checkpoint Host budget exhausted' in str(e.get('error','')) for e in events),events
                    assert not any(c.get('finish_reason') for e in events for c in e.get('choices',[])),events
                else:
                    assert status==500 and 'checkpoint Host budget exhausted' in json.loads(raw)['error'],raw
                status,raw=request(name+'-recovery',[{'role':'user','content':'Say hello.'}])
                assert status==500 and 'checkpoint Host budget exhausted' in json.loads(raw)['error'],raw
                with opener.open(base+'/api/monitor/snapshot',timeout=5) as response:
                    usage=json.load(response)['kvmem']
                assert usage['physical_checkpoint_bytes']<4096,usage
                assert child.poll() is None
                print('PASS',name,'late commit error, rollback, repeated rejection without retained state',flush=True)
            trace=log_path.read_text(errors='replace')
            peaks=[int(n) for n in re.findall(r'peak_bytes=(\d+)',trace)]
            assert peaks and max(peaks)<=args.host_mib*1024*1024
            (args.out/'result.json').write_text(json.dumps({'passed':True,'peak_checkpoint_bytes':max(peaks)}))
        finally:
            child.terminate()
            try:child.wait(15)
            except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':main()
