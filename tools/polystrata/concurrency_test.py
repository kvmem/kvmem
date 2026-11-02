"""Single-GPU native-slot isolation and retained-prefix HTTP acceptance; not a long-context quality comparison."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,os,subprocess,sys,time,urllib.request,urllib.error
ap=argparse.ArgumentParser(description=__doc__)
for name in ('source','worker','model','tokenizer','out'):ap.add_argument('--'+name,type=Path,required=True)
ap.add_argument('--gpu',required=True)
ap.add_argument('--port',type=int,default=18287)
ap.add_argument('--lanes',type=int,choices=range(2,9),default=2)
ap.add_argument('--block-tokens',type=int,choices=(32,64),default=64)
ap.add_argument('--host-mib',type=int,default=256)
ap.add_argument('--mtp-drafts',type=int,choices=range(4),default=2)
ap.add_argument('--kv-type',choices=('f16','q8_0','int8','int8k'))
ap.add_argument('--q8',action='store_true');a=ap.parse_args()
for name in ('source','worker','model','tokenizer','out'):setattr(a,name,getattr(a,name).resolve())
out=a.out;out.mkdir(parents=True,exist_ok=True)
port=a.port;base=f'http://127.0.0.1:{port}';source=a.source
cmd=[sys.executable,str(Path(__file__).with_name('server.py')),'--source',str(source),'--worker',str(a.worker),
 '--model',str(a.model),'--tokenizer',str(a.tokenizer),
 '--gpu',a.gpu,'--port',str(port),'--kvmem-block-tokens',str(a.block_tokens),'--context','8192','--budget','512','--reserve','512',
 '--log',str(out/'native.log'),'--host-mib',str(a.host_mib),'--prefill','128','--max-tokens','64','--concurrency',str(a.lanes),'--retained-sessions',str(max(4,a.lanes+1)),
 '--mtp-drafts',str(a.mtp_drafts),'--kv-type',a.kv_type or ('q8_0' if a.q8 else 'f16'),'--','--threads','8','--stage-mib','256','--tail-skip','0','--expert-swaps','0','--kvmem-verify-transfers']
# Verify-transfers belongs to native diagnostics; the wrapper intentionally owns all --kvmem-* flags.
cmd.remove('--kvmem-verify-transfers')
env=dict(os.environ,KVMEM_POLY_SESSION_METRICS='1',KVMEM_POLY_PROFILE='1')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def req(path,payload=None,timeout=300):
 data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode('utf-8')
 with opener.open(urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'}),timeout=timeout) as r:return json.load(r)
result={'status':'running','command':cmd,'gpu_uuid':a.gpu,'lanes':a.lanes,'requests':[]}
log=(out/'server.log').open('w',encoding='utf-8');p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
try:
 deadline=time.monotonic()+180
 while True:
  if p.poll() is not None:raise RuntimeError('frontend exited '+str(p.returncode))
  try:req('/health',timeout=1);break
  except (OSError,urllib.error.URLError):
   if time.monotonic()>deadline:raise RuntimeError('health timeout')
   time.sleep(.25)
 result['props']=req('/props');assert result['props']['kvmem']['max_active_requests']==a.lanes,result['props']
 assert result['props']['kvmem']['kv_type']==(a.kv_type or ('q8_0' if a.q8 else 'f16')),result['props']
 labels=[f'LANE-{i}-KEY-{7319+i}' for i in range(a.lanes)]
 bodies=[]
 for i,label in enumerate(labels):
  text=('Context line for this independent request.\n'*(8+i*3))+f'Your assigned label is {label}. Write a numbered list with 60 entries. Every entry must contain exactly this label followed by its entry number. Start the list immediately.'
  bodies.append({'model':'kvmem-polystrata-qwen35','messages':[{'role':'user','content':text}], 'temperature':0,'max_tokens':128 if a.lanes==8 else 64,'chat_template_kwargs':{'enable_thinking':False}})
 peak=0
 with ThreadPoolExecutor(max_workers=a.lanes) as pool:
  futures=[pool.submit(req,'/v1/chat/completions',b) for b in bodies]
  while any(not f.done() for f in futures):
   states=req('/slots',timeout=10);peak=max(peak,sum(bool(s.get('is_processing')) for s in states));time.sleep(.04)
  responses=[f.result() for f in futures]
 result['peak_active']=peak
 for i,(body,response,label) in enumerate(zip(bodies,responses,labels)):
  text=response['choices'][0]['message']['content'];result['requests'].append({'request':body,'response':response})
  assert label in text,(label,response)
  assert not any(other in text for other in labels if other!=label),response
 assert peak==a.lanes,('not all native slots overlapped',peak,a.lanes)
 follows=[]
 for body,response in zip(bodies,responses):
  follows.append({**body,'max_tokens':24,'messages':body['messages']+[{'role':'assistant','content':response['choices'][0]['message']['content']},{'role':'user','content':'Reply only with your assigned label.'}]})
 with ThreadPoolExecutor(max_workers=a.lanes) as pool: second=list(pool.map(lambda b:req('/v1/chat/completions',b),follows))
 for label,body,response in zip(labels,follows,second):
  result['requests'].append({'request':body,'response':response});assert label in response['choices'][0]['message']['content'],response
  assert response['usage']['prompt_tokens_details']['cached_tokens']>0,response
 result['metrics']=req('/metrics');result['status']='pass'
 print(json.dumps({'status':'pass','lanes':a.lanes,'peak_active':peak,'cached':[r['usage']['prompt_tokens_details']['cached_tokens'] for r in second]}),flush=True)
except Exception as e:
 result.update(status='fail',error=str(e));raise
finally:
 subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True)
 p.wait(timeout=30);log.close();(out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
