"""Core HTTP Query/Pressure checks on either backend; uses an existing local model.

--argv-json is a JSON argv array (or {"argv": [...]}); the private server's
port and selected device are set here. Requests, responses and per-request native
traces are retained. No reference answer is inserted in later messages.
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['llamacpp', 'ninfer'], required=True)
    p.add_argument('--argv-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--tool-repeats', type=int, default=160,
                   help='observation repetitions; increase to exceed a larger append reserve')
    p.add_argument('--resident-lines', type=int, default=80,
                   help='reference lines for the B < prompt <= B+R resident-view fixture')
    p.add_argument('--reject-replay', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    command = json.loads(a.argv_json.read_text(encoding='utf-8-sig'))
    if isinstance(command, dict):
        command = command['argv']
    command = list(command)
    expected_spec = command[command.index('--spec')+1] if '--spec' in command else 'none'
    scorer = command[command.index('--kvmem-query-scorer') + 1] if '--kvmem-query-scorer' in command else 'page-softmax'
    query_limit = int(command[command.index('--kvmem-query-max-tokens') + 1]) if '--kvmem-query-max-tokens' in command else 512
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    if '--port' in command:
        command[command.index('--port')+1] = str(port)
    else:
        command += ['--port', str(port)]
    if a.reject_replay:
        command += ['--kvmem-query-replay-max-tokens', '8']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', KVMEM_TRACE='1')
    env.pop('KVMEM_ATTENTION_VERIFY', None)
    env['PATH'] = str(Path(command[0]).parent) + os.pathsep + env.get('PATH', '')
    report = {'gpu': a.gpu, 'backend': a.backend, 'argv': command, 'checks': [], 'requests': []}
    log_path = a.output / 'server.log'
    def save():
        (a.output/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    def check(name, condition):
        report['checks'].append({'name': name, 'pass': bool(condition)})
        save()
        if not condition:
            raise AssertionError(name)
        print('PASS '+name, flush=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def call(path, body=None, timeout=180):
        request = urllib.request.Request('http://127.0.0.1:'+str(port)+path,
            None if body is None else json.dumps(body).encode(), {'Content-Type':'application/json'})
        try:
            with opener.open(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            raise RuntimeError(e.read().decode()) from e
    def ask(name, messages, maximum=16, session='policy-A', reset=False):
        offset = log_path.stat().st_size
        body = {'messages':messages, 'temperature':0, 'seed':42, 'max_tokens':maximum,
                'enable_thinking':False, 'cache_reset':reset, 'prompt_cache_key':session,
                'kvmem':{'conversation_id':session}}
        (a.output/(name+'.request.json')).write_text(json.dumps(body,indent=2),encoding='utf-8')
        started = time.monotonic()
        response = call('/v1/chat/completions',body)
        elapsed = time.monotonic()-started
        trace = log_path.read_bytes()[offset:].decode('utf-8',errors='replace')
        (a.output/(name+'.trace.log')).write_text(trace,encoding='utf-8')
        (a.output/(name+'.response.json')).write_text(json.dumps(response,indent=2),encoding='utf-8')
        report['requests'].append({'phase':name,'elapsed_seconds':elapsed,'usage':response.get('usage')})
        check(name+' succeeds', bool(response.get('choices')) and response['usage']['completion_tokens']>0)
        for start,end in set(re.findall(r'query_begin=(\d+) query_end=(\d+)',trace)):
            check(name+' Q bounded',int(end)-int(start)<=query_limit)
        if a.backend == 'ninfer':
            captured = [int(n) for n in re.findall(r'KVMEM_QUERY scorer=\S+[^\n]* rows=(\d+)',trace)]
            spans = [(int(x),int(y)) for x,y in re.findall(
                r'KVMEM_PREFILL_PLAN[^\n]* query_begin=(\d+) query_end=(\d+)[^\n]* capture_query=1',trace)]
            if captured:
                check(name+' scores every planned Query row', bool(spans) and
                      all(n in {y-x for x,y in spans} for n in captured))
        for rows in re.findall(r'replayed_rows=(\d+)|replay_rows=(\d+)',trace):
            check(name+' replay bounded',int(next(x for x in rows if x)) <= (8 if a.reject_replay else 2048))
        return response,trace
    save()
    with log_path.open('w',encoding='utf-8') as log:
        server = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+180
            while True:
                if server.poll() is not None:
                    raise RuntimeError('server exited before readiness')
                try:
                    call('/health',timeout=2)
                    break
                except (OSError, urllib.error.URLError, RuntimeError):
                    if time.monotonic()>deadline:
                        raise TimeoutError('server readiness')
                    time.sleep(.25)
            if a.backend == 'ninfer':
                props = call('/props')
                (a.output/'props.json').write_text(json.dumps(props, indent=2), encoding='utf-8')
                if expected_spec in ('mtp', 'dflash2'):
                    check('worker reports requested speculation',
                          props.get('speculative', {}).get('algorithm') == expected_spec)
                if '--tensor-parallel' in command:
                    execution = props.get('execution', {})
                    check('worker reports heads TP', execution.get('mode') == 'tensor' and
                          execution.get('tensor_attention') == 'heads')
                    shares = execution.get('tensor_shares', [])
                    check('worker reports actual rank shares', len(shares) >= 2)
                elif '--stage-layers' in command:
                    expected_layers = [int(x) for x in command[command.index('--stage-layers')+1].split(',')]
                    check('worker reports requested layer pipeline',
                          props.get('execution', {}).get('mode') == 'layer' and
                          props['execution'].get('stage_layers') == expected_layers)
            document = ''.join(f'Record {i}: the river flows through the valley.\n' for i in range(240))
            document += '\nReply briefly to acknowledge the reference text.'
            messages = [{'role':'user','content':document}]
            response,trace = ask('document',messages,8,reset=True)
            if a.backend == 'ninfer' and not a.reject_replay:
                check('long document exercises full Query limit',
                      re.search(r'KVMEM_QUERY scorer=\S+[^\n]* rows='+str(query_limit)+r'\b',trace) is not None)
            if a.reject_replay:
                check('over-limit never replays','KVMEM_QUERY_REPLAY from=' not in trace and
                      'path=query_replay ' not in trace)
            messages += [{'role':'assistant','content':response['choices'][0]['message'].get('content') or 'OK'},
                         {'role':'user','content':'What flows through the valley? Answer briefly.'}]
            response,trace = ask('question',messages,32)
            check('question reuses document prefix',response['usage'].get('prompt_cache_hit_tokens',
                response['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0))>0)
            # A real token-aligned user range, not the preceding document tail.
            if a.backend=='ninfer':
                spans=re.findall(r'query_begin=(\d+) query_end=(\d+)',trace)
                check('question span observed',bool(spans))
                check('question span excludes document',all(0<int(y)-int(x)<64 for x,y in spans))
            else:
                check('question has true user source','query_source=user' in trace)
            tool_base = messages + [{'role':'assistant','content':'','tool_calls':[
                {'id':'read1','type':'function','function':{'name':'read_reference','arguments':'{}'}}]}]
            tools = tool_base+[{'role':'tool','tool_call_id':'read1','content':
                'Observation: the valley river is blue.\n'*a.tool_repeats}]
            for name,current in [('tool',tools)]:
                _,trace=ask(name,current,16)
                check(name+' no cold Query selection', 'KVMEM_QUERY scorer=' not in trace and
                      'KVMEM_RETRIEVAL logical=' not in trace and 'KVMEM_QUERY_REPLAY from=' not in trace and
                      'path=query_replay ' not in trace and 'cached_q_reselect' not in trace)
                check(name+' actually uses Pressure','KVMEM_PRESSURE ' in trace)
            ask('other-session',[{'role':'user','content':'Reply briefly.'}],8,session='policy-B')
            resumed,_=ask('resume',tools,16)
            check('resume restores a native prefix',resumed['usage'].get('prompt_cache_hit_tokens',
                resumed['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0))>0)
            cold=[{'role':'system','content':'Independent cold tool request.'}]+tools
            cold_response,trace=ask('cold-tool',cold,16,session='policy-cold',reset=True)
            check('cold-tool starts without reused prefix',cold_response['usage'].get('prompt_cache_hit_tokens',
                cold_response['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0))==0)
            check('cold-tool no cold Query selection','KVMEM_QUERY scorer=' not in trace and
                'KVMEM_RETRIEVAL logical=' not in trace and 'KVMEM_QUERY_REPLAY from=' not in trace and
                'path=query_replay ' not in trace)
            check('cold-tool actually uses Pressure','KVMEM_PRESSURE ' in trace)
            rewrite=messages[:-1]+[{'role':'user','content':'Describe the river briefly.'}]
            ask('rewrite',rewrite,16)
            # Disconnect after a published streaming token, then reuse/edit the
            # same session. The following request must have a valid native prefix.
            body={'messages':rewrite,'temperature':0,'max_tokens':512,'stream':True,
                  'enable_thinking':False,'prompt_cache_key':'policy-A',
                  'kvmem':{'conversation_id':'policy-A'}}
            req=urllib.request.Request('http://127.0.0.1:'+str(port)+'/v1/chat/completions',
                json.dumps(body).encode(),{'Content-Type':'application/json'})
            with opener.open(req,timeout=180) as stream:
                while True:
                    line=stream.readline()
                    if not line or line.startswith(b'data:'):
                        break
            ask('after-cancel',messages,16)
            check('health after cancel',call('/health') is not None)
            fits=[{'role':'user','content':''.join(
                f'Record {i}: the river flows through the valley.\n' for i in range(a.resident_lines))+
                '\nReply briefly.'}]
            fit_response,trace=ask('fits-device',fits,8,session='policy-fits',reset=True)
            if '--kvmem-budget' in command and '--kvmem-gen-reserve' in command:
                budget = int(command[command.index('--kvmem-budget')+1])
                reserve = int(command[command.index('--kvmem-gen-reserve')+1])
                fit_tokens = fit_response['usage']['prompt_tokens']
                check('resident fixture crosses B', fit_tokens > budget)
                check('resident fixture fits B+R', fit_tokens <= budget + reserve)
            check('resident history above B avoids Query replay','path=query_replay ' not in trace and
                'KVMEM_QUERY_REPLAY from=' not in trace and 'KVMEM_QUERY scorer=' not in trace)
            check('resident view captures no unused Query', 'capture_query=0' in trace if a.backend=='ninfer'
                else 'q_rows=0' in trace)
            follow=fits+[{'role':'assistant','content':'','tool_calls':[
                {'id':'read_after_fit','type':'function','function':{'name':'read_reference','arguments':'{}'}}]},
                {'role':'tool','tool_call_id':'read_after_fit','content':'Observation: the valley river is blue.\n'*a.tool_repeats}]
            response,trace=ask('tool-after-resident',follow,16,session='policy-fits')
            check('tool works without cached Q','KVMEM_PRESSURE ' in trace and 'KVMEM_QUERY scorer=' not in trace)
            follow+=[response['choices'][0]['message'],{'role':'user','content':'What color is the river in the latest observation?'}]
            _,trace=ask('new-query-after-resident',follow,16,session='policy-fits')
            if not a.reject_replay:
                check('new Query captures fresh features','KVMEM_QUERY scorer='+scorer in trace)
            if a.backend == 'ninfer' and expected_spec in ('mtp', 'dflash2'):
                after = call('/props')
                (a.output/'props-after.json').write_text(json.dumps(after, indent=2), encoding='utf-8')
                stats = after['speculative']['statistics']
                check('speculation actually drafts', stats['rounds'] > 0 and stats['drafted_tokens'] > 0)
                check('accepted tokens stay within drafted tokens',
                      0 <= stats['accepted_tokens'] <= stats['drafted_tokens'])
            report['status']='passed'
            save()
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill();server.wait()

if __name__=='__main__':
    main()
