#!/usr/bin/env python3
"""Opt-in zero-client-change retrieval regressions against a running NInfer.

Launch the prepared server with --model-id minimal-retention --max-context 32768
--max-concurrency 2 --kv-dtype fp8 --prefill-chunk 256 --kvmem-budget 6144
--kvmem-gen-reserve 512 --kvmem-host-mib 4096 --kvmem-sink-tokens 128
--kvmem-recent-tokens 512 --kvmem-sessions 4. No external tools are executed.
Run: python3 tests/test_ninfer_task_retention_http.py --server http://127.0.0.1:18087 --output retention-results.json

The 22 retrieval checks cover system/task retention, reminder rewrites, human
corrections, closed tasks, history fact mapping, semantic tool selection, local
tool-result continuation, cache reuse and two concurrent requests. Two separate
model-output-selection cases remain opt-in reproductions: the fixed model also
fails them with full attention. They are excluded from retrieval qualification
explicitly, rather than relabeled as passes. Sparse retrieval is approximate.
"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import re
import time
import urllib.error
import urllib.request

SERVER = ''
def tools():
    decoys = []
    for i in range(40):
        decoys.append({'type':'function','function':{'name':f'archive_operation_{i}',
            'description':f'Legacy archival operation {i}; only for old unrelated documents. '+
                          'Stores historical shipping notes and unrelated catalog metadata. '*5,
            'parameters':{'type':'object','properties':{'record':{'type':'string'}},'required':['record']}}})
    decoys.insert(20, {'type':'function','function':{'name':'confirm_inventory',
        'description':'Confirm a current inventory audit ticket before preparing the final answer.',
        'parameters':{'type':'object','properties':{'ticket':{'type':'string'},'mode':{'type':'string'},
            'quantity':{'type':'integer'}},'required':['ticket','mode','quantity']}}})
    return decoys

def history(case, definitions=False):
    # The critical directive lies beyond the sink and outside the recent tail.
    system = 'You are a precise assistant. '+ 'General archival background; preserve ordinary historical records. '*35
    system += f'\nFor every final textual answer, begin exactly SYSTEM_{case}. Tool calls are permitted without this prefix. '
    system += 'Do not invent missing historical facts. Follow the current user task. '
    messages = [{'role':'system','content':system}]
    messages += [{'role':'user','content':'Read the two configuration files for later reference.'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'alpha','type':'function','function':{'name':'archive_operation_0','arguments':'{"record":"alpha.cfg"}'}}]},
        {'role':'tool','tool_call_id':'alpha','content':f'alpha.cfg: project {case}, retrieval checksum ALPHA_7913; timeout_seconds=37.'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'beta','type':'function','function':{'name':'archive_operation_1','arguments':'{"record":"beta.cfg"}'}}]},
        {'role':'tool','tool_call_id':'beta','content':f'beta.cfg: project {case}, validation checksum BETA_4628; retry_limit=11.'},
        {'role':'assistant','content':'Recorded both files.'}]
    for i in range(10):
        messages += [{'role':'user','content':f'Archive unrelated entry {i}. '+
            'The violet observatory catalogs stars while the coastal warehouse files invoices. '*35},
            {'role':'assistant','content':'Archived.'}]
    body = {'model':'minimal-retention','messages':messages,'temperature':0,'seed':20261013,
            'enable_thinking':False,'max_tokens':160,'stream':True,'stream_options':{'include_usage':True}}
    if definitions: body['tools']=tools()
    return body

def quality_fixtures():
    cases=[]
    task=history('TASK')
    task['messages'] += [{'role':'user','content':'For the current task return exactly TASK_BOUND_4821 after the required system prefix.'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'bulk','type':'function','function':{'name':'archive_operation_0','arguments':'{"record":"bulk"}'}}]},
        {'role':'tool','tool_call_id':'bulk','content':'Read-only diagnostic archive; no new task. '*220},
        {'role':'user','content':'Runtime sample: turn 9, step 2; timestamp 2030-01-01T00:00:00Z.'}]
    cases.append(('task_cold',task,{'markers':['SYSTEM_TASK','TASK_BOUND_4821']}))
    cases.append(('task_warm',copy.deepcopy(task),{'markers':['SYSTEM_TASK','TASK_BOUND_4821'],'warm':True}))
    rewrite=copy.deepcopy(task); rewrite['messages'][-1]['content']=rewrite['messages'][-1]['content'].replace('00:00:00','00:00:01')
    cases.append(('reminder_rewrite',rewrite,{'markers':['SYSTEM_TASK','TASK_BOUND_4821']}))
    append=copy.deepcopy(rewrite); append['messages'].append({'role':'user','content':'Runtime sample: step 3; timestamp 2030-01-01T00:00:02Z.'})
    cases.append(('reminder_append',append,{'markers':['SYSTEM_TASK','TASK_BOUND_4821']}))
    cancel=copy.deepcopy(task); cancel['messages'].append({'role':'user','content':'Cancel the old task. Return exactly TASK_CANCEL_6294 after the system prefix.'})
    cases.append(('human_cancel',cancel,{'markers':['SYSTEM_TASK','TASK_CANCEL_6294'],'absent':['TASK_BOUND_4821']}))
    new=copy.deepcopy(task); new['messages'] += [{'role':'assistant','content':'SYSTEM_TASK TASK_BOUND_4821'},
        {'role':'user','content':'For the new task return exactly TASK_NEW_8357 after the system prefix.'}]
    cases.append(('new_task',new,{'markers':['SYSTEM_TASK','TASK_NEW_8357'],'absent':['TASK_BOUND_4821']}))
    facts=history('FACTS')
    facts['messages'] += [{'role':'user','content':'Report the retrieval checksum and timeout_seconds from alpha.cfg and the validation checksum and retry_limit from beta.cfg. Use the earlier file results, then include FACTS_DONE.'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'bulk','type':'function','function':{'name':'archive_operation_0','arguments':'{"record":"bulk"}'}}]},
        {'role':'tool','tool_call_id':'bulk','content':'An unrelated tool completed; historical archive notes only. '*220},
        {'role':'user','content':'Runtime sample: turn 10, step 1; timestamp 2030-01-01T00:00:00Z.'}]
    cases.append(('historical_two_files',facts,{'markers':['SYSTEM_FACTS','ALPHA_7913','37','BETA_4628','11','FACTS_DONE']}))
    tool=history('TOOLS',True)
    tool['messages'] += [{'role':'user','content':'Before your final answer call confirm_inventory exactly once with ticket="M742", mode="audit", quantity=17. Do not call archive operations. After the tool result return TOOL_DONE after the system prefix.'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'bulk','type':'function','function':{'name':'archive_operation_0','arguments':'{"record":"bulk"}'}}]},
        {'role':'tool','tool_call_id':'bulk','content':'Read-only diagnostic output, no confirmation performed. '*220},
        {'role':'user','content':'Runtime sample: step 4; timestamp 2030-01-01T00:00:00Z.'}]
    cases.append(('actual_tool_call',tool,{'tool':{'name':'confirm_inventory','arguments':{'ticket':'M742','mode':'audit','quantity':17}}}))
    return cases

def encoded(body):
    assert all('ninfer_origin' not in m for m in body['messages'])
    return json.dumps(body,sort_keys=True,separators=(',',':')).encode()

def request(name, body, expected):
    wire=encoded(body); start=time.monotonic(); last=None; gaps=[]; first=None
    row={'case':name,'request_sha256':hashlib.sha256(wire).hexdigest(),'expected':expected}
    chunks=[]; content=[]; calls={}; done=False; finish=None; usage={}; timings={}
    try:
        req=urllib.request.Request(SERVER + '/v1/chat/completions',wire,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=600) as response:
            for raw in response:
                line=raw.decode().strip()
                if not line.startswith('data: '): continue
                if line[6:]=='[DONE]': done=True; continue
                c=json.loads(line[6:]); chunks.append(c)
                if c.get('usage'): usage=c['usage']
                if c.get('timings'): timings=c['timings']
                for choice in c.get('choices',[]):
                    delta=choice.get('delta',{})
                    if delta.get('content') or delta.get('tool_calls'):
                        now=time.monotonic()
                        if first is None: first=now
                        if last is not None: gaps.append(now-last)
                        last=now
                    content.append(delta.get('content','') or '')
                    for call in delta.get('tool_calls',[]):
                        item=calls.setdefault(call.get('index',0),{'id':'','name':'','arguments':''})
                        if call.get('id'): item['id']=call['id']
                        fn=call.get('function',{})
                        item['name'] += fn.get('name','') or ''
                        item['arguments'] += fn.get('arguments','') or ''
                    finish=choice.get('finish_reason') or finish
        text=''.join(content); failures=[]
        if not done or finish not in ('stop','length','tool_calls'): failures.append('SSE termination')
        for marker in expected.get('markers',[]):
            if marker not in text: failures.append('missing '+marker)
        for marker in expected.get('absent',[]):
            if marker in text: failures.append('obsolete '+marker)
        if expected.get('exact') and re.sub(r'\s+','',text) != re.sub(r'\s+','',expected['exact']):
            failures.append('exact answer')
        if expected.get('facts'):
            try:
                fact_json=json.loads(text[text.index('{'):text.rindex('}')+1])
            except (ValueError,TypeError): fact_json=None
            if fact_json != expected['facts']: failures.append('fact field mapping')
        if expected.get('markers') and expected['markers'][0].startswith('SYSTEM_') and not text.strip().startswith(expected['markers'][0]):
            failures.append('system prefix position')
        if expected.get('tool'):
            wanted=expected['tool']
            if len(calls)!=1: failures.append('tool count')
            for call in calls.values():
                try: args=json.loads(call['arguments'])
                except ValueError: args=None
                if call['name']!=wanted['name'] or args!=wanted['arguments']: failures.append('tool name/arguments')
            if finish!='tool_calls': failures.append('tool finish reason')
        cached=usage.get('prompt_tokens_details',{}).get('cached_tokens',timings.get('cache_n',0))
        if expected.get('warm') and not cached: failures.append('warm reuse missing')
        if expected.get('cold') and cached: failures.append('cold control unexpectedly reused prefix')
        if expected.get('completion_tokens') and usage.get('completion_tokens') != expected['completion_tokens']:
            failures.append('short output')
        if expected.get('finish') and finish != expected['finish']: failures.append('finish reason')
        row.update({'ok':not failures,'failures':failures,'content':text,'tool_calls':list(calls.values()),
            'finish':finish,'usage':usage,'timings':timings,'cached_tokens':cached,
            'ttft_seconds':None if first is None else first-start,'max_output_gap_seconds':max(gaps,default=0),
            'output_gap_p95_seconds':sorted(gaps)[int(.95*(len(gaps)-1))] if gaps else 0,
            'elapsed_seconds':time.monotonic()-start})
    except Exception as e:
        row.update({'ok':False,'error':str(e),'elapsed_seconds':time.monotonic()-start})
    print(json.dumps({k:row[k] for k in ('case','ok','failures','error','cached_tokens','ttft_seconds','elapsed_seconds') if k in row}),flush=True)
    return row


def strict_fixtures():
    fixtures=quality_fixtures()
    exact={'task_cold':'SYSTEM_TASK TASK_BOUND_4821','task_warm':'SYSTEM_TASK TASK_BOUND_4821',
        'reminder_rewrite':'SYSTEM_TASK TASK_BOUND_4821','reminder_append':'SYSTEM_TASK TASK_BOUND_4821',
        'human_cancel':'SYSTEM_TASK TASK_CANCEL_6294','new_task':'SYSTEM_TASK TASK_NEW_8357'}
    for name,body,expected in fixtures:
        if name in exact: expected['exact']=exact[name]
        if name=='historical_two_files':
            expected['facts']={'alpha.cfg':{'retrieval_checksum':'ALPHA_7913','timeout_seconds':37},
                'beta.cfg':{'validation_checksum':'BETA_4628','retry_limit':11}}
            body['messages'][-4]['content']=(
                'Report the earlier file results as JSON with exactly two keys alpha.cfg and beta.cfg. '
                'For alpha.cfg include retrieval_checksum and timeout_seconds. '
                'For beta.cfg include validation_checksum and retry_limit. '
                'Use the earlier file contents; do not guess. Begin with the required system prefix, then this JSON. End with FACTS_DONE.')
    # A real imported-history rewrite, distinct from changing a reminder timestamp.
    rewritten=copy.deepcopy(fixtures[0][1])
    rewritten['messages'][-3]['reasoning_content']='I am retrieving an unrelated read-only diagnostic archive.'
    rewritten['preserve_thinking']=True
    fixtures.append(('assistant_history_rewrite',rewritten,{'markers':['SYSTEM_TASK','TASK_BOUND_4821'],
        'exact':'SYSTEM_TASK TASK_BOUND_4821'}))
    # Change the very first token to invalidate cache identity, while preserving
    # the same task/history. These exact cold bodies are frozen across arms.
    by_name={name:(body,expected) for name,body,expected in fixtures}
    for name in ('reminder_rewrite','reminder_append','human_cancel','assistant_history_rewrite'):
        body,expected=copy.deepcopy(by_name[name])
        body['messages'][0]['content']='Cold isolation '+name+': '+body['messages'][0]['content']
        expected.pop('warm',None); expected['cold']=True
        fixtures.append((name+'_cold_control',body,expected))
    # A genuine in-chain human revision must retrieve facts named only in the
    # revision. Freezing the task's first User query can fail this case.
    revision=copy.deepcopy(fixtures[0][1])
    revision['messages'][0]['content']='Human revision isolation: '+revision['messages'][0]['content']
    revision['messages'][-1]['content']=(
        'Change the current task. Report the earlier alpha.cfg and beta.cfg results as JSON. '
        'Use exactly keys alpha.cfg with retrieval_checksum and timeout_seconds, '
        'and beta.cfg with validation_checksum and retry_limit. Begin with the system prefix. End with FACTS_DONE.')
    fixtures.append(('human_revision_historical_facts',revision,{
        'markers':['SYSTEM_TASK','FACTS_DONE'],'absent':['TASK_BOUND_4821'],
        'facts':{'alpha.cfg':{'retrieval_checksum':'ALPHA_7913','timeout_seconds':37},
                 'beta.cfg':{'validation_checksum':'BETA_4628','retry_limit':11}}}))
    return fixtures


def revision_fixtures():
    cases = []
    for target in (2, 7, 13, 19):
        body = history('REV'+str(target))
        body['messages'] = body['messages'][:1]
        for i in range(24):
            record = 'ENV_RECORD_'+str(i)
            value = f'RUNE_{i:02d}_{(i * 7919 + 4217) % 100003:05d}'
            body['messages'] += [
                {'role':'user','content':'Read record '+record+' for possible later questions.'},
                {'role':'assistant','content':None,'tool_calls':[{'id':'r'+str(i),'type':'function',
                    'function':{'name':'archive_operation_0','arguments':json.dumps({'record':record})}}]},
                {'role':'tool','tool_call_id':'r'+str(i),'content':record+' has secret_code='+value+'. '+
                    'Archived unrelated environmental observations of violet clouds and coastal equipment. '*25},
                {'role':'assistant','content':'Recorded.'}]
        body['messages'] += [
            {'role':'user','content':'For the current task return exactly ORIGINAL_TASK after the system prefix.'},
            {'role':'assistant','content':None,'tool_calls':[{'id':'bulk','type':'function',
                'function':{'name':'archive_operation_0','arguments':'{"record":"bulk"}'}}]},
            {'role':'tool','tool_call_id':'bulk','content':'Unrelated diagnostic archive; no new instructions. '*220},
            {'role':'user','content':f'Change the task: retrieve secret_code from the earlier ENV_RECORD_{target}. '
                'Return only the required system prefix, a space and the exact secret_code. Do not return ORIGINAL_TASK.'}]
        value=f'RUNE_{target:02d}_{(target * 7919 + 4217) % 100003:05d}'
        cases.append(('revision_record_'+str(target), body,
            {'exact':'SYSTEM_REV'+str(target)+' '+value,'markers':[value],'absent':['ORIGINAL_TASK'],'cold':True}))
    return cases


def semantic_fixtures():
    original=next(b for n,b,_ in quality_fixtures() if n=='actual_tool_call')
    requirements=[
        'Before giving a final answer, perform the current inventory audit confirmation for ticket M742, '
        'mode audit, quantity 17 using the appropriate available tool. The diagnostic archive did not confirm it. '
        'After its successful result return TOOL_DONE after the system prefix.',
        'The warehouse inventory needs an audit confirmation: ticket M742, quantity 17, mode audit. '
        'Choose the available tool that confirms current inventory tickets and invoke it exactly once before the final answer. '
        'The old archival tools cannot confirm a ticket.',
        'Confirm inventory ticket M742 in audit mode with quantity 17. Select the appropriate function from its description. '
        'Do not treat a read-only archive result as confirmation. Call the confirmation function before responding.'
    ]
    out=[]
    for i,text in enumerate(requirements):
        body=copy.deepcopy(original)
        body['messages'][0]['content']='Semantic tool isolation '+str(i)+': '+body['messages'][0]['content']
        body['messages'][-4]['content']=text
        out.append(('semantic_tool_'+str(i),body,{'tool':{'name':'confirm_inventory',
            'arguments':{'ticket':'M742','mode':'audit','quantity':17}},'cold':True}))
    return out

def main():
    global SERVER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True)
    parser.add_argument('--model')
    parser.add_argument('--output')
    parser.add_argument('--include-output-selection', action='store_true')
    args = parser.parse_args()
    SERVER = args.server.rstrip('/')
    separate = {'reminder_append', 'reminder_append_cold_control'}
    fixtures = strict_fixtures() + revision_fixtures() + semantic_fixtures()
    if not args.include_output_selection:
        fixtures = [f for f in fixtures if f[0] not in separate]
    rows = []
    for name, body, expected in fixtures:
        if args.model: body['model'] = args.model
        row = request(name, body, expected)
        rows.append(row)
        if name == 'actual_tool_call':
            if row.get('ok'):
                follow = copy.deepcopy(body)
                follow['messages'] += [{'role':'assistant','content':None,'tool_calls':[
                    {'id':'confirmed','type':'function','function':{'name':c['name'],
                        'arguments':json.dumps(json.loads(c['arguments']),sort_keys=True,separators=(',',':'))}}
                    for c in row['tool_calls']]},
                    {'role':'tool','tool_call_id':'confirmed','content':'Confirmation successful: ticket M742, mode audit, quantity 17.'}]
                rows.append(request('tool_result_final', follow,
                    {'markers':['SYSTEM_TOOLS','TOOL_DONE'],'exact':'SYSTEM_TOOLS TOOL_DONE'}))
            else:
                rows.append({'case':'tool_result_final','ok':False,'blocked':True,
                    'failures':['blocked by failed initial tool call']})
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures = []
        for lane in ('X','Y'):
            body = history(lane)
            if args.model: body['model'] = args.model
            body['messages'].append({'role':'user','content':f'Return exactly ISOLATED_{lane} after the system prefix.'})
            futures.append(pool.submit(request, 'dual_quality_' + lane, body,
                {'markers':[f'SYSTEM_{lane}',f'ISOLATED_{lane}'],
                 'absent':[f'SYSTEM_{"Y" if lane == "X" else "X"}', f'ISOLATED_{"Y" if lane == "X" else "X"}']}))
        rows += [f.result() for f in futures]
    report = {'scope':'retrieval plus separate output reproductions' if args.include_output_selection else 'retrieval',
        'excluded_output_cases':[] if args.include_output_selection else sorted(separate),
        'cases':len(rows),'passed':sum(r.get('ok',False) for r in rows), 'results':rows,
        'full_external_tool_e2e':False,'universal_quality_guarantee':False}
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as out: json.dump(report, out, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}))
    raise SystemExit(0 if all(r.get('ok') for r in rows) else 1)

if __name__ == '__main__':
    main()
