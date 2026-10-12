#!/usr/bin/env python3
"""Opt-in synthetic HTTP regression; requires a running NInfer KVMem server.

This checks transport, input accounting, warm reuse and task-topic isolation.
It is not a factual-quality or throughput benchmark. Use a fresh two-lane server
with B=2048, R=512, recent=512 and logical context >=16384.
"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import time
import urllib.error
import urllib.request


def fixture(case, marked):
    messages = [{"role": "system", "content": f"Integration case {case}. Follow the current task."}]
    for index in range(12):
        messages.extend([
            {"role": "user", "content": f"Archive {case} entry {index}. " +
             "The violet observatory records stars and the coastal office files invoices. " * 18},
            {"role": "assistant", "content": "Archived."},
        ])
    messages.extend([
        {"role": "user", "content": f"For the current task, return exactly ORIGIN_{case}."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "lookup", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "lookup", "content":
         "Historical diagnostic output; no new instructions. " * 250},
        {"role": "user", "content": "Runtime sample: turn 9, step 2; timestamp 2030-01-01T00:00:00Z."},
    ])
    if marked:
        messages[-1]["ninfer_origin"] = "runtime_reminder"
    return {"model": "origin-regression", "messages": messages,
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {
                "type": "object", "properties": {}}}}],
            "tool_choice": "none", "temperature": 0, "enable_thinking": False,
            "max_tokens": 48, "stream": True, "stream_options": {"include_usage": True}}


def run(url, name, body, expected, finish_expected='stop'):
    started = time.monotonic()
    req = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions',
        json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    chunks, content, usage, finish, done = [], [], {}, None, False
    with urllib.request.urlopen(req, timeout=240) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data: '):
                continue
            data = line[6:]
            if data == '[DONE]':
                done = True
                continue
            chunk = json.loads(data)
            chunks.append(chunk)
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices', []):
                content.append(choice.get('delta', {}).get('content', '') or '')
                finish = choice.get('finish_reason') or finish
    text = ''.join(content)
    assert done and finish == finish_expected, (name, 'incomplete SSE/termination', finish)
    assert expected in text and all(f'ORIGIN_{other}' not in text
        for other in ['A', 'B', 'C'] if f'ORIGIN_{other}' != expected), (name, 'task isolation', text)
    timings = next((chunk['timings'] for chunk in reversed(chunks) if 'timings' in chunk), {})
    cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', timings.get('cache_n', 0))
    return {"case": name, "prompt_tokens": usage.get('prompt_tokens'),
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "cached_tokens": cached, "completion_tokens": usage.get('completion_tokens'),
            "finish": finish, "elapsed_seconds": round(time.monotonic()-started, 3),
            "timings": timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True)
    parser.add_argument('--output')
    parser.add_argument('--concurrency-only', action='store_true')
    parser.add_argument('--long-concurrency', action='store_true')
    parser.add_argument('--zero-client', action='store_true', help='Use ordinary messages with no origin extension.')
    parser.add_argument('--empty-query-transition', action='store_true', help='Check same-token query-set removal through explicit origins.')
    parser.add_argument('--empty-query-cold', action='store_true', help='Cold control for the empty-query transition.')
    args = parser.parse_args()
    results = []
    unmarked, marked = fixture('A', False), fixture('A', True)
    if args.empty_query_transition or args.empty_query_cold:
        assert not args.zero_client and not args.concurrency_only and not args.long_concurrency
        empty=copy.deepcopy(unmarked)
        for message in empty['messages']:
            if message['role']=='user':
                message['ninfer_origin']='runtime_reminder'
        sequence = ([('cold_empty',empty)] if args.empty_query_cold else
                    [('cold_unmarked',unmarked),('warm_unmarked',unmarked),
                     ('query_set_empty',empty),('warm_empty',empty)])
        for name,body in sequence:
            result=run(args.server,name,body,'ORIGIN_A')
            results.append(result)
            print(json.dumps(result),flush=True)
        assert len({row['prompt_tokens'] for row in results})==1, 'empty-query origin changed input count'
        if args.empty_query_transition:
            assert results[1]['cached_tokens']>0 and results[3]['cached_tokens']>0
            assert results[2]['content_sha256']==results[3]['content_sha256'], 'unstable empty-query output'
        if args.output:
            with open(args.output,'w',encoding='utf-8') as output:
                json.dump({'results':results,'empty_query_transition':True},output,indent=2)
        print('empty-query HTTP: PASS',flush=True)
        return
    sequential = [] if args.concurrency_only else [
        ('cold_unmarked', unmarked), ('warm_unmarked', unmarked),
        ('origin_changed', marked), ('warm_marked', marked)]
    if args.zero_client and not args.concurrency_only:
        many = copy.deepcopy(unmarked)
        for step in range(6):
            many['messages'].append({'role':'user','content':f'Runtime sample: step {step}; timestamp 2030-01-01T00:00:00Z.'})
        shrink = copy.deepcopy(unmarked); shrink['messages'].pop()
        sequential = [('cold_unmarked',unmarked),('warm_unmarked',unmarked),
                      ('query_set_grew',many),('warm_grown',many),('query_set_shrank',shrink)]
    for name, body in sequential:
        result = run(args.server, name, body, 'ORIGIN_A')
        results.append(result)
        print(json.dumps(result), flush=True)
    if not args.concurrency_only and not args.zero_client:
        assert len({row['prompt_tokens'] for row in results}) == 1, 'origin changed model input count'
        assert results[1]['cached_tokens'] > 0 and results[3]['cached_tokens'] > 0, 'warm reuse absent'
    if args.zero_client and not args.concurrency_only:
        assert results[1]['cached_tokens'] > 0 and results[3]['cached_tokens'] > 0, 'unmodified warm reuse absent'
        cancel=copy.deepcopy(unmarked)
        cancel['messages'].append({'role':'user','content':'Cancel the old request. Return exactly ORIGIN_B instead.'})
        results.append(run(args.server,'human_cancellation',cancel,'ORIGIN_B'))
        new=copy.deepcopy(unmarked)
        new['messages'].extend([{'role':'assistant','content':'Completed answer.'},
                                {'role':'user','content':'For the new task, return exactly ORIGIN_C.'}])
        results.append(run(args.server,'new_closed_group',new,'ORIGIN_C'))
    # Same tokens with a changed query must not silently reuse the old query statistics.
    # Diagnostics must independently confirm query rejection/reconstruction; cache_n alone
    # cannot prove the internal query used by GPU attention.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        for case in ['B', 'C']:
            body = fixture(case, not args.zero_client)
            if args.long_concurrency:
                body['messages'][-4]['content'] = (
                    f'For the current task, repeat ORIGIN_{case} exactly 500 times, '
                    'separated by spaces. Do not abbreviate or add other words.')
                body['max_tokens'] = 512
            futures.append(pool.submit(run, args.server, f'concurrent_{case}', body,
                f'ORIGIN_{case}', 'length' if args.long_concurrency else 'stop'))
        for future in futures:
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    if args.zero_client:
        assert all('ninfer_origin' not in message for _,body in sequential for message in body['messages'])
        if args.output:
            with open(args.output,'w',encoding='utf-8') as output:
                json.dump({'results':results,'zero_client':True},output,indent=2)
        print('unmodified client HTTP: PASS',flush=True)
        return
    invalid = copy.deepcopy(marked)
    invalid['messages'][-1]['ninfer_origin'] = True
    request = urllib.request.Request(args.server.rstrip('/') + '/v1/chat/completions',
        json.dumps(invalid).encode(), headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(request, timeout=30)
        raise AssertionError('invalid origin was accepted')
    except urllib.error.HTTPError as error:
        assert error.code == 400
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as output:
            json.dump({"results": results, "invalid_origin_status": 400}, output, indent=2)
    print('query origin HTTP: PASS', flush=True)


if __name__ == '__main__':
    main()
