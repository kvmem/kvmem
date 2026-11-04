"""Content-cache HTTP regression against an already configured, dedicated server.

The launcher owns GPU selection, model loading and all cache budgets. Pass those
settings explicitly for the evidence manifest; this script never starts a GPU
process or changes server configuration. Native logs must be readable locally.
Example (substitute actual launch settings):
  python tests/content-cache-regression.py --backend llama --url http://127.0.0.1:8080 \
    --model MODEL_ALIAS --gpu GPU_UUID --budget-tokens B --reserve-tokens R \
    --host-mib H --sessions N --native-log SERVER_LOG --out NEW_DIRECTORY

Use --replay DIRECTORY to recheck saved evidence without HTTP, or --self-test for
the model-free evidence/parser tests. Timings include tracing and are diagnostic.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid


TOOL = {'name': 'record_review', 'description': 'Record the current review status.',
        'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def trace_since(path, offset):
    with path.open('rb') as stream:
        stream.seek(0, 2)
        assert stream.tell() >= offset, 'native log rotated during the test'
        stream.seek(offset)
        return stream.read()


def response_parts(response, protocol):
    usage = response['usage']
    if protocol == 'responses':
        text = ''.join(part.get('text', '') for item in response.get('output', [])
                       if item.get('type') == 'message' for part in item.get('content', [])
                       if part.get('type') == 'output_text')
        return {'role': 'assistant', 'content': text}, usage['input_tokens'], usage.get('input_tokens_details', {}).get('cached_tokens')
    return response['choices'][0]['message'], usage['prompt_tokens'], usage.get('prompt_tokens_details', {}).get('cached_tokens', usage.get('prompt_cache_hit_tokens'))


def sse_events(lines):
    event, data = '', []
    for raw in lines:
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                yield event, '\n'.join(data)
            event, data = '', []
        elif line.startswith('data:'):
            data.append(line[5:].removeprefix(' '))
        elif line.startswith('event:'):
            event = line[6:].strip()
    if data:
        yield event, '\n'.join(data)


def parse_sse(lines, protocol):
    """Return real terminal usage; never infer it from deltas or cached quotes."""
    identity, model, usage, terminal, done = None, None, None, None, False
    message, calls = {'role': 'assistant', 'content': ''}, {}
    for event_name, data in sse_events(lines):
        if data == '[DONE]':
            done = True
            continue
        assert not done, 'SSE data arrived after [DONE]'
        event = json.loads(data)
        if not isinstance(event, dict) or 'error' in event or event_name == 'error':
            raise RuntimeError('SSE error: ' + data)
        if protocol == 'responses':
            kind = event.get('type', event_name)
            if kind in ('response.failed', 'error'):
                raise RuntimeError('Responses SSE failure: ' + data)
            response = event.get('response', {})
            if response.get('id'):
                assert identity in (None, response['id']), 'Responses SSE mixed response IDs'
                identity = response['id']
            if kind in ('response.completed', 'response.incomplete'):
                assert terminal is None, 'duplicate Responses terminal event'
                assert response.get('id') == identity and identity, 'Responses terminal event lacks its response ID'
                terminal, usage = response, response.get('usage')
            continue
        if event.get('id'):
            assert identity in (None, event['id']), 'Chat SSE mixed completion IDs'
            identity = event['id']
        model = event.get('model', model)
        for choice in event.get('choices', []):
            assert choice.get('index', 0) == 0, 'fixture expects one Chat choice'
            delta = choice.get('delta') or {}
            assert delta.get('role') in (None, 'assistant'), 'unexpected Chat delta role'
            for field in ('content', 'reasoning_content', 'refusal'):
                text = delta.get(field)
                if text is not None:
                    assert isinstance(text, str), 'Chat text delta must be a string or null'
                    message[field] = message.get(field, '') + text
            for fragment in delta.get('tool_calls', []):
                call = calls.setdefault(fragment['index'], {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                call['id'] += fragment.get('id') or ''
                for field in ('name', 'arguments'):
                    call['function'][field] += fragment.get('function', {}).get(field) or ''
            if choice.get('finish_reason') is not None:
                terminal = choice['finish_reason']
        if event.get('usage') is not None and terminal is not None:
            usage = event['usage']
    assert identity and terminal is not None, 'SSE lacks response ID or terminal event'
    counts = ('input_tokens', 'output_tokens') if protocol == 'responses' else ('prompt_tokens', 'completion_tokens')
    assert isinstance(usage, dict) and all(type(usage.get(key)) is int and usage[key] >= 0 for key in counts), 'SSE lacks genuine terminal usage'
    if protocol == 'responses':
        return terminal
    assert done, 'Chat SSE ended without [DONE]'
    if calls:
        message['tool_calls'] = [calls[index] for index in sorted(calls)]
    return {'id': identity, 'object': 'chat.completion', 'model': model, 'usage': usage,
            'choices': [{'index': 0, 'message': message, 'finish_reason': terminal}]}


def request_body(messages, key, model, limit, protocol, extra):
    body = dict(extra, model=model, temperature=0)
    if protocol == 'chat':
        body.update(messages=copy.deepcopy(messages), max_tokens=limit,
                    tools=[{'type': 'function', 'function': TOOL}])
    else:
        items = []
        for message in messages:
            if message['role'] == 'tool':
                items.append({'type': 'function_call_output', 'call_id': message['tool_call_id'], 'output': message['content']})
            else:
                if message.get('content'):
                    items.append({'role': message['role'], 'content': message['content']})
                for call in message.get('tool_calls', []):
                    items.append({'type': 'function_call', 'call_id': call['id'], **call['function']})
        body.update(input=items, max_output_tokens=limit, store=False,
                    tools=[{'type': 'function', **TOOL}])
    if key is not None:
        body['prompt_cache_key'] = key
    return body


def monitor_rows(snapshot):
    return snapshot.get('active', []) + snapshot.get('recent', [])


def match_observation(rows, response_id, client_id, before_ids, prompt, limit, backend):
    """Use wire IDs, or Poly's unique limit+prompt in a new monitor record.

    Poly currently does not publish its wire response ID. Each request therefore
    gets a distinct output limit, and old monitor IDs are excluded. Ambiguous or
    stale records are rejected; array order is never used as request identity.
    """
    direct = [r for r in rows if (response_id and r.get('native_request_id') == response_id)
              or (client_id and r.get('client_request_id') == client_id)]
    matches = direct
    method = 'wire_id'
    if not matches and backend == 'polystrata':
        matches = [r for r in rows if r.get('id') not in before_ids
                   and r.get('limit') == limit and r.get('prompt_tokens') == prompt]
        method = 'new_monitor_id+unique_limit+prompt'
    if len(matches) > 1:
        raise AssertionError('ambiguous native request observation')
    return (matches[0], method) if matches else (None, None)


def check_observation(row, hot_min, parent_min):
    observed = row['observation']
    actual, prompt = observed.get('actual_cached_tokens'), row['prompt_tokens']
    assert type(prompt) is int and prompt > 0, 'invalid prompt token count'
    assert observed.get('cache_semantics_revision') == 2, 'native cache semantics revision 2 is required'
    assert type(actual) is int and 0 <= actual <= prompt, 'native actual is missing/invalid; an admission quote is insufficient'
    path, source = observed.get('restore_path'), observed.get('cache_source')
    assert path == ('miss' if actual == 0 else 'snapshot' if source == 'shared' else 'continuation' if source == 'own' else None), 'native restore path/source disagree'
    assert actual == 0 or source in ('own', 'shared'), 'unknown successful native source'
    assert actual != 0 or source == 'miss', 'native zero-hit source must be miss'
    assert observed.get('prompt_tokens') == prompt, 'native observation belongs to a different prompt'
    stage = row['stage']
    threshold = hot_min if stage == 'hot' else parent_min if stage == 'parent' else 0
    if stage != 'document':
        assert actual > 0, f'{row["label"]}: native cache miss (check saved physical capacity evidence)'
        assert actual / prompt >= threshold, f'{row["label"]}: native hit {actual}/{prompt} below {threshold:.1%}'
    return {'label': row['label'], 'stage': stage, 'key_mode': row['key_mode'],
            'native_actual': actual, 'prompt_tokens': prompt, 'hit_ratio': actual / prompt,
            'usage_cached_quote': row.get('usage_cached_quote'), 'restore_path': path}


def check_tool_trace(backend, trace):
    assert trace.strip(), 'tool continuation requires isolated native trace evidence'
    assert not re.search(r'KVMEM_QUERY_REPLAY(?!_SKIP)', trace), 'tool continuation replayed a cached/old Query'
    if backend == 'polystrata':
        policies = re.findall(r'KVMEM_QUERY source=(\w+) recall=(\d+)', trace)
        assert policies and all(source == 'tool_or_continuation' and recall == '0' for source, recall in policies), 'missing/contradictory append/resident-only tool observation'
        spans = re.findall(r'KVMEM_REUSE [^\r\n]*query_begin=(\d+) query_end=(\d+)', trace)
        assert spans and all(a == b == '0' for a, b in spans), 'tool continuation used Query rows'
    elif backend == 'radiance':
        spans = re.findall(r'KVMEM_QUERY_SPAN [^\r\n]*begin=(-?\d+) end=(-?\d+) proven=(\d+)', trace)
        assert spans and all(a == b == '-1' and proven == '0' for a, b, proven in spans), 'tool continuation has a nonempty/unproven Query span'
        pressure = re.findall(r'KVMEM_RESELECT [^\r\n]*restore=(\d+) mode=([^\s]+)', trace)
        assert all(n == '0' and mode in ('prefill-pressure-attention', 'decode-pressure-attention') for n, mode in pressure), 'tool Pressure restored cold history'
    else:
        lines = [line for line in trace.splitlines() if 'path=append_pressure' in line]
        assert lines, 'tool continuation did not report append_pressure'
        if backend == 'llama':
            assert any(re.search(r'query_source=none .*q_rows=0 replay_rows=0', line) for line in lines), 'tool continuation used cold Query features'
            assert not re.search(r'query_source=user|q_rows=[1-9]|replay_rows=[1-9]|query_reused=[1-9]', trace), 'tool continuation reused/captured Query features'
        else:
            assert any('reason=no_query' in line and 'planned_replay_rows=0' in line and 'capture_query=0' in line for line in lines), 'tool continuation captured/replayed Query rows'
            assert not re.search(r'capture_query=[1-9]|planned_replay_rows=[1-9]', trace), 'tool continuation captured/replayed Query rows'
    return {'append_resident_only': True,
            'pressure_exercised': bool(re.search(r'KVMEM_(?:RESELECT|DECODE_PRESSURE|PRESSURE)\b', trace))}


def check_native_pair(pair, members):
    sample = pair.get('native_active_sample')
    assert sample, f'pair {pair["pair"]}: native concurrency unqualified; no simultaneous native-active sample'
    expected = {r['observation']['id'] for r in members}
    proven = {r['id'] for r in sample['active'] if r.get('status') in ('reading', 'generating')
              and r.get('cache_semantics_revision') == 2 and type(r.get('actual_cached_tokens')) is int}
    assert len(expected) == 2 and expected <= proven, 'native concurrent sample does not contain both correlated requests'


def validate(record, trace):
    settings, rows = record['settings'], record['requests']
    assert len(rows) == 6 + 2 * settings['pairs'], 'incomplete workload; raw evidence is retained'
    assert sum(r['stage'] == 'first_question' for r in rows) == 3
    assert all(sum(r['stage'] == stage for r in rows) == 1 for stage in ('document', 'parent', 'tool'))
    hot = [r for r in rows if r['stage'] == 'hot']
    assert len(hot) == 2 * settings['pairs'] and {r['key_mode'] for r in hot} == {'same', 'different', 'none'}
    for pair in range(settings['pairs']):
        members = [r for r in hot if r['pair'] == pair]
        assert len(members) == 2
        assert max(r['started_monotonic'] for r in members) < min(r['ended_monotonic'] for r in members), f'pair {pair}: HTTP requests did not overlap'
        check_native_pair(record['pairs'][pair], members)
    results = [check_observation(r, settings['hot_min'], settings['parent_min']) for r in rows]
    return {'status': 'pass', 'requests': results, 'tool': check_tool_trace(settings['backend'], trace),
            'timing_scope': 'diagnostic HTTP wall time with native tracing; no performance comparison'}


class Probe:
    def __init__(self, args):
        self.args, self.rows, self.pairs, self.lock = args, [], [], threading.Lock()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.tag = 'content-cache-' + uuid.uuid4().hex
        self.before_ids = {r['id'] for r in monitor_rows(self.call(args.monitor_path))}

    def call(self, path, body=None, client_id=None, timeout=None):
        headers = {'Content-Type': 'application/json'}
        if client_id:
            headers['X-Request-ID'] = client_id
        request = urllib.request.Request(self.args.url.rstrip('/') + path,
                    None if body is None else json.dumps(body).encode('utf-8'), headers)
        try:
            with self.opener.open(request, timeout=timeout or self.args.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode('utf-8', errors='replace')) from error

    def stream_generation(self, body, stem, client_id):
        path = '/v1/responses' if self.args.protocol == 'responses' else '/v1/chat/completions'
        request = urllib.request.Request(self.args.url.rstrip('/') + path, json.dumps(body).encode('utf-8'),
                    {'Content-Type': 'application/json', 'X-Request-ID': client_id})
        with stem.with_suffix('.response.sse').open('wb') as wire:
            try:
                with self.opener.open(request, timeout=self.args.timeout) as response:
                    def lines():
                        for line in response:
                            wire.write(line)
                            yield line
                    return parse_sse(lines(), self.args.protocol)
            except urllib.error.HTTPError as error:
                raw = error.read(); wire.write(raw)
                raise RuntimeError(raw.decode('utf-8', errors='replace')) from error

    def sample_pair(self, pair, limits, stop):
        # A native actual is emitted only after materialization. Requiring two
        # such active records excludes same-key queued/serialized HTTP requests.
        result = {'pair': pair, 'native_active_sample': None}
        self.pairs.append(result)
        with (self.args.out / f'pair-{pair}.active.jsonl').open('w', encoding='utf-8') as output:
            while not stop.is_set():
                try:
                    snapshot = self.call(self.args.monitor_path, timeout=min(2, self.args.timeout))
                    active = [r for r in snapshot.get('active', []) if r.get('limit') in limits]
                    sample = {key: snapshot.get(key) for key in ('server_instance', 'sampled_at_ms')}
                    sample['active'] = active
                    output.write(json.dumps(sample) + '\n')
                    eligible = [r for r in active if r.get('status') in ('reading', 'generating')
                                and r.get('cache_semantics_revision') == 2 and type(r.get('actual_cached_tokens')) is int]
                    if len({r['id'] for r in eligible}) == 2:
                        result['native_active_sample'] = sample
                        break
                except Exception as error:
                    result['last_sampling_error'] = str(error)
                stop.wait(self.args.monitor_interval)

    def turn(self, index, label, history, stage, key_mode='same', pair=None, barrier=None):
        args, limit = self.args, self.args.max_tokens + index
        key = None if key_mode == 'none' else self.tag if key_mode == 'same' else self.tag + '-' + label
        body = request_body(history, key, args.model, limit, args.protocol, args.request_options)
        if args.stream:
            body['stream'] = True
            if args.protocol == 'chat':
                body['stream_options'] = {'include_usage': True}
        stem = args.out / f'{index:02d}-{label}'
        dump(stem.with_suffix('.request.json'), body)
        if barrier:
            barrier.wait(timeout=args.timeout)
        started = time.monotonic()
        client_id = self.tag + '-' + label
        response = self.stream_generation(body, stem, client_id) if args.stream else self.call(
            '/v1/responses' if args.protocol == 'responses' else '/v1/chat/completions', body, client_id)
        ended = time.monotonic()
        elapsed = ended - started
        dump(stem.with_suffix('.response.json'), response)
        message, prompt, quote = response_parts(response, args.protocol)
        assert message.get('content'), f'{label}: response has no text'
        deadline = time.monotonic() + args.monitor_timeout
        while True:
            snapshot = self.call(args.monitor_path)
            observation, method = match_observation(monitor_rows(snapshot), response.get('id'), self.tag + '-' + label,
                                        self.before_ids, prompt, limit, args.backend)
            if observation and observation.get('actual_cached_tokens') is not None and observation.get('status') == 'completed':
                break
            if time.monotonic() >= deadline:
                dump(stem.with_suffix('.monitor.json'), snapshot)
                raise AssertionError(f'{label}: no correlated terminal native actual observation')
            time.sleep(.1)
        dump(stem.with_suffix('.monitor.json'), snapshot)
        row = {'index': index, 'label': label, 'stage': stage, 'key_mode': key_mode, 'pair': pair,
               'prompt_tokens': prompt, 'usage_cached_quote': quote, 'observation': observation,
               'observation_correlation': method, 'seconds': elapsed,
               'started_monotonic': started, 'ended_monotonic': ended}
        with self.lock:
            self.rows.append(row)
        history.append(message)
        print(f'{label}: native_actual={observation["actual_cached_tokens"]}/{prompt} quote={quote}', flush=True)

    def workload(self):
        args = self.args
        document = '\n'.join(f'Record {i:04d}: coastal station maintenance completed; tide gauge inspection recorded.' for i in range(args.lines))
        document += '\nStation Alder uses an ultraviolet spectrometer. Supervisor: Mara Lin. Station code: ALDER-73.'
        history = [{'role': 'system', 'content': 'Read the reference notes. Acknowledge the document with only Noted. Answer later questions briefly using the notes.'},
                   {'role': 'user', 'content': 'Reference notes:\n' + document}]
        self.turn(0, 'document', history, 'document')
        branches = {name: copy.deepcopy(history) for name in ('P', 'A', 'B')}
        questions = ['Which instrument is assigned to Station Alder?', 'Who supervises Station Alder?', 'What is the code of Station Alder?']
        for index, (branch, question) in enumerate(zip(branches, questions), 1):
            branches[branch].append({'role': 'user', 'content': question})
            self.turn(index, 'first-' + branch, branches[branch], 'first_question')
        with ThreadPoolExecutor(max_workers=2) as pool:
            for pair in range(args.pairs):
                pending = []
                barrier = threading.Barrier(2)
                stop = threading.Event()
                sampler = threading.Thread(target=self.sample_pair, args=(pair, {args.max_tokens + 4 + pair * 2 + i for i in (0, 1)}, stop))
                sampler.start()
                errors = []
                try:
                    for lane, branch in enumerate(('A', 'B')):
                        branches[branch].append({'role': 'user', 'content': f'Review step {pair}, branch {branch}: repeat the instrument and supervisor of Station Alder.'})
                        pending.append(pool.submit(self.turn, 4 + pair * 2 + lane, f'pair-{pair}-{branch}', branches[branch], 'hot', ('same', 'different', 'none')[pair % 3], pair, barrier))
                    # Await both tasks even if one fails; retain both responses.
                    for future in pending:
                        try: future.result()
                        except Exception as error: errors.append(error)
                finally:
                    stop.set()
                    sampler.join()
                if errors:
                    raise errors[0]
        parent = branches['P']
        parent.append({'role': 'user', 'content': 'Confirm the instrument and station code of Station Alder.'})
        self.turn(4 + 2 * args.pairs, 'parent-return', parent, 'parent', 'different')
        # Extend the last accepted assistant turn with a tool call. This also
        # exercises the template's rewritten assistant suffix without a new Q.
        parent[-1]['tool_calls'] = [{'id': 'content_review', 'type': 'function', 'function': {'name': TOOL['name'], 'arguments': '{}'}}]
        parent.append({'role': 'tool', 'tool_call_id': 'content_review', 'content': '\n'.join(
            f'Review row {i}: instrument ultraviolet spectrometer; supervisor Mara Lin; status recorded.' for i in range(args.tool_lines)) + '\nAcknowledge the tool result briefly.'})
        offset = args.native_log.stat().st_size
        try:
            self.turn(5 + 2 * args.pairs, 'tool-continuation', parent, 'tool', 'none')
        finally:
            (args.out / 'tool.trace.log').write_bytes(trace_since(args.native_log, offset))


def self_test():
    class EvidenceTests(unittest.TestCase):
        def test_native_actual_not_quote_and_thresholds(self):
            row = {'label': 'first-A', 'stage': 'first_question', 'key_mode': 'same', 'prompt_tokens': 100,
                   'usage_cached_quote': 99, 'observation': {'cache_semantics_revision': 2, 'prompt_tokens': 100,
                   'actual_cached_tokens': 76, 'cache_source': 'shared', 'restore_path': 'snapshot'}}
            check_observation(row, .95, .95)  # A legal initial native boundary can be 76%.
            for patch in ({'actual_cached_tokens': None}, {'cache_semantics_revision': 1}, {'restore_path': 'continuation'}):
                bad = copy.deepcopy(row); bad['observation'].update(patch)
                with self.assertRaises(AssertionError): check_observation(bad, .95, .95)
            row['stage'] = 'parent'
            with self.assertRaises(AssertionError): check_observation(row, .95, .95)

        def test_monitor_identity_never_array_order(self):
            old = {'id': 'old', 'limit': 24, 'prompt_tokens': 100}
            fresh = dict(old, id='new')
            self.assertEqual(match_observation([old, fresh], '', '', {'old'}, 100, 24, 'polystrata')[0], fresh)
            self.assertIsNone(match_observation([fresh], 'other', '', set(), 100, 24, 'llama')[0])
            with self.assertRaises(AssertionError): match_observation([fresh, dict(fresh, id='new2')], '', '', set(), 100, 24, 'polystrata')

        def test_tools_require_positive_native_evidence(self):
            traces = {'llama': 'KVMEM_PREFILL_DECISION path=append_pressure reason=ordinary_continuation query_source=none query_reused=0 q_rows=0 replay_rows=0',
                      'ninfer': 'KVMEM_PREFILL_PLAN path=append_pressure reason=no_query planned_replay_rows=0 capture_query=0',
                      'polystrata': 'KVMEM_QUERY source=tool_or_continuation recall=0\nKVMEM_REUSE query_begin=0 query_end=0',
                      'radiance': 'KVMEM_QUERY_SPAN begin=-1 end=-1 proven=0\nKVMEM_RESELECT restore=0 mode=prefill-pressure-attention'}
            for backend, trace in traces.items():
                check_tool_trace(backend, trace)
                with self.assertRaises(AssertionError): check_tool_trace(backend, '')
                with self.assertRaises(AssertionError): check_tool_trace(backend, trace + '\nKVMEM_QUERY_REPLAY from=32')
            with self.assertRaises(AssertionError): check_tool_trace('radiance', traces['radiance'].replace('restore=0', 'restore=16'))
            for backend, bad in {'llama': 'query_source=user q_rows=12', 'ninfer': 'capture_query=1',
                                 'polystrata': 'KVMEM_QUERY source=resident recall=1'}.items():
                with self.assertRaises(AssertionError): check_tool_trace(backend, traces[backend] + '\n' + bad)

        def test_responses_tool_conversion_and_missing_key(self):
            messages = [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c', 'function': {'name': 'record_review', 'arguments': '{}'}}]},
                        {'role': 'tool', 'tool_call_id': 'c', 'content': 'Recorded.'}]
            body = request_body(messages, None, 'model', 24, 'responses', {})
            self.assertNotIn('prompt_cache_key', body)
            self.assertEqual([x['type'] for x in body['input']], ['function_call', 'function_call_output'])
            self.assertEqual(body['tools'][0]['name'], 'record_review')

        def test_native_concurrency_rejects_client_overlap_and_queued_rows(self):
            members = [{'observation': {'id': name}} for name in ('a', 'b')]
            active = [{'id': name, 'status': 'generating', 'cache_semantics_revision': 2, 'actual_cached_tokens': 90} for name in ('a', 'b')]
            check_native_pair({'pair': 0, 'native_active_sample': {'active': active}}, members)
            for sample in (None, {'active': active[:1]}, {'active': [active[0], dict(active[1], status='queued')]},
                           {'active': [active[0], dict(active[1], actual_cached_tokens=None)]}):
                with self.assertRaises(AssertionError): check_native_pair({'pair': 0, 'native_active_sample': sample}, members)

        def test_chat_sse_null_role_usage_and_truncated_stream(self):
            wire = (': heartbeat\n\ndata: {"id":"chat-1","choices":[{"delta":{"role":"assistant","content":null}}]}\n\n'
                    'data: {"id":"chat-1","choices":[{"delta":{"content":"Noted."},"finish_reason":"stop"}]}\n\n'
                    'data: {"id":"chat-1","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":3}}\n\n'
                    'data: [DONE]\n\n')
            result = parse_sse(wire.encode().splitlines(keepends=True), 'chat')
            self.assertEqual(result['choices'][0]['message']['content'], 'Noted.')
            self.assertEqual(result['usage']['prompt_tokens'], 100)
            for bad in (wire.replace('data: [DONE]\n\n', ''), wire.replace('"usage":', '"missing_usage":'),
                        wire.replace('"finish_reason":"stop"', '"finish_reason":null')):
                with self.assertRaises(AssertionError): parse_sse(bad.encode().splitlines(keepends=True), 'chat')

        def test_responses_sse_terminal_full_response_and_errors(self):
            response = {'id': 'resp-1', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Noted.'}]}],
                        'usage': {'input_tokens': 100, 'output_tokens': 3}}
            wire = ('event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"ignored partial"}\n\n'
                    'event: response.incomplete\ndata: ' + json.dumps({'type': 'response.incomplete', 'response': response}) + '\n\n')
            self.assertEqual(parse_sse(wire.encode().splitlines(keepends=True), 'responses'), response)
            for event in ({'type': 'response.output_text.delta', 'delta': 'no terminal'},
                          {'type': 'response.completed', 'response': dict(response, usage=None)},
                          {'type': 'response.failed', 'response': {'error': {'message': 'native failure'}}}):
                with self.assertRaises((AssertionError, RuntimeError)):
                    parse_sse(('data: ' + json.dumps(event) + '\n\n').encode().splitlines(keepends=True), 'responses')
    return 0 if unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceTests)).wasSuccessful() else 1


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--self-test', action='store_true')
    cli.add_argument('--replay', type=Path, help='validate an existing evidence directory; no HTTP or GPU')
    cli.add_argument('--backend', choices=['llama', 'ninfer', 'polystrata', 'radiance'])
    cli.add_argument('--url', help='dedicated already-running server; do not share with other clients')
    cli.add_argument('--model', help='the loaded API model alias')
    cli.add_argument('--gpu', help='launcher-selected GPU UUID/model, recorded verbatim')
    for name in ('budget-tokens', 'reserve-tokens', 'host-mib', 'sessions'):
        cli.add_argument('--' + name, type=int, help='actual launcher configuration, recorded verbatim')
    cli.add_argument('--native-log', type=Path, help='append-only native server log; isolate the last tool request')
    cli.add_argument('--out', type=Path)
    cli.add_argument('--protocol', choices=['chat', 'responses'], default='chat')
    cli.add_argument('--stream', action='store_true', help='standard SSE; required when the backend exposes live native progress only to streaming consumers')
    cli.add_argument('--monitor-path', default='/api/monitor/snapshot')
    cli.add_argument('--pairs', type=int, default=6, help='at least six overlapping pairs; cycles same/different/no key')
    cli.add_argument('--lines', type=int, default=96)
    cli.add_argument('--tool-lines', type=int, default=64)
    cli.add_argument('--max-tokens', type=int, default=16, help='base output cap; each request adds its index for unambiguous Poly correlation')
    cli.add_argument('--hot-min', type=float, default=.9, help='minimum native actual/prompt for hot branches')
    cli.add_argument('--parent-min', type=float, default=.9, help='minimum native actual/prompt for the old parent return')
    cli.add_argument('--timeout', type=float, default=180)
    cli.add_argument('--monitor-timeout', type=float, default=10)
    cli.add_argument('--monitor-interval', type=float, default=.02, help='native active sampling interval; absent overlap evidence fails qualification')
    cli.add_argument('--request-options', type=json.loads, default={}, help='JSON backend sampling/template options; core fixture fields take precedence')
    args = cli.parse_args()
    if args.self_test:
        return self_test()
    if args.replay:
        record = json.loads((args.replay / 'evidence.json').read_text(encoding='utf-8'))
        print(json.dumps(validate(record, (args.replay / 'tool.trace.log').read_text(encoding='utf-8', errors='replace')), indent=2))
        return 0
    required = ('backend', 'url', 'model', 'gpu', 'budget_tokens', 'reserve_tokens', 'host_mib', 'sessions', 'native_log', 'out')
    if any(getattr(args, name) is None for name in required):
        cli.error('online mode requires ' + ', '.join('--' + name.replace('_', '-') for name in required))
    if args.pairs < 6 or args.lines < 16 or args.tool_lines < 1 or args.max_tokens < 1:
        cli.error('need >=6 pairs, >=16 document lines and positive tool-lines/max-tokens')
    if min(args.budget_tokens, args.reserve_tokens, args.host_mib, args.sessions, args.timeout, args.monitor_timeout, args.monitor_interval) <= 0:
        cli.error('launcher settings and timeouts must be positive')
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (args.hot_min, args.parent_min)):
        cli.error('hit thresholds must be finite ratios in [0,1]')
    if args.max_tokens + 5 + 2 * args.pairs > args.reserve_tokens:
        cli.error('unique per-request output limits must fit the declared reserve')
    if not isinstance(args.request_options, dict) or not args.native_log.is_file():
        cli.error('request-options must be an object and native-log must exist')
    if set(args.request_options) & {'messages', 'input', 'tools', 'store', 'prompt_cache_key', 'promptCacheKey', 'kvmem', 'stream', 'stream_options', 'max_tokens', 'max_output_tokens'}:
        cli.error('request-options cannot override fixture messages, cache keys, tools or response shape')
    args.out.mkdir(parents=True, exist_ok=False)
    record = {'settings': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              'scope': 'functional native cache proof; budgets declared by launcher, capacity failures retained; no performance comparison', 'requests': []}
    probe = None
    log_offset = args.native_log.stat().st_size
    try:
        probe = Probe(args)
        dump(args.out / 'monitor-before.json', probe.call(args.monitor_path))
        probe.workload()
        record['requests'] = sorted(probe.rows, key=lambda row: row['index'])
        record['pairs'] = sorted(probe.pairs, key=lambda pair: pair['pair'])
        report = validate(record, (args.out / 'tool.trace.log').read_text(encoding='utf-8', errors='replace'))
        dump(args.out / 'report.json', report)
        record['status'] = 'pass'
        print('PASS content-selected branches, keys, concurrency, old parent and native tool continuation')
        return 0
    except Exception as error:
        record.update(status='fail', error=str(error))
        raise
    finally:
        if probe:
            record['requests'] = sorted(probe.rows, key=lambda row: row['index'])
            record['pairs'] = sorted(probe.pairs, key=lambda pair: pair['pair'])
            try: dump(args.out / 'monitor-after.json', probe.call(args.monitor_path))
            except Exception as error: record['final_monitor_error'] = str(error)
        try: (args.out / 'run.trace.log').write_bytes(trace_since(args.native_log, log_offset))
        except Exception as error: record['native_log_error'] = str(error)
        dump(args.out / 'evidence.json', record)


if __name__ == '__main__':
    raise SystemExit(main())
