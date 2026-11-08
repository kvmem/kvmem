"""Live props/slots/timings regression. Uses a supplied model, never downloads one."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--server', required=True)
p.add_argument('--model', required=True)
p.add_argument('--output', required=True)
p.add_argument('--mtp', action='store_true')
p.add_argument('--mcp-echo-server', type=Path, help='Path to the llama.cpp mcp_echo_server.py test fixture')
a = p.parse_args()
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
base = f'http://127.0.0.1:{port}'
checks = []
headers = {'Authorization': 'Bearer api-test-only', 'Content-Type': 'application/json'}

def check(name, ok):
    checks.append({'name': name, 'pass': bool(ok)})
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)
    if not ok:
        raise AssertionError(name)

def request(path, data=None, auth=True, method=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(base + path,
        data=None if data is None else json.dumps(data).encode(),
        headers=headers if auth else {}, method=method)
    try:
        with opener.open(req, timeout=180) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)

args = [a.server, '-m', a.model, '--host', '127.0.0.1', '--port', str(port),
        '-c', '8192', '-n', '128', '--api-key', 'api-test-only', '--threads-http', '4',
        '--reasoning-effort', 'none', '--temp', '0', '--no-webui', '--kv-dtype', 'q8_0',
        '--kvmem-budget', '4096', '--kvmem-gen-reserve', '2048']
args += ['--spec-type', 'draft-mtp' if a.mtp else 'none']
if a.mcp_echo_server:
    if not a.mcp_echo_server.is_file():
        p.error('MCP echo server fixture is missing')
    config = out / 'mcp servers.json'
    config.write_text(json.dumps({'mcpServers': {'demo': {
        'command': sys.executable, 'args': [str(a.mcp_echo_server.resolve())]}}}), encoding='utf-8')
    args += ['--mcp-servers-config', str(config)]
log = (out / 'server.log').open('wb')
proc = subprocess.Popen(args, stdout=log, stderr=log,
    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
try:
    deadline = time.monotonic() + 240
    while True:
        if proc.poll() is not None:
            raise RuntimeError('server exited; see server.log')
        try:
            if request('/health', auth=False)[0] == 200:
                break
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError('startup')
        time.sleep(1)
    check('slots requires authentication', request('/slots', auth=False)[0] == 401)
    if a.mcp_echo_server:
        check('tool listing requires authentication', request('/tools', auth=False)[0] == 401)
        status, raw, _ = request('/tools')
        check('MCP adder discovered', status == 200 and any(
            item['definition']['function']['name'] == 'demo_add' for item in json.loads(raw)))
        status, raw, _ = request('/tools', {'tool': 'demo_add', 'params': {'a': 3, 'b': 5}})
        check('MCP adder invoked', status == 200 and json.loads(raw)['plain_text_response'] == '8')
    check('public preflight with CORS', request('/slots', auth=False, method='OPTIONS')[0] == 204)
    status, body, _ = request('/props')
    props = json.loads(body)
    check('props exposes effective template and slots', status == 200 and props['chat_template']
          and props['endpoint_slots'] and props['total_slots'] == 1)
    check('idle slots available', request('/slots?fail_on_no_slot=1')[0] == 200)
    prompt = [{'role': 'user', 'content':
        'Write numbers from 1 through 500 in ascending order, separated by commas. Output only the list.'}]
    responses = []
    all_samples = []
    for stream in (False, True):
        payload = {'messages': prompt, 'max_tokens': 128, 'stream': stream, 'temperature': 0,
                   'reasoning_effort': 'none'}
        if stream:
            payload['stream_options'] = {'include_usage': True}
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(request, '/v1/chat/completions', payload)
            samples = []
            while not future.done():
                started = time.monotonic()
                code, raw, _ = request('/slots')
                sample = json.loads(raw)[0]
                samples.append(sample)
                check('slots polling responsive', code == 200 and time.monotonic() - started < 3)
                if sample['is_processing']:
                    check('busy flag query is exact', request('/slots?not_fail_on_no_slot=1')[0] == 200)
                    # Generation may finish between these requests, so either result is valid.
                    check('busy availability status', request('/slots?fail_on_no_slot=1')[0] in (200, 503))
                time.sleep(0.1)
            status, raw, _ = future.result()
        check('chat succeeds', status == 200)
        check('live progress is visible', any(s['is_processing'] and
              0 < s['next_token'][0]['n_decoded'] < 128 for s in samples))
        active = [s for s in samples if s['is_processing']]
        check('same-task counters increase', all(x['next_token'][0]['n_decoded'] <= y['next_token'][0]['n_decoded']
              for x, y in zip(active, active[1:]) if x['id_task'] == y['id_task']))
        if stream:
            chunks = [json.loads(line[6:]) for line in raw.decode().splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
            check('SSE stable created and DONE', len({c['created'] for c in chunks}) == 1 and b'data: [DONE]' in raw)
            response = next(c for c in chunks if 'usage' in c)
        else:
            response = json.loads(raw)
        timings, usage = response['timings'], response['usage']
        check('timings cache count matches usage', timings['cache_n'] == usage['prompt_cache_hit_tokens'])
        check('timings excludes cached prompt tokens', timings['prompt_n'] + timings['cache_n'] == usage['prompt_tokens'])
        check('timings generation count', timings['predicted_n'] == usage['completion_tokens'] == 128)
        check('slot returns idle', not json.loads(request('/slots')[1])[0]['is_processing'])
        responses.append(response)
        all_samples += samples
    check('repeated prompt has cache hits', responses[1]['timings']['cache_n'] > 0)
    for field in ('max_tokens', 'max_completion_tokens'):
        for value in (4294967297, 18446744073709551615):
            status, raw, _ = request('/v1/chat/completions', {'messages': prompt, field: value,
                'temperature': 0, 'reasoning_effort': 'none'})
            # A positive cap is first clamped to n_ctx, then checked against
            # the space left after tokenizing the nonempty prompt.
            check(f'{field} {value} does not wrap to one', status == 400 and
                  b'prompt + max_tokens exceeds n_ctx' in raw)
    status, _, _ = request('/v1/chat/completions', {'messages': prompt, 'max_tokens': -2})
    check('invalid request leaves slot idle', status == 400 and not json.loads(request('/slots')[1])[0]['is_processing'])
    (out / 'responses.json').write_text(json.dumps(responses, indent=2), encoding='utf-8')
    (out / 'slots.json').write_text(json.dumps(all_samples, indent=2), encoding='utf-8')
finally:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log.close()
    (out / 'results.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
