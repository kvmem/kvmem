"""Gemma-4 cold-history recall through the ordinary two-turn HTTP protocol.

Supply an existing model with --model or KVMEM_TEST_GEMMA4_MODEL. Without one,
CTest skips this test; no model or runtime dependency is downloaded.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import gpu_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--model', default=os.environ.get('KVMEM_TEST_GEMMA4_MODEL'))
    parser.add_argument('--gpu', default=os.environ.get('KVMEM_TEST_GPU'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.model:
        return 77
    args.out.mkdir(parents=True, exist_ok=True)
    env = gpu_env.apply_gpu(dict(os.environ), args.gpu)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str(args.exe.resolve()), '-m', args.model, '--host', '127.0.0.1', '--port', str(port),
               '-ngl', '99', '-c', '16384', '-b', '256', '-ub', '256', '--no-ui', '--no-think',
               '--spec-type', 'none', '--kv-dtype', 'q8_0', '--kvmem', '--kvmem-budget', '2048',
               '--kvmem-gen-reserve', '512', '--kvmem-block-tokens', '64', '--kvmem-sink-tokens', '128',
               '--kvmem-recent-tokens', '256', '--kvmem-host-mib', '8192', '--kvmem-trace']
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f'http://127.0.0.1:{port}'
    log_path = args.out / 'server.log'
    with log_path.open('wb') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=log,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            deadline = time.monotonic() + 180
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f'server exited: {child.returncode}')
                try:
                    with http.open(base + '/health', timeout=3) as response:
                        if response.status == 200:
                            break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('server readiness')
                    time.sleep(.5)

            def request(name, messages):
                body = dict(messages=messages, temperature=0, seed=42, max_tokens=32)
                (args.out / (name + '-request.json')).write_text(json.dumps(body, indent=2), encoding='utf8')
                offset = log_path.stat().st_size
                start = time.perf_counter()
                req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                             headers={'Content-Type': 'application/json'})
                with http.open(req, timeout=300) as response:
                    data = json.load(response)
                (args.out / (name + '-response.json')).write_text(json.dumps(data, indent=2), encoding='utf8')
                trace = log_path.read_bytes()[offset:].decode(errors='replace')
                (args.out / (name + '-trace.log')).write_text(trace, encoding='utf8')
                return data, trace, time.perf_counter() - start

            records = [f'Record {i:04d}: Routine maintenance was completed for storage racks and inspection equipment.'
                       for i in range(600)]
            records[143] = 'Record 0143: The access code for the violet observatory is EMERALD-2719.'
            messages = [{'role': 'system', 'content': 'Answer briefly from the supplied records.'},
                        {'role': 'user', 'content': 'Read this reference document and reply only RECORDED.\n' + '\n'.join(records)}]
            document, _, document_seconds = request('document', messages)
            assert document['usage']['prompt_tokens'] > 8192, 'fixture did not exceed the KV budget'
            messages.append(document['choices'][0]['message'])
            messages.append({'role': 'user', 'content': 'What is the access code for the violet observatory? Reply with only its code.'})
            answer, trace, question_seconds = request('question', messages)
            assert 'EMERALD-2719' in (answer['choices'][0]['message'].get('content') or ''), 'cold-history answer missing'
            assert answer['usage']['prompt_cache_hit_tokens'] > 0, 'document prefix was not reused'
            span = re.search(r'method=native_role .*tokens=\[(\d+),(\d+)\)', trace)
            scored = re.search(r'KVMEM_QUERY scorer=page-softmax .*rows=(\d+)', trace)
            assert span and scored, 'native user Query or its Q/K features were not captured'
            begin, end = map(int, span.groups())
            assert 0 < end - begin < 64 and int(scored[1]) == end - begin, 'wrong user Query coverage'
            assert 'query_source=user' in trace and 'replay=1' in trace, 'user Query replay was not exercised'
            result = dict(passed=True, document_seconds=document_seconds, question_seconds=question_seconds,
                          query_span=[begin, end], query_rows=int(scored[1]),
                          prefix_hit_tokens=answer['usage']['prompt_cache_hit_tokens'])
            (args.out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf8')
            print(json.dumps(result))
        finally:
            child.terminate()
            try:
                child.wait(15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    return 0


if __name__ == '__main__':
    sys.exit(main())
