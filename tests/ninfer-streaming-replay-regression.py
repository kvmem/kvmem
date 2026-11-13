"""Real HTTP regression for default full Query replay and resident-only tool append.

Uses the public production server, a small physical KV window, and transfer
verification. This is a correctness test, not a throughput measurement.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scorer', choices=['page-softmax', 'meanq-cosine'], default='page-softmax')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = {key: value for key, value in os.environ.items() if not key.startswith('NINFER_')}
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID')
    command = [str(args.worker), str(args.model), '--host', '127.0.0.1', '--port', str(port),
        '--max-context', '16384', '--max-concurrency', '1', '--prefill-chunk', '512',
        '--spec', 'none', '--kvmem', '--kvmem-budget', '2048', '--kvmem-gen-reserve', '1024',
        '--kvmem-host-mib', '8192', '--kvmem-sink-tokens', '64', '--kvmem-recent-tokens', '64',
        '--kvmem-sessions', '1', '--kv-dtype', 'int8', '--device-profile', 'auto',
        '--kvmem-recall-strategy', 'legacy', '--kvmem-query-scorer', args.scorer,
        '--kvmem-verify-transfers', '--log-level', 'info']
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report = dict(argv=command, gpu=args.gpu, scope='correctness; transfer verification enabled')

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def call(route, body=None):
        request = urllib.request.Request(f'http://127.0.0.1:{port}' + route,
            data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with client.open(request, timeout=600) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode(errors='replace')) from error

    save('launch.json', report)
    log_path = args.output / 'server.log'
    with log_path.open('wb') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            deadline = time.monotonic() + 300
            while True:
                if process.poll() is not None:
                    raise RuntimeError('server exited during startup')
                try:
                    props = call('/props')
                    break
                except (OSError, RuntimeError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('server startup')
                    time.sleep(.2)
            assert props['execution']['monitor_devices'][0]['uuid'] == args.gpu
            assert 'KVMEM_LIMITS query_max_tokens=0 query_replay_max_tokens=0' in log_path.read_text(errors='replace')
            document = '\n'.join(f'Record {i:04d}: the river flows past the orchard and the stone bridge.' for i in range(320))
            messages = [dict(role='system', content='Read the supplied records. Reply only OK when no question is asked.'),
                        dict(role='user', content=document + '\nRead these records and reply only OK.')]
            body = dict(model=props['model_alias'], messages=messages, enable_thinking=False, max_tokens=16, temperature=0, seed=42)
            save('document-request.json', body)
            offset = log_path.stat().st_size
            response = call('/v1/chat/completions', body)
            save('document-response.json', response)
            trace = log_path.read_bytes()[offset:].decode(errors='replace')
            replays = re.findall(r'KVMEM_QUERY_REPLAY from=(\d+) to=(\d+) query_tokens=(\d+) replay_rows=(\d+)', trace)
            assert len(replays) == 1, replays
            end, begin, query, rows = map(int, replays[0])
            assert query > 2048 and rows == end - begin and rows > 3072, replays
            refreshed = [tuple(map(int, x)) for x in re.findall(
                r'KVMEM_REPLAY_QUERY_SELECTION logical=(\d+) query_tokens=(\d+) candidates=(\d+) selected=(\d+)', trace)]
            assert refreshed and all(q == query and n == (at + 63) // 64 and 0 < kept <= 32 and begin < at < end
                                     for at, q, n, kept in refreshed), refreshed
            assert [x[0] for x in refreshed] == sorted(set(x[0] for x in refreshed))
            before, after = trace.split('KVMEM_QUERY_REPLAY ', 1)
            assert 'KVMEM_PRESSURE phase=prefill' in before
            assert 'KVMEM_PRESSURE phase=prefill' not in after
            report.update(query_tokens=query, replay_rows=rows, refreshes=refreshed)

            # A supplied tool result is appended after the already accepted user
            # turn. It has no new user Query and must not reactivate its old Q.
            messages += [response['choices'][0]['message'],
                dict(role='assistant', content='', tool_calls=[dict(id='lookup-1', type='function',
                    function=dict(name='lookup', arguments='{}'))]),
                dict(role='tool', tool_call_id='lookup-1', content='\n'.join(
                    f'Appendix {i:04d}: the garden path continues beside the river.' for i in range(180)))]
            offset = log_path.stat().st_size
            save('tool-request.json', dict(body, messages=messages))
            result = call('/v1/chat/completions', dict(body, messages=messages))
            save('tool-response.json', result)
            trace = log_path.read_bytes()[offset:].decode(errors='replace')
            assert result['timings']['cache_n'] > 0, result['timings']
            assert 'KVMEM_PRESSURE phase=prefill' in trace
            assert 'KVMEM_QUERY_REPLAY ' not in trace and 'KVMEM_REPLAY_QUERY_SELECTION ' not in trace
            report['tool_append_resident_only'] = True
            report['passed'] = True
            save('report.json', report)
            print(json.dumps(report), flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)


if __name__ == '__main__':
    main()
