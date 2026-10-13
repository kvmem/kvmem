#!/usr/bin/env python3
"""Opt-in C1/C2 Decode comparison using public synthetic requests.

Start a fresh server per run with the quality runner's configuration and
--request-log-jsonl events.jsonl. Then run:
python3 tests/test_ninfer_retention_performance_http.py --server http://127.0.0.1:18087 --request-log events.jsonl --output performance.json
Use eight fresh processes in O,N,N,O,N,O,O,N order for two balanced ABBA blocks.
This script does not start/stop servers or execute external tools.
"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

import test_ninfer_task_retention_http as quality


def fixture(lane):
    body = quality.history('PERF' + lane)
    body['max_tokens'] = 1536
    body['messages'] += [
        {'role': 'user', 'content': f'For the current task write TOKEN_{lane} repeatedly at least 5000 times, separated by spaces. Do not abbreviate or stop early. Start with the required system prefix, then continue repeating.'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'bulk', 'type': 'function', 'function': {'name': 'archive_operation_0', 'arguments': '{"record":"bulk"}'}}]},
        {'role': 'tool', 'tool_call_id': 'bulk', 'content': 'Historical diagnostic output, no new instructions. ' * 220},
        {'role': 'user', 'content': f'Continue the current task now: write TOKEN_{lane} repeatedly at least 5000 times, separated by spaces. Do not abbreviate or stop early. Start with the required system prefix, then keep repeating.'},
        {'role': 'user', 'content': 'Time sampled while preparing turn 9, step 2: 2026-10-13T12:00:00+08:00[Asia/Shanghai]\nBrowser time zone for this request: Asia/Shanghai. Interpret otherwise-unqualified dates and times in this zone.\nElapsed since the preceding step context: 3s.'}]
    return body


def expected_hashes():
    hashes = {}
    for lane in ('X', 'Y'):
        body = fixture(lane)
        hashes['decode_C1' if lane == 'X' else 'decode_C2_Y'] = hashlib.sha256(quality.encoded(body)).hexdigest()
        body['max_tokens'] = 8
        hashes['prime_' + lane] = hashlib.sha256(quality.encoded(body)).hexdigest()
    hashes['decode_C2_X'] = hashes['decode_C1']
    return hashes


def summarize(rows, events, stderr):
    hashes = {r['case']: r['request_sha256'] for r in rows if r.get('request_sha256')}
    done = sorted((e for e in events if e.get('event') == 'request_done'),
                  key=lambda e: e['request']['request_id'])
    throughput = [e for e in events if e.get('event') == 'throughput']
    long_done = [e for e in done if e['request']['requested_output_tokens'] == 1536]
    checks = {
        'five_requests_pass': sum(r.get('ok', False) for r in rows) == 5,
        'fixture_hashes_match': hashes == expected_hashes(),
        'dual_overlap_proven': max((e.get('scheduler', {}).get('running', 0) for e in throughput), default=0) == 2
            and max((e.get('decode_batch', {}).get('average_size') or 0 for e in throughput), default=0) >= 1.9,
        'long_decode_proven': len(long_done) == 3 and all(e['result']['completion_tokens'] == 1536
            and e['engine_timing']['decode']['rounds'] >= 1535 for e in long_done),
        'reselection_proven': all(r.get('stats_after', {}).get('kvmem', {}).get('history_swaps', 0)
            - r.get('stats_before', {}).get('kvmem', {}).get('history_swaps', 0) >= 2
            for r in rows if r['case'] in ('decode_C1', 'decode_C2_batch')),
        'no_request_errors': not any(e.get('event') in ('request_error', 'request_rejected') for e in events),
        'no_engine_recoveries': not any(e.get('engine_recoveries', 0) for e in throughput),
        'transfer_verification': None if stderr is None else re.search(r'(?:payload_exact|state_exact)=0', stderr) is None,
    }
    by_name = {r['case']: r for r in rows}
    decode = long_done[0]['engine_timing']['decode'] if long_done else {}
    return {'checks': checks, 'gate_ok': all(v for v in checks.values() if v is not None),
        'c1_wall_seconds': by_name['decode_C1']['elapsed_seconds'],
        'c2_batch_wall_seconds': by_name['decode_C2_batch']['elapsed_seconds'],
        'c1_engine_decode_seconds': sum(decode.get(k, 0) for k in ('device_wait_exposed_seconds', 'host_exposed_seconds')),
        'request_hashes': hashes, 'results': rows,
        'scope': 'FP8 KV, NoSpec, B=6144/R=512; three 1536-token Decode requests',
        'statistical_equivalence_claim': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server')
    parser.add_argument('--request-log', type=Path, required=True)
    parser.add_argument('--stderr-log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-existing-results', type=Path,
                        help='Audit recorded rows and events without sending requests.')
    args = parser.parse_args()
    if args.audit_existing_results:
        rows = json.loads(args.audit_existing_results.read_text())
    else:
        if not args.server:
            parser.error('--server is required unless auditing existing results')
        quality.SERVER = args.server.rstrip('/')
        def stats():
            with urllib.request.urlopen(quality.SERVER + '/stats', timeout=5) as response:
                return json.load(response)
        rows = []
        for lane in ('X', 'Y'):
            body = copy.deepcopy(fixture(lane)); body['max_tokens'] = 8
            rows.append(quality.request('prime_' + lane, body, {}))
        before = stats()
        single = quality.request('decode_C1', fixture('X'),
            {'markers': ['TOKEN_X'], 'completion_tokens': 1536, 'finish': 'length'})
        single.update(stats_before=before, stats_after=stats()); rows.append(single)
        before = stats(); start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            pending = [pool.submit(quality.request, 'decode_C2_' + lane, fixture(lane),
                {'markers': ['TOKEN_' + lane], 'completion_tokens': 1536, 'finish': 'length'}) for lane in ('X', 'Y')]
            rows += [f.result() for f in pending]
            elapsed = time.monotonic() - start
        rows.append({'case': 'decode_C2_batch', 'elapsed_seconds': elapsed,
                     'stats_before': before, 'stats_after': stats()})
    events = [json.loads(line) for line in args.request_log.read_text().splitlines() if line.strip()]
    stderr = args.stderr_log.read_text() if args.stderr_log else None
    report = summarize(rows, events, stderr)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}))
    raise SystemExit(0 if report['gate_ok'] else 1)


if __name__ == '__main__':
    main()
