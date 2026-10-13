#!/usr/bin/env python3
"""Opt-in cold/warm concurrent replay of two caller-provided Chat request files.

Run against a fresh owned server:
python3 tests/test_ninfer_real_replay_http.py --server http://127.0.0.1:18087 --request case1.json --request case2.json --output real-results.json
Requests are canonically JSON-encoded without changing their fields. The output
contains response text and must be kept private for private inputs. No external
tools are executed. Inputs must request streaming; this loop expects stop rather
than length/tool_calls. Transport/reuse checks do not establish semantic correctness.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path

import test_ninfer_task_retention_http as quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True)
    parser.add_argument('--request', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.request) != 2:
        parser.error('provide exactly two --request files')
    quality.SERVER = args.server.rstrip('/')
    bodies = [json.loads(path.read_text(encoding='utf-8')) for path in args.request]
    rows = []
    for phase in ('fresh', 'warm'):
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            pending = [pool.submit(quality.request, phase + '_case' + str(i + 1), body,
                {'finish': 'stop', **({'warm': True} if phase == 'warm' else {})})
                for i, body in enumerate(bodies)]
            rows += [f.result() for f in pending]
        args.output.write_text(json.dumps({'results': rows,
            'scope': 'transport, finish and warm reuse; semantic review required',
            'external_tools_executed': False}, indent=2) + '\n', encoding='utf-8')
        if not all(row.get('ok') for row in rows):
            raise SystemExit(1)
    print(json.dumps({'cases': len(rows), 'passed': sum(row.get('ok', False) for row in rows)}))


if __name__ == '__main__':
    main()
