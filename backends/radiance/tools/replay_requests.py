"""Replay 1-5 saved JSON requests under the existing resource-monitored harness.

Payloads are copied verbatim except the served model alias. This is a diagnostic
lane, never a replacement for the canonical 33-request source-history test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import agent_history
import smoke_server


def read_plan(path):
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not 1 <= len(entries) <= 5:
        raise ValueError('diagnostic plan must contain 1-5 requests')
    payloads = []
    for item in entries:
        source = (path.parent / item['request']).resolve()
        raw = source.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload.get('messages'), list) or payload.get('stream', False):
            raise ValueError('saved non-streaming chat request required')
        payload['model'] = 'bringup'
        payloads.append((item, payload, hashlib.sha256(raw).hexdigest()))
    return payloads


def make_runner(payloads):
    def replay(request, output, context, request_limit, chunk_tokens, rows, **unused):
        for index, (item, payload, sha256) in enumerate(payloads):
            (output / f'replay-{index:03d}.request.json').write_text(json.dumps(payload, indent=2))
            start = time.monotonic()
            response = request('/v1/chat/completions', payload, timeout=1800)
            elapsed = time.monotonic() - start
            (output / f'replay-{index:03d}.response.json').write_text(json.dumps(response, indent=2))
            choice = response['choices'][0]
            content = choice['message'].get('content') or ''
            terms = item.get('expected_terms', [])
            row = {'round': index + 1, 'name': item['name'], 'request_seconds': elapsed,
                   'usage': response['usage'], 'timings': response.get('timings'),
                   'content': content, 'reasoning_content': choice['message'].get('reasoning_content'),
                   'finish_reason': choice['finish_reason'], 'source_sha256': sha256,
                   'expected_terms': terms, 'terms_present': all(x in content for x in terms),
                   'literal_tool_call': '<tool_call>' in content,
                   'pass': response['usage']['completion_tokens'] > 0 and choice['finish_reason'] in ('stop', 'length')}
            rows.append(row)
            (output / 'rounds.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
        return {'diagnostic_request_replay': True, 'pass': all(r['pass'] for r in rows),
                'quality_checks': {'expected_terms_present': [r['terms_present'] for r in rows]},
                'pass_scope': 'saved-request completion only; expected terms reported separately'}
    return replay


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--plan', type=Path, required=True)
    args, remainder = parser.parse_known_args()
    if any(x in remainder for x in ('--agent-requests', '--cases', '--rounds', '--prompt-file')):
        parser.error('replay plan owns the diagnostic requests')
    payloads = read_plan(args.plan)
    agent_history.run = make_runner(payloads)
    sys.argv = [sys.argv[0], *remainder, '--agent-requests', str(len(payloads))]
    return smoke_server.main()


if __name__ == '__main__':
    raise SystemExit(main())
