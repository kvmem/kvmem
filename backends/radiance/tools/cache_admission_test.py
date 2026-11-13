"""Native Radiance cache admission regression on a dedicated one-group worker.

Use --kvmem-cache-min-tokens 4096 (or default 0), --kvmem-sessions 1,
--max-num-seqs 2, and enough host/SSD capacity to avoid quota evictions.
The shared HTTP suite checks real accepted history, not token allowances.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import socket
import time
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('cache_admission_http', ROOT/'tests/cache-admission-http.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


class RadianceProbe(shared.Probe):
    def __init__(self, url, output):
        # Windows urllib discovers the system proxy; <local> does not reliably
        # bypass literal loopback IPs. A proxy can continue its upstream stream
        # after our client closes, invalidating the native cancellation check.
        if urllib.parse.urlsplit(url).hostname in ('127.0.0.1', 'localhost', '::1'):
            urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
        super().__init__(url, output)

    def disconnect_stream(self, response):
        # Explicitly end both directions of the Windows HTTP connection. Closing
        # only the buffered response can leave transport ownership ambiguous.
        (self.output/'disconnect-peer.json').write_text(json.dumps(dict(peer=response.fp.raw._sock.getpeername())))
        response.fp.raw._sock.shutdown(socket.SHUT_RDWR)

    def status(self):
        snapshot = self.http('/api/monitor/snapshot')
        state = dict(snapshot['kvmem'])
        state['active_sessions'] = len(snapshot['active'])
        # The test worker's launch contract is one retained conversation group.
        state['retained_sessions_limit'] = 1
        return state

    def idle(self):
        stable = 0
        for _ in range(600):
            state = self.status()
            busy = any(state.get(k, 0) for k in ('active_sessions', 'parking_sessions',
                       'finishing_sessions', 'io_pending_sequences', 'active_archive_bytes', 'active_checkpoint_bytes'))
            stable = 0 if busy else stable + 1
            if stable >= 3:
                self.observations.append(state)
                (self.output/'observations.json').write_text(json.dumps(self.observations, indent=2))
                return state
            time.sleep(.05)
        (self.output/'idle-timeout.json').write_text(json.dumps(self.http('/api/monitor/snapshot'), indent=2))
        raise AssertionError('Radiance requests or cache I/O did not retire')

    def run(self, long_history=False, cancel_only=False):
        # Establish an actual disconnect before the expensive long-history run.
        # Completion at the requested limit must not masquerade as cancellation.
        body = dict(model=self.model, messages=self.messages(128, 'disconnect-preflight'),
                    max_tokens=2048, temperature=0, enable_thinking=False, ignore_eos=True, stream=True)
        request = urllib.request.Request(self.url+'/v1/chat/completions', data=json.dumps(body).encode(),
                                         headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            for line in response:
                if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                    event = json.loads(line[6:])
                    if any(c.get('delta', {}).get('content') for c in event.get('choices', [])):
                        self.disconnect_stream(response)
                        break
        self.idle()
        snapshot = self.http('/api/monitor/snapshot')
        (self.output/'disconnect-preflight.json').write_text(json.dumps(snapshot, indent=2))
        assert snapshot['recent'][0]['status'] == 'disconnected', snapshot['recent'][0]
        assert snapshot['recent'][0]['generated_tokens'] < 2048, snapshot['recent'][0]
        if cancel_only:
            return dict(validated=True, scope='transport cancellation only', rows=self.rows)
        result = super().run(long_history)
        if result['minimum']:
            messages = self.messages(4095, 'crossing')
            response = self.request('crossing-before-restore', messages, 3, ignore_eos=True)
            after = self.idle()
            assert after['cache_candidate_tokens'] == 4097, after
            follow = messages + [response['choices'][0]['message'],
                                 {'role':'user','content':'Continue with the next number.'}]
            resumed = self.request('crossing-restore', follow, 1, ignore_eos=True)
            assert self.cached_tokens(resumed) >= 4095, resumed['usage']
            result['status'] = self.idle()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--long', action='store_true')
    parser.add_argument('--cancel-only', action='store_true')
    args = parser.parse_args()
    probe = RadianceProbe(args.url, args.out)
    try:
        result = probe.run(args.long, args.cancel_only)
        result['backend'] = 'radiance'
    except BaseException as error:
        result = dict(validated=False, error=str(error), rows=probe.rows)
        raise
    finally:
        (args.out/'result.json').write_text(json.dumps(result, indent=2))
    print('Radiance cache admission regression passed', flush=True)


if __name__ == '__main__':
    main()
