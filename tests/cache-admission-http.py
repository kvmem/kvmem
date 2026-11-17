"""NInfer/llama.cpp retention admission regression against a dedicated server.

Run once with --kvmem-cache-min-tokens 4096, and once with the default 0.
The server must use --kvmem-sessions 1, at least two active lanes, enough Host
capacity for the fixtures, and a context of at least 16384 (131072 for --long).
Run both legacy and layerwise recall. This checks cache behavior, not model quality.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


class Probe:
    def __init__(self, url, output, backend='ninfer', final_token_committed=None):
        self.url = url.rstrip('/')
        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.observations = []
        self.backend = backend
        self.final_token_committed = (backend == 'llamacpp' if final_token_committed is None
                                      else bool(final_token_committed))
        self.fixtures = {}
        self.calibration = []
        self.model = self.http('/v1/models')['data'][0]['id']

    def http(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'{path}: {error.code}: {error.read().decode()}') from error

    def status(self):
        return self.http('/props')['kvmem']['usage']

    def disconnect_stream(self, response):
        # Most transports close the underlying socket with the response context.
        # Backends may strengthen this to an explicit connection shutdown.
        pass

    def idle(self):
        for _ in range(200):
            status = self.status()
            if status['active_sessions'] == 0:
                self.observations.append(status)
                (self.output / 'observations.json').write_text(json.dumps(self.observations, indent=2))
                return status
            time.sleep(.05)
        raise AssertionError('native requests did not retire')

    def request(self, label, messages, maximum=1, **extra):
        body = dict(model=self.model, messages=messages, max_tokens=maximum,
                    temperature=0, seed=17, enable_thinking=False, **extra)
        (self.output / f'{label}.request.json').write_text(json.dumps(body, indent=2), encoding='utf-8')
        start = time.monotonic()
        response = self.http('/v1/chat/completions', body)
        (self.output / f'{label}.response.json').write_text(json.dumps(response, indent=2), encoding='utf-8')
        row = dict(label=label, seconds=time.monotonic()-start, usage=response['usage'])
        self.rows.append(row)
        print(json.dumps(row), flush=True)
        return response

    def messages(self, target, label):
        key = (target, label)
        if key in self.fixtures:
            return self.fixtures[key]
        padding = max(0, target - 100)
        for _ in range(12):
            messages = [{'role':'system','content':'Reply with only OK.'},
                        {'role':'user','content':f'Document {label}. Remember this text for the next question.\n'
                         + ' a' * padding + '\nAcknowledge with OK.'}]
            if self.backend == 'llamacpp' and label == 'crossing':
                # Its API does not offer ignore_eos. Request a multi-token answer
                # and still require exactly three actually returned tokens below.
                messages[0]['content'] = 'Reply with the numbers from 1 to 20, separated by spaces.'
                messages[1]['content'] = messages[1]['content'].replace('Acknowledge with OK.', 'List the numbers from 1 to 20.')
            if self.backend == 'llamacpp':
                # This worker has no tokenize/template HTTP endpoints. Calibrate
                # against native usage before any retention assertions begin.
                response = self.http('/v1/chat/completions', dict(model=self.model, messages=messages,
                    max_tokens=1, temperature=0, seed=17, enable_thinking=False, ignore_eos=True))
                count = response['usage']['prompt_tokens']
                self.calibration.append(dict(label=label, target=target, padding=padding, usage=response['usage']))
                (self.output/'calibration.json').write_text(json.dumps(self.calibration, indent=2))
            else:
                rendered = self.http('/apply-template', dict(model=self.model, messages=messages,
                                                             enable_thinking=False))['prompt']
                count = len(self.http('/tokenize', dict(model=self.model, content=rendered))['tokens'])
            if count == target:
                self.fixtures[key] = messages
                return messages
            padding += target-count
            assert padding >= 0, 'fixture is smaller than its chat template'
        raise AssertionError(f'cannot construct exact {target}-token prompt (last {count})')

    def run(self, long_history=False):
        # llama.cpp ordinary decode commits the last returned token. Its MTP
        # path and NInfer leave that token unevaluated until the next step.
        # Compare the same accepted-history frontiers on both native paths.
        final_token = int(self.final_token_committed)
        if self.backend == 'llamacpp':
            fixtures = [(128,'default'), (6000,'default-document'), (192,'default-incidental')] if not self.status()['cache_min_tokens'] else [
                (4095-final_token,'4095'), (4096-final_token,'4096'),
                (4097-final_token,'4097'), (4095-final_token,'crossing'),
                (100000 if long_history else 6000,'retained-document'), (128,'incidental-short'),
                (160,'concurrent-0'), (161,'concurrent-1'), (128,'cancelled-short'), (192,'large-allowance')]
            for target, label in fixtures:
                self.messages(target, label)
        before = self.idle()
        minimum = before['cache_min_tokens']
        assert minimum in (0, 4096), before
        assert before['retained_sessions_limit'] == 1, before
        if minimum == 0:
            response = self.request('default-short', self.messages(128, 'default'), ignore_eos=True)
            after = self.idle()
            assert response['usage']['prompt_tokens'] == 128
            assert after['cache_candidate_tokens'] == 128 + final_token, after
            assert after['retained_sessions'] == 1 and after['cache_below_minimum'] == 0, after
            # Negative control: without filtering, a distinct short history can
            # still replace the long history in the one-group retained pool.
            self.request('default-document', self.messages(6000, 'default-document'), ignore_eos=True)
            saved = self.idle()
            self.request('default-incidental-short', self.messages(192, 'default-incidental'), ignore_eos=True)
            after = self.idle()
            assert after['cache_candidate_tokens'] == 192 + final_token and after['retained_sessions'] == 1, after
            assert after['history_retention_evictions'] > saved['history_retention_evictions'], after
            assert after['cache_below_minimum'] == 0, after
            return dict(minimum=minimum, status=after, rows=self.rows, validated=True)

        for length in (4095, 4096, 4097):
            before = self.idle()
            response = self.request(f'boundary-{length}', self.messages(length-final_token, str(length)), ignore_eos=True)
            after = self.idle()
            assert response['usage']['prompt_tokens'] == length-final_token, response['usage']
            assert response['usage']['completion_tokens'] == 1, response['usage']
            assert after['cache_candidate_tokens'] == length, after
            assert after['cache_below_minimum'] - before['cache_below_minimum'] == int(length < minimum), after
            if length >= minimum:
                assert after['retained_sessions'] == 1, after

        # Two accepted output steps carry a short input across the threshold.
        before = self.idle()
        response = self.request('output-crossing', self.messages(4095-final_token, 'crossing'), maximum=3, ignore_eos=True)
        after = self.idle()
        assert response['usage']['prompt_tokens'] == 4095-final_token and response['usage']['completion_tokens'] == 3
        assert after['cache_candidate_tokens'] == 4097 and after['cache_below_minimum'] == before['cache_below_minimum'], after

        length = 100000 if long_history else 6000
        document = self.messages(length, 'retained-document')
        acknowledgement = self.request('document', document, ignore_eos=True)['choices'][0]['message']
        saved = self.idle()
        assert saved['cache_candidate_tokens'] == length+final_token and saved['retained_sessions'] == 1, saved
        short = self.messages(128, 'incidental-short')
        self.request('incidental-short', short, ignore_eos=True)
        after = self.idle()
        assert after['retained_sessions'] == saved['retained_sessions'], after
        assert after['history_retention_evictions'] == saved['history_retention_evictions'], after
        assert after['history_capacity_evictions'] == saved['history_capacity_evictions'], after
        assert after['cache_below_minimum'] == saved['cache_below_minimum'] + 1, after

        # Reuse an ordinary conversation with its document in the earlier turn.
        # Do not combine the long document with the question in the final message.
        followup = document + [{'role':'assistant','content':acknowledgement.get('content') or ''},
                               {'role':'user','content':'What word was repeated in the document? Reply briefly.'}]
        response = self.request('short-followup', followup, maximum=8, ignore_eos=True)
        reused = self.cached_tokens(response)
        # The llama legacy checkpoint precedes its bounded 512-token Query;
        # recurrent state cannot be restored at an arbitrary token prefix.
        reuse_floor = length - (1024 if self.backend == 'llamacpp' else 64)
        assert reused >= reuse_floor, response['usage']
        assert response['usage']['prompt_tokens'] - length < 100, response['usage']
        after = self.idle()
        assert after['cache_candidate_tokens'] >= length, after

        # Two independent short requests must not claim the long group's slot.
        baseline = after
        requests = [self.messages(160 + i, 'concurrent-' + str(i)) for i in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.request, 'concurrent-' + str(i), messages, 16, ignore_eos=True)
                       for i, messages in enumerate(requests)]
            for future in futures:
                future.result()
        after = self.idle()
        assert after['cache_below_minimum'] == baseline['cache_below_minimum'] + 2, after
        assert after['retained_sessions'] == baseline['retained_sessions'], after
        assert after['history_retention_evictions'] == baseline['history_retention_evictions'], after
        assert after['history_capacity_evictions'] == baseline['history_capacity_evictions'], after

        # Closing a short streaming branch must release its working state without
        # deleting the retained source. Even its whole allowance is below N.
        baseline = after
        body = dict(model=self.model, messages=self.messages(128, 'cancelled-short'),
                    max_tokens=2048, temperature=0, seed=17, enable_thinking=False,
                    ignore_eos=True, stream=True)
        (self.output / 'cancelled-short.request.json').write_text(json.dumps(body, indent=2))
        req = urllib.request.Request(self.url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=900) as response:
            while line := response.readline():
                if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                    continue
                event = json.loads(line[6:])
                if any(choice.get('delta', {}).get('content') for choice in event.get('choices', [])):
                    (self.output / 'cancelled-short.first-event.txt').write_bytes(line)
                    self.disconnect_stream(response)
                    break
        after = self.idle()
        assert after['retained_sessions'] == baseline['retained_sessions'], after
        assert after['history_retention_evictions'] == baseline['history_retention_evictions'], after
        assert after['history_capacity_evictions'] == baseline['history_capacity_evictions'], after
        response = self.request('source-after-cancellation', followup, maximum=1, ignore_eos=True)
        assert self.cached_tokens(response) >= reuse_floor, response['usage']
        after = self.idle()

        # A large output allowance must not make an actually short history eligible.
        # Calibrate a deterministic stop instead of requiring logit_bias support.
        short = self.messages(192, 'large-allowance')
        first = self.request('short-stop-calibration', short, ignore_eos=True)
        stop = first['choices'][0]['message'].get('content')
        assert stop, first
        baseline = self.idle()
        self.request('large-allowance-short-stop', short, maximum=4096, stop=[stop])
        after = self.idle()
        assert after['cache_candidate_tokens'] < minimum, after
        assert after['cache_below_minimum'] == baseline['cache_below_minimum'] + 1, after
        assert after['history_retention_evictions'] == baseline['history_retention_evictions'], after
        return dict(minimum=minimum, long_history_tokens=length, status=after, rows=self.rows, validated=True)

    def cached_tokens(self, response):
        usage = response['usage']
        return usage.get('prompt_cache_hit_tokens', 0) if self.backend == 'llamacpp' else usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--long', action='store_true')
    parser.add_argument('--backend', choices=('ninfer','llamacpp'),default='ninfer')
    parser.add_argument('--final-token-committed', type=int, choices=(0,1), default=None,
                        help='Override native frontier accounting; use 0 for llama.cpp MTP')
    args = parser.parse_args()
    probe = Probe(args.url, args.out, args.backend, args.final_token_committed)
    try:
        result = probe.run(args.long)
    except BaseException as error:
        (args.out/'result.json').write_text(json.dumps(dict(validated=False, error=str(error), rows=probe.rows), indent=2))
        raise
    (args.out/'result.json').write_text(json.dumps(result, indent=2))
    print(args.backend+' cache admission regression passed', flush=True)


if __name__ == '__main__':
    main()
