"""Model-free evidence guards for PolyStrata performance probes."""
import copy
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

ERROR_BODY_LIMIT = 64 * 1024


def record_http_turn(opener, url, body, stage, native_path, row, save, timeout=240):
    """Save intent before sending, and preserve evidence on transport failure.

    ``save`` publishes the caller's whole result object, preferably atomically.
    The HTTP exception still propagates after its bounded response body is saved.
    """
    native_path = Path(native_path)
    offset = native_path.stat().st_size if native_path.exists() else 0
    turn = {'stage': stage, 'request': copy.deepcopy(body), 'status': 'running',
            'native_log_offset': offset}
    row['turns'].append(turn)
    save()
    start = time.monotonic()
    try:
        request = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'),
                                         headers={'Content-Type': 'application/json'})
        with opener.open(request, timeout=timeout) as response:
            turn['http_status'] = getattr(response, 'status', 200)
            turn['response'] = json.load(response)
        turn['status'] = 'received'
    except urllib.error.HTTPError as error:
        turn.update(status='fail', http_status=error.code, error=repr(error))
        try:
            data = error.read(ERROR_BODY_LIMIT + 1)
            turn['error_body'] = data[:ERROR_BODY_LIMIT].decode('utf-8', errors='replace')
            turn['error_body_truncated'] = len(data) > ERROR_BODY_LIMIT
        except Exception as read_error:
            turn['error_body_read_error'] = repr(read_error)
        finally:
            error.close()
        raise
    except Exception as error:
        turn.update(status='fail', error=repr(error))
        raise
    finally:
        turn['wall_s'] = time.monotonic() - start
        try:
            data = native_path.read_bytes() if native_path.exists() else b''
            if len(data) < offset:
                raise ValueError('Native log was truncated during request')
            turn['native'] = data[offset:].decode('utf-8', errors='replace')
        except Exception as log_error:
            turn['native_log_error'] = repr(log_error)
        save()
    return turn


def verify_question_query(text, prompt_tokens, cached_tokens, query_limit, replay_limit):
    spans = [tuple(map(int, match)) for match in re.findall(
        r'^KVMEM_QUERY query_begin=(\d+) query_end=(\d+) prefix=(\d+)', text, re.M)]
    replays = [tuple(map(int, match)) for match in re.findall(
        r'^KVMEM_QUERY replay=1 reason=\S+ begin=(\d+) end=(\d+)', text, re.M)]
    if len(spans) != 1 or len(replays) != 1:
        raise ValueError('Question turn must contain its own Query span and replay')
    begin, end, prefix = spans[0]
    replay_begin, replay_end = replays[0]
    if not (cached_tokens <= begin == prefix and 0 < end - begin <= query_limit and end <= prompt_tokens):
        raise ValueError('Question Query span is outside its accepted prefix/token bounds')
    if not (replay_begin == begin and end <= replay_end <= prompt_tokens and 0 < replay_end - replay_begin <= replay_limit):
        raise ValueError('Question replay does not match its Query span/budget')
    return {'query_span': list(spans[0]), 'replay_span': list(replays[0]), 'scope': 'question_turn_only'}


def validate_performance_input(raw):
    if raw.get('smoke_only') or raw.get('measurement_kind') == 'smoke':
        raise ValueError('Smoke-only runs are runner validation, not performance samples')
    if raw.get('status') != 'pass':
        raise ValueError('Only completed, passing runs can be summarized')


def input_provenance(raw):
    provenance = {key: raw[key] for key in
                  ('started', 'workers', 'rounds', 'measurement_kind', 'reconstructed_harness') if key in raw}
    provenance['smoke_only'] = raw.get('smoke_only')
    return provenance
