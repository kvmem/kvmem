"""Verify payload identity when a native converter changes only the draft head.

This reads complete containers; it never edits them. Common weight offsets may
move, so compare each logical weight's payload separately, not whole-file SHA.
"""
import argparse
import hashlib
import json
from pathlib import Path
from inspect_rad import inspect


def payload_hash(stream, entry):
    stream.seek(entry['offset'])
    remaining = entry['bytes']
    h = hashlib.sha256()
    while remaining:
        chunk = stream.read(min(remaining, 8 * 1024 * 1024))
        if not chunk:
            raise ValueError('truncated weight payload')
        remaining -= len(chunk)
        h.update(chunk)
    return h.hexdigest()


def compare(reference, candidate):
    a, b = (inspect(p, include_entries=True) for p in (reference, candidate))
    left = {e['name']: e for e in a['weight_entries'] if not e['name'].startswith('dflash.')}
    right = {e['name']: e for e in b['weight_entries']}
    missing = sorted(set(left) - set(right))
    extra = sorted(set(right) - set(left))
    equal, different = [], []
    with reference.open('rb') as x, candidate.open('rb') as y:
        for name in sorted(set(left) & set(right)):
            l, r = left[name], right[name]
            lh, rh = payload_hash(x, l), payload_hash(y, r)
            row = {'name': name, 'reference_sha256': lh, 'candidate_sha256': rh,
                   'reference_bytes': l['bytes'], 'candidate_bytes': r['bytes'],
                   'encoding_equal': l['encoding_sha256'] == r['encoding_sha256'],
                   'relative_planes_equal': l['relative_planes'] == r['relative_planes']}
            matches = lh == rh and l['shape'] == r['shape'] and row['encoding_equal'] and row['relative_planes_equal']
            (equal if matches else different).append(row)
    return {'reference': str(reference), 'candidate': str(candidate),
            'all_reference_non_dflash_payloads_equal': not missing and not different,
            'equal_count': len(equal), 'missing': missing, 'different': different,
            'candidate_extra': extra, 'payload_hashes': equal,
            'scope': 'weight payload identity; tokenizer, metadata and kernel arithmetic are separate checks'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'payload_hashes'}, indent=2))
    raise SystemExit(0 if result['all_reference_non_dflash_payloads_equal'] else 1)
