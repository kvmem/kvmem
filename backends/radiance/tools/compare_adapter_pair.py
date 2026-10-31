"""Compare a same-model/same-budget adapter A/B; never starts inference.

Different executable hashes are expected. Kernels, architecture plugins, SDK,
fixed inputs, effective command options and harness must still match. A speed
exception requires review, never automatic promotion to a larger budget.
"""
import argparse
import json
from pathlib import Path

from summarize_task2 import summarize


def normalized_command(command):
    result = []
    i = 1  # The adapter executable is the deliberately changed variable.
    while i < len(command):
        if command[i] in ('--port',):  # Harness-selected ephemeral HTTP port.
            i += 2
        else:
            result.append(command[i])
            i += 1
    return result


def compare(a, b):
    x, y = a.get('rounds', []), b.get('rounds', [])
    checks = {
        'drivers_pass': bool(a.get('pass') and b.get('pass')),
        'same_model': bool(a.get('model_sha256')) and a.get('model_sha256') == b.get('model_sha256'),
        'same_kernels': bool(a.get('kernel_sha256')) and a.get('kernel_sha256') == b.get('kernel_sha256'),
        'same_plugins': bool(a.get('plugin_sha256')) and a.get('plugin_sha256') == b.get('plugin_sha256'),
        'same_sdk': bool(a.get('sdk')) and a.get('sdk') == b.get('sdk'),
        'same_runtime': bool(a.get('runtime_libraries')) and a.get('runtime_libraries') == b.get('runtime_libraries'),
        'same_harness': bool(a.get('harness_sha256')) and a.get('harness_sha256') == b.get('harness_sha256'),
        'same_options': bool(a.get('command')) and bool(b.get('command')) and normalized_command(a['command']) == normalized_command(b['command']),
        'same_inputs': bool(x) and len(x) == len(y) and all(
            p.get('source_sha256') and p.get('source_sha256') == q.get('source_sha256') for p, q in zip(x, y)),
        'different_executables': bool(a.get('exe_sha256') and b.get('exe_sha256')) and a.get('exe_sha256') != b.get('exe_sha256'),
    }
    ratios, outputs_equal = [], []
    eligible = bool(x) and len(x) == len(y)
    for p, q in zip(x, y):
        u, v = p.get('timings', {}), q.get('timings', {})
        valid = min(u.get('predicted_n', 0), v.get('predicted_n', 0)) >= 32 and min(
            u.get('predicted_per_second', 0), v.get('predicted_per_second', 0)) > 0
        eligible = eligible and valid
        ratios.append(v['predicted_per_second'] / u['predicted_per_second'] if valid else None)
        outputs_equal.append(all(p.get(k) == q.get(k) for k in ('content', 'reasoning_content', 'finish_reason')))
    checks['eligible_outputs'] = eligible
    checks['every_round_decode_within_5_percent'] = bool(ratios) and all(r is not None and r >= .95 for r in ratios)
    return {
        'pass': all(checks.values()), 'checks': checks, 'round_decode_ratios': ratios,
        'round_outputs_identical': outputs_equal,
        'reference_exe_sha256': a.get('exe_sha256'), 'candidate_exe_sha256': b.get('exe_sha256'),
        'native_pools': [a.get('native_pool'), b.get('native_pool')],
        'windows_adapter_memory': [a.get('windows_adapter_memory'), b.get('windows_adapter_memory')],
        'linux_gpu_memory': [a.get('linux_gpu_memory'), b.get('linux_gpu_memory')],
        'scope': 'Same-budget adapter screen, not accuracy or larger-budget qualification. Any round regression >5% stops automatic progression; review acceptance/cycle/transfer evidence.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    a, b = [json.loads((p / 'result.json').read_text()) for p in (args.reference, args.candidate)]
    result = compare(a, b)
    result['summaries'] = [summarize(p) for p in (args.reference, args.candidate)]
    print(json.dumps(result, indent=2))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
