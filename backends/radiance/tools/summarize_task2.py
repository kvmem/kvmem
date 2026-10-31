"""Read saved Radiance task-two evidence; never launches a model.

Use README-style incremental input counts, not summed cache-miss tokens (which
include the deliberate native drafter-window refill on repeated requests).
"""
import argparse
import json
from pathlib import Path


def cached_followup(row):
    # Saved-request replay reports native cache counts but does not add the
    # canonical agent harness's retained_prefix convenience field.
    if 'retained_prefix' in row:
        return bool(row['retained_prefix'])
    cached = (row.get('timings') or {}).get('cache_n')
    if cached is None:
        cached = row.get('usage', {}).get('prompt_tokens_details', {}).get('cached_tokens', 0)
    return cached > 0


def summarize(folder):
    folder = Path(folder)
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    rows = result.get('rounds', [])
    if not rows or not all(r.get('timings') for r in rows):
        raise ValueError('missing completed requests/native timings')
    tool = rows[1:]
    pseconds = sum(r['timings']['prompt_ms'] for r in rows) / 1000
    dseconds = sum(r['timings']['predicted_ms'] for r in tool) / 1000
    final = rows[-1]
    final_prompt = final['usage']['prompt_tokens']
    # Same task-two convention as teacher's test-gdn-task2.py: first input's
    # initial prediction position is excluded once, not once per resumed request.
    new_positions = final_prompt - 1
    total_output = sum(r['timings']['predicted_n'] for r in tool)
    drafts = sum(r['timings'].get('draft_n', 0) for r in tool)
    accepted = sum(r['timings'].get('draft_n_accepted', 0) for r in tool)
    gpu = result.get('linux_gpu_memory', {})
    board = []
    for name, peak in gpu.get('peak_bytes', {}).items():
        total = gpu.get('initial', {}).get(name, {}).get('mem_info_vram_total')
        used = peak.get('mem_info_vram_used')
        if used is not None:
            board.append({'device': name, 'peak_MiB': used / 2**20,
                          'free_at_peak_MiB': (total-used)/2**20 if total is not None else None})
    available, swap, n = [], [], 0
    memory = folder / 'memory.jsonl'
    if memory.exists():
        for line in memory.read_text(encoding='utf-8').splitlines():
            sample = json.loads(line)
            n += 1
            if 'VmSwap' in sample:
                swap.append(int(sample['VmSwap'].split()[0]))
            # Keep raw sampling evidence authoritative if a version lacks this field.
            if 'system_MemAvailable' in sample:
                available.append(int(sample['system_MemAvailable'].split()[0]))
    rss = result.get('sampled_memory_peaks', {}).get('running', {}).get('VmRSS_KiB')
    return {
        'source': str(folder), 'driver_pass': result.get('pass'),
        'requests_completed': len(rows), 'final_prompt_tokens': final_prompt,
        'final_output_tokens': final['usage']['completion_tokens'],
        'context': result.get('agent_history', {}).get('close_to_context_limit'),
        'cached_followup_requests': sum(cached_followup(r) for r in tool),
        'followup_requests': len(tool), 'new_input_positions': new_positions,
        'total_prefill_seconds': pseconds,
        'first_pass_prefill_tps': None, # no separate first-pass timer: do not substitute native PP
        'effective_prefill_tps': new_positions / pseconds if pseconds else None,
        'tool_decode_tps_readme_count': total_output / dseconds if dseconds else None,
        'tool_decode_tps_native_count': (total_output - len(tool)) / dseconds if dseconds else None,
        'final_code_decode_tps_readme_count': final['timings']['predicted_n'] * 1000 / final['timings']['predicted_ms'],
        'final_code_decode_tps_native_count': final['timings']['predicted_per_second'],
        'draft_acceptance_percent': 100*accepted/drafts if drafts else None,
        'draft_tokens': drafts, 'accepted_draft_tokens': accepted,
        'request_wall_seconds': sum(r['request_seconds'] for r in rows),
        'board_memory': board, 'running_RSS_peak_MiB': rss/1024 if rss is not None else None,
        'process_swap_peak_MiB': max(swap)/1024 if swap else None,
        'system_available_min_MiB': min(available)/1024 if available else None,
        'memory_samples': n, 'exe_sha256': result.get('exe_sha256'),
        'kernel_sha256': result.get('kernel_sha256'), 'model_sha256': result.get('model_sha256'),
        'transport': result.get('agent_history', {}).get('transport'),
        'quality_separate': result.get('quality_checks'),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    data = summarize(args.folder)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if args.output:
        with args.output.open('x', encoding='utf-8') as out:
            out.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
