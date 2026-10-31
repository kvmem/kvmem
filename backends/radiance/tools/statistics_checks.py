"""Compare evaluated native KV positions with API-emitted token counts.

Pinned Radiance scheduler.cpp commits ctx0 + 1 + n_accepted before its
per-token max_tokens sink clamp. A terminal verify can therefore evaluate
accepted positions which are never emitted. No later request may reuse this
generated suffix; the adapter resumes its separately retained prompt lease.
"""


def frontier_matches(item, row, *, draft_depth=0, output_limit=None, context_limit=0):
    usage = row.get('usage') or {}
    prompt = usage.get('prompt_tokens', 0)
    output = usage.get('completion_tokens', 0)
    if prompt <= 0 or output <= 0:
        return False
    total = prompt + output
    frontier = int(item.get('tokens', -1))
    if total - 1 <= frontier <= total:
        return True
    # Keep the ordinary bound for EOS, unknown limits, and non-speculative runs.
    capped = (output_limit is not None and output_limit > 0 and output == output_limit)
    capped = capped or (context_limit > 0 and total == context_limit)
    timing = row.get('timings') or {}
    return (row.get('finish_reason') == 'length' and capped and draft_depth > 0
            and timing.get('draft_n', 0) > 0
            and total < frontier <= total + draft_depth - 1)
