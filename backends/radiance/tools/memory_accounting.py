"""Read-only accounting of a recorded native KV allocation and .rad directory.

Does not choose a budget, change pool geometry, load weights or claim capacity
qualification. Geometry mirrors the pinned adapter for EXPLANATION only; native
allocation logs provide page costs. Container payload bytes are not GPU bytes.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re

from inspect_rad import inspect

MIB = 1024**2


def physical_tokens(budget, block, batch, reserve, native_blocks, recycle=False, prefix_demote=False):
    if min(budget, block, batch, reserve) <= 0 or not native_blocks or min(native_blocks) <= 0:
        raise ValueError('positive selection geometry required')
    block = math.lcm(block, *native_blocks)
    if budget % block or budget < 2 * block:
        raise ValueError('budget must cover sink and tail in complete aligned blocks')
    append = (math.ceil(batch / block) + 1) * block
    generation = math.ceil(reserve / block) * block
    if prefix_demote and not recycle:
        raise ValueError('prefix demotion requires page recycling')
    live = (budget if prefix_demote else 2 * budget) + 2 * append
    total = live + (0 if recycle else budget) + generation
    if total > 2**31 - 1:
        raise ValueError('native page index range exceeded')
    return {'aligned_block_tokens': block, 'append_slack_per_view': append,
            'live_and_scratch_or_prefix_tokens': live,
            'extra_prefix_allowance_tokens': 0 if recycle else budget,
            'page_recycle_enabled': recycle,
            'pressure_prefix_demote': prefix_demote,
            'generation_reserve_tokens': generation, 'physical_tokens': total}


def native_geometry(log):
    pools = re.findall(r'KVMEM_POOL resident_tokens=(\d+) carved_bytes=(\d+) logical_context=(\d+)', log)
    if len(pools) != 1:
        raise ValueError('need one completed native pool declaration, not an incomplete/planning log')
    tokens, carved, context = map(int, pools[0])
    pattern = (r'^\s+(\S+)\s+full\s+(\d+) layers\s+block\s+(\d+) tok\s+'
               r'page\s+([\d.]+) (B|KiB|MiB)\s+(\d+) blocks')
    groups = []
    for name, layers, block, size, unit, count in re.findall(pattern, log, re.M):
        page_bytes = float(size) * {'B': 1, 'KiB': 1024, 'MiB': MIB}[unit]
        if not page_bytes.is_integer():
            raise ValueError('rounded native page size cannot be treated as exact')
        block, count, page_bytes = int(block), int(count), int(page_bytes)
        if block * count != tokens:
            raise ValueError('group capacity differs from observer pool')
        groups.append({'name': name, 'layers': int(layers), 'page_tokens': block,
                       'page_bytes_from_log': page_bytes, 'blocks': count})
    if not groups:
        raise ValueError('no FULL native groups in log')
    paged = sum(g['page_bytes_from_log'] * g['blocks'] for g in groups)
    if not 0 < paged <= carved:
        raise ValueError('paged allocation inconsistent with carved pool')
    return {'resident_tokens': tokens, 'logical_context': context, 'carved_bytes': carved,
            'full_groups': groups, 'attention_bytes': paged,
            'non_full_or_alignment_bytes': carved - paged,
            'precision_note': 'Page sizes come from human-readable native log; confirm exact layout before changing allocations.'}


def compare_budgets(log, budgets, block, batch, reserve):
    native = native_geometry(log)
    demote = 'KVMEM_POOL_MODE native_recycle_v2 ' in log
    recycle = demote or 'KVMEM_POOL_MODE native_recycle_v1 ' in log
    rows = []
    for budget in budgets:
        geometry = physical_tokens(budget, block, batch, reserve,
                                   [g['page_tokens'] for g in native['full_groups']], recycle, demote)
        attention = sum(geometry['physical_tokens'] // g['page_tokens'] * g['page_bytes_from_log']
                        for g in native['full_groups'])
        total = attention + native['non_full_or_alignment_bytes']
        rows.append({'logical_selection_budget': budget, **geometry,
                     'calculated_attention_mib': attention / MIB,
                     'calculated_total_kv_mib': total / MIB,
                     'delta_vs_recorded_kv_mib': (total - native['carved_bytes']) / MIB,
                     'is_capacity_test': False})
    return {'recorded_native_pool': native, 'calculated_budget_scenarios': rows}


def weight_groups(entries, target_layers):
    if type(target_layers) is not int or target_layers <= 0:
        raise ValueError('explicit checkpoint target layer count required')
    totals = collections.Counter()
    classes = collections.Counter()
    for row in entries:
        name = row['name']
        layer = re.match(r'^blk\.(\d+)\.', name)
        if layer:
            group = 'target_blocks' if int(layer[1]) < target_layers else 'auxiliary_blocks'
        elif name.startswith('mtp.'):
            group = 'mtp_named_weights'
        elif name.startswith('token_embd.'):
            group = 'token_embedding'
        elif name.startswith('v.'):
            group = 'vision'
        elif name.startswith('output.'):
            group = 'target_output_head'
        else:
            group = 'other'
        totals[group] += row['bytes']
        classes[re.sub(r'^blk\.\d+\.', 'blk.*.', name)] += row['bytes']
    return {'container_payload_mib_by_group': {k: v / MIB for k, v in totals.items()},
            'container_largest_classes_mib': dict((k, v / MIB) for k, v in classes.most_common(15)),
            'scope': 'Stored payload only. Embedding/vision placement and runtime packing must be read from native logs, not inferred.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-log', required=True, type=Path)
    p.add_argument('--budgets', type=int, nargs='+', required=True)
    p.add_argument('--block', type=int, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--output-reserve', type=int, required=True)
    p.add_argument('--model', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if bool(a.model) != bool(a.config):
        p.error('--model and --config must be supplied together')
    data = a.native_log.read_bytes()
    result = compare_budgets(data.decode(), a.budgets, a.block, a.batch, a.output_reserve)
    result['source_log'] = str(a.native_log)
    result['source_log_sha256'] = hashlib.sha256(data).hexdigest()
    result['scope'] = 'Read-only accounting; no inference, no new capacity qualification, no model checksum validation.'
    if a.model:
        config = json.loads(a.config.read_text())
        config = config.get('text_config', config)
        container = inspect(a.model, include_entries=True)
        result['container'] = {'path': str(a.model), 'file_bytes': container['file_bytes'],
                               **weight_groups(container['weight_entries'], config['num_hidden_layers'])}
    value = json.dumps(result, indent=2) + '\n'
    if a.output:
        with a.output.open('x', encoding='utf-8') as out:
            out.write(value)
    else:
        print(value, end='')


if __name__ == '__main__':
    main()
