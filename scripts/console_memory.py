"""Validate the native planner's report for this terminal startup memory failure."""
import json
import re

MIB = 1024 * 1024
KV_STORAGE = {'bf16': 0, 'int8': 1, 'fp8': 2, 'rk8v4': 3, 'nvfp4': 4,
              'k8v4': 5, 'rk4v4': 6, 'rk4v4-e8': 7, 'rk2v4-e8': 8}
RESERVATION = re.compile(
    r'requested Engine runtime reservation requires (\d+) bytes, but only (\d+) '
    r'bytes are available for runtime capacity(?! on device)', re.I)
OOM = re.compile(r'cudaErrorMemoryAllocation|CUDA[^\r\n]{0,160}(?:out of memory|memory allocation failed)'
                 r'|(?:out of memory|memory allocation failed)[^\r\n]{0,160}CUDA'
                 r'|(?:minimum Engine runtime reservation|automatic KV headroom) requires \d+ bytes', re.I)


def diagnose(log, parameters, token_settings):
    log = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', log)
    fatal = list(re.finditer(r'(?m)^.*\bFATAL\b[^\r\n]*', log))
    if fatal:
        log = log[fatal[-1].start():]
    matches = list(RESERVATION.finditer(log))
    if not matches and not OOM.search(log):
        return None
    result = {'kind': 'gpu_memory', 'basis': 'native_planner', 'reason': '',
              'gpu': parameters.get('Gpu', ''), 'verified': False, 'options': []}
    if parameters.get('RecallStrategy', 'legacy') == 'layerwise':
        result['reason'] = 'layerwise_planning_unavailable'
        return result
    if parameters.get('Backend') != 'ninfer' or ',' in parameters.get('Gpu', '') or parameters.get('TensorParallel'):
        result['reason'] = 'single_ninfer_only'
        return result
    if not matches or ' | KVMEM_MEMORY_PLAN ' not in log:
        result['reason'] = 'missing_native_plan'
        return result
    try:
        required, available = map(int, matches[-1].groups())
        report_text = log[matches[-1].end():].split(' | KVMEM_MEMORY_PLAN ', 1)[1].splitlines()[0]
        if len(report_text) > 8192:
            raise ValueError('oversized report')
        report = json.loads(report_text)
        integer = lambda x: type(x) is int and 0 <= x < 2**63
        if (report['version'] != 1 or report['required_bytes'] != required
                or report['available_bytes'] != available or required <= available
                or not integer(report['safety_bytes']) or report['safety_bytes'] < 128*MIB
                or not integer(report['planning_us'])):
            raise ValueError('invalid admission evidence')
        effective = (token_settings or {}).get('effective', {})
        original = report['original']
        for key in ('Budget', 'Reserve', 'Prefill', 'Context'):
            if type(original[key]) is not int or original[key] != effective.get(key):
                raise ValueError('different launch settings')
        if (original['Concurrency'] != parameters.get('Concurrency', 1)
                or original['kv_storage'] != KV_STORAGE.get(parameters.get('KvType', 'int8'))):
            raise ValueError('different launch format')
        reserve = min(original['Reserve'], 4096)
        prefill = min(original['Prefill'], reserve) // 128 * 128
        minimum = max(128, max(1, effective['SinkTokens']//64)*64 + effective['RecentTokens']//64*64,
                      256-reserve)
        maximum = min(original['Budget'], original['Context']-reserve)
        options = report['options']
        if len(options) != 3 or [o['id'] for o in options] != ['int8', 'nvfp4', 'rk8v4']:
            raise ValueError('invalid formats')
        validated = []
        for item in options:
            if type(item['available']) is not bool:
                raise ValueError('invalid availability')
            option = {'id': item['id'], 'available': item['available'], 'reason': item['reason'],
                      'minimum_budget': minimum, 'safety_bytes': report['safety_bytes'], 'changes': {}}
            if item['available']:
                budget, runtime = item['budget'], item['runtime_bytes']
                fast = parameters.get('FastPrefill', False) if item['id'] != 'nvfp4' else False
                if (not integer(budget) or budget % 64 or not minimum <= budget <= maximum
                        or item['reserve'] != reserve or item['prefill'] != prefill or prefill < 128
                        or type(item['fast_prefill']) is not bool or item['fast_prefill'] != fast
                        or not integer(runtime) or runtime == 0
                        or runtime + report['safety_bytes'] > available):
                    raise ValueError('invalid recommendation')
                option.update(estimated_runtime_bytes=runtime, changes={
                    'KvType': item['id'], 'Budget': budget, 'Reserve': reserve,
                    'Prefill': prefill, 'FastPrefill': fast})
            elif item['reason'] not in ('minimum_does_not_fit', 'planning_unavailable'):
                raise ValueError('invalid unavailable reason')
            validated.append(option)
        result.update(required_bytes=required, available_bytes=available,
                      planning_us=report['planning_us'], graph_basis='static',
                      original={'Budget': original['Budget'], 'Reserve': original['Reserve'],
                                'KvType': parameters.get('KvType', 'int8')}, options=validated)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        result['reason'] = 'invalid_native_plan'
    return result


def validate_budget_plan(output, parameters, token_settings):
    """Bind a successful one-shot report to the exact effective launch settings."""
    try:
        lines = [line.removeprefix('KVMEM_BUDGET_PLAN ') for line in output.splitlines()
                 if line.startswith('KVMEM_BUDGET_PLAN ')]
        if len(lines) != 1 or len(lines[0]) > 8192:
            raise ValueError('missing or oversized budget plan')
        report = json.loads(lines[0])
        numeric = ('version', 'budget', 'minimum_budget', 'maximum_budget', 'reserve', 'prefill',
                   'context', 'concurrency', 'kv_storage', 'available_bytes', 'runtime_bytes',
                   'safety_bytes', 'planning_us')
        if any(type(report[k]) is not int or not 0 <= report[k] < 2**63 for k in numeric):
            raise ValueError('invalid numeric field')
        effective = token_settings['effective']
        minimum = max(128, max(1, effective['SinkTokens']//64)*64 + effective['RecentTokens']//64*64,
                      256-effective['Reserve'])
        maximum = (effective['Context']-effective['Reserve'])//64*64
        if (report['version'] != 1 or type(report['available']) is not bool
                or any(report[k] != effective[v] for k,v in
                       (('reserve','Reserve'),('prefill','Prefill'),('context','Context')))
                or report['concurrency'] != parameters.get('Concurrency',1)
                or report['kv_storage'] != KV_STORAGE.get(parameters.get('KvType','int8'))
                or report['minimum_budget'] != minimum or report['maximum_budget'] != maximum
                or report['safety_bytes'] != max(128*MIB, report['available_bytes']//20)):
            raise ValueError('different planning settings')
        if report['available']:
            if (not minimum <= report['budget'] <= maximum or report['budget'] % 64
                    or report['runtime_bytes'] <= 0
                    or report['runtime_bytes'] + report['safety_bytes'] > report['available_bytes']):
                raise ValueError('invalid budget boundary')
        elif report['budget'] != 0 or report['runtime_bytes'] != 0:
            raise ValueError('invalid unavailable result')
        return dict(report, basis='native_planner', verified=False,
                    changes={'Budget': report['budget']} if report['available'] else {})
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        raise ValueError('后端 budget 测算报告无效或与当前设置不符 / Invalid or mismatched budget plan') from error
