"""Read the worker contract without loading a model or initializing a GPU."""
import copy
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
from .recall import require_recall_features, resource_defaults, token_constraints, mtp_history_default


def strategy_profile(capabilities, selected):
    """Return only an explicitly declared strategy, never a legacy substitute."""
    if selected not in ('legacy', 'layerwise'):
        raise ValueError('Unknown recall strategy')
    profiles = capabilities.get('recall_strategies', {})
    profile = profiles.get(selected) if isinstance(profiles, dict) else None
    if selected == 'layerwise' and (type(capabilities.get('recall_strategy_revision')) is not int
            or capabilities['recall_strategy_revision'] < 1 or not isinstance(profile, dict)
            or profile.get('current_layer_q') is not True or profile.get('query_replay') is not False
            or profile.get('position_policy') != 'original'):
        return None
    if selected == 'layerwise' and isinstance(profile, dict):
        resource_defaults(profile)
        token_constraints(profile)
        specs = profile.get('speculative', {})
        if isinstance(specs, dict):
            for spec in specs.values():
                if not isinstance(spec, dict):
                    continue
                token_constraints(profile, spec)
                mtp_history_default(spec)
                policies = spec.get('history_policy_caps', {})
                if isinstance(policies, dict):
                    for policy in policies.values():
                        if isinstance(policy, dict):
                            token_constraints(profile, policy)
    return copy.deepcopy(profile) if isinstance(profile, dict) else None


def _identity(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


@lru_cache(maxsize=16)
def _digest(path, identity):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): hasher.update(chunk)
    if _identity(Path(path)) != identity:
        raise ValueError('Worker changed during verification; retry after the update completes')
    return hasher.hexdigest()


def worker_digest(path):
    path = Path(path).resolve()
    return _digest(str(path), _identity(path))


def probe_worker(worker, backend, environment=None):
    path = Path(worker).resolve()
    identity = _identity(path)
    probe = dict(capture_output=True, encoding='utf-8', errors='replace', timeout=10, cwd=path.parent,
                 env={**os.environ, **(environment or {}), 'CUDA_VISIBLE_DEVICES': '-1',
                      'HIP_VISIBLE_DEVICES': '-1', 'ROCR_VISIBLE_DEVICES': '-1'},
                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        run = subprocess.run([str(path), '--kvmem-capabilities'], **probe)
        value = json.loads(run.stdout)
        if (run.returncode or not isinstance(value, dict) or value.get('schema_version') != 1
                or value.get('backend') != backend or value.get('query_scoring_revision', 0) < 2
                or not value.get('independent_graph_lrfu') or not value.get('signed_lrfu_hits')):
            raise ValueError('incompatible worker contract')
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise ValueError('Worker does not support this launch contract; update the complete package. ' + str(path)) from error
    if backend == 'ninfer':
        if _identity(path) != identity:
            raise ValueError('Worker changed during capability probing; retry after the update completes')
        # The native capability document predates these switches. With no
        # artifact argument, successful parsing terminates before loading a model.
        try:
            switches = subprocess.run([str(path), '--thinking', '--no-preserve-thinking'], **probe)
            value['explicit_thinking_switches'] = bool(
                switches.returncode and 'artifact path is required' in switches.stderr)
        except (OSError, subprocess.TimeoutExpired):
            value['explicit_thinking_switches'] = False
    digest = worker_digest(path)
    if _identity(path) != identity:
        raise ValueError('Worker changed during capability probing; retry after the update completes')
    value.update(executable_sha256=digest, path=str(path))
    return value


def supports_inkling_fp8(capabilities):
    models = capabilities.get('model_kv_formats')
    formats = models.get('inkling2_loop') if isinstance(models, dict) else None
    return isinstance(formats, list) and 'fp8' in formats


def require_launch_features(capabilities, parameters):
    if (parameters.get('Backend') == 'ninfer' and capabilities.get('explicit_thinking_switches') is False
            and (parameters.get('ThinkingMode') == 'on' or parameters.get('PreserveThinking') is False)):
        raise ValueError('当前 NInfer 无法在启动时显式开启思考或关闭历史思考保留，请更新后端。 / '
                         'Update the NInfer worker to explicitly enable thinking or disable reasoning history.')
    if (parameters.get('Backend') == 'llamacpp' and '--mcp-servers-config' in parameters.get('WorkerArgs', [])
            and capabilities.get('mcp_stdio') is not True):
        raise ValueError('Update the llama.cpp worker to a version with native stdio MCP support')
    if parameters.get('Lora') and capabilities.get('fixed_lora') is not True:
        raise ValueError('This worker does not support fixed LoRA; update the llama.cpp worker')
    if parameters.get('Backend') == 'polystrata' and parameters.get('KvType') == 'fp8' and not supports_inkling_fp8(capabilities):
        raise ValueError('This worker does not support Inkling FP8 KV with KVMem')
    require_recall_features(capabilities, parameters)
    if (parameters.get('Backend') == 'ninfer' and parameters.get('RecallStrategy', 'legacy') == 'legacy'
            and any(parameters.get(key, 0) == 0 for key in ('QueryMaxTokens', 'QueryReplayMaxTokens'))):
        revision = capabilities.get('legacy_query_replay_revision')
        if type(revision) is not int or revision < 1:
            raise ValueError('This worker does not support NInfer full Query/replay (zero token limits); '
                             'update the complete NInfer package, or explicitly set both QueryMaxTokens '
                             'and QueryReplayMaxTokens to positive limits')
    admission_revision = capabilities.get('cache_admission_revision')
    if parameters.get('CacheMinTokens', 0) and (
            type(admission_revision) is not int or admission_revision < 1):
        raise ValueError('This worker does not support CacheMinTokens; update the complete backend package')
    checkpoint_revision = capabilities.get('checkpoint_recompute_revision')
    if parameters.get('CheckpointExtraRecomputeTokens', 0) and (
            type(checkpoint_revision) is not int or checkpoint_revision < 1):
        raise ValueError('This worker does not support checkpoint recompute limits; update the complete NInfer package')
    revision = capabilities.get('heads_tensor_parallel_revision')
    if parameters.get('TensorParallel') and (type(revision) is not int or revision < 1):
        raise ValueError('This worker does not support KVMem heads tensor parallelism; update the complete NInfer package')
