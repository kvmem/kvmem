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
    try:
        run = subprocess.run([str(path), '--kvmem-capabilities'], capture_output=True,
            encoding='utf-8', errors='replace', timeout=10, cwd=path.parent,
            env={**os.environ, **(environment or {}), 'CUDA_VISIBLE_DEVICES': '-1', 'HIP_VISIBLE_DEVICES':'-1', 'ROCR_VISIBLE_DEVICES':'-1'},
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        value = json.loads(run.stdout)
        if (run.returncode or not isinstance(value, dict) or value.get('schema_version') != 1
                or value.get('backend') != backend or value.get('query_scoring_revision', 0) < 2
                or not value.get('independent_graph_lrfu') or not value.get('signed_lrfu_hits')):
            raise ValueError('incompatible worker contract')
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise ValueError('Worker does not support this launch contract; update the complete package. ' + str(path)) from error
    digest = worker_digest(path)
    if _identity(path) != identity:
        raise ValueError('Worker changed during capability probing; retry after the update completes')
    value.update(executable_sha256=digest, path=str(path))
    return value


def require_launch_features(capabilities, parameters):
    if (parameters.get('Backend') == 'llamacpp' and '--mcp-servers-config' in parameters.get('WorkerArgs', [])
            and capabilities.get('mcp_stdio') is not True):
        raise ValueError('Update the llama.cpp worker to a version with native stdio MCP support')
    if parameters.get('Lora') and capabilities.get('fixed_lora') is not True:
        raise ValueError('This worker does not support fixed LoRA; update the llama.cpp worker')
    require_recall_features(capabilities, parameters)
    checkpoint_revision = capabilities.get('checkpoint_recompute_revision')
    if parameters.get('CheckpointExtraRecomputeTokens', 0) and (
            type(checkpoint_revision) is not int or checkpoint_revision < 1):
        raise ValueError('This worker does not support checkpoint recompute limits; update the complete NInfer package')
    revision = capabilities.get('heads_tensor_parallel_revision')
    if parameters.get('TensorParallel') and (type(revision) is not int or revision < 1):
        raise ValueError('This worker does not support KVMem heads tensor parallelism; update the complete NInfer package')
