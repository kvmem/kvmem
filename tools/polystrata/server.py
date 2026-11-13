#!/usr/bin/env python3
"""Launch the pinned PolyStrata frontend with a separately built KVMem worker."""
from __future__ import annotations

import argparse
import threading
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from launcher.recall import LAYERWISE_RESOURCES, recall_args, require_recall_features, validate_recall
from launcher.worker import probe_worker


def normalize_tokens(args, parser):
    block = args.kvmem_block_tokens
    changes = []
    if args.recall_strategy == 'layerwise':
        for key in ('budget', 'reserve', 'sink', 'recent'):
            value = getattr(args, key)
            if value is None and key in ('sink', 'recent'):
                continue
            if type(value) is not int or value < (block if key in ('budget', 'reserve') else 0) or value > 2147483647 or value % block:
                parser.error(f'Layerwise {key} requires an exact native block-aligned value')
        if not 1 <= args.prefill <= 2147483647:
            parser.error('Prefill must be positive and <= 2147483647')
        return changes
    for key, label, minimum in (('budget', 'Budget', 128), ('reserve', 'Reserve', block),
                                ('sink', 'SinkTokens', 0), ('recent', 'RecentTokens', block)):
        requested = getattr(args, key)
        if not minimum <= requested <= 2147483647:
            parser.error(f'{label} must be between {minimum} and 2147483647')
        effective = requested // block * block
        if effective < minimum:
            parser.error(f'{label} must retain at least {minimum} tokens after alignment')
        setattr(args, key, effective)
        if effective != requested:
            changes.append(dict(parameter=label, requested=requested, effective=effective,
                                reason=f'alignment to {block} tokens'))
    if not 1 <= args.prefill <= 2147483647:
        parser.error('Prefill must be positive and <= 2147483647')
    if args.prefill > args.reserve:
        changes.append(dict(parameter='Prefill', requested=args.prefill, effective=args.reserve,
                            reason='Reserve limit'))
        args.prefill = args.reserve
    for change in changes:
        print(f"KVMEM_CONFIG {change['parameter']}: {change['requested']} -> {change['effective']} tokens ({change['reason']})", file=sys.stderr)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "worker", "model"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, help="explicit same-model export; omitted: export this GGUF to the local model-specific cache")
    parser.add_argument("--backend", choices=("cuda", "hip"), default="cuda")
    parser.add_argument("--gpu", required=True, help="CUDA GPU UUID, or HIP device ordinal with --backend hip")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--model-alias", default="kvmem-polystrata-qwen35")
    parser.add_argument("--chat-template-file", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18200)
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--budget", type=int, default=32768)
    parser.add_argument("--kvmem-block-tokens", type=int, choices=(32, 64), default=32,
                        help="KVMem block size in tokens; fixed for this worker (default: 32)")
    parser.add_argument("--reserve", type=int, default=1024)
    parser.add_argument("--host-mib", type=int)
    parser.add_argument("--disk-path", type=Path)
    parser.add_argument("--disk-mib", type=int, default=0)
    parser.add_argument("--verify-transfers", action="store_true", help="verify restored native KV and recurrent state bytes")
    parser.add_argument("--prefill", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking-mode", choices=("default", "on", "off"), default="default")
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"))
    parser.add_argument("--preserve-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vram-reserve-mib", type=int, default=256)
    parser.add_argument("--sink", type=int)
    parser.add_argument("--recent", type=int)
    parser.add_argument("--recall-strategy", choices=("legacy", "layerwise"), default="legacy")
    parser.add_argument("--layerwise-mtp-history", choices=("upstream-global", "first-layer"))
    parser.add_argument("--draft-method", choices=("none", "mtp", "guess"))
    for key, (flag, default) in LAYERWISE_RESOURCES.items():
        parser.add_argument(flag.replace('--kvmem-', '--'), type=int)
    parser.add_argument("--score-agg", choices=("mean", "max"), default=None)
    parser.add_argument("--query-scorer", choices=("page-softmax","meanq-cosine"), default="page-softmax")
    parser.add_argument("--graph-alpha", type=float)
    parser.add_argument("--query-max-tokens", type=int, default=512)
    parser.add_argument("--replay-max-tokens", type=int, default=2048)
    parser.add_argument("--kv-type", choices=("f16","q8_0","int8","int8k"), default="f16")
    parser.add_argument("--lrfu", action="store_true")
    parser.add_argument("--no-lrfu", action="store_true")
    parser.add_argument("--history-reuse", action=argparse.BooleanOptionalAction, default=True, help="reuse identical managed conversation prefixes")
    parser.add_argument("--retained-sessions", type=int, default=None, help="1..16 retained histories including the GPU owner; N>1 requires reuse")
    parser.add_argument("--cache-min-tokens", type=int, default=0, help="Minimum complete restorable history to retain; 0 disables filtering")
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=1,
                        help="1..8 active requests in one GPU worker; each slot owns B+R KV")
    parser.add_argument("--draft-model", type=Path, help="matching Gemma 4 assistant GGUF for MTP")
    parser.add_argument("--mtp-drafts", type=int, choices=range(4), default=0,
                        help="opt-in MTP draft tokens (1..3); Gemma requires --draft-model")
    parser.add_argument("--vision-worker", type=Path, help="strata-vision executable; requires --mmproj")
    parser.add_argument("--mmproj", type=Path, help="matching multimodal projection GGUF; requires --vision-worker")
    parser.add_argument("--vision-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--vision-tokens", type=int, default=1024,
                        help="maximum merged image tokens per image")
    parser.add_argument("--vision-threads", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument("--lrfu-lambda", type=float, default=0.5)
    parser.add_argument("--lrfu-top-frac", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.host_mib is None:
        args.host_mib = 16384 if args.recall_strategy == 'layerwise' else 4096
    if args.recall_strategy == 'legacy':
        if args.sink is None: args.sink = 64
        if args.recent is None: args.recent = 64
    draft_method = args.draft_method or ('mtp' if args.mtp_drafts else 'none')
    if (draft_method == 'none') != (args.mtp_drafts == 0):
        parser.error('draft method and physical draft count disagree')
    recall_parameters = dict(Backend='polystrata', RecallStrategy=args.recall_strategy,
        Model=str(args.model), Gpu=args.gpu, Concurrency=args.concurrency,
        SpeculativeMode=draft_method, DraftTokens=args.mtp_drafts, DraftPolicy='adaptive',
        KvType=args.kv_type, BlockTokens=args.kvmem_block_tokens, Reserve=args.reserve,
        Vision=bool(args.vision_worker), VisionResidency='cpu' if args.vision_device == 'cpu' else 'resident')
    if args.draft_model: recall_parameters['DraftModel'] = str(args.draft_model)
    if args.layerwise_mtp_history: recall_parameters['LayerwiseMtpHistory'] = args.layerwise_mtp_history
    for key, (flag, default) in LAYERWISE_RESOURCES.items():
        value = getattr(args, flag.removeprefix('--kvmem-').replace('-', '_'))
        if value is not None: recall_parameters[key] = value
    try: validate_recall(recall_parameters)
    except ValueError as error: parser.error(str(error))
    if args.thinking_budget is not None and not 0 <= args.thinking_budget <= 2147483647:
        parser.error("thinking budget must be a nonnegative integer")
    if (args.thinking_mode == "off" and args.reasoning_effort not in (None, "none")) or \
            (args.thinking_mode == "on" and args.reasoning_effort == "none"):
        parser.error("thinking mode conflicts with reasoning effort")
    if args.retained_sessions is None:args.retained_sessions=max(4,args.concurrency+1)
    if not 0 <= args.cache_min_tokens <= 2147483647:
        parser.error("cache-min-tokens must be between 0 and 2147483647")
    if not 0 <= args.disk_mib <= (1<<20) or bool(args.disk_path) != bool(args.disk_mib):
        parser.error("SSD requires both --disk-path and positive --disk-mib")
    if not 1 <= args.retained_sessions <= 16 or (args.retained_sessions > 1 and not args.history_reuse):
        parser.error("retained sessions must be 1..16 and multiple histories require --history-reuse")
    if bool(args.vision_worker) != bool(args.mmproj):
        parser.error("vision requires both --vision-worker and --mmproj")
    if not 1 <= args.vision_tokens <= 16384 or not 1 <= args.vision_threads <= 1024:
        parser.error("invalid vision token or thread limit")
    block = args.kvmem_block_tokens
    token_adjustments = normalize_tokens(args, parser)
    checks = [
        ((block if args.recall_strategy=='layerwise' else 128) <= args.budget <= 2147483647 and args.budget % block == 0, f"Budget must fit native block size {block}"),
        (block <= args.reserve <= 2147483647 and args.reserve % block == 0, f"Reserve must be >={block} and a multiple of {block}"),
        (args.budget + args.reserve <= args.context <= 2147483647, "Context must be >= Budget + Reserve"),
        (1 <= args.prefill <= (2147483647 if args.recall_strategy=='layerwise' else args.reserve), "Prefill exceeds its admitted limit"),
        (0 <= args.max_tokens <= args.context, "MaxTokens must be 0 (unlimited) or positive and <= Context"),
        (args.sink is None or args.sink >= 0 and args.sink % block == 0, f"SinkTokens must be nonnegative and a multiple of {block}"),
        (args.recent is None or args.recent >= (0 if args.recall_strategy=='layerwise' else block) and args.recent % block == 0, f"RecentTokens must fit native block size {block}"),
        (args.sink is None or args.recent is None or args.sink + args.recent <= args.budget, "SinkTokens + RecentTokens must be <= Budget"),
        (1 <= args.host_mib <= (1 << 24), "HostMiB is outside the supported range"),
        (1 <= args.port <= 65535, "Port must be 1..65535"),
        (256 <= args.vram_reserve_mib <= 65536, "VramReserveMiB must be 256..65536; the worker enforces a 256 MiB minimum"),
    ]
    for valid, message in checks:
        if not valid: parser.error(message)
    if args.lrfu and args.no_lrfu:
        parser.error("conflicting LRFU options")
    if args.query_scorer=="meanq-cosine":
        if args.score_agg is None: args.score_agg="max"
        if args.graph_alpha is None: args.graph_alpha=0
    else:
        if args.score_agg is None: args.score_agg="mean"
        if args.graph_alpha is None: args.graph_alpha=.2
    if not (math.isfinite(args.graph_alpha) and 0<=args.graph_alpha<=1 and
            1<=args.query_max_tokens<=8192 and 1<=args.replay_max_tokens<=65536):
        parser.error("invalid bounded Query or graph settings")
    if not (math.isfinite(args.lrfu_lambda) and 0 <= args.lrfu_lambda <= 1 and
            math.isfinite(args.lrfu_top_frac) and 0 < args.lrfu_top_frac <= 1):
        parser.error("invalid finite LRFU coefficients")
    for name in ("source", "worker", "tokenizer", "model", "vision_worker", "mmproj"):
        if getattr(args, name) is None:
            continue
        setattr(args, name, getattr(args, name).resolve())
    if not (args.source / "serve/server.py").is_file() or not (args.source / "serve/kvmem.py").is_file():
        parser.error("--source must contain the prepared PolyStrata frontend")
    if not args.worker.is_file() or not args.model.is_file():
        parser.error("worker or model is missing")
    automatic_tokenizer = args.tokenizer is None
    if automatic_tokenizer:
        from assets import tokenizer_path
        try:
            args.tokenizer = tokenizer_path(args.model, args.source, prepare=not args.dry_run)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    from assets import tokenizer_files
    if not (automatic_tokenizer and args.dry_run) and not all((args.tokenizer / name).is_file() for name in tokenizer_files(args.tokenizer)):
        parser.error("worker, model or same-GGUF exported tokenizer/template is missing")
    if args.vision_worker and (not args.vision_worker.is_file() or not args.mmproj.is_file()):
        parser.error("vision worker or mmproj is missing")
    gpu_uuids = [args.gpu.strip()]
    if args.backend == "cuda" and not re.fullmatch(r"GPU-[0-9a-fA-F-]{8,64}",gpu_uuids[0]):
        parser.error("this integration requires one GPU UUID")
    if args.backend == "hip" and (not re.fullmatch(r"[0-9]{1,10}", gpu_uuids[0]) or int(gpu_uuids[0]) > 2147483647):
        parser.error("the HIP backend requires one non-negative device ordinal")
    if args.backend == "hip":
        print("KVMEM_EXPERIMENTAL: HIP integration has not been qualified on AMD hardware; "
              "compile/link checks do not establish inference correctness.", file=sys.stderr)
    if args.host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("STRATA_API_KEY"):
        parser.error("a non-loopback listener requires STRATA_API_KEY")
    extra = args.worker_args[1:] if args.worker_args[:1] == ["--"] else args.worker_args
    # The console's native options cover both processes. Only serve.server owns MCP.
    frontend_args = []
    if "--mcp-config" in extra:
        index = extra.index("--mcp-config")
        if extra.count("--mcp-config") != 1 or index + 1 == len(extra):
            parser.error("--mcp-config requires exactly one configuration file")
        path = Path(extra[index + 1])
        if extra[index + 1].startswith("--") or not path.is_file():
            parser.error("--mcp-config requires an existing configuration file")
        frontend_args = ["--mcp-config", str(path.resolve())]
        del extra[index:index + 2]
    reserved = r"^--(kvmem-|model(?:=|$)|max-context(?:=|$)|prompt-batch(?:=|$)|window(?:=|$)|lookup(?:=|$)|draft-model(?:=|$)|prompt-cache|conversation-cache|batch(?:=|$)|slots(?:=|$)|kv(?:=|$)|layer-split|images|image-pad)"
    if any(re.match(reserved, option) for option in extra):
        parser.error("worker arguments may not override the wrapper's model, memory or M1 capability settings")
    if args.mtp_drafts and "--no-draft" in extra:
        parser.error("--no-draft conflicts with --mtp-drafts")
    if not any(option == "--vram-reserve-mib" or option.startswith("--vram-reserve-mib=") for option in extra):
        extra += ["--vram-reserve-mib", str(args.vram_reserve_mib)]
    # Preserve the mapped-expert default used before poly.9. Its automatic RAM
    # pool can reduce decode throughput; an explicit limit (or -2 for auto) wins.
    if not any(re.match(r"^--ram-experts-mib(?:$|=|\s)", option) for option in extra):
        extra += ["--ram-experts-mib", "-1"]
    worker_args = ["--model", str(args.model), "--max-context", str(args.context),
                   "--prompt-batch", str(args.prefill), "--window", str(1+args.mtp_drafts), "--lookup", "0",
                   "--prompt-cache", "0", "--conversation-cache-mib", "0", "--conversation-cache-slots", "0",
                   "--kvmem-budget", str(args.budget), "--kvmem-gen-reserve", str(args.reserve),
                   "--kvmem-block-tokens", str(block),
                   "--kvmem-host-mib", str(args.host_mib), "--kvmem-sink-tokens", str(args.sink),
                   "--kvmem-recent-tokens", str(args.recent), "--kvmem-score-agg", args.score_agg,
                   "--kvmem-query-scorer",args.query_scorer,"--kvmem-graph-alpha",str(args.graph_alpha),
                   "--kvmem-query-max-tokens",str(args.query_max_tokens),
                   "--kvmem-query-replay-max-tokens",str(args.replay_max_tokens),
                   "--kv","fp16" if args.kv_type=="f16" else args.kv_type]
    if args.recall_strategy == 'layerwise':
        omitted = {'--kvmem-score-agg', '--kvmem-query-scorer', '--kvmem-graph-alpha',
                   '--kvmem-query-max-tokens', '--kvmem-query-replay-max-tokens'}
        if args.sink is None: omitted.add('--kvmem-sink-tokens')
        if args.recent is None: omitted.add('--kvmem-recent-tokens')
        filtered = []
        index = 0
        while index < len(worker_args):
            if worker_args[index] in omitted:
                index += 2
            else:
                filtered.append(worker_args[index]); index += 1
        worker_args = filtered + recall_args(recall_parameters)
        if any(re.match(r'^--(?:kvmem(?:$|-)|mtp|no-draft|draft-model|window|batch)', option) for option in extra):
            parser.error('Use named layerwise settings; worker arguments cannot override the recall or draft owner')
        if not args.dry_run:
            try: require_recall_features(probe_worker(args.worker, 'polystrata'), recall_parameters)
            except ValueError as error: parser.error(str(error))
    if args.concurrency>1:
        worker_args += ["--batch",str(args.concurrency)]
    worker_args += ["--kvmem-mtp"] if args.mtp_drafts else ["--no-draft"]
    if args.verify_transfers:
        worker_args += ["--kvmem-verify-transfers"]
    if args.draft_model:
        if not args.mtp_drafts or not args.draft_model.is_file() or args.draft_model.suffix.lower() != '.gguf':
            parser.error('--draft-model requires enabled MTP and an existing matching GGUF')
        worker_args += ['--draft-model', str(args.draft_model.resolve())]
    if args.disk_path:
        worker_args += ["--kvmem-disk-path",str(args.disk_path.resolve()),"--kvmem-disk-mib",str(args.disk_mib)]
    if args.recall_strategy == 'layerwise':
        pass
    elif args.lrfu or (args.query_scorer=="meanq-cosine" and not args.no_lrfu):
        worker_args += ["--kvmem-lrfu", "--kvmem-lrfu-lambda", str(args.lrfu_lambda),
                        "--kvmem-lrfu-top-frac", str(args.lrfu_top_frac)]
    else:
        worker_args += ["--no-kvmem-lrfu"]
    if args.history_reuse:
        worker_args += ["--kvmem-history-reuse"]
    if args.retained_sessions != 1:
        worker_args += ["--kvmem-sessions", str(args.retained_sessions)]
    if args.cache_min_tokens:
        worker_args += ["--kvmem-cache-min-tokens", str(args.cache_min_tokens)]
        if not args.dry_run:
            from launcher.worker import require_launch_features
            try: require_launch_features(probe_worker(args.worker, 'polystrata'),
                                         {**recall_parameters, 'CacheMinTokens': args.cache_min_tokens})
            except ValueError as error: parser.error(str(error))
    worker_args += extra
    indices = [0]
    if args.backend == "hip":
        indices = [int(gpu_uuids[0])]
    elif not args.dry_run:
        devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
        found = {row.split(",")[1].strip(): int(row.split(",")[0]) for row in devices.splitlines()}
        if any(uuid not in found for uuid in gpu_uuids):
            parser.error("one or more GPU UUIDs were not found")
        indices = [found[uuid] for uuid in gpu_uuids]
    config = {
        "token_adjustments": token_adjustments,
        "exe": str(args.worker), "args": worker_args, "cwd": str(args.source),
        "tokenizer": str(args.tokenizer), "model_name": args.model_alias,
        "backend": args.backend, "gpu": indices[0],
        "env": {"KVMEM_POLY_STORAGE_METRICS": "1",
                "KVMEM_POLY_SESSION_METRICS": "1",
                "STRATA_FATTN": os.environ.get("STRATA_FATTN", "1"),
                "STRATA_FDEC": os.environ.get("STRATA_FDEC", "1")},
        "warm_start": False, "kvmem_default_max_tokens": args.max_tokens or None,
    }
    if args.backend == "cuda":
        config["env"]["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        config["kvmem_monitor_gpu_uuid"] = gpu_uuids[0]
    if args.log:
        config["log"]=str(args.log.resolve())
    effort = args.reasoning_effort
    if args.thinking_mode == "off": effort = "none"
    elif args.thinking_mode == "on" and effort is None: effort = "medium"
    if effort is not None: config["reasoning_effort"] = effort
    if args.thinking_budget is not None: config["reasoning_budget_tokens"] = args.thinking_budget
    config["preserve_thinking"] = args.preserve_thinking
    config["kvmem_mtp_policy"] = "native-adaptive" if args.mtp_drafts else "off"
    config["kvmem_mtp_drafts_max"] = args.mtp_drafts
    config["client_reasoning_history"] = True
    config["kvmem_console_only"] = True
    config["kvmem_vram_reserve_mib"] = args.vram_reserve_mib
    if args.chat_template_file:
        config["kvmem_chat_template"]=str(args.chat_template_file.resolve())
    if args.vision_worker:
        config["vision"] = {
            "exe": str(args.vision_worker), "mmproj": str(args.mmproj), "model": str(args.model),
            "gpu": args.vision_device == "gpu", "max_tokens": args.vision_tokens,
            "threads": args.vision_threads,
        }
        if args.vision_device == "gpu":
            config["vision"]["cuda_device"] = indices[0]
    if args.dry_run:
        config["tokenizer_preparation"] = "automatic_from_model_on_start" if automatic_tokenizer else "explicit_export"
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    sys.path.insert(0, str(args.source))
    from serve import server
    with tempfile.TemporaryDirectory(prefix="kvmem-polystrata-") as directory:
        # Keep native counters available to the monitor even without an explicit log path.
        if not args.log:
            config['log'] = str(Path(directory) / 'native.log')
        path = Path(directory) / "strata-kvmem.json"
        path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        sys.argv = ["polystrata-server", "--engine", "strata", "--config", str(path),
                    "--host", args.host, "--port", str(args.port)] + frontend_args
        # The native worker writes to its private log. Forward only its verified
        # scoring contract so the supervising console can confirm the launch.
        stop = threading.Event()
        def relay_scoring():
            seen = set()
            while not stop.wait(.2):
                try:
                    with open(config['log'], encoding='utf-8', errors='replace') as stream:
                        lines = stream.read(65536).splitlines(keepends=True)
                except OSError:
                    continue
                for line in lines:
                    marker = line.split(' ', 1)[0]
                    recall_report = marker == 'INFO' and 'recall_strategy=layerwise' in line
                    if line.endswith('\n') and (marker in {'KVMEM_SCORING', 'KVMEM_LIMITS', 'KVMEM_BLOCKS'} or recall_report) and marker not in seen:
                        print(line.rstrip(), flush=True)
                        seen.add(marker)
                required = {'INFO', 'KVMEM_BLOCKS'} if args.recall_strategy == 'layerwise' else {'KVMEM_SCORING', 'KVMEM_LIMITS', 'KVMEM_BLOCKS'}
                if required.issubset(seen):
                    return
        thread = threading.Thread(target=relay_scoring, daemon=True)
        thread.start()
        try:
            return server.main()
        finally:
            stop.set()
            thread.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())
