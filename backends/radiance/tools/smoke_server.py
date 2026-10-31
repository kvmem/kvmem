"""Bounded engine bring-up probe; not a KVMem retrieval acceptance benchmark."""
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request

HASH_METHODS = {}

def digest(path):
    path = path.resolve()
    # Large weights on a Windows-mounted volume otherwise cross the WSL file
    # boundary for EVERY hash. Hash the same file natively, still in full, with
    # a read-only sharing lock and metadata stability checks. No cached digest.
    if os.environ.get("WSL_INTEROP") and re.match(r"^/mnt/[a-z]/", str(path)):
        helper = Path(__file__).with_name("hash_windows_file.ps1")
        try:
            before = path.stat()
            native = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
            native_helper = subprocess.check_output(["wslpath", "-w", str(helper)], text=True).strip()
            result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-File", native_helper, "-LiteralPath", native], text=True, capture_output=True, check=True, timeout=180)
            value = json.loads(result.stdout.lstrip("\ufeff"))
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or value["bytes"] != after.st_size:
                raise RuntimeError("file changed across native hashing")
            if not re.fullmatch(r"[a-f0-9]{64}", value["sha256"]):
                raise ValueError("invalid SHA-256 response")
            HASH_METHODS[str(path)] = "Windows native full-file SHA-256 via read-only stream"
            return value["sha256"]
        except (OSError, subprocess.SubprocessError, ValueError, KeyError):
            pass # portable full-file fallback; no unverified hash is substituted
    HASH_METHODS[str(path)] = "Python full-file SHA-256"
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_evidence(path):
    if path.is_file():
        return {"model_bytes": path.stat().st_size, "model_sha256": digest(path)}
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in
                   {".json", ".safetensors", ".jinja", ".txt"})
    if not any(p.suffix == ".safetensors" for p in files):
        raise FileNotFoundError(f"no checkpoint weights in {path}")
    return {"model_bytes": sum(p.stat().st_size for p in files),
            "model_files": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sdk", required=True, type=Path)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--embedding-placement", choices=("ram", "vram"), default="vram")
    parser.add_argument("--gpu-headroom-mib", type=int, default=1024)
    parser.add_argument("--min-system-available-mib", type=int, default=1024,
                        help="stop only this server if system available RAM stays below this for 3 seconds; 0 disables")
    parser.add_argument("--windows-gpu-counters", action="store_true",
                        help="WSL: record Windows adapter-wide dedicated/shared memory, including other processes")
    parser.add_argument("--linux-gpu-counters", action="store_true",
                        help="require native Linux AMD board VRAM/GTT counters; not process VRAM")
    parser.add_argument("--max-process-swap-mib", type=int, default=0,
                        help="stop this test if process swap exceeds this for 3 seconds; 0 disables")
    parser.add_argument("--observe-statistics", action="store_true",
                        help="require the real engine Q/K observer evidence; not a retrieval test")
    parser.add_argument("--selection-budget", type=int, default=0)
    parser.add_argument("--selection-block", type=int, default=128)
    parser.add_argument("--sink-tokens", type=int, default=None,
                        help="explicit always-kept prefix; omitted uses the engine/common KVMem default")
    parser.add_argument("--recent-tokens", type=int, default=None,
                        help="explicit always-kept suffix; omitted uses the engine/common KVMem default")
    parser.add_argument("--replay-tokens", type=int, default=1024)
    parser.add_argument("--host-archive-mib", type=int, default=256)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--drafter", choices=("auto", "off", "mtp", "dflash"), default="auto",
                        help="reuse native drafter; MTP requires MTP weights, never silently falls back to DFlash")
    parser.add_argument("--speculative-tokens", type=int, default=None,
                        help="native Radiance draft depth; 0 is the historical non-speculative baseline")
    parser.add_argument("--kv-cache-dtype", choices=("bf16", "fp8"), default="bf16",
                        help="native Radiance KV encoding; explicit in every recorded command")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--rounds", type=int, default=1, choices=range(1, 6),
                        help="one to five growing-history short requests; not the final 33-round test")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--require-bounded-pool", action="store_true",
                        help="require logical input longer than the actual native pool")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--require-frozen-query", action="store_true")
    parser.add_argument("--require-query-replay", action="store_true")
    parser.add_argument("--require-prefix-resume", action="store_true",
                        help="require exact native prefix/state reuse on every request after the first")
    parser.add_argument("--check-reserve-rejection", action="store_true",
                        help="send an over-reserve request first, require HTTP 400, then prove service remains healthy")
    parser.add_argument("--expected-answer", default="OK")
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--cases", type=Path, help="JSON list of explicit messages/expected records")
    parser.add_argument("--agent-requests", type=int, choices=range(1, 34), default=0,
                        help="teacher source-file workload: 1/5 development gates, 33 final context gate")
    parser.add_argument("--agent-chunk-tokens", type=int, default=8192)
    parser.add_argument("--agent-greedy", action="store_true",
                        help="diagnostic only: replace teacher temperature 1 with 0; recorded in requests")
    parser.add_argument("--agent-min-decode-tps", type=float, default=0,
                        help="pause for inspection below this decode rate on a >=32-token agent response; 0 disables")
    parser.add_argument("--agent-max-draft-step-ms", type=float, default=0,
                        help="fixed-depth draft-cycle wall-time guard; 0 disables, not a kernel timer")
    parser.add_argument("--agent-performance-pause-seconds", type=int, default=0,
                        help="on performance guard, retain this service for explicit review up to this many seconds; 0 exits immediately")
    args = parser.parse_args()
    from draft_control import resolve as resolve_drafter, verify_log as verify_drafter_log
    try:
        drafter = resolve_drafter(args.model, args.drafter, args.speculative_tokens)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.speculative_tokens = drafter["depth"]
    if args.agent_requests and (args.cases or args.prompt_file or args.disable_thinking or args.rounds != 1):
        parser.error("agent workload controls its own messages/thinking; do not mix short-fixture options")
    expected_rounds = args.agent_requests or args.rounds
    if args.agent_chunk_tokens <= 0:
        parser.error("agent chunk size must be positive")
    if args.speculative_tokens < 0:
        parser.error("speculative tokens must be nonnegative")
    if args.gpu_headroom_mib < 0:
        parser.error("GPU headroom must be nonnegative")
    if args.min_system_available_mib < 0:
        parser.error("minimum available RAM must be nonnegative")
    if args.max_process_swap_mib < 0:
        parser.error("maximum process swap must be nonnegative")
    if not 0 <= args.agent_performance_pause_seconds <= 3600:
        parser.error("performance review wait must be between 0 and 3600 seconds")
    if args.agent_performance_pause_seconds and (not args.agent_requests or
            not (args.agent_min_decode_tps > 0 or args.agent_max_draft_step_ms > 0)):
        parser.error("performance review pause needs an agent workload and an enabled speed guard")
    if min(args.replay_tokens, args.host_archive_mib, args.batch) <= 0:
        parser.error("replay, archive and batch limits must be positive")
    if any(x is not None and x < -1 for x in (args.sink_tokens, args.recent_tokens)):
        parser.error("keep bands must be -1 (auto) or nonnegative token counts")
    if not args.selection_budget and any(x is not None for x in (args.sink_tokens, args.recent_tokens)):
        parser.error("keep bands require selection")
    cases = json.loads(args.cases.read_text()) if args.cases else None
    if cases is not None and (not isinstance(cases, list) or len(cases) != args.rounds):
        parser.error("--cases must contain exactly --rounds records")
    args.output.mkdir(parents=True, exist_ok=False)
    for path in (args.exe, args.home / "kernels" / "libr4d.so"):
        if not path.is_file():
            raise FileNotFoundError(path)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [str(args.exe), "--model", str(args.model), "--radiance-home", str(args.home),
               "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "bringup",
               "--tp", "1", "--max-model-len", str(args.context), "--max-num-seqs", "1",
               "--max-num-batched-tokens", str(args.batch), "--num-speculative-tokens", str(args.speculative_tokens),
               "--gpu-headroom-mib", str(args.gpu_headroom_mib), "--kv-cache-dtype", args.kv_cache_dtype,
               "--placement", "all_vram", "--embedding-placement", args.embedding_placement,
               "--host-pool-mib", "0", "--prefix-cache-host-mib", "0",
               "--mm-max-patches", "0",
               "--no-prefix-cache", "--http-threads", "4", "--deterministic"]
    if args.selection_budget:
        if not args.observe_statistics or args.selection_budget < 1 or args.selection_block < 1:
            parser.error("selection requires statistics observation and positive geometry")
        command += ["--kvmem-budget", str(args.selection_budget),
                    "--kvmem-block-tokens", str(args.selection_block),
                    "--kvmem-replay-tokens", str(args.replay_tokens),
                    "--kvmem-host-mib", str(args.host_archive_mib)]
        for flag, value in (("--kvmem-sink-tokens", args.sink_tokens), ("--kvmem-recent-tokens", args.recent_tokens)):
            if value is not None:
                command += [flag, str(value)]
    if args.require_bounded_pool and not args.selection_budget:
        parser.error("bounded pool qualification requires selection")
    if args.check_reserve_rejection and (not args.selection_budget or args.context < args.replay_tokens + 1024):
        parser.error("reserve rejection check needs selection and context >= replay_tokens + 1024")
    env = os.environ.copy()
    if "microsoft" in Path('/proc/sys/kernel/osrelease').read_text().lower():
        env["HSA_ENABLE_DXG_DETECTION"] = "1"
    else:
        env.pop("HSA_ENABLE_DXG_DETECTION", None)
    env["LD_LIBRARY_PATH"] = f"{args.sdk}/lib:{args.sdk}/lib/llvm/lib"
    purpose = "Radiance engine Q/K statistics bridge (retrieval not enabled)" if args.observe_statistics else \
              "Radiance standalone engine bring-up (KVMem not connected)"
    if args.selection_budget:
        purpose = "KVMem/Radiance single-rank bounded working-set qualification"
    report = {"purpose": purpose,
              "drafter_selection": drafter,
              "harness_sha256": {name: digest(Path(__file__).with_name(name))
                  for name in ("smoke_server.py", "agent_history.py")},
              "command": command, "exe_sha256": digest(args.exe),
              "kernel_sha256": digest(args.home / "kernels" / "libr4d.so"),
              "model": str(args.model), **model_evidence(args.model),
              "sdk": str(args.sdk), "pass": False}
    if args.drafter != "auto" and "RADIANCE_NO_DRAFT" in env:
        parser.error("unset RADIANCE_NO_DRAFT: it prevents drafting but retains drafter weights; use --drafter off")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path, payload=None, timeout=2):
        data = json.dumps(payload).encode() if payload is not None else None
        q = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                   headers={"Content-Type": "application/json"})
        with opener.open(q, timeout=timeout) as response:
            return json.load(response)
    child = None
    gpu_sampler = None
    gpu_log = None
    gpu_stop = args.output / "gpu-sampler.stop"
    stop = threading.Event()
    thread = None
    phase = "loading"
    memory_guard = {"minimum_available_mib": args.min_system_available_mib,
                    "continuous_seconds": 3, "triggered": False}
    report["memory_guard"] = memory_guard
    memory_guard["maximum_process_swap_mib"] = args.max_process_swap_mib
    linux_cards = []
    start = time.monotonic()
    try:
        if args.linux_gpu_counters:
            from linux_probe_metrics import discover_amd_cards, sample_amd_cards
            linux_cards = discover_amd_cards()
            if not linux_cards:
                raise RuntimeError('native AMD board counters requested but none found')
            report["linux_gpu_memory"] = {
                "scope": "AMD board-wide VRAM/GTT, includes all processes; not process VRAM",
                "initial": sample_amd_cards(linux_cards), "peak_bytes": {}}
        if args.windows_gpu_counters:
            def windows_path(path):
                return subprocess.check_output(["wslpath", "-w", str(path.resolve())], text=True).strip()
            gpu_log = (args.output / "gpu-sampler.log").open("x")
            # Process-scoped policy only for this local read-only counter helper;
            # do not change the machine/user execution policy or security tools.
            gpu_sampler = subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass", "-File",
                windows_path(Path(__file__).with_name("sample_windows_gpu.ps1")),
                "-OutputPath", windows_path(args.output / "gpu-adapters.csv"),
                "-StopPath", windows_path(gpu_stop),
                "-MaxSeconds", str(args.startup_timeout + expected_rounds * (1800 if args.agent_requests else 180) + 60)],
                stdout=gpu_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        with (args.output / "server.stdout.log").open("x") as out, \
                (args.output / "server.stderr.log").open("x") as err:
            child = subprocess.Popen(command, env=env, stdout=out, stderr=err,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            report["pid"] = child.pid
            def sample_impl():
                low_since = None
                swap_since = None
                last_gpu_sample = -1.0
                with (args.output / "memory.jsonl").open("x") as stream:
                    while not stop.wait(0.1):
                        try:
                            status = Path(f"/proc/{child.pid}/status").read_text()
                        except FileNotFoundError:
                            break
                        row = {"elapsed_s": round(time.monotonic() - start, 3), "phase": phase}
                        for line in status.splitlines():
                            key, _, value = line.partition(":")
                            if key in {"VmRSS", "VmHWM", "RssAnon", "RssFile", "VmSwap", "Threads"}:
                                row[key] = value.strip()
                        # Process residency and system pressure are distinct.
                        # Record the latter as well for long, unattended runs.
                        for line in Path('/proc/meminfo').read_text().splitlines():
                            key, _, value = line.partition(':')
                            if key in {'MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree', 'Committed_AS', 'CommitLimit'}:
                                row['system_' + key] = value.strip()
                        now = time.monotonic()
                        if linux_cards and now - last_gpu_sample >= 1:
                            row['linux_gpu'] = sample_amd_cards(linux_cards)
                            last_gpu_sample = now
                            for card, counters in row['linux_gpu'].items():
                                peaks = report['linux_gpu_memory']['peak_bytes'].setdefault(card, {})
                                for field in ('mem_info_vram_used', 'mem_info_gtt_used'):
                                    if field in counters:
                                        peaks[field] = max(peaks.get(field, 0), counters[field])
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        available_kib = int(row['system_MemAvailable'].split()[0])
                        swap_kib = int(row.get('VmSwap', '0 kB').split()[0])
                        if args.max_process_swap_mib and swap_kib > args.max_process_swap_mib * 1024:
                            swap_since = now if swap_since is None else swap_since
                        else:
                            swap_since = None
                        if swap_since is not None and now - swap_since >= 3:
                            memory_guard.update(triggered=True, reason='process swap limit', evidence=row)
                            (args.output / 'memory-guard.json').write_text(json.dumps(memory_guard, indent=2))
                            if child.poll() is None:
                                child.terminate()
                            break
                        if args.min_system_available_mib and available_kib < args.min_system_available_mib * 1024:
                            now = time.monotonic()
                            low_since = now if low_since is None else low_since
                            if now - low_since >= 3:
                                memory_guard.update(triggered=True, reason='system available RAM', evidence=row)
                                (args.output / "memory-guard.json").write_text(json.dumps(memory_guard, indent=2))
                                if child.poll() is None:
                                    child.terminate()
                                break
                        else:
                            low_since = None
            def sample():
                try:
                    sample_impl()
                except Exception:
                    memory_guard["sampler_error"] = traceback.format_exc()
                    if child.poll() is None:
                        child.terminate()
            thread = threading.Thread(target=sample, daemon=True)
            thread.start()
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f"server exited during startup: {child.returncode}")
                try:
                    report["health"] = request("/health")
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            else:
                raise TimeoutError("server startup deadline exceeded")
            report["ready_s"] = time.monotonic() - start
            report["drafter_check"] = verify_drafter_log(
                drafter, (args.output / "server.stderr.log").read_text(errors="replace"))
            if not report["drafter_check"]["pass"]:
                raise RuntimeError("loaded drafter differs from requested mode; stopping before inference")
            phase = "running"
            maps = Path(f"/proc/{child.pid}/maps").read_text()
            (args.output / "loaded-maps.txt").write_text(maps)
            libraries = sorted({line.split()[-1] for line in maps.splitlines()
                                if "libamdhip64" in line or "libhsa-runtime64" in line})
            report["runtime_libraries"] = libraries
            report["plugin_sha256"] = {path: digest(Path(path)) for path in sorted({line.split()[-1]
                for line in maps.splitlines() if str(args.home) + "/" in line and line.split()[-1].endswith(".so")})}
            if not libraries or any(not lib.startswith(str(args.sdk) + "/") for lib in libraries):
                raise RuntimeError("GPU runtime missing or not from the selected SDK")
            report["models"] = request("/v1/models")
            if args.check_reserve_rejection:
                oversized = {"model": "bringup", "messages": [{"role": "user", "content": "Reply OK."}],
                             "max_tokens": args.replay_tokens + 512, "temperature": 0,
                             "chat_template_kwargs": {"enable_thinking": False}}
                try:
                    request("/v1/chat/completions", oversized, timeout=10)
                except urllib.error.HTTPError as error:
                    body = error.read().decode("utf-8", errors="replace")
                    report["reserve_rejection"] = {"http_status": error.code, "body": body}
                    if error.code != 400 or "--kvmem-replay-tokens" not in body:
                        raise RuntimeError("request was not rejected by the adapter reserve guard") from error
                else:
                    raise RuntimeError("over-reserve request unexpectedly admitted")
                report["health_after_rejection"] = request("/health")
            content = args.prompt_file.read_text() if args.prompt_file else "Reply with the single word OK."
            messages, rounds = [], []
            report["rounds"] = rounds
            if args.agent_requests:
                from agent_history import run as run_agent_history
                report.update(run_agent_history(request, args.output, args.context,
                    args.agent_requests, args.agent_chunk_tokens, rounds,
                    temperature=0.0 if args.agent_greedy else 1.0, logprobs=args.logprobs,
                    min_decode_tps=args.agent_min_decode_tps,
                    max_draft_step_ms=args.agent_max_draft_step_ms, draft_depth=args.speculative_tokens,
                    performance_pause_seconds=args.agent_performance_pause_seconds,
                    service_alive=lambda: child.poll() is None))
                report['agent_sampling'] = {'temperature': 0.0 if args.agent_greedy else 1.0,
                                          'logprobs': args.logprobs,
                                          'teacher_default': not args.agent_greedy}
            for index in range(0 if args.agent_requests else args.rounds):
                if cases is not None:
                    messages = cases[index]["messages"]
                    expected = cases[index]["expected"]
                else:
                    messages.append({"role": "user", "content": content})
                    expected = args.expected_answer
                prompt = {"model": "bringup", "messages": messages, "max_tokens": 128, "temperature": 0}
                if args.logprobs:
                    prompt.update(logprobs=True, top_logprobs=5)
                if args.disable_thinking:
                    prompt["chat_template_kwargs"] = {"enable_thinking": False}
                suffix = "" if args.rounds == 1 else f"-{index + 1}"
                (args.output / f"request{suffix}.json").write_text(json.dumps(prompt, indent=2))
                request_start = time.monotonic()
                response = request("/v1/chat/completions", prompt, timeout=180)
                elapsed = time.monotonic() - request_start
                (args.output / f"response{suffix}.json").write_text(json.dumps(response, indent=2, ensure_ascii=False))
                choice = response["choices"][0]
                answer = choice.get("message", {}).get("content") or ""
                expected_values = expected if isinstance(expected, list) else [expected]
                answer_ok = answer.strip().rstrip(".!。！").upper() in [str(x).upper() for x in expected_values]
                row = {"round": index + 1, "request_seconds": elapsed, "usage": response.get("usage"),
                       "timings": response.get("timings"),
                       "finish_reason": choice.get("finish_reason"), "content": answer,
                       "pass": answer_ok and choice.get("finish_reason") == "stop"}
                rounds.append(row)
                (args.output / "rounds.json").write_text(json.dumps(rounds, indent=2))
                if args.rounds == 1:
                    report.update({k: v for k, v in row.items() if k != "round"})
                if not row["pass"]:
                    break
                messages.append({"role": "assistant", "content": answer})
            if not args.agent_requests:
                report["pass"] = len(rounds) == args.rounds and all(row["pass"] for row in rounds)
    except BaseException:
        report["error"] = traceback.format_exc()
    finally:
        report["hash_methods"] = dict(HASH_METHODS)
        stop.set()
        if gpu_sampler:
            gpu_stop.touch(exist_ok=False)
            try:
                report["gpu_sampler_exit"] = gpu_sampler.wait(timeout=15)
            except subprocess.TimeoutExpired:
                gpu_sampler.terminate()
                report["gpu_sampler_exit"] = "stop timeout; bounded sampler has its own deadline"
            if gpu_log:
                gpu_log.close()
            path = args.output / "gpu-adapters.csv"
            peaks = {}
            if path.exists():
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    for row in csv.DictReader(stream):
                        key = row["adapter_path"]
                        peaks[key] = max(peaks.get(key, 0), int(row["used_bytes"]))
            report["windows_adapter_memory"] = {
                "peak_bytes_by_counter": peaks,
                "scope": "adapter-wide including other processes; not process VRAM",
                "available": bool(peaks)}
        if thread:
            thread.join(timeout=3)
        if memory_guard["triggered"] or memory_guard.get("sampler_error"):
            report["pass"] = False
        if child:
            report["exit_before_cleanup"] = child.poll()
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            report["exit_after_cleanup"] = child.returncode
            if child.returncode != 0:
                report["pass"] = False
        log_path = args.output / "server.stderr.log"
        log_lines = log_path.read_text(errors="replace").splitlines() if log_path.exists() else []
        report["drafter_check"] = verify_drafter_log(drafter, "\n".join(log_lines))
        report["pass"] = report["pass"] and report["drafter_check"]["pass"]
        if args.observe_statistics:
            from statistics_checks import frontier_matches
            evidence = []
            for line in log_lines:
                if "KVMEM_STATISTICS " in line:
                    evidence.append(dict(re.findall(r"(\w+)=([^ ]+)", line)))
            report["statistics"] = evidence
            # Count the evaluated prefix, not just API-emitted tokens. Native
            # terminal verify can evaluate accepted rows beyond max_tokens;
            # require a proven output/context limit and the fixed draft bound.
            stats_ok = len(evidence) == expected_rounds and len(report.get("rounds", [])) == expected_rounds
            for index, (item, row) in enumerate(zip(evidence, report.get("rounds", []))):
                output_limit = None
                for prefix in ("agent", "replay"):
                    saved = args.output / f"{prefix}-{index:03d}.request.json"
                    if saved.exists():
                        output_limit = json.loads(saved.read_text()).get("max_tokens")
                        break
                stats_ok = stats_ok and item.get("complete") == "1" and int(item.get("layers", 0)) > 0 and \
                    float(item.get("mean_k_l1", 0)) > 0 and frontier_matches(item, row,
                        draft_depth=args.speculative_tokens, output_limit=output_limit, context_limit=args.context)
            stats_ok = stats_ok and len({item.get("request", item.get("session")) for item in evidence}) == len(evidence)
            if not (args.require_prefix_resume or args.agent_requests):
                stats_ok = stats_ok and len({item.get("incarnation") for item in evidence}) == len(evidence)
            report["statistics_pass"] = stats_ok
            report["pass"] = report["pass"] and stats_ok
            resumed = [dict(re.findall(r"(\w+)=([^ ]+)", line)) for line in log_lines
                       if "KVMEM_PREFIX_RESUME " in line]
            report["prefix_resumes"] = resumed
            retained = [dict(re.findall(r"(\w+)=([^ ]+)", line)) for line in log_lines
                        if "KVMEM_PREFIX_RETAIN " in line]
            report["prefix_retained"] = retained
            if args.require_prefix_resume:
                resume_ok = len(resumed) == expected_rounds - 1 and expected_rounds > 1
                resume_ok = resume_ok and len(retained) == expected_rounds
                for index, (item, row) in enumerate(zip(resumed, report.get("rounds", [])[1:]), 1):
                    cached = (row.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                    previous_prompt = report["rounds"][index - 1]["usage"]["prompt_tokens"]
                    # Only the proven history prefix is reusable: generation-only
                    # chat template suffixes are recomputed, never falsely cached.
                    resume_ok = resume_ok and 0 < int(item["frontier"]) == cached <= previous_prompt and \
                        item["exact_prefix"] == item["same_query"] == "1"
                    resume_ok = resume_ok and index <= len(retained) and \
                        int(retained[index - 1]["frontier"]) == cached
                report["prefix_resume_pass"] = resume_ok
                report["pass"] = report["pass"] and resume_ok
            frozen = []
            for line in log_lines:
                if "KVMEM_QUERY_FROZEN session=" in line:
                    frozen.append(dict(re.findall(r"(\w+)=([^ ]+)", line)))
            report["frozen_queries"] = frozen
            replays = []
            for line in log_lines:
                if "KVMEM_QUERY_REPLAY_DONE session=" in line:
                    replays.append(dict(re.findall(r"(\w+)=([^ ]+)", line)))
            report["query_replays"] = replays
            if args.require_query_replay:
                replay_ok = len(replays) == args.rounds and len({x["session"] for x in replays}) == args.rounds
                for item, row in zip(replays, report.get("rounds", [])):
                    replay_ok = replay_ok and 0 <= int(item["begin"]) < int(item["end"]) == row["usage"]["prompt_tokens"] \
                        and item["before_sampler"] == "1"
                report["query_replay_pass"] = replay_ok
                report["pass"] = report["pass"] and replay_ok
            if args.require_frozen_query:
                query_ok = len(frozen) == args.rounds and len({x["session"] for x in frozen}) == args.rounds
                for item, row in zip(frozen, report.get("rounds", [])):
                    query_ok = query_ok and 0 <= int(item["begin"]) < int(item["end"]) <= row["usage"]["prompt_tokens"] \
                        and int(item["tokens"]) == int(item["end"]) - int(item["begin"])
                report["frozen_query_pass"] = query_ok
                report["pass"] = report["pass"] and query_ok
            if args.selection_budget:
                from selection_checks import untrimmed_within_reserve
                rows = []
                for line in log_lines:
                    if "KVMEM_RESELECT session=" in line:
                        rows.append(dict(re.findall(r"(\w+)=([^ ]+)", line)))
                report["selections"] = rows
                selected_ok = bool(rows) and all(int(row["visible"]) <= args.selection_budget and
                    row["original_positions"] == "1" for row in rows) and \
                    any(int(row["spill"]) > 0 for row in rows) and (
                        any(row["mode"] == "mean-key" for row in rows) or
                        (args.require_query_replay and bool(replays) and all(int(x["begin"]) == 0 for x in replays)))
                if not rows and evidence and all(untrimmed_within_reserve(
                        item, resumed, args.selection_budget, args.replay_tokens, replays) for item in evidence):
                    selected_ok = True # deliberate dense-sized budget: selection is unnecessary
                for item in evidence:
                    session_rows = [row for row in rows if row["session"] == item["session"]]
                    if (int(item["tokens"]) > args.selection_budget + args.selection_block and
                            not untrimmed_within_reserve(item, resumed, args.selection_budget, args.replay_tokens, replays)):
                        selected_ok = selected_ok and bool(session_rows) and any(int(r["spill"]) > 0 for r in session_rows)
                report["selection_pass"] = selected_ok
                report["restore_observed"] = any(int(row["restore"]) > 0 for row in rows)
                report["pass"] = report["pass"] and selected_ok
                pool_rows = []
                for line in log_lines:
                    if "KVMEM_POOL resident_tokens=" in line:
                        pool_rows.append(dict(re.findall(r"(\w+)=([^ ]+)", line)))
                report["native_pool"] = pool_rows
                if args.require_bounded_pool:
                    pool_ok = len(pool_rows) == 1 and int(pool_rows[0]["carved_bytes"]) > 0 and \
                        int(pool_rows[0]["resident_tokens"]) < args.context and any(
                            row["usage"]["prompt_tokens"] > int(pool_rows[0]["resident_tokens"])
                            for row in report.get("rounds", []))
                    report["bounded_pool_pass"] = bool(pool_ok)
                    report["pass"] = report["pass"] and bool(pool_ok)
        report["elapsed_s"] = time.monotonic() - start
        peaks = {}
        memory_path = args.output / "memory.jsonl"
        for line in (memory_path.read_text().splitlines() if memory_path.exists() else []):
            row = json.loads(line)
            group = peaks.setdefault(row["phase"], {})
            for key in ("VmRSS", "RssAnon", "RssFile", "VmSwap"):
                if key in row:
                    group[key + "_KiB"] = max(group.get(key + "_KiB", 0), int(row[key].split()[0]))
        report["sampled_memory_peaks"] = peaks
        (args.output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
