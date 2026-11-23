#!/usr/bin/env python3
"""Model-level protocol, paging and same-configuration performance evidence."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time


class Worker:
    def __init__(self, args, log):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        self.log = log.open("w", encoding="utf-8")
        self.sample_memory = args.scenario != "performance"
        command = [str(args.worker), "--model", str(args.model), "--serve", "--max-context", str(args.context),
                   "--threads", str(args.threads), "--prompt-batch", str(args.batch), "--stage-mib", str(args.stage),
                   "--window", str(1+getattr(args,"mtp_drafts",0)), "--lookup", "0", "--tail-skip", "0", "--expert-swaps", str(args.swaps),
                   "--prompt-cache", "0", "--conversation-cache-mib", "0", "--conversation-cache-slots", "0"]
        if not getattr(args,"mtp_drafts",0): command += ["--no-draft"]
        if getattr(args,"ignore_eos",False): command += ["--eos-ids", "2147483647"]
        if getattr(args,"kv_type",None): command += ["--kv", "fp16" if args.kv_type=="f16" else args.kv_type]
        if args.whole:
            command += ["--kv-whole"]
        if args.no_graphs:
            command += ["--no-graphs"]
        if getattr(args, "batch_slots", 0):
            command += ["--batch", str(args.batch_slots)]
        if getattr(args, "vram_reserve_mib", None) is not None:
            command += ["--vram-reserve-mib", str(args.vram_reserve_mib)]
        if args.mode == "kvmem":
            if getattr(args,"mtp_drafts",0): command += ["--kvmem-mtp"]
            if getattr(args,"disk_mib",0):
                command += ["--kvmem-disk-path",str(args.disk_path),"--kvmem-disk-mib",str(args.disk_mib)]
                env["KVMEM_POLY_STORAGE_METRICS"]="1"
            command += ["--kvmem-budget", str(args.budget), "--kvmem-gen-reserve", str(args.reserve),
                        "--kvmem-host-mib", str(args.host_mib), "--kvmem-sink-tokens", str(args.sink),
                        "--kvmem-recent-tokens", str(args.recent), "--kvmem-score-agg", args.aggregation]
            if not getattr(args,"legacy",False): command += ["--kvmem-query-scorer",getattr(args,"scorer","page-softmax")]
            if args.verify:
                command += ["--kvmem-verify-transfers"]
            if args.lrfu:
                command += ["--kvmem-lrfu"]
            if getattr(args, "history_reuse", False):
                command += ["--kvmem-history-reuse"]
            if getattr(args, "sessions", 1) != 1:
                command += ["--kvmem-sessions", str(args.sessions)]
            if getattr(args, "cache_min_tokens", 0):
                command += ["--kvmem-cache-min-tokens", str(args.cache_min_tokens)]
        if getattr(args, "session_metrics", False):
            env["KVMEM_POLY_SESSION_METRICS"] = "1"
        if args.fault:
            env["KVMEM_POLY_TEST_FAULT"] = args.fault
        if args.profile:
            env["KVMEM_POLY_PROFILE"] = "1"
        self.command = command
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, bufsize=1, env=env, creationflags=0x08000000 if os.name == "nt" else 0)
        self.lines = queue.Queue()
        def read():
            for line in self.process.stdout:
                self.lines.put((line.rstrip(), time.perf_counter()))
            self.lines.put((None, time.perf_counter()))
        threading.Thread(target=read, daemon=True).start()
        self.info = []
        while True:
            line = self.take(180)
            self.info.append(line)
            if line.startswith("READY "):
                break
            if line.startswith("ERR "):
                raise RuntimeError(line)

    def take(self, timeout=120):
        try:
            line, self.received_at = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("worker protocol timed out") from None
        if line is None:
            raise RuntimeError(f"worker ended: {self.process.poll()}")
        return line

    def request(self, ids, span, count, stop_after=None, stop_prefill=False):
        memory = {}
        if self.sample_memory and os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                    (name, ctypes.c_uint64) for name in ("total_phys", "available_phys", "total_pagefile",
                                                      "available_pagefile", "total_virtual", "available_virtual", "extended_virtual")]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                memory = {name:getattr(status, name) for name in ("available_phys", "available_pagefile", "available_virtual")}
        first, last = span
        keys = f" query_begin={first} query_end={last} temperature=0 top_k=20"
        self.process.stdin.write(f"GEN {count}{keys} {','.join(map(str, ids))}\n")
        self.process.stdin.flush()
        tokens, progress, arrivals = [], [], []
        started = time.perf_counter()
        stopped = False
        while True:
            line = self.take()
            if line.startswith("T "):
                tokens.append(int(line.split()[1]))
                arrivals.append((self.received_at-started)*1000)
                if stop_after is not None and len(tokens) >= stop_after and not stopped:
                    self.process.stdin.write("STOP\n")
                    self.process.stdin.flush()
                    stopped = True
            elif line.startswith("PP "):
                progress.append(line)
                if stop_prefill and not stopped:
                    self.process.stdin.write("STOP\n")
                    self.process.stdin.flush()
                    stopped = True
            elif line.startswith("ERR "):
                return {"error": line, "wall_ms": (time.perf_counter()-started)*1000, "memory_before":memory}
            elif line.startswith("DONE "):
                fields = line.split()
                return {"tokens": tokens, "prompt_tokens": int(fields[2]), "prompt_ms": float(fields[3]),
                        "decode_ms": float(fields[4]), "finish": fields[5], "resume": int(fields[8]),
                        "draft_accepted":int(fields[6]), "draft_offered":int(fields[7]),
                        "expert_hits": int(fields[9]), "expert_total": int(fields[10]),
                        "read_tokens": int(fields[14]), "wall_ms": (time.perf_counter()-started)*1000,
                        "token_arrival_ms": arrivals,
                        "progress": progress, "protocol": line, "memory_before":memory}

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=20)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "worker", "model", "tokenizer", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "kvmem"), required=True)
    parser.add_argument("--scenario", choices=("smoke", "paging", "performance", "fault", "admission"), default="smoke")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--reserve", type=int, default=4096)
    parser.add_argument("--host-mib", type=int, default=4096)
    parser.add_argument("--sink", type=int, default=64)
    parser.add_argument("--recent", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--stage", type=int, default=512)
    parser.add_argument("--vram-reserve-mib", type=int, default=None,
                        help="explicit native expert-cache VRAM reserve for matched comparisons")
    parser.add_argument("--swaps", type=int, default=64)
    parser.add_argument("--aggregation", choices=("mean", "max"), default="mean")
    parser.add_argument("--scorer",choices=("page-softmax","meanq-cosine"),default="page-softmax")
    parser.add_argument("--kv-type",choices=("f16","q8_0"),default="f16")
    parser.add_argument("--mtp-drafts",type=int,choices=range(4),default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--lrfu", action="store_true")
    parser.add_argument("--whole", action="store_true")
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--fault", choices=("prepare", "spill", "restore", "publish"))
    parser.add_argument("--lengths", default="256,2048,8192")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--layout", choices=("system", "user"), default="system")
    parser.add_argument("--recoveries", type=int, default=1, help="fresh long requests after the one-shot fault")
    parser.add_argument("--query-mode", choices=("full", "explicit-tail"), default="full")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    for name in ("source", "worker", "model", "tokenizer", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source))
    sys.path.insert(0, str(args.source / "tools"))
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, LITERAL_TAGS
    from serve import kvmem
    tok = Tokenizer.from_dir(args.tokenizer)
    template = ChatTemplate(args.tokenizer / "chat_template.jinja")
    class Frontend:
        literals = LITERAL_TAGS
        def render_prompt(self, messages, tools, kwargs):
            return template.render(messages, tools=tools, enable_thinking=False)
    frontend = Frontend()
    frontend.tok = tok
    def prompt(messages):
        return kvmem.encode(frontend, messages, None, {})
    def document_prompt(document, question):
        if args.layout == "system":
            return prompt([{"role":"system", "content":document}, {"role":"user", "content":question}])
        messages = [{"role":"user", "content":document+"\n"+question}]
        ids = prompt(messages)
        if args.query_mode == "explicit-tail":
            selected = kvmem.resolve_query(frontend, messages, None, {}, ids,
                                          {"message":0,"start":len(document)+1,"end":len(messages[0]["content"])})
            if selected != ids:
                raise RuntimeError("explicit query metadata changed input token IDs")
            return selected
        return ids
    def corpus(length, target=True):
        lines = ["Reference records. Answer from these records; do not invent keys.\n"]
        for i in range(10000):
            if i == 70 and target:
                lines.append("The access key for Project Atlas is MARBLE-4827. Project Atlas uses MARBLE-4827.\n")
            else:
                lines.append(f"Record {i}: Project Cedar-{i} has inspection date 2031-04-19 and serial REF-{i:05d}.\n")
            if i % 32 == 0 and len(tok.encode("".join(lines))) >= length:
                break
        return "".join(lines)
    cases = []
    if args.scenario == "smoke":
        for label, text in (("arithmetic", "What is 2 + 3? Reply only with the number."),
                            ("unicode", "请只输出数字：7乘以8等于多少？")):
            cases.append((label, prompt([{"role":"user", "content":text}]), 32))
    elif args.scenario in ("paging", "fault", "admission"):
        document = corpus(12000)
        question = "Find Project Atlas in the reference records. What is its exact access key? " \
                   "Copy the access key from its record. Reply only with that key, without explanation."
        cases.append(("cold_atlas", document_prompt(document, question), 48))
    else:
        question = "Using the records, write a detailed numbered list of twenty observations. " \
                   "Discuss their organization, serial numbers and inspection dates. Continue until the list is complete."
        for length in map(int, args.lengths.split(",")):
            cases.append((f"prompt_{length}", document_prompt(corpus(length, False), question), args.max_new))
    result = {"mode":args.mode, "scenario":args.scenario, "gpu_uuid":args.gpu, "worker":str(args.worker),
              "context":args.context, "budget":args.budget, "reserve":args.reserve, "repeats":args.repeats,
              "cases":[], "status":"running"}
    result["worker_sha256"] = hashlib.sha256(args.worker.read_bytes()).hexdigest()
    result["model_bytes"] = args.model.stat().st_size
    result["layout"] = args.layout
    result["query_mode"] = args.query_mode
    result["machine"] = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total",
                                                  "--format=csv,noheader"], text=True).splitlines()
    worker = Worker(args, args.out / "worker.log")
    result["command"], result["info"] = worker.command, worker.info
    def save():
        (args.out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    try:
        for label, ids, count in cases:
            entry = {"name":label, "input_tokens":len(ids), "query":ids.kvmem_query, "runs":[]}
            (args.out / (label+".json")).write_text(json.dumps({"ids":ids, "query":ids.kvmem_query}), encoding="utf-8")
            repetitions = args.repeats if args.scenario == "performance" else 1
            if args.scenario == "performance":
                for _ in range(2):
                    warm = worker.request(ids, ids.kvmem_query, count)
                    if "error" in warm:
                        raise RuntimeError(warm["error"])
            for repeat in range(repetitions):
                run = worker.request(ids, ids.kvmem_query, count)
                if "tokens" in run:
                    run["text"] = tok.decode(run["tokens"])
                    run["prefill_tok_s"] = len(ids)*1000/max(run["prompt_ms"], 1e-6)
                    run["decode_tok_s"] = max(len(run["tokens"])-1, 0)*1000/max(run["decode_ms"], 1e-6)
                    if run["resume"] != 0:
                        raise RuntimeError("performance/correctness request unexpectedly reused a prefix")
                entry["runs"].append(run)
                print(json.dumps({"case":label, "repeat":repeat, **{k:v for k,v in run.items() if k not in ("tokens","progress")}}, ensure_ascii=False), flush=True)
            if args.scenario in ("fault", "admission"):
                if "error" not in entry["runs"][0]:
                    raise RuntimeError("the intended fault/budget refusal did not occur")
                recovery_ids = (prompt([{"role":"user", "content":"What is 2 + 3? Reply only with the number."}])
                                if args.scenario == "admission" else ids)
                recovery = worker.request(recovery_ids, recovery_ids.kvmem_query, count)
                if "error" in recovery:
                    raise RuntimeError("fresh request did not recover after injected failure: "+recovery["error"])
                recovery["text"] = tok.decode(recovery["tokens"])
                expected = "5" if args.scenario == "admission" else "MARBLE-4827"
                if expected not in recovery["text"]:
                    raise RuntimeError("recovery did not produce the expected answer")
                entry["recovery"] = recovery
                if args.scenario == "fault":
                    entry["recovery_repeats"] = []
                    for _ in range(args.recoveries - 1):
                        fresh = worker.request(ids, ids.kvmem_query, count)
                        if "error" in fresh or "MARBLE-4827" not in tok.decode(fresh["tokens"]):
                            raise RuntimeError("repeated fresh request did not recover after the one-shot failure")
                        entry["recovery_repeats"].append(fresh)
            elif any("error" in run for run in entry["runs"]):
                raise RuntimeError(entry["runs"][0].get("error", "request failed"))
            if args.scenario == "paging" and "MARBLE-4827" not in entry["runs"][0].get("text", ""):
                raise RuntimeError("cold-record answer did not contain the expected access key")
            if args.scenario == "smoke":
                expected = "5" if label == "arithmetic" else "56"
                actual = entry["runs"][0].get("text", "").replace("<|im_end|>", "").strip()
                if actual != expected:
                    raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")
            if args.scenario == "performance":
                entry["median"] = {key:statistics.median(run[key] for run in entry["runs"])
                                    for key in ("prompt_ms","decode_ms","prefill_tok_s","decode_tok_s","wall_ms")}
            result["cases"].append(entry)
            save()
        if args.scenario == "paging":
            paging_ids = cases[0][1]
            prefill_cancel = worker.request(paging_ids, paging_ids.kvmem_query, 48, stop_prefill=True)
            if prefill_cancel.get("finish") != "cancel" or prefill_cancel.get("tokens"):
                raise RuntimeError("STOP did not cancel inside prefill")
            ids = prompt([{"role":"user", "content":"Write a long numbered list of facts about arithmetic."}])
            cancelled = worker.request(ids, ids.kvmem_query, 512, stop_after=3)
            if cancelled.get("finish") != "cancel":
                raise RuntimeError("STOP did not cancel generation")
            recovered = worker.request(ids, ids.kvmem_query, 24)
            if "error" in recovered:
                raise RuntimeError("request after cancellation failed")
            invalid = []
            for span in ((-1, 3), (2, 2), (0, len(ids)+1)):
                refusal = worker.request(ids, span, 24)
                if "error" not in refusal:
                    raise RuntimeError("an invalid query span was admitted")
                invalid.append(refusal)
            final = worker.request(ids, ids.kvmem_query, 24)
            if "error" in final:
                raise RuntimeError("query-span rejection poisoned a fresh request")
            result["cancellation"] = {"prefill_cancel":prefill_cancel, "cancelled":cancelled, "recovered":recovered,
                                      "invalid_spans":invalid, "after_rejection":final}
        result["status"] = "pass"
    except Exception as error:
        result["status"], result["error"] = "fail", str(error)
        raise
    finally:
        save()
        worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
