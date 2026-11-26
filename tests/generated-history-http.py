"""Exercise a KVMem chat server via ordinary messages, retaining native traces."""

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--exe", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--gpu", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument(
    "--native-template",
    action="store_true",
    help="check the stock Qwen3.5 template, which drops the historical empty think wrapper",
)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
url = f"http://127.0.0.1:{port}"
http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, KVMEM_TRACE="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
args = [
    str(a.exe.resolve()),
    "--api-key",
    "fixture-one,fixture-two",
    "-m",
    str(a.model.resolve()),
    "--port",
    str(port),
    "-c",
    "12288",
    "-b",
    "4096",
    "-ub",
    "4096",
    "--kvmem-budget",
    "6144",
    "--kvmem-gen-reserve",
    "4096",
    "--kvmem-sessions",
    "1",
    "--spec-type",
    "none",
    "-ngl",
    "99",
    "-fa",
    "on",
]
if not a.native_template:
    args += [
        "--chat-template",
        "{% for m in messages %}{{ '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
    ]
data = {"argv": args, "gpu": a.gpu, "turns": []}
log = a.out / "server.log"


def ask(name, msgs, maximum=16, stream=False, *, key="fixture-one", **extra):
    start = log.stat().st_size
    body = dict(messages=msgs, max_tokens=maximum, temperature=0, seed=27, stream=stream, **extra)
    (a.out / (name + ".request.json")).write_text(
        json.dumps(body, ensure_ascii=False), encoding="utf8"
    )
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    with http.open(req, timeout=120) as resp:
        wire = resp.read()
    (a.out / (name + ".response")).write_bytes(wire)
    if stream:
        frames = [json.loads(x[6:]) for x in wire.decode().splitlines() if x.startswith("data: {")]
        text = "".join(
            (f["choices"][0]["delta"].get("content") or "") for f in frames if f["choices"]
        )
        reply = {"role": "assistant", "content": text}
        usage = next((f["usage"] for f in frames if f.get("usage")), {})
    else:
        obj = json.loads(wire)
        reply = obj["choices"][0]["message"]
        usage = obj["usage"]
    trace = log.read_bytes()[start:].decode(errors="replace")
    found = re.findall(r"generated_history_rows=(\d+)", trace)
    row = dict(
        name=name,
        history=int(found[-1]) if found else None,
        usage=usage,
        query=re.findall(r"query_loc method=native_role[^\n]+", trace),
    )
    if not stream:
        metric = obj["kvmem"]["cache_reuse"]
        assert metric["generated_history_tokens"] == row["history"]
        assert metric["actual_cached_tokens"] == usage["prompt_cache_hit_tokens"]
    data["turns"].append(row)
    (a.out / "results.json").write_text(json.dumps(data, indent=2), encoding="utf8")
    print(name, row, flush=True)
    return reply, row


def ask_responses(name, items, stream):
    body = {
        "input": items,
        "model": "test",
        "max_output_tokens": 16,
        "stream": stream,
        "temperature": 0,
    }
    (a.out / (name + ".request.json")).write_text(json.dumps(body), encoding="utf8")
    headers = {"Content-Type": "application/json", "Authorization": "Bearer fixture-one"}
    req = urllib.request.Request(url + "/v1/responses", json.dumps(body).encode(), headers)
    with http.open(req, timeout=120) as resp:
        wire = resp.read()
    (a.out / (name + ".response")).write_bytes(wire)
    if stream:
        frames = [
            json.loads(line[6:])
            for line in wire.decode().splitlines()
            if line.startswith("data: {")
        ]
        event = next(
            f for f in frames if f["type"] in ("response.completed", "response.incomplete")
        )
        obj = event["response"]
    else:
        obj = json.loads(wire)
    return obj


def user(s):
    return {"role": "user", "content": s}


with log.open("wb") as fout:
    proc = subprocess.Popen(args, env=env, stdout=fout, stderr=fout)
    try:
        for _ in range(200):
            if proc.poll() is not None:
                raise RuntimeError(f"startup {proc.returncode}, see {log}")
            try:
                with http.open(url + "/health", timeout=1):
                    break
            except OSError:
                time.sleep(0.3)
        else:
            raise RuntimeError("health timeout")
        startup = re.search(r"KVMEM_STARTUP ready=([^\n]+)", log.read_text(errors="replace"))
        assert startup
        config = json.loads(startup[1])
        assert config["ubatch_actual"] == config["batch_actual"] == 4096
        # The Qwen vocabulary has separate a/b tokens (64/65) and a merged ab.
        # GBNF constrains the test generation only; the follow-up sends plain messages.
        forced = "root ::= " + " ".join(["<[64]> <[65]>"] * 48)
        msgs = [user("Write the repeating string.")]
        msg, first = ask("first", msgs, 96, grammar=forced)
        assert msg["content"] == "ab" * 48, msg
        follow = msgs + [msg, user("Which two letters did you repeat?")]
        hot, h = ask("hot", follow)
        if a.native_template:
            assert h["history"] == 0 and h["query"]
            # Qwen's stock template includes the reasoning wrapper again on a tool continuation.
            tool_msgs = msgs + [msg, {"role": "tool", "content": "The operation completed."}]
            _, tool = ask("tool-follow", tool_msgs)
            assert tool["history"] == first["usage"]["prompt_tokens"] + 96
            definition = [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
            parent = [user("Use the weather tool for Beijing.")]
            actual, r = ask("real-call", parent, 120, tools=definition, tool_choice="required")
            calls = actual["tool_calls"]
            echoed = parent + [
                actual,
                {"role": "tool", "tool_call_id": calls[0]["id"], "content": "Sunny, 20 Celsius."},
            ]
            _, resumed = ask("real-result", echoed, 32, tools=definition)
            assert (
                resumed["history"] == r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]
            )
            raise SystemExit(0)
        assert (
            h["history"] == first["usage"]["prompt_tokens"] + first["usage"]["completion_tokens"]
            and h["query"]
        )
        _, isolated = ask("other-key", follow, key="fixture-two")
        assert isolated["history"] == 0
        _, canonical = ask("canonical", follow, cache_reset=True)
        assert (
            canonical["history"] == 0
            and canonical["usage"]["prompt_tokens"] < h["usage"]["prompt_tokens"]
        )
        for i in range(4):
            ask(
                "displace-" + str(i),
                [{"role": "system", "content": str(i) * 300}, user("Reply OK.")],
                4,
            )
        cold, c = ask("cold", follow)
        assert (
            c["history"] == h["history"]
            and c["usage"]["prompt_tokens"] == h["usage"]["prompt_tokens"]
        )
        assert c["usage"]["prompt_cache_hit_tokens"] == 0 < h["usage"]["prompt_cache_hit_tokens"]
        assert hot == cold
        # A changed latest reply may still fork from the first proven turn.
        edited = follow + [
            {"role": "assistant", "content": "edited second answer"},
            user("One more."),
        ]
        _, ancestor = ask("ancestor", edited)
        assert ancestor["history"] == h["history"]
        _, changed = ask("edit", [msgs[0], dict(msg, content=msg["content"] + "x"), follow[-1]])
        assert changed["history"] == 0
        sm = [user("Write the word lighthouse.")]
        sr, first = ask("stream", sm, 32, True, stream_options={"include_usage": True})
        _, follow_stream = ask("stream-follow", sm + [sr, user("Spell it again.")])
        assert (
            follow_stream["history"]
            == first["usage"]["prompt_tokens"] + first["usage"]["completion_tokens"]
        )
        # Same text and structured parent, different token path. The ordinary
        # tokenizer shortens 96 IDs to 48, and the retained record becomes ambiguous.
        duplicate, second = ask("second-path", msgs, 96, grammar='root ::= "' + "ab" * 48 + '"')
        assert duplicate == msg
        if second["usage"]["completion_tokens"] != 96:
            _, ambiguous = ask("ambiguous", follow)
            assert ambiguous["history"] == 0
        else:
            raise AssertionError("fixture did not produce a different token path")
        for stream in (False, True):
            items = [user(f"Reply with just hello. Test {stream}.")]
            response = ask_responses("responses-" + str(stream), items, stream)
            follow_response = ask_responses(
                "responses-follow-" + str(stream),
                items + response["output"] + [user("Again?")],
                False,
            )
            expected = response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
            assert follow_response["kvmem"]["cache_reuse"]["generated_history_tokens"] == expected
        stop_parent = [user("Output ab with no explanation.")]
        clean, _ = ask("clean", stop_parent, 3, grammar="root ::= <[64]> <[65]>")
        cut, _ = ask("cut", stop_parent, 3, grammar="root ::= <[64]> <[65]> <[55]>", stop=["X"])
        assert clean == cut and clean["content"] == "ab"
        _, after_cut = ask("after-cut", stop_parent + [cut, user("Continue.")])
        assert after_cut["history"] == 0
    finally:
        proc.terminate()
        proc.wait(timeout=20)
