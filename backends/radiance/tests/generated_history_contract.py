"""Check generated histories on a dedicated, running Radiance KVMem server.

Use a byte-preserving ChatML template, 4096-token prefill, one retained session,
and a vocabulary where an ordinary generated answer can differ from its re-tokenization.
Use --native-template to also accept a proven template rewrite fallback.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request


def message(role, content):
    return {"role": role, "content": content}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-template", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def save(name, value):
        (args.output / (name + ".json")).write_text(
            json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")

    def api(path, body, stream=False):
        request = urllib.request.Request(
            args.url.rstrip("/") + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with opener.open(request, timeout=300) as reply:
                return reply.read().decode() if stream else json.load(reply)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{error.code}: {error.read().decode()}") from error

    def chat(name, messages, limit, **options):
        body = dict(model=args.model, messages=messages, max_tokens=limit,
                    temperature=0, reasoning_effort="none") | options
        save(name + "-request", body)
        reply = api("/v1/chat/completions", body, options.get("stream", False))
        save(name + "-response", reply)
        return reply

    def render(messages):
        return api("/apply-template", dict(model=args.model, messages=messages,
                                          reasoning_effort="none"))["prompt"]

    def tokens(text):
        return api("/tokenize", dict(content=text, add_special=False, parse_special=True))["tokens"]

    def selected(reply):
        return reply["timings"]["cache_reuse"]["generated_history_tokens"]

    # Obtain accepted token IDs through the ordinary logprobs interface. Raw
    # GBNF is unavailable on some renderers; no sampler or tokenizer is mocked.
    vocab = json.load(opener.open(args.url.rstrip("/") + "/get_tokenizer_info", timeout=10))
    turn_end = vocab["eot_token_id"] if vocab.get("eot_token_id") is not None else vocab["eos_token_id"]

    def visible_ids(entries):
        ids = [entry["id"] for entry in entries]
        if ids and ids[-1] == turn_end and not entries[-1]["bytes"]:
            ids.pop()
        return ids

    long_string = "ab" * 45 + "a" * 60
    constrained = {"response_format": {"type": "json_schema", "json_schema": {
        "name": "fixture", "schema": {"type": "string", "enum": [long_string]}}},
        "temperature": 1.3, "top_p": 1.0, "seed": 41, "frequency_penalty": 1.5}
    prompts = [
        "Return the requested JSON string.",
        "Return 300 lowercase a characters without spaces, commentary or Markdown.",
        "Print the word ab 200 times consecutively, without spaces or explanation.",
        "Print this JSON compactly many times: {\"code\":\"abc\",\"i\":123456789}.",
        "Print all integers from 1 to 120 separated only by a comma.",
        "Write a Python program that prints the Fibonacci sequence.",
        "Write 100 digits of pi, then explain how the Chudnovsky algorithm works.",
    ]
    # Penalise repeated tokens: several vocab entries spell runs of different
    # lengths, giving a verifiable noncanonical path through the same text.
    repetitions = {"temperature": 1.4, "top_p": 1.0, "seed": 67,
                   "top_k": 0, "frequency_penalty": 1.8, "presence_penalty": 1.2}
    for index, prompt in enumerate(prompts):
        first = [message("user", prompt)]
        settings = constrained if index == 0 else repetitions
        one = chat(f"first-{index}", first, 60, logprobs=True, **settings)
        choice = one["choices"][0]
        assistant = choice["message"]
        generated = visible_ids(choice["logprobs"]["content"])
        base = render(first)
        full = render(first + [assistant, message("user", "Write one letter.")])
        entries = choice["logprobs"]["content"]
        raw_reply = b"".join(bytes(entry["bytes"]) for entry in entries).decode()
        prefix = base + raw_reply
        canonical = tokens(full)
        if args.native_template and not full.startswith(prefix):
            follow = first + [assistant, message("user", "Write one letter.")]
            fallback = chat("native-fallback", follow, 3)
            assert selected(fallback) == 0
            assert fallback["usage"]["prompt_tokens"] == len(canonical)
            save("result", {"result": "pass", "native_template": "transformed-history-fallback"})
            return
        # A parser may trim whitespace. Such a reply has no reusable byte
        # seam even if the remaining text is a prefix of the follow-up.
        if full.startswith(prefix):
            prefix_ids = tokens(base) + generated
            effective = prefix_ids + tokens(full[len(prefix):])
            if effective != canonical:
                break
    else:
        raise AssertionError("No observed generation produced a noncanonical history on this model")
    follow = first + [assistant, message("user", "Write one letter.")]
    compare = {"seed": 137, "logprobs": True, "top_logprobs": 5}
    warm = chat("warm", follow, 3, **compare)
    assert selected(warm) == len(prefix_ids), (selected(warm), len(prefix_ids))
    assert warm["usage"]["prompt_tokens"] == len(effective)

    separate = chat("other-owner", follow, 3, prompt_cache_key="different-owner")
    assert selected(separate) == 0
    assert separate["usage"]["prompt_tokens"] == len(canonical)
    reset = chat("reset", follow, 3, cache_reset=True)
    assert selected(reset) == 0
    assert reset["usage"]["prompt_tokens"] == len(canonical)
    for i in range(3):
        unrelated = [message("user", f"Unrelated branch {i} " + str(i) * 500)]
        chat(f"evict-{i}", unrelated, 2, prompt_cache_key=f"eviction-{i}")
    cold = chat("cold", follow, 3, **compare)
    assert selected(cold) == len(prefix_ids)
    assert cold["usage"]["prompt_tokens"] == len(effective)
    # Check the actual cold logical input against a direct, raw IDs request.
    # Different backend prefill and restore numerics can change closely ranked
    # tokens. Preserve both distributions and measure them independently.
    chat("displace-direct", [message("user", "Unrelated control " + "x" * 512)], 2,
         prompt_cache_key="displace-direct")
    direct_body = dict(model=args.model, prompt=effective, temperature=0, max_tokens=3,
                       seed=137, logprobs=5, prompt_cache_key="direct-ids-reference")
    save("direct-ids-request", direct_body)
    direct = api("/v1/completions", direct_body)
    save("direct-ids-response", direct)
    assert direct["usage"]["prompt_tokens"] == len(effective)
    # The native prefix cache may retain independently reusable prompt pages.
    cold_choice = cold["choices"][0]
    # The chat reader can trim surrounding whitespace; request logprobs retain
    # native accepted IDs before formatting on both frontends.
    direct_entries = direct["choices"][0]["logprobs"]["content"]
    assert [entry["id"] for entry in direct_entries] == [
        entry["id"] for entry in cold_choice["logprobs"]["content"]], (cold_choice, direct)


    # A separate cache owner ensures that only the streaming publication can
    # explain the next hit even when the greedy output is identical.
    stream_first = first
    stream_owner = {"prompt_cache_key": "streaming-owner"}
    events = chat("stream", stream_first, 60, stream=True, logprobs=True, **stream_owner, **settings)
    chunks = [json.loads(line[6:]) for line in events.splitlines() if line.startswith("data: {")]
    content = "".join(chunk["choices"][0]["delta"].get("content") or ""
                      for chunk in chunks if chunk["choices"])
    stream_entries = [entry for chunk in chunks if chunk["choices"]
                  for entry in ((chunk["choices"][0].get("logprobs") or {}).get("content") or [])]
    stream_ids = visible_ids(stream_entries)
    stream_raw = b"".join(bytes(entry["bytes"]) for entry in stream_entries).decode()
    assert content and stream_ids
    stream_follow = stream_first + [message("assistant", content), message("user", "Write one letter.")]
    stream_base = render(stream_first)
    stream_text = render(stream_follow)
    assert stream_text.startswith(stream_base + stream_raw)
    stream_prefix_ids = tokens(stream_base) + stream_ids
    stream_effective = stream_prefix_ids + tokens(stream_text[len(stream_base + stream_raw):])
    stream_reply = chat("stream-follow", stream_follow, 2, **stream_owner)
    assert selected(stream_reply) == len(stream_prefix_ids)
    assert stream_reply["usage"]["prompt_tokens"] == len(stream_effective)
    save("reference-ids", {"warm": effective, "cold": effective, "stream-follow": stream_effective,
                           "other-owner": canonical})
    factual = first + [assistant, message("user", "What is two plus two? Reply with the digit only.")]
    # Repopulate the same lineage with the streamed owner's first turn.
    factual_warm = chat("factual-warm", factual, 6, **stream_owner, **compare)
    chat("displace-factual", [message("user", "Unrelated fact " + "x" * 512)], 2,
         prompt_cache_key="displace-factual")
    factual_cold = chat("factual-cold", factual, 6, **stream_owner, **compare)
    assert selected(factual_warm) == selected(factual_cold) == len(stream_prefix_ids)
    assert factual_warm["choices"][0]["message"]["content"].strip() == "4"
    assert factual_cold["choices"][0]["message"]["content"].strip() == "4"
    save("result", {"result": "pass", "canonical": len(canonical), "effective": len(effective),
                    "generated_prefix_ids": len(prefix_ids), "generated_ids": generated,
                    "effective_ids_sha256": hashlib.sha256(
                        b"".join(x.to_bytes(4, "little", signed=True) for x in effective)).hexdigest(),
                    "warm_cache": warm["timings"]["cache_n"], "cold_cache": cold["timings"]["cache_n"],
                    "other_owner_prompt": separate["usage"]["prompt_tokens"]})


if __name__ == "__main__":
    main()
