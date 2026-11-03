"""Audit the archived 2026-11-03 CPU PNG run without loading a model or GPU.

Usage: python check_cpu_image_evidence.py evidence.tar.gz --sha256 <archive hash>
The original worker's checks are retained as historical evidence. This verifier
also checks raw answers, request identity, encoder sizes, cancellation and metrics.
"""
import argparse
import array
import base64
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import tarfile
import zlib


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def check_colors(text, left, right):
    pairs = re.findall(r"\b(left|right)\s*=\s*([a-z]+)\b", text.lower())
    require(set(pairs) == {("left", left), ("right", right)}, "incorrect or contradictory side colors")


def check_embedding(raw, rows, hidden=5120):
    require(len(raw) == rows * hidden * 2, "encoder byte count differs from rows * hidden * BF16")
    words = array.array("H")
    words.frombytes(raw)
    if sys.byteorder != "little":
        words.byteswap()
    require(all((x & 0x7F80) != 0x7F80 for x in words), "non-finite encoder output")
    require(any(x & 0x7FFF for x in words), "all-zero encoder output")


def check_png(raw, size, reverse=False):
    require(raw[:8] == b"\x89PNG\r\n\x1a\n", "invalid PNG signature")
    offset, compressed, dimensions, ended = 8, bytearray(), None, False
    while offset < len(raw):
        require(offset + 12 <= len(raw), "truncated PNG chunk")
        length = struct.unpack_from(">I", raw, offset)[0]
        kind, payload = raw[offset + 4:offset + 8], raw[offset + 8:offset + 8 + length]
        require(offset + length + 12 <= len(raw), "truncated PNG data")
        crc = struct.unpack_from(">I", raw, offset + 8 + length)[0]
        require(zlib.crc32(kind + payload) & 0xFFFFFFFF == crc, "PNG CRC mismatch")
        if kind == b"IHDR":
            dimensions = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            ended = True
        offset += length + 12
    require(ended and dimensions == (size, size, 8, 2, 0, 0, 0), "unexpected PNG geometry/format")
    pixels = zlib.decompress(compressed)
    a, b = (b"\x00\x00\xff", b"\xff\x00\x00") if reverse else (b"\xff\x00\x00", b"\x00\x00\xff")
    require(pixels == (b"\0" + a * (size // 2) + b * (size // 2)) * size, "PNG pixels differ from expected halves")


def verify(archive, expected_sha256):
    archive = Path(archive)
    with archive.open("rb") as stream:
        actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    require(actual_sha == expected_sha256, "archive SHA256 differs from the recorded run")
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        require(len(names) == len(set(names)), "duplicate archive names")
        for member in members:
            name = PurePosixPath(member.name)
            require(member.isfile() and not name.is_absolute() and ".." not in name.parts and ":" not in member.name and "\\" not in member.name, "non-file or unsafe archive member")
        content = {m.name: tar.extractfile(m).read() for m in members}
    read = lambda name: json.loads(content[name])
    manifest = read("evidence-manifest.json")["files"]
    require(set(content) == set(manifest) | {"evidence-manifest.json"}, "archive inventory mismatch")
    for name, record in manifest.items():
        require(len(content[name]) == record["bytes"] and hashlib.sha256(content[name]).hexdigest() == record["sha256"], "member checksum mismatch: " + name)

    result, summary, engine = read("result.json"), read("summary.json"), read("engine-result.json")
    profile = result["profile"]
    require(result["exit"] == 0 and summary["result"] == result and profile["completed"], "run did not complete")
    require(len(profile["checks"]) == 57 and all(x["pass"] for x in profile["checks"]), "historical worker checks did not pass")
    require(profile["exe_sha256"] == engine["server_sha256"] and profile["source_delta_sha256"] == engine["source_manifest_sha256"], "source/binary evidence mismatch")
    command = profile["command"]
    for option, value in {"--vision-residency": "cpu", "--mm-max-patches": "1024", "--num-speculative-tokens": "3", "--max-num-seqs": "2"}.items():
        require(command.count(option) == 1 and command[command.index(option) + 1] == value, "unexpected launch option: " + option)
    folder = "run-v1/cpu-fp8-f32-mtp3-c2/"
    require("vision: CPU execution" in content[folder + "server.log"].decode(), "CPU placement log missing")
    requests = read(folder + "requests.json")
    require(len(requests) == profile["requests"] == 9, "unexpected response count")
    indexed = {r["label"]: r for r in requests}
    require(len(indexed) == 9, "duplicate request labels")
    answers = {}
    for label, row in indexed.items():
        response = read(folder + label + ".response.json")
        native, usage = row["native_monitor"], row["usage"]
        require(response["usage"] == usage and response["id"] == native["native_request_id"], "request/native identity mismatch")
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        require(0 <= cached <= usage["prompt_tokens"] and cached == native["actual_cached_tokens"], "cache counters disagree")
        require(math.isfinite(row["wall_s"]) and row["wall_s"] >= native["ttft_ms"] / 1000, "invalid timing relationship")
        answers[label] = response["choices"][0]["message"]["content"].strip()
    for label in ("01-small-image", "04-large-image", "06-parallel-0"):
        check_colors(answers[label], "red", "blue")
    for label in ("03-changed-image", "06-parallel-1"):
        check_colors(answers[label], "blue", "red")
    require(answers["02-shared-image-followup"].lower() == "blue", "incorrect image followup")
    require(re.findall(r"\b(red|blue)\b", answers["05-two-images"].lower()) == ["red", "blue"], "incorrect two-image order")
    require(answers["08-text-after-image-error"] == "4" and answers["10-text-after-cancel"] == "7", "post-error/disconnect recovery failed")
    follow = indexed["02-shared-image-followup"]
    require(follow["usage"]["prompt_tokens_details"]["cached_tokens"] == 192 and follow["native_monitor"]["cache_source"] == "shared", "shared followup evidence differs")
    require(indexed["03-changed-image"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, "changed image reused old KV")
    invalid = read(folder + "07-invalid.response.json")
    require(invalid["status"] == 400, "invalid PNG was not rejected")
    events = [json.loads(line[6:]) for line in content[folder + "09-cancel.events.txt"].decode().splitlines() if line.startswith("data: {")]
    require(events and len({x["id"] for x in events}) == 1, "invalid disconnect event identity")
    cancelled_id = events[0]["id"]
    snapshot = read(folder + "final.monitor.json")
    require(any(x["native_request_id"] == cancelled_id and x["status"] == "disconnected" for x in snapshot["recent"]), "stream did not disconnect natively")
    for key in ("parking_sessions", "finishing_sessions", "io_pending_sequences", "active_archive_bytes", "active_checkpoint_bytes"):
        require(snapshot["kvmem"][key] == 0, "pending ownership not drained: " + key)
    for i, (size, reverse) in enumerate(((128, False), (128, True), (512, False))):
        raw = content[f"image-{i}.png"]
        check_png(raw, size, reverse)
        require(hashlib.sha256(raw).hexdigest() == profile["image_sha256"][i], "fixture identity mismatch")
    for label, expected_images in {"01-small-image": [0], "03-changed-image": [1], "04-large-image": [2], "05-two-images": [0, 1], "06-parallel-0": [0], "06-parallel-1": [1]}.items():
        body = read(folder + label + ".request.json")
        urls = [p["image_url"]["url"] for p in body["messages"][0]["content"] if p["type"] == "image_url"]
        expected_urls = ["data:image/png;base64," + base64.b64encode(content[f"image-{i}.png"]).decode() for i in expected_images]
        require(urls == expected_urls, "request does not carry the recorded image: " + label)
    embeddings = read("embedding-summary.json")
    require(len(embeddings) == 3, "unexpected encoder output count")
    for item in embeddings:
        raw = content["embeddings/" + item["file"]]
        t, h, w = item["grid"]
        require(item["rows"] == t * h * w // 4 and h % 2 == w % 2 == 0, "incorrect merged image grid")
        require(item["bytes"] == len(raw) and item["sha256"] == hashlib.sha256(raw).hexdigest(), "encoder output hash mismatch")
        check_embedding(raw, item["rows"])
    require({x["rows"] for x in embeddings} == {64, 256}, "missing image token size")
    require(len({x["sha256"] for x in embeddings if x["rows"] == 64}) == 2, "different images encoded identically")
    samples = [json.loads(line) for line in content[folder + "samples.jsonl"].decode().splitlines()]
    require(max(len(x.get("monitor", {}).get("active", [])) for x in samples) >= 2, "two native requests did not overlap")
    for field in ("gpu_vram_used_bytes", "VmRSS_bytes", "VmHWM_bytes", "VmSwap_bytes"):
        require(max(x[field] for x in samples if x.get(field) is not None) == profile["sampled_peak_" + field], "peak sampling mismatch: " + field)
    require(profile["returncode_after_cleanup"] == 0, "server did not exit cleanly")
    require(all(p.get("process_info") == "No running processes detected" for board in summary["gpu_processes"] for p in board["process_list"]), "GPU still occupied at cleanup")
    return {"passed": True, "archive_sha256": actual_sha, "verified_members": len(content), "normal_responses": len(requests), "historical_checks": 57, "encoder_rows": sorted({x["rows"] for x in embeddings}), "native_disconnected_id": cancelled_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.archive, args.sha256), indent=2))
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        parser.exit(1, f"CPU image evidence failed: {error}\n")


if __name__ == "__main__":
    main()
