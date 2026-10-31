#!/usr/bin/env python3
"""Compare a first-user replay from zero with the same model's dense execution.

Identical tokens and top-5 log-probabilities are required. This is a replay
mechanics test, not evidence of sparse retrieval quality on arbitrary history.
Run under the external GPU lock; each server finishes before its successor starts.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "home", "model", "sdk", "output", "prompt-file"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-answer", default="OK")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    common = [sys.executable, str(Path(__file__).with_name("smoke_server.py"))]
    for name in ("exe", "home", "model", "sdk", "prompt-file"):
        common += ["--" + name, str(getattr(args, name.replace("-", "_")))]
    common += ["--disable-thinking", "--observe-statistics", "--logprobs",
               "--expected-answer", args.expected_answer]
    results = []
    for name, extra in (("dense", []), ("replay", ["--selection-budget", "64", "--selection-block", "16",
                                                  "--require-frozen-query", "--require-query-replay"])):
        rc = subprocess.run(common + ["--output", str(args.output / name)] + extra, check=False).returncode
        if rc:
            raise RuntimeError(f"{name} qualification failed: {rc}")
        results.append(json.loads((args.output / name / "response.json").read_text())["choices"][0])
    a, b = results
    if a["message"] != b["message"] or a["finish_reason"] != b["finish_reason"]:
        raise RuntimeError("dense/replay output differs")
    left, right = a["logprobs"]["content"], b["logprobs"]["content"]
    if not left or len(left) != len(right):
        raise RuntimeError("missing or unequal token logprobs")
    errors = []
    for x, y in zip(left, right):
        if x["token"] != y["token"]:
            raise RuntimeError("sampled token differs")
        pairs = [(x, y)]
        tx, ty = x["top_logprobs"], y["top_logprobs"]
        if len(tx) != 5 or [p["token"] for p in tx] != [p["token"] for p in ty]:
            raise RuntimeError("top-5 token identities differ")
        pairs += list(zip(tx, ty))
        for p, q in pairs:
            error = abs(p["logprob"] - q["logprob"])
            if not math.isfinite(error):
                raise RuntimeError("non-finite logprob comparison")
            errors.append(error)
    result = {"pass": max(errors) <= args.tolerance, "compared_values": len(errors),
              "max_abs_logprob_difference": max(errors), "tolerance": args.tolerance,
              "scope": "first-user query replay from zero versus dense, same executable/model/runtime",
              "response": a["message"]}
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
