#!/usr/bin/env python3
"""Deterministic five-case history recall: dense reference then sparse replay.

The fact is outside the first sink block and the recent tail. Both variants must
answer it correctly; a dense failure is a failed fixture, not a sparse success.
This small-model gate complements, rather than replaces, final agent tests.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "home", "model", "sdk", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--embedding-placement", choices=("ram", "vram"), default="vram")
    parser.add_argument("--windows-gpu-counters", action="store_true")
    parser.add_argument("--filler-lines", type=int, default=30)
    parser.add_argument("--filler-step", type=int, default=8)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--require-bounded-pool", action="store_true")
    args = parser.parse_args()
    if args.filler_lines < 0 or args.filler_step < 0 or args.context < 1:
        parser.error("invalid fixture/context size")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for index, (project, code) in enumerate(zip(
            ("Falcon", "Orchid", "Cedar", "Aurora", "Maple"),
            ("AMBER", "COBALT", "VIOLET", "SILVER", "CORAL"))):
        prefix = "Project archive: routine records are listed below. " * 12
        fact = f"\nImportant record: The access code for project {project} is {code}.\n"
        filler = "\n".join(f"Routine report {i}: the inspection is complete and no action is required."
                           for i in range(args.filler_lines + index * args.filler_step))
        cases.append({"messages": [
            {"role": "user", "content": prefix + fact + filler},
            {"role": "assistant", "content": "I have read the archive."},
            {"role": "user", "content": f"What is the access code for project {project}? Reply with only the code."}],
            "expected": code})
    case_path = args.output / "cases.json"
    case_path.write_text(json.dumps(cases, indent=2))
    common = [sys.executable, str(Path(__file__).with_name("smoke_server.py"))]
    for name in ("exe", "home", "model", "sdk"):
        common += ["--" + name, str(getattr(args, name))]
    common += ["--disable-thinking", "--observe-statistics", "--rounds", "5",
               "--context", str(args.context), "--cases", str(case_path),
               "--embedding-placement", args.embedding_placement]
    if args.windows_gpu_counters:
        common += ["--windows-gpu-counters"]
    summary = {}
    for name, extra in (("dense", []), ("sparse", ["--selection-budget", str(args.budget),
            "--selection-block", "16", "--require-frozen-query", "--require-query-replay"])):
        if name == "sparse" and args.require_bounded_pool:
            extra += ["--require-bounded-pool"]
        rc = subprocess.run(common + ["--output", str(args.output / name)] + extra, check=False).returncode
        result = json.loads((args.output / name / "result.json").read_text())
        summary[name] = {"exit_code": rc, "pass": result["pass"], "rounds": result.get("rounds", []),
                         "restore_observed": result.get("restore_observed")}
        (args.output / "comparison.json").write_text(json.dumps(summary, indent=2))
        if rc:
            raise RuntimeError(f"{name} history recall failed; preserved evidence for diagnosis")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
