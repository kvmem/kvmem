#!/usr/bin/env python3
"""Five fixed tool-history turns; exact native KV/state reuse, not final agent33."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "home", "model", "sdk", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--tool-lines", type=int, default=35)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--replay-tokens", type=int, default=1024)
    p.add_argument("--budget", type=int, default=256)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    messages = [
        {"role": "system", "content": "You are checking routine reports. Respond with only OK, no commentary."},
        {"role": "user", "content": "Review the reports I will provide. After each report, reply with only OK."}]
    cases = []
    for index in range(5):
        # The small reference model may acknowledge or request the next report.
        # Both are structurally meaningful; arbitrary prose/garbage is not.
        # After the dense run the candidate is held to its EXACT chosen answer.
        next_call = ("<tool_call>\n<function=read_report>\n<parameter=index>\n" +
                     str(index) + "\n</parameter>\n</function>\n</tool_call>")
        cases.append({"messages": copy.deepcopy(messages), "expected": ["OK", next_call]})
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"report-{index}", "type": "function",
                "function": {"name": "read_report", "arguments": json.dumps({"index": index})}}]},
            {"role": "tool", "tool_call_id": f"report-{index}", "content": "\n".join(
                f"Report {index} row {line}: inspection completed; everything is normal."
                for line in range(a.tool_lines)) +
                "\nReport complete. No further tool calls are needed. Acknowledge with only OK."}]
    casefile = a.output / "cases.json"
    casefile.write_text(json.dumps(cases, indent=2))
    cmd = [sys.executable, str(Path(__file__).with_name("smoke_server.py"))]
    for name in ("exe", "home", "model", "sdk"):
        cmd += ["--" + name, str(getattr(a, name))]
    cmd += ["--cases", str(casefile), "--rounds", "5", "--context", str(a.context),
            "--replay-tokens", str(a.replay_tokens), "--embedding-placement", "ram",
            "--observe-statistics", "--disable-thinking"]
    dense = subprocess.run(cmd + ["--output", str(a.output / "dense")], check=False).returncode
    if dense:
        raise RuntimeError("dense reference failed: this fixture cannot qualify sparse continuation")
    reference = json.loads((a.output / "dense" / "rounds.json").read_text())
    for case, row in zip(cases, reference):
        case["expected"] = row["content"]
    casefile = a.output / "candidate-cases.json"
    casefile.write_text(json.dumps(cases, indent=2))
    cmd[cmd.index("--cases") + 1] = str(casefile)
    return subprocess.run(cmd + ["--output", str(a.output / "evidence"),
        "--selection-budget", str(a.budget), "--selection-block", "16",
        "--require-prefix-resume", "--require-bounded-pool"], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
