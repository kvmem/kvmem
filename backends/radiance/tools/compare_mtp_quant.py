"""Audit complete model payloads: only the requested MTP projections may change."""
import argparse
import json
from pathlib import Path

from compare_rad_target import compare
from inspect_rad import inspect
from mtp_quant_recipe import draft_names


def judge(result, allowed):
    changed = {row["name"] for row in result["different"]}
    return {"pass": not result["missing"] and not result["candidate_extra"]
                     and changed == set(allowed),
            "allowed_changed_weights": sorted(allowed),
            "actual_changed_weights": sorted(changed),
            "unexpected_changes": sorted(changed - set(allowed)),
            "expected_but_unchanged": sorted(set(allowed) - changed),
            "unchanged_weight_count": result["equal_count"],
            "changed_payload_bytes_saved": sum(row["reference_bytes"] - row["candidate_bytes"]
                                                for row in result["different"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for model in (args.reference, args.candidate):
        info = inspect(model)
        if info["entries_by_name_prefix"].get("dflash") or not info["entries_by_name_prefix"].get("mtp"):
            raise ValueError("audit requires two MTP containers without DFlash")
    result = compare(args.reference, args.candidate)
    audit = judge(result, draft_names(json.loads(args.config.read_text())))
    result["draft_quantization_audit"] = audit
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(audit, indent=2))
    return 0 if audit["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
