"""Optional draft-only recipe for the pinned Qwen dense native MTP path.

Reuse Radiance libquant/libr4d; never quantize target layers by a broad glob.
The layer index comes from config, not a hard-coded 27B layer count. This pin
declares one MTP block. Its fc uses plain BF16 gemm_nt, so leave fc and norms
alone instead of changing native arithmetic to accommodate a recipe.
"""
import argparse
import hashlib
import json
from pathlib import Path

PROJECTIONS = ("attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down")
QUANT = {
    "mxfp4": "rtn codes=fp4_e2m1 group=32 scale=e8m0",
    "fp8": "rtn codes=fp8_e4m3 block=128x128 scale=bf16",
}


def draft_names(config):
    text = config.get("text_config", config)
    if not isinstance(text, dict) or text.get("model_type") not in {"qwen3_5_text", "qwen3_5"}:
        raise ValueError("only the pinned Qwen dense MTP name map is supported")
    count = text.get("num_hidden_layers")
    if type(count) is not int or not 1 <= count <= 10000:
        raise ValueError("num_hidden_layers must be a positive integer")
    if type(text.get("mtp_num_hidden_layers")) is not int or text["mtp_num_hidden_layers"] != 1:
        raise ValueError("this native name map declares exactly one MTP block")
    if text.get("num_experts", 0) or text.get("num_local_experts", 0):
        raise ValueError("routed MTP projections need their own recipe; not dense")
    return [f"blk.{count}.{name}.weight" for name in PROJECTIONS]


def recipe(config, base, precision):
    if precision not in QUANT:
        raise ValueError("unsupported precision")
    names = draft_names(config)
    return ("# Draft projections only. Generated from checkpoint text_config.\n"
            "# Native libquant/libr4d recipe; requires rad-convert --plan-only first.\n"
            "# Target weights, BF16 mtp.fc, norms and existing U2 draft head are unchanged.\n"
            + "".join(f"{name} {QUANT[precision]}\n" for name in names)
            + "\n# Original target/output/draft-head rules follow unchanged.\n" + base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--precision", required=True, choices=tuple(QUANT))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = recipe(config, args.base.read_text(), args.precision)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(result)
    print(json.dumps({"recipe": str(args.output), "precision": args.precision,
                      "allowed_changed_weights": draft_names(config),
                      "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                      "base_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
