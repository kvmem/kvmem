"""Select/verify existing Radiance drafters; no model or engine modifications.

The pinned qwen35 plugin selects DFlash ahead of MTP from container metadata.
Turning speculation off is max_spec=0, not RADIANCE_NO_DRAFT (which retains
the weights and therefore cannot measure the memory saving).
"""
import re

from inspect_rad import inspect


def declared_integer(value):
    # Native converters also store flattened config integers as string metadata.
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    raise ValueError(f"invalid integer in native drafter metadata: {value!r}")


def resolve(model, mode, depth=None):
    if mode not in ("auto", "off", "mtp", "dflash"):
        raise ValueError(f"unknown drafter: {mode}")
    if depth is not None and depth < 0:
        raise ValueError("speculative tokens must be nonnegative")
    if mode == "auto":
        return {"requested": mode, "depth": 0 if depth is None else depth,
                "expected_native": None}
    if mode == "off":
        if depth not in (None, 0):
            raise ValueError("--drafter off requires zero speculative tokens")
        return {"requested": mode, "depth": 0, "expected_native": "off"}
    if not model.is_file() or model.suffix != ".rad":
        raise ValueError("explicit drafter preflight currently requires a complete .rad container")
    container = inspect(model)
    if container["arch_id"] not in ("qwen35", "qwen35moe"):
        raise ValueError("explicit drafter selection is validated only for the pinned qwen35 plugin")
    meta = container["draft_metadata"]
    groups = container["entries_by_name_prefix"]
    has_dflash = "DFlash" in str(meta.get("draft.architectures", ""))
    if mode == "mtp":
        if not groups.get("mtp", 0):
            raise ValueError("model contains no mtp.* weights; use an MTP-bearing container converted without --draft-model")
        if has_dflash:
            raise ValueError("pinned upstream prioritizes DFlash; use an MTP-only container, not renamed metadata")
        if declared_integer(meta.get("qwen35.nextn_predict_layers", meta.get("mtp_num_hidden_layers", 0))) <= 0:
            raise ValueError("MTP weights exist but the container declares no NextN layer")
        chosen = 3 if depth is None else depth
        if chosen == 0:
            raise ValueError("--drafter mtp needs a positive depth; use off to disable")
        native = "mtp"
    else:
        if not has_dflash or not groups.get("dflash", 0):
            raise ValueError("model does not contain a declared DFlash drafter and weights")
        block = declared_integer(meta.get("draft.dflash_config.block_size"))
        if block < 2 or block & (block - 1):
            raise ValueError("DFlash container does not declare a valid trained block size")
        chosen = block - 1 if depth is None else depth
        if chosen != block - 1:
            raise ValueError(f"DFlash trained block requires {block - 1} speculative tokens")
        native = "dflash2"
    return {"requested": mode, "depth": chosen, "expected_native": native,
            "container_preflight": container,
            "note": "directory check only; runtime must still verify the actual drafter"}


def verify_log(selection, log):
    declarations = re.findall(r"speculation: '([^']+)'", log)
    expected = selection["expected_native"]
    if expected is None:
        ok = True  # historical auto mode: record without asserting a specific drafter
    elif expected == "off":
        ok = not declarations
    else:
        ok = bool(declarations) and all(name == expected for name in declarations)
    return {"requested": selection["requested"], "expected": expected,
            "declared": declarations, "pass": ok}
