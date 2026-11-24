#!/usr/bin/env python3
"""Static contract checks for the hipfire KVMem adapter.

The shim is deliberately independent of llama.cpp/ninfer/polystrata. This test
locks the pieces that can silently drift:

* the indexed checkout layout and pin digests;
* the C ABI version in the header and in the Rust crate;
* every Rust `#[repr(C)]` type is registered (new ones cannot appear silently);
* the two configuration structs mirrored by the C ABI still exist on both sides.

Rust-side checks are skipped when no hipfire checkout is available. Run the CI
copy with `--hipfire-dir` after `scripts/prepare-backends.py --backend hipfire`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "backends/hipfire/hipfire_kvmem_shim.h"
SOURCE = ROOT / "backends/hipfire/hipfire_kvmem_shim.cpp"
VERSIONS = ROOT / "backends/versions.json"
UPSTREAM = ROOT / "backends/hipfire/upstream.json"

# Rust `#[repr(C)]` types currently present in crates/hipfire-kvmem/src/lib.rs.
# Only the two config structs are mirrored by the C ABI today. The rest are
# opaque handles or Rust-owned records; registering them here makes a new
# `#[repr(C)]` declaration fail this test until someone decides its ABI.
KNOWN_REPR_C = {
    "BlockCopies", "BlockDescriptor", "CacheOwnerHints", "ContentCandidate",
    "ContentSlot", "CoverageInfo", "HkBackendIdentity", "HkCheckpointPlan",
    "HkCurrentLayerQuery", "HkExecutionView", "HkHostRecord", "HkHostSource",
    "HkKvRegions", "HkKvSnapshot", "HkKvStore", "HkLayerMemorySnapshot",
    "HkLayerPlan", "HkLayerRequestSpec", "HkLayerSelection", "HkLayerwiseEnvelope",
    "HkLayerwiseImportStart", "HkLayerwisePlan", "HkLayerwiseRequest",
    "HkLayerwiseWindowPlan", "HkLegacyReplayPlan", "HkMemoryCatalog",
    "HkMemoryDescriptor", "HkMemorySnapshot", "HkNativeArchive", "HkNativePayload",
    "HkOutputAcceptance", "HkOutputMass", "HkOutputPressure", "HkOutputSnapshot",
    "HkPackedSnapshot", "HkPayloadLayout", "HkPrefillAcceptance", "HkPrefillMass",
    "HkPrefillPressure", "HkPrefillSnapshot", "HkPressureChoice", "HkPressureChoices",
    "HkProbeSelection", "HkQueryLayout", "HkQueryPlaneBytes", "HkQueryScoreFrame",
    "HkRawStore", "HkReservation", "HkRetrievalCheckpoint", "HkSpeculativeStatistics",
    "HkSpill", "HkStorage", "HkTape", "HkTapeChunk", "KvStoreConfig",
    "LayerQueryGeometry", "LayerwiseIdentity", "LayerwiseWork", "MemoryBudget",
    "MemoryCapabilities", "MemoryStamp", "NativeCurrentLayerQuery", "OutputOptions",
    "OutputScopeRow", "OutputScopeState", "PackedSelection", "PayloadPlane",
    "PrefillOptions", "PrefixPackInfo", "PrefixRange", "QueryProjectionPlane",
    "QueryRange", "RawStoreConfig", "SessionIdentity", "ViewBlock", "WindowState",
}
MIRRORED_CONFIG = {
    "RawStoreConfig": "hk_raw_store_config",
    "KvStoreConfig": "hk_kv_store_config",
}
REPR_C = re.compile(
    r"#\[repr\(C\)\]\s*(?:(?:#\[[^\]]*\])\s*)*(?:pub(?:\([^)]*\))?\s+)?struct "
    r"([A-Za-z_][A-Za-z0-9_]*)")
ABI_MACRO = re.compile(r"^\s*#define\s+HK_ABI_VERSION\s+([0-9]+)u?\s*$", re.MULTILINE)
RUST_ABI = re.compile(r"pub const HK_ABI_VERSION:\s*u32\s*=\s*([0-9]+)\s*;")


def _pin_module():
    spec = importlib.util.spec_from_file_location(
        "hipfire_pin", ROOT / "scripts/hipfire_pin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _struct_names(text: str) -> set[str]:
    return set(REPR_C.findall(text))


def _struct_fields(header: str, c_name: str) -> list[str]:
    match = re.search(rf"struct\s+{re.escape(c_name)}\s*\{{(.*?)\n\}}", header, re.S)
    if not match:
        return []
    fields = []
    for line in match.group(1).splitlines():
        line = re.sub(r"//.*", "", line).strip()
        field = re.match(r"[A-Za-z_][A-Za-z0-9_:\s]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", line)
        if field:
            fields.append(field.group(1))
    return fields


def _rust_fields(text: str, struct: str) -> list[str]:
    match = re.search(rf"pub struct\s+{re.escape(struct)}\s*\{{(.*?)\n\}}", text, re.S)
    if not match:
        return []
    fields = []
    for line in match.group(1).splitlines():
        field = re.match(r"\s*pub\s+([A-Za-z_][A-Za-z0-9_]*)\s*:", line)
        if field:
            fields.append(field.group(1))
    return fields


def header_guard_errors(header: str, source: str) -> list[str]:
    errors: list[str] = []
    for c_name in MIRRORED_CONFIG.values():
        fields = _struct_fields(header, c_name)
        if not fields:
            continue
        for token in (f"sizeof({c_name})", f"alignof({c_name})"):
            if f"static_assert({token}" not in header:
                errors.append(f"shim header has no static_assert for {token}")
        for field in fields:
            token = f"offsetof({c_name}, {field})"
            if f"static_assert({token}" not in header:
                errors.append(f"shim header has no static_assert for {token}")
    for include in re.findall(r'^\s*#\s*include\s+"([^"]+)"', source, re.MULTILINE):
        if include == "hipfire_kvmem_shim.h":
            continue
        if include.startswith("kvmem/") and include.endswith((".h", ".hpp")):
            continue
        errors.append(f"shim include is outside the KVMem allowlist: {include}")
    return errors


def rust_entrypoint_errors(text: str) -> list[str]:
    return [] if text.count("ensure_abi()?") >= 5 else [
        "Rust crate does not check ensure_abi() at all five entry points"
    ]


def check(hipfire_dir: Path | None, require_rust: bool = False) -> list[str]:
    errors: list[str] = []

    versions = json.loads(VERSIONS.read_text(encoding="utf-8"))
    entry = versions.get("hipfire")
    if not isinstance(entry, dict):
        errors.append("versions.json has no hipfire entry")
        entry = {}
    if entry.get("layout") != "external-checkout-with-patches":
        errors.append("hipfire entry is not external-checkout-with-patches")
    manifest_rel = entry.get("manifest")
    if manifest_rel != "backends/hipfire/upstream.json":
        errors.append(f"hipfire manifest is {manifest_rel!r}, expected backends/hipfire/upstream.json")
    if not UPSTREAM.is_file():
        errors.append("backends/hipfire/upstream.json is missing")
        return errors
    upstream = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    if upstream.get("layout") != entry.get("layout"):
        errors.append("upstream.json layout differs from versions.json")
    for key in ("repository", "base_commit", "patched_tree"):
        if not isinstance(upstream.get(key), str) or not upstream[key]:
            errors.append(f"upstream.json has no {key}")
    if upstream.get("series") in {"grouped-v1", "grouped-v2", "grouped-v3", "grouped-v4", "grouped-v5", "grouped-v6"}:
        archive_name = upstream.get("previous_series")
        if not isinstance(archive_name, str) or not archive_name:
            errors.append("grouped manifest has no previous_series")
        else:
            archive = (UPSTREAM.parent / archive_name).resolve()
            if not archive.is_relative_to(ROOT) or not archive.is_file():
                errors.append(f"individual patch archive is missing: {archive_name}")
            else:
                archived = json.loads(archive.read_text(encoding="utf-8"))
                if len(archived.get("patches", [])) != 73:
                    errors.append("individual patch archive does not contain 73 patches")
                if archived.get("patched_tree") != upstream.get("patched_tree"):
                    errors.append("individual patch archive final tree differs from upstream.json")
    patches = upstream.get("patches")
    if not isinstance(patches, list) or not patches:
        errors.append("upstream.json has no patch series")
    else:
        for patch in patches:
            path = patch.get("path") if isinstance(patch, dict) else None
            if not isinstance(path, str):
                errors.append(f"invalid patch entry: {patch!r}")
                continue
            resolved = (UPSTREAM.parent / path).resolve()
            if not resolved.is_relative_to(ROOT):
                errors.append(f"patch path escapes the repository: {path}")
            elif not resolved.is_file():
                errors.append(f"patch file is missing: {path}")

    header = HEADER.read_text(encoding="utf-8")
    macro = ABI_MACRO.search(header)
    if not macro:
        errors.append("HK_ABI_VERSION is missing from the shim header")
    else:
        pin_module = _pin_module()
        try:
            pin = pin_module.read_pin(ROOT)
        except pin_module.PinError as error:
            errors.append(str(error))
        else:
            if pin["abi_version"] != macro.group(1):
                errors.append(
                    f"pin abi_version {pin['abi_version']} != header {macro.group(1)}")
        try:
            pin_module.check_pin(ROOT, os.environ.get("KVMEM_EXPECTED_COMMIT"))
        except pin_module.PinError as error:
            errors.append(f"pin check failed: {error}")

    for rust_name, c_name in MIRRORED_CONFIG.items():
        if not _struct_fields(header, c_name):
            errors.append(f"shim header has no {c_name} fields")
    errors.extend(header_guard_errors(header, SOURCE.read_text(encoding="utf-8")))

    if hipfire_dir is None:
        if require_rust:
            errors.append("Rust contract checks require a prepared hipfire checkout")
        else:
            print("hipfire checkout not provided; skipping Rust contract checks")
        return errors
    lib = hipfire_dir / "crates/hipfire-kvmem/src/lib.rs"
    abi = hipfire_dir / "crates/hipfire-kvmem/src/abi.rs"
    if not abi.is_file():
        if require_rust:
            errors.append(f"Rust layout guard is missing: {abi}")
        else:
            print("hipfire checkout is not prepared with the P0 contract; "
                  "skipping Rust contract checks")
        return errors
    if not lib.is_file():
        errors.append(f"Rust crate source is missing: {lib}")
        return errors
    text = lib.read_text(encoding="utf-8")
    rust_abi = RUST_ABI.search(text)
    if not rust_abi:
        errors.append("Rust crate has no pub const HK_ABI_VERSION: u32")
    elif macro and rust_abi.group(1) != macro.group(1):
        errors.append(f"Rust ABI {rust_abi.group(1)} != header ABI {macro.group(1)}")
    errors.extend(rust_entrypoint_errors(text))
    repr_types = _struct_names(text)
    unknown = sorted(repr_types - KNOWN_REPR_C)
    if unknown:
        errors.append(f"unregistered #[repr(C)] structs: {', '.join(unknown)}")
    for rust_name in MIRRORED_CONFIG:
        if rust_name not in repr_types:
            errors.append(f"Rust crate has no #[repr(C)] {rust_name}")
    for rust_name, c_name in MIRRORED_CONFIG.items():
        c_fields = _struct_fields(header, c_name)
        r_fields = _rust_fields(text, rust_name)
        if r_fields and c_fields and c_fields != r_fields:
            errors.append(
                f"{rust_name} fields {r_fields} != {c_name} fields {c_fields}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hipfire-dir", type=Path, default=None,
                        help="prepared hipfire checkout (default: HIPFIRE_DIR or ../hipfire)")
    parser.add_argument("--no-rust", action="store_true",
                        help="skip Rust-side checks even when a checkout exists")
    parser.add_argument("--require-rust", action="store_true",
                        help="fail when the checkout lacks the P0 Rust crate")
    args = parser.parse_args(argv)
    hipfire_dir = None if args.no_rust else (args.hipfire_dir or (
        Path(os.environ["HIPFIRE_DIR"]) if os.environ.get("HIPFIRE_DIR") else None))
    if not args.no_rust and hipfire_dir is None:
        candidate = ROOT.parent / "hipfire"
        hipfire_dir = candidate if (candidate / ".git").exists() else None
    errors = check(hipfire_dir, args.require_rust)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("hipfire KVMem contract OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
