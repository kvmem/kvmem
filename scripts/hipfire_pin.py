#!/usr/bin/env python3
"""Generate or verify backends/hipfire/kvmem.pin.

The pin binds hipfire's Rust build to the exact KVMem headers, host-source
manifest and C ABI shim that were validated. Run --write after changing any of
those files. The recorded commit is provenance only; the digests are the hard
check. KVMEM_EXPECTED_COMMIT (or --expected-commit) additionally requires the
current checkout's HEAD, which is useful in CI.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PIN = Path("backends/hipfire/kvmem.pin")
INCLUDE_DIR = Path("kvmem/include")
HOST_MANIFEST = Path("kvmem/host-sources.txt")
SHIM_FILES = (
    Path("backends/hipfire/hipfire_kvmem_shim.h"),
    Path("backends/hipfire/hipfire_kvmem_shim.cpp"),
)
ABI_RE = re.compile(r"^\s*#define\s+HK_ABI_VERSION\s+([0-9]+)u?\s*$", re.MULTILINE)


class PinError(RuntimeError):
    pass


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise PinError(f"cannot read {path}: {error}") from error


def _digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_read(root / relative))
        digest.update(b"\0")
    return digest.hexdigest()


def _include_files(root: Path) -> list[Path]:
    directory = root / INCLUDE_DIR
    if not directory.is_dir():
        raise PinError(f"include directory not found: {directory}")
    files = [path.relative_to(root / INCLUDE_DIR)
             for path in directory.rglob("*") if path.is_file()]
    if not files:
        raise PinError(f"include directory is empty: {directory}")
    return files


def _host_manifest(root: Path) -> Path:
    path = root / HOST_MANIFEST
    if not path.is_file():
        raise PinError(f"host source manifest not found: {path}")
    return Path(HOST_MANIFEST.name)


def _shim_files(root: Path) -> list[Path]:
    for relative in SHIM_FILES:
        if not (root / relative).is_file():
            raise PinError(f"shim file not found: {root / relative}")
    return list(SHIM_FILES)


def _abi_version(root: Path) -> str:
    header = _read(root / SHIM_FILES[0]).decode("utf-8", errors="replace")
    match = ABI_RE.search(header)
    if not match:
        raise PinError(f"HK_ABI_VERSION is missing from {SHIM_FILES[0]}")
    return match.group(1)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, check=False)
    except OSError:
        return None


def _git_head(root: Path) -> str | None:
    result = _git(root, "rev-parse", "HEAD")
    if result is None or result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _check_commit(root: Path, expected: str | None) -> None:
    if not expected:
        return
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", expected):
        raise PinError(f"invalid expected commit: {expected!r}")
    head = _git_head(root)
    if head is None:
        raise PinError(
            f"cannot verify expected commit {expected}: {root} has no readable git HEAD")
    if not head.startswith(expected.lower()):
        raise PinError(f"commit mismatch: expected {expected}, actual {head}")


def _fields(root: Path) -> dict[str, str]:
    include = _include_files(root)
    host = _host_manifest(root)
    shim = _shim_files(root)
    return {
        "abi_version": _abi_version(root),
        "host_sources_sha256": _digest(root / "kvmem", [host]),
        "include_sha256": _digest(root / INCLUDE_DIR, include),
        "shim_sha256": _digest(root, shim),
    }


def read_pin(root: Path) -> dict[str, str]:
    path = root / PIN
    if not path.is_file():
        raise PinError(f"pin file not found: {path}")
    values: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise PinError(f"{path}:{lineno}: expected key=value")
        key, value = line.split("=", 1)
        key = key.strip()
        if key in values:
            raise PinError(f"{path}:{lineno}: duplicate key: {key}")
        values[key] = value.strip()
    for key in ("commit", "abi_version", "host_sources_sha256", "include_sha256", "shim_sha256"):
        if key not in values:
            raise PinError(f"{path}: missing {key}")
    return values


def write_pin(root: Path) -> dict[str, str]:
    fields = _fields(root)
    head = _git_head(root)
    commit = head or "unknown"
    dirty = _git(root, "status", "--porcelain")
    if dirty is not None and dirty.returncode == 0 and dirty.stdout.strip():
        print("warning: working tree is dirty; pin digests describe the current files",
              file=sys.stderr)
    values = {"commit": commit, **fields}
    lines = ["# generated by scripts/hipfire_pin.py --write"]
    for key in ("commit", "abi_version", "host_sources_sha256", "include_sha256", "shim_sha256"):
        lines.append(f"{key}={values[key]}")
    path = root / PIN
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".kvmem-pin-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return values


def check_pin(root: Path, expected_commit: str | None) -> dict[str, str]:
    recorded = read_pin(root)
    fields = _fields(root)
    if recorded["abi_version"] != fields["abi_version"]:
        raise PinError(
            f"abi_version mismatch: pin {recorded['abi_version']}, header {fields['abi_version']}")
    for key in ("host_sources_sha256", "include_sha256", "shim_sha256"):
        if recorded[key] != fields[key]:
            raise PinError(f"{key} mismatch: expected {recorded[key]}, actual {fields[key]}")
    _check_commit(root, expected_commit)
    return recorded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="repository root (default: the checkout containing this script)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify the pin (default)")
    mode.add_argument("--write", action="store_true", help="rewrite the pin atomically")
    parser.add_argument("--expected-commit", default=None,
                        help="require HEAD to start with this commit (default: KVMEM_EXPECTED_COMMIT)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    expected = args.expected_commit or os.environ.get("KVMEM_EXPECTED_COMMIT")
    try:
        if args.write:
            values = write_pin(root)
            print(f"wrote {root / PIN} (commit {values['commit']})")
        else:
            check_pin(root, expected)
            print(f"{root / PIN}: digest OK")
    except PinError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
