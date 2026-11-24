#!/usr/bin/env python3
"""Keep kvmem/host-sources.txt in sync with kvmem/src/host/*.cpp.

The CMake build and hipfire's build.rs consume the same manifest. `--check`
verifies that it is a duplicate-free, existing, complete set; `--write`
preserves the current order, appends new sources sorted, and drops stale ones.
"""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import sys


ROOT = Path(__file__).resolve().parents[1]


class ManifestError(RuntimeError):
    pass


def _sources_root(root: Path) -> Path:
    return root / "kvmem"


def _manifest_path(root: Path) -> Path:
    return _sources_root(root) / "host-sources.txt"


def _discover(root: Path) -> list[str]:
    host_dir = _sources_root(root) / "src/host"
    if not host_dir.is_dir():
        raise ManifestError(f"host source directory not found: {host_dir}")
    entries = sorted(
        path.relative_to(_sources_root(root)).as_posix()
        for path in host_dir.glob("*.cpp")
        if path.is_file()
    )
    return entries


def _read_entries(root: Path, require_exists: bool) -> list[str]:
    manifest = _manifest_path(root)
    if not manifest.is_file():
        raise ManifestError(f"source manifest not found: {manifest}")
    sources_root = _sources_root(root)
    entries: list[str] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\\" in line:
            raise ManifestError(f"{manifest}:{lineno}: use '/' in source paths")
        path = PurePosixPath(line)
        if path.is_absolute() or ".." in path.parts:
            raise ManifestError(f"{manifest}:{lineno}: path must stay under kvmem/: {line}")
        if path.parts[:2] != ("src", "host") or path.suffix != ".cpp":
            raise ManifestError(
                f"{manifest}:{lineno}: expected src/host/*.cpp, got: {line}")
        entry = path.as_posix()
        if entry in seen:
            raise ManifestError(f"{manifest}:{lineno}: duplicate source: {entry}")
        seen.add(entry)
        if require_exists and not (sources_root / Path(*path.parts)).is_file():
            raise ManifestError(f"{manifest}:{lineno}: source does not exist: {entry}")
        entries.append(entry)
    if not entries:
        raise ManifestError(f"{manifest}: no sources listed")
    return entries


def _validate_set(entries: list[str], discovered: list[str]) -> None:
    listed = set(entries)
    actual = set(discovered)
    missing = sorted(actual - listed)
    extra = sorted(listed - actual)
    if missing:
        raise ManifestError(
            "source manifest is missing: " + ", ".join(missing))
    if extra:
        raise ManifestError(
            "source manifest lists files that are not src/host/*.cpp: "
            + ", ".join(extra))


def check(root: Path) -> tuple[list[str], list[str]]:
    entries = _read_entries(root, require_exists=True)
    discovered = _discover(root)
    _validate_set(entries, discovered)
    return entries, discovered


def write(root: Path) -> list[str]:
    existing = _read_entries(root, require_exists=False)
    discovered = _discover(root)
    actual = set(discovered)
    kept = [entry for entry in existing if entry in actual]
    entries = kept + sorted(actual - set(kept))
    _validate_set(entries, discovered)
    manifest = _manifest_path(root)
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="repository root (default: the checkout containing this script)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="verify the manifest (default)")
    mode.add_argument("--write", action="store_true",
                      help="rewrite the manifest instead of checking it")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.write:
            entries = write(root)
            print(f"wrote {_manifest_path(root)} ({len(entries)} sources)")
        else:
            entries, _ = check(root)
            print(f"{_manifest_path(root)}: {len(entries)} sources OK")
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
