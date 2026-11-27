#!/usr/bin/env python3
"""Repack the hipfire patch chain into semantic group patches.

The individual patch series is kept for history (``upstream-75.json`` and the
original patch files); ``upstream.json`` points at a small ordered set of group
patches so a reviewer can see subsystem boundaries without losing the verified
final tree. Group patches are generated from real tree snapshots: each group is
the binary diff between the tree before and after its underlying patches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
HIPFIRE_DIR = ROOT / "backends" / "hipfire"
DEFAULT_MANIFEST = HIPFIRE_DIR / "upstream.json"
DEFAULT_ARCHIVE = HIPFIRE_DIR / "upstream-75.json"

# Each underlying patch belongs to exactly one group and groups stay in the
# verified sequential order. Keep this list in sync with the ids in the
# individual manifest; validation rejects missing, extra or reordered ids.
GROUPS = (
    {
        "id": "escha-baseline",
        "file": "hipfire-group-00-escha-baseline.patch",
        "patches": ("escha-baseline",),
    },
    {
        "id": "fork-base",
        "file": "hipfire-group-01-fork-base.patch",
        "patches": ("tool-recover", "codex-responses", "bonsai-render"),
    },
    {
        "id": "kvmem-stack",
        "file": "hipfire-group-02-kvmem-stack.patch",
        # Groups 02-09 shared file locations at their boundaries (the later
        # group rewrote lines introduced by the earlier one), so their tree
        # diffs are folded into one contiguous patch. The final tree is
        # unchanged; only the review split is coarser.
        "patches": (
            "kvmem", "rdna3-opt", "spec-accept", "batch-restore",
            "policy-knobs", "query-span", "softmax-simd",
            "layerwise-foundation", "layerwise-function",
            "layerwise-complete", "mtp-foundation", "mtp-function",
            "prefix-pin", "nonq8-mask", "ar-head-aux-coarchive",
            "layerwise-suffix-overflow", "kvmem-metrics",
            "layerwise-window-import", "import-admission",
            "import-coverage-fallback", "endpoint-coverage",
            "checkpoint-policy", "checkpoint-policy-wiring",
            "keep-trace", "content-cache", "content-retention",
            "dflash2-kvmem", "speculative-interface",
            "query-span-identity", "kv-snapshot", "content-directory",
            "content-ssd", "content-pack", "content-pack-wire",
            "dn-persist", "endpoint-dn", "content-single-tier",
            "snapshot-origin-inherit", "snapshot-origin-accounting",
            "prefix-demote-reclaim", "content-hit-metrics",
            "env-docs-config-owned", "single-tier-cleanup",
            "cold-import-staging",
            "media-prefix-anchor", "vl-media-identity",
            "vl-media-refreeze", "vl-reuse-finish",
            "layerwise-vl-admission", "media-replay-probe",
            "content-pack-io",
            "rdna3-weight-prefill", "mtp-complete", "legacy-contract",
            "think-recovery", "legacy-budget-clamp",
            "think-budget-emitter", "think-knobs-answer-room",
            "sequential-loop-recovery",
            "fork-k2-k13-requeue", "kvmem-full-takeover",
            "ui-capability-inspect", "layerwise-feed-capture",
        ),
    },
)


class RepackError(RuntimeError):
    pass


def git(directory: Path, *args: str, check: bool = True, env=None, text: bool = True):
    return subprocess.run(
        ["git", "-c", "core.longpaths=true", "-c", "core.autocrlf=false",
         "-C", str(directory), *args],
        check=check, capture_output=True, text=text, env=env,
    )


def temporary_index():
    from contextlib import contextmanager

    @contextmanager
    def _index():
        handle, filename = tempfile.mkstemp(prefix="kvmem-repack-", suffix=".index")
        os.close(handle)
        os.unlink(filename)
        env = dict(os.environ, GIT_INDEX_FILE=filename)
        try:
            yield env
        finally:
            if Path(filename).exists():
                os.unlink(filename)

    return _index()


def source_tree(directory: Path, base: str = "HEAD") -> str:
    """Snapshot the worktree through a private index without touching its index."""
    with temporary_index() as env:
        git(directory, "read-tree", base, env=env)
        git(directory, "add", "-A", env=env)
        return git(directory, "write-tree", env=env).stdout.strip()


def load_manifest(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RepackError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("patches"), list):
        raise RepackError(f"{path}: manifest has no patches list")
    return data


def validate_groups(patches: list[dict], groups=GROUPS) -> None:
    expected: list[str] = []
    for group in groups:
        if not group.get("id") or not group.get("file") or not group.get("patches"):
            raise RepackError(f"invalid group: {group!r}")
        expected.extend(group["patches"])
    actual = [item.get("id") for item in patches]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("extra " + ", ".join(extra))
        if not detail:
            detail.append("order differs")
        raise RepackError("group definitions do not cover the individual series: " + "; ".join(detail))

    known: set[str] = set()
    for item in patches:
        identity = item.get("id")
        requires = item.get("requires", [])
        bad = [dep for dep in requires if dep not in known]
        if bad:
            raise RepackError(f"{identity}: dependency does not precede it: {', '.join(bad)}")
        known.add(identity)


def resolve_patch(manifest_path: Path, entry: dict) -> Path:
    path = Path(entry["path"])
    if path.is_absolute():
        raise RepackError(f"{entry.get('id')}: patch path must be relative: {path}")
    resolved = (manifest_path.parent / path).resolve()
    if not resolved.is_file():
        raise RepackError(f"{entry.get('id')}: patch file is missing: {resolved}")
    return resolved


def relpath_from_manifest(manifest_path: Path, patch_path: Path) -> str:
    return Path(os.path.relpath(patch_path, manifest_path.parent)).as_posix()


def generate(repo: Path, source_manifest: Path, source: dict, groups,
             out_dir: Path) -> tuple[list[dict], str]:
    """Generate group patches from ``source`` and return (entries, final_tree)."""
    validate_groups(source["patches"], groups)
    by_id = {item["id"]: item for item in source["patches"]}
    base_commit = source["base_commit"]
    base_tree = git(repo, "rev-parse", f"{base_commit}^{{tree}}").stdout.strip()

    parent = Path(tempfile.mkdtemp(prefix="kvmem-repack-"))
    worktree = parent / "tree"
    entries: list[dict] = []
    try:
        git(repo, "worktree", "add", "--detach", "--force", str(worktree), base_commit)
        previous_tree = base_tree
        previous_id = None
        for group in groups:
            for identity in group["patches"]:
                patch = resolve_patch(source_manifest, by_id[identity])
                git(worktree, "apply", "--check", str(patch))
                git(worktree, "apply", "--whitespace=nowarn", str(patch))
            current_tree = source_tree(worktree)
            if current_tree == previous_tree:
                raise RepackError(f"{group['id']}: group produced no tree change")
            diff = git(worktree, "diff", "--binary", previous_tree, current_tree,
                       text=False).stdout
            if not diff:
                raise RepackError(f"{group['id']}: empty binary diff")
            destination = out_dir / group["file"]
            destination.write_bytes(diff)
            entries.append({
                "id": group["id"],
                "path": relpath_from_manifest(source_manifest, destination),
                "sha256": hashlib.sha256(diff).hexdigest(),
                "requires": [previous_id] if previous_id else [],
                "after_tree": current_tree,
            })
            previous_tree = current_tree
            previous_id = group["id"]
        if previous_tree != source["patched_tree"]:
            raise RepackError(
                f"repacked tree {previous_tree} differs from manifest patched_tree "
                f"{source['patched_tree']}")
        return entries, previous_tree
    finally:
        if worktree.exists():
            git(repo, "worktree", "remove", "--force", str(worktree), check=False)
        git(repo, "worktree", "prune", check=False)
        shutil.rmtree(parent, ignore_errors=True)


def verify_series(repo: Path, manifest_path: Path, manifest: dict) -> str:
    """Apply the manifest series in an isolated index and verify every tree."""
    patches = manifest["patches"]
    with temporary_index() as env:
        git(repo, "read-tree", manifest["base_commit"], env=env)
        for item in patches:
            patch = resolve_patch(manifest_path, item)
            git(repo, "apply", "--cached", "--whitespace=nowarn", str(patch), env=env)
            actual = git(repo, "write-tree", env=env).stdout.strip()
            if actual != item["after_tree"]:
                raise RepackError(
                    f"{item['id']}: patch produces {actual}, expected {item['after_tree']}")
    return patches[-1]["after_tree"]


def grouped_output(source: dict, entries: list[dict], patched_tree: str,
                   archive_name: str) -> dict:
    output = dict(source)
    output["patches"] = entries
    output["patched_tree"] = patched_tree
    output["series"] = "grouped-v8"
    output["previous_series"] = archive_name
    return output


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def is_grouped(manifest: dict, groups=GROUPS) -> bool:
    return [item.get("id") for item in manifest["patches"]] == [group["id"] for group in groups]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE,
                        help="individual series retained for history")
    parser.add_argument("--hipfire-dir", type=Path, default=None,
                        help="prepared hipfire checkout (default: ../hipfire)")
    parser.add_argument("--write", action="store_true", help="generate group patches and rewrite the manifest")
    parser.add_argument("--check", action="store_true", help="verify the current manifest series")
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    current = load_manifest(manifest_path)
    if args.check:
        if not is_grouped(current):
            raise RepackError("upstream.json is still the individual series; run --write first")
        repo = (args.hipfire_dir or (ROOT.parent / "hipfire")).resolve()
        verify_series(repo, manifest_path, current)
        print(f"hipfire: {len(current['patches'])} group patches OK "
              f"({current['patched_tree'][:12]})")
        return 0
    if not args.write:
        parser.error("choose --write or --check")

    if is_grouped(current):
        if not args.archive.is_file():
            raise RepackError(f"grouped manifest has no individual archive: {args.archive}")
        source_manifest = args.archive.resolve()
        source = load_manifest(source_manifest)
    else:
        source_manifest = manifest_path
        source = current
    validate_groups(source["patches"])

    archive = args.archive.resolve()
    if not archive.exists():
        archive.write_bytes(manifest_path.read_bytes())
        print(f"archived individual series: {archive}")

    repo = (args.hipfire_dir or (ROOT.parent / "hipfire")).resolve()
    out_dir = manifest_path.parent / ".." / "patches"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="kvmem-groups-", dir=out_dir))
    try:
        entries, patched_tree = generate(repo, source_manifest, source, GROUPS, staging)
        for entry in entries:
            source_file = staging / Path(entry["path"]).name
            destination = out_dir / source_file.name
            os.replace(source_file, destination)
            entry["path"] = relpath_from_manifest(manifest_path, destination)
        output = grouped_output(source, entries, patched_tree, relpath_from_manifest(manifest_path, archive))
        write_json(manifest_path, output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"hipfire: repacked {len(source['patches'])} patches into {len(entries)} groups")
    for entry in entries:
        print(f"  {entry['id']:24} {entry['sha256'][:12]} {entry['after_tree'][:12]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RepackError as error:
        raise SystemExit(f"error: {error}")
