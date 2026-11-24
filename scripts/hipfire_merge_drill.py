#!/usr/bin/env python3
"""Run the scheduled upstream merge drill for the hipfire KVMem series.

The drill never touches the developer checkout:

1. clone the canonical upstream into a temporary directory;
2. check out the pinned base and apply the grouped KVMem series;
3. create a local commit and merge the current upstream branch into it;
4. report conflict files/hunks, then (when the merge is clean) run the
   contract/pin checks and the host-only default and kvmem cargo checks.

The result is written as Markdown for a GitHub job summary. A completed drill
exits 0 even when it finds conflicts; ``--strict`` makes conflicts and failed
host checks exit 1. Setup, clone, patch, tree-reconstruction and merge failures
always exit 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = "https://github.com/warpfront/hipfire.git"
DEFAULT_REF = "master"


class DrillError(RuntimeError):
    """A setup or repository operation prevented the drill from running."""


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
        timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except FileNotFoundError as error:
        raise DrillError(f"command not found: {command[0]}") from error
    except subprocess.TimeoutExpired as error:
        raise DrillError(f"command timed out after {timeout}s: {' '.join(command)}") from error
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise DrillError(f"command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(["git", "-C", str(repo), *args], check=check)


def load_series(root: Path) -> tuple[Path, dict, list[dict]]:
    versions_path = root / "backends" / "versions.json"
    try:
        versions = json.loads(versions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DrillError(f"cannot read {versions_path}: {error}") from error
    entry = versions.get("hipfire")
    if not isinstance(entry, dict):
        raise DrillError("backends/versions.json has no hipfire entry")
    manifest_rel = entry.get("manifest", "backends/hipfire/upstream.json")
    manifest_path = (root / manifest_rel).resolve()
    if not manifest_path.is_relative_to(root.resolve()):
        raise DrillError(f"hipfire manifest escapes the repository: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DrillError(f"cannot read {manifest_path}: {error}") from error
    if manifest.get("layout") != entry.get("layout"):
        raise DrillError("versions.json layout differs from the hipfire manifest")
    for key in ("repository", "base_commit", "patched_tree", "patches"):
        if key not in manifest:
            raise DrillError(f"{manifest_path} is missing {key}")
    if not isinstance(manifest["repository"], str) or not manifest["repository"]:
        raise DrillError(f"{manifest_path} has no repository URL")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["base_commit"]):
        raise DrillError(f"{manifest_path} has an invalid base_commit")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["patched_tree"]):
        raise DrillError(f"{manifest_path} has an invalid patched_tree")
    if not isinstance(manifest["patches"], list):
        raise DrillError(f"{manifest_path} patches is not a list")

    patches: list[dict] = []
    seen: set[str] = set()
    paths: set[Path] = set()
    for item in manifest["patches"]:
        if not isinstance(item, dict):
            raise DrillError(f"invalid patch entry: {item!r}")
        identity = item.get("id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise DrillError(f"invalid or duplicate patch id: {identity!r}")
        dependencies = item.get("requires", [])
        if not isinstance(dependencies, list) or any(not isinstance(dep, str) for dep in dependencies):
            raise DrillError(f"{identity}: requires must be a list of patch ids")
        if any(dependency not in seen for dependency in dependencies):
            raise DrillError(f"{identity}: dependency does not precede it")
        path = item.get("path")
        if not isinstance(path, str) or not path:
            raise DrillError(f"{identity}: invalid patch path")
        patch_path = (manifest_path.parent / path).resolve()
        if not patch_path.is_relative_to(root.resolve()) or patch_path in paths:
            raise DrillError(f"{identity}: invalid or duplicate patch path: {patch_path}")
        if not patch_path.is_file():
            raise DrillError(f"{identity}: patch is missing: {patch_path}")
        digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        if digest != item.get("sha256"):
            raise DrillError(f"{identity}: patch hash differs from the manifest")
        expected_tree = item.get("after_tree")
        if not isinstance(expected_tree, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_tree):
            raise DrillError(f"{identity}: invalid after_tree")
        seen.add(identity)
        paths.add(patch_path)
        patches.append({"id": identity, "path": patch_path, "after_tree": expected_tree})

    if not patches:
        raise DrillError("hipfire manifest has no patches")
    if patches[-1]["after_tree"] != manifest["patched_tree"]:
        raise DrillError("hipfire final patch tree differs from patched_tree")
    return manifest_path, manifest, patches


def clone_upstream(url: str, destination: Path) -> None:
    command = ["git", "clone", "--quiet", "--no-tags"]
    if url.startswith(("http://", "https://", "git@")):
        # The full worktree is still needed for cargo, but history blobs are lazy.
        command.append("--filter=blob:none")
    command.extend([url, str(destination)])
    run(command, timeout=1800)


def resolve_ref(repo: Path, ref: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    for candidate in (f"origin/{ref}", ref):
        result = git(repo, "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}", check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    raise DrillError(f"upstream ref not found: {ref}")


def conflict_details(repo: Path, files: list[str]) -> dict:
    hunks = 0
    markers: list[str] = []
    for name in files:
        path = repo / name
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        count = text.count("<<<<<<<")
        if count:
            hunks += count
            markers.append(f"{name} ({count})")
    return {"files": files, "hunks": hunks, "markers": markers}


def host_checks(repo: Path, root: Path, timeout: int) -> list[dict]:
    env = os.environ.copy()
    env["KVMEM_ROOT"] = str(root)
    env["KVMEM_REQUIRE_PIN"] = "1"
    env.pop("KVMEM_EXPECTED_COMMIT", None)
    checks = [
        (
            "contract",
            [sys.executable, str(root / "tests/test_hipfire_contract.py"),
             "--hipfire-dir", str(repo), "--require-rust"],
        ),
        (
            "pin",
            [sys.executable, str(root / "scripts/hipfire_pin.py"), "--check"],
        ),
    ]
    results = []
    for label, command in checks:
        try:
            result = run(command, cwd=root, env=env, timeout=timeout, check=False)
            ok = result.returncode == 0
            output = (result.stdout + "\n" + result.stderr).strip()
        except DrillError as error:
            ok = False
            output = str(error)
        results.append({
            "label": label,
            "command": " ".join(command),
            "ok": ok,
            "tail": "\n".join(output.splitlines()[-12:]),
        })
    return results


def cargo_check(repo: Path, root: Path, label: str, args: list[str], timeout: int) -> dict:
    env = os.environ.copy()
    env["KVMEM_ROOT"] = str(root)
    env["KVMEM_REQUIRE_PIN"] = "1"
    env.pop("KVMEM_EXPECTED_COMMIT", None)
    env["CARGO_INCREMENTAL"] = "0"
    command = ["cargo", *args]
    try:
        result = run(command, cwd=repo, env=env, timeout=timeout, check=False)
        ok = result.returncode == 0
        output = (result.stdout + "\n" + result.stderr).strip()
    except DrillError as error:
        ok = False
        output = str(error)
    tail = "\n".join(output.splitlines()[-12:])
    return {"label": label, "command": " ".join(command), "ok": ok, "tail": tail}


def write_summary(report: dict) -> str:
    lines = [
        "# hipfire upstream merge drill",
        "",
        f"- generated: {report['generated']}",
        f"- upstream: `{report['upstream']}` @ `{report['ref']}` (`{report['upstream_commit'] or '?'}`)",
        f"- pinned base: `{report['base_commit']}`",
        f"- patch groups: {report['groups_applied']}/{report['groups_total']}",
        f"- final patch tree: `{report['patch_tree'] or '?'}`",
        f"- status: **{report['status']}**",
        "",
    ]
    if report.get("error"):
        lines.extend(["## Error", "", "```", report["error"], "```", ""])
    if report.get("failed_patch"):
        lines.extend([f"Failed patch: `{report['failed_patch']}`", ""])
    conflicts = report.get("conflicts", {})
    if conflicts or report["status"] == "conflicts":
        files = conflicts.get("files", [])
        lines.extend([
            "## Merge conflicts",
            "",
            f"- files: {len(files)}",
            f"- hunks: {conflicts.get('hunks', 0)}",
        ])
        if files:
            lines.append("- paths:")
            lines.extend(f"  - `{name}`" for name in files[:30])
            if len(files) > 30:
                lines.append(f"  - ... {len(files) - 30} more")
        lines.append("")
    checks = report.get("checks", [])
    if checks:
        lines.extend(["## Host checks", "", "| check | result |", "|---|---|"])
        for check in checks:
            lines.append(f"| {check['label']} | {'pass' if check['ok'] else 'FAIL'} |")
        for check in checks:
            if not check["ok"]:
                lines.extend(["", f"### {check['label']} ({check['command']})", "",
                              "```", check["tail"], "```"])
        lines.append("")
    if report.get("temp_dir"):
        lines.extend([f"Kept drill checkout: `{report['temp_dir']}`", ""])
    return "\n".join(lines)


def drill(args: argparse.Namespace) -> tuple[dict, int]:
    root = args.kvmem_root.resolve()
    manifest_path, manifest, patches = load_series(root)
    upstream = str(args.upstream_dir) if args.upstream_dir else (
        args.upstream_repository or DEFAULT_UPSTREAM)
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(manifest_path),
        "manifest_repository": manifest["repository"],
        "upstream": upstream,
        "ref": args.upstream_ref,
        "upstream_commit": None,
        "base_commit": manifest["base_commit"],
        "groups_total": len(patches),
        "groups_applied": 0,
        "patch_tree": None,
        "status": "error",
        "error": None,
        "failed_patch": None,
        "conflicts": {},
        "checks": [],
        "temp_dir": None,
    }
    temp = Path(tempfile.mkdtemp(prefix="hipfire-merge-drill-"))
    repo = temp / "hipfire"
    try:
        clone_upstream(upstream, repo)
        upstream_commit = resolve_ref(repo, args.upstream_ref)
        report["upstream_commit"] = upstream_commit
        git(repo, "config", "user.email", "merge-drill@invalid")
        git(repo, "config", "user.name", "KVMem merge drill")
        git(repo, "checkout", "--quiet", "--detach", manifest["base_commit"])
        for item in patches:
            result = git(repo, "apply", "--check", str(item["path"]), check=False)
            if result.returncode != 0:
                report["failed_patch"] = item["id"]
                report["status"] = "patch-failed"
                report["error"] = (result.stderr or result.stdout).strip()
                return report, 1
            git(repo, "apply", "--whitespace=nowarn", str(item["path"]))
            report["groups_applied"] += 1
        git(repo, "add", "-A")
        tree = git(repo, "write-tree").stdout.strip()
        report["patch_tree"] = tree
        if tree != manifest["patched_tree"]:
            report["status"] = "tree-mismatch"
            report["error"] = (
                f"applied series produces {tree}, manifest expects {manifest['patched_tree']}"
            )
            return report, 1
        git(repo, "commit", "--quiet", "-m", "kvmem: pinned patch series")
        merge = git(repo, "merge", "--no-commit", "--no-ff", upstream_commit, check=False)
        conflict_names = [
            line for line in git(
                repo, "diff", "--name-only", "--diff-filter=U", check=False
            ).stdout.splitlines() if line
        ]
        if merge.returncode != 0 or conflict_names:
            report["conflicts"] = conflict_details(repo, conflict_names)
            report["status"] = "conflicts" if conflict_names else "merge-failed"
            if not conflict_names:
                report["error"] = (merge.stderr or merge.stdout).strip()
        else:
            report["status"] = "clean"
            if not args.skip_cargo:
                report["checks"] = host_checks(repo, root, args.cargo_timeout)
                report["checks"].extend([
                    cargo_check(repo, root, "cargo check (default)",
                                ["check", "-p", "hipfire-arch-qwen35"], args.cargo_timeout),
                    cargo_check(repo, root, "cargo check (kvmem)",
                                ["check", "-p", "hipfire-arch-qwen35", "--features", "kvmem"],
                                args.cargo_timeout),
                ])
                if any(not check["ok"] for check in report["checks"]):
                    report["status"] = "check-failed"
        if not args.keep:
            shutil.rmtree(temp, ignore_errors=True)
        else:
            report["temp_dir"] = str(temp)
        if args.strict and report["status"] != "clean":
            return report, 1
        return report, 0
    except DrillError as error:
        report["error"] = str(error)
        report["status"] = "error"
        if args.keep:
            report["temp_dir"] = str(temp)
        else:
            shutil.rmtree(temp, ignore_errors=True)
        return report, 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kvmem-root", type=Path, default=ROOT)
    parser.add_argument("--upstream-dir", type=Path, default=None,
                        help="clone this repository/path instead of the canonical upstream")
    parser.add_argument("--upstream-repository", default=None,
                        help="override the canonical upstream URL")
    parser.add_argument("--upstream-ref", default=os.environ.get("UPSTREAM_REF", DEFAULT_REF))
    parser.add_argument("--summary", type=Path, default=None,
                        help="append the Markdown report to this file (GitHub step summary)")
    parser.add_argument("--json", type=Path, default=None, help="write the machine-readable report")
    parser.add_argument("--skip-cargo", "--skip-checks", dest="skip_cargo",
                        action="store_true",
                        help="skip the host checks (contract, pin and cargo)")
    parser.add_argument("--cargo-timeout", type=int, default=1800)
    parser.add_argument("--keep", action="store_true", help="keep the temporary drill checkout")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 on conflicts and failed host checks")
    args = parser.parse_args(argv)
    report, code = drill(args)
    summary = write_summary(report)
    print(summary)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(summary + "\n")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DrillError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
