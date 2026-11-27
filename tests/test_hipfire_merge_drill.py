#!/usr/bin/env python3
"""Tests for the hipfire upstream merge drill."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hipfire_merge_drill.py"


def run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None,
        check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, check=check,
    )


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return run(["git", "-C", str(repo), *args])


def commit(repo: Path, message: str) -> None:
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=test", "-c", "user.email=test@example.invalid",
        "commit", "-q", "-m", message)


def tree_with_patch(repo: Path, base: str, patch: Path) -> str:
    handle = tempfile.NamedTemporaryFile(prefix="drill-index-", delete=False)
    index = handle.name
    handle.close()
    os.unlink(index)
    env = dict(os.environ, GIT_INDEX_FILE=index)
    try:
        run(["git", "-C", str(repo), "read-tree", base], env=env)
        run(["git", "-C", str(repo), "apply", "--cached", "--whitespace=nowarn",
             str(patch)], env=env)
        return run(["git", "-C", str(repo), "write-tree"], env=env).stdout.strip()
    finally:
        if Path(index).exists():
            os.unlink(index)


class DrillFixture:
    def __init__(self, root: Path, upstream: Path):
        self.root = root
        self.upstream = upstream
        upstream.mkdir()
        git(upstream, "init", "-q", "-b", "master")
        (upstream / "shared.txt").write_text("base\n", encoding="utf-8")
        commit(upstream, "base")
        self.base = git(upstream, "rev-parse", "HEAD").stdout.strip()
        self.patch = root / "backends" / "patches" / "test.patch"
        self.patch.parent.mkdir(parents=True)
        (upstream / "shared.txt").write_text("kvmem\n", encoding="utf-8")
        diff = git(upstream, "diff", "--no-color", "--", "shared.txt").stdout
        self.patch.write_text(diff, encoding="utf-8")
        git(upstream, "checkout", "--", "shared.txt")
        self.patched_tree = tree_with_patch(upstream, self.base, self.patch)
        self.manifest = root / "backends" / "hipfire" / "upstream.json"
        self.manifest.parent.mkdir(parents=True)
        (root / "backends" / "versions.json").write_text(json.dumps({
            "hipfire": {
                "layout": "external-checkout-with-patches",
                "manifest": "backends/hipfire/upstream.json",
            },
        }), encoding="utf-8")
        self.manifest.write_text(json.dumps({
            "repository": str(upstream),
            "base_commit": self.base,
            "patched_tree": self.patched_tree,
            "layout": "external-checkout-with-patches",
            "patches": [{
                "id": "test-group",
                "path": "../patches/test.patch",
                "sha256": hashlib.sha256(self.patch.read_bytes()).hexdigest(),
                "requires": [],
                "after_tree": self.patched_tree,
            }],
        }), encoding="utf-8")

    def upstream_add(self, name: str, text: str) -> None:
        (self.upstream / name).write_text(text, encoding="utf-8")
        commit(self.upstream, f"add {name}")

    def upstream_change_shared(self, text: str) -> None:
        (self.upstream / "shared.txt").write_text(text, encoding="utf-8")
        commit(self.upstream, "change shared")


class MergeDrillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hipfire-drill-test-")
        base = Path(self.temp.name)
        self.root = base / "kvmem"
        self.root.mkdir()
        self.upstream = base / "upstream.git-root"
        self.fixture = DrillFixture(self.root, self.upstream)
        self.report = base / "report.json"
        self.summary = base / "summary.md"

    def tearDown(self):
        self.temp.cleanup()

    def run_drill(self, *extra: str) -> subprocess.CompletedProcess:
        return run([
            sys.executable, str(SCRIPT),
            "--kvmem-root", str(self.root),
            "--upstream-dir", str(self.upstream),
            "--skip-cargo",
            "--json", str(self.report),
            "--summary", str(self.summary),
            *extra,
        ], check=False)

    def report_data(self) -> dict:
        return json.loads(self.report.read_text(encoding="utf-8"))

    def test_default_upstream_is_canonical(self):
        spec = importlib.util.spec_from_file_location("hipfire_merge_drill", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            module.DEFAULT_UPSTREAM, "https://github.com/warpfront/hipfire.git")

    def test_clean_merge_reports_no_conflicts(self):
        self.fixture.upstream_add("upstream.txt", "upstream\n")
        result = self.run_drill()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.report_data()
        self.assertEqual(report["status"], "clean")
        self.assertEqual(report["groups_applied"], 1)
        self.assertEqual(report["conflicts"].get("files", []), [])
        self.assertTrue(report["upstream_commit"])
        self.assertIn("status: **clean**", self.summary.read_text(encoding="utf-8"))

    def test_conflicting_upstream_commit_is_counted(self):
        self.fixture.upstream_change_shared("upstream\n")
        result = self.run_drill()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.report_data()
        self.assertEqual(report["status"], "conflicts")
        self.assertEqual(report["conflicts"]["files"], ["shared.txt"])
        self.assertEqual(report["conflicts"]["hunks"], 1)
        self.assertEqual(report["checks"], [])

    def test_strict_conflict_exits_one(self):
        self.fixture.upstream_change_shared("upstream\n")
        result = self.run_drill("--strict")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.report_data()["status"], "conflicts")

    def test_failed_patch_is_reported(self):
        self.fixture.patch.write_text(
            "diff --git a/shared.txt b/shared.txt\n"
            "--- a/shared.txt\n"
            "+++ b/shared.txt\n"
            "@@ -1 +1 @@\n"
            "-not-the-base\n"
            "+kvmem\n",
            encoding="utf-8",
        )
        manifest = json.loads(self.fixture.manifest.read_text(encoding="utf-8"))
        manifest["patches"][0]["sha256"] = hashlib.sha256(
            self.fixture.patch.read_bytes()).hexdigest()
        self.fixture.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        result = self.run_drill()
        self.assertEqual(result.returncode, 1, result.stderr)
        report = self.report_data()
        self.assertEqual(report["status"], "patch-failed")
        self.assertEqual(report["failed_patch"], "test-group")
        self.assertEqual(report["groups_applied"], 0)


if __name__ == "__main__":
    unittest.main()
