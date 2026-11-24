#!/usr/bin/env python3
"""Tests for the hipfire group-patch repacker."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def repack_module():
    spec = importlib.util.spec_from_file_location(
        "repack_hipfire_series", ROOT / "scripts/repack_hipfire_series.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(directory: Path, *args: str, check: bool = True, text: bool = True):
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        check=check, capture_output=True, text=text)


class RepackFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kvmem-repack-test-")
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "test@example.invalid")
        git(self.repo, "config", "user.name", "test")
        (self.repo / "a.txt").write_text("one\n", encoding="utf-8")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "base")
        self.base = git(self.repo, "rev-parse", "HEAD").stdout.strip()

        patches = self.root / "patches"
        patches.mkdir()
        (self.repo / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
        git(self.repo, "add", "-A")
        (patches / "p1.patch").write_bytes(
            git(self.repo, "diff", "--binary", "--cached", "HEAD", text=False).stdout)
        git(self.repo, "commit", "-qm", "p1")
        self.tree1 = git(self.repo, "rev-parse", "HEAD^{tree}").stdout.strip()

        (self.repo / "b.txt").write_text("bee\n", encoding="utf-8")
        git(self.repo, "add", "-A")
        (patches / "p2.patch").write_bytes(
            git(self.repo, "diff", "--binary", "--cached", "HEAD", text=False).stdout)
        git(self.repo, "commit", "-qm", "p2")
        self.tree2 = git(self.repo, "rev-parse", "HEAD^{tree}").stdout.strip()
        git(self.repo, "reset", "--hard", self.base)

        self.manifest = {
            "base_commit": self.base,
            "patched_tree": self.tree2,
            "patches": [
                {"id": "p1", "path": "../patches/p1.patch",
                 "sha256": hashlib.sha256((patches / "p1.patch").read_bytes()).hexdigest(),
                 "requires": [], "after_tree": self.tree1},
                {"id": "p2", "path": "../patches/p2.patch",
                 "sha256": hashlib.sha256((patches / "p2.patch").read_bytes()).hexdigest(),
                 "requires": ["p1"], "after_tree": self.tree2},
            ],
        }
        self.manifest_path = self.root / "manifest" / "upstream.json"
        self.manifest_path.parent.mkdir()
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.groups = (
            {"id": "g1", "file": "g1.patch", "patches": ("p1",)},
            {"id": "g2", "file": "g2.patch", "patches": ("p2",)},
        )
        self.out = self.root / "out"
        self.out.mkdir()

    def tearDown(self):
        self.temp.cleanup()


class GroupValidationTest(RepackFixture):
    def test_each_patch_appears_once_in_order(self):
        module = repack_module()
        module.validate_groups(self.manifest["patches"], self.groups)
        with self.assertRaises(module.RepackError):
            module.validate_groups(self.manifest["patches"][:1], self.groups)
        swapped = tuple(reversed(self.groups))
        with self.assertRaises(module.RepackError):
            module.validate_groups(self.manifest["patches"], swapped)


class GenerateTest(RepackFixture):
    def test_group_patches_reconstruct_the_final_tree(self):
        module = repack_module()
        entries, final = module.generate(
            self.repo, self.manifest_path, self.manifest, self.groups, self.out)
        self.assertEqual(final, self.tree2)
        self.assertEqual([entry["id"] for entry in entries], ["g1", "g2"])
        self.assertEqual(entries[1]["requires"], ["g1"])
        self.assertTrue(all((self.out / Path(entry["path"]).name).is_file()
                            for entry in entries))

        verify = self.root / "verify"
        git(self.repo, "worktree", "add", "--detach", "--force", str(verify), self.base)
        for entry in entries:
            patch = self.out / Path(entry["path"]).name
            git(verify, "apply", "--check", str(patch))
            git(verify, "apply", str(patch))
        self.assertEqual(module.source_tree(verify), self.tree2)

    def test_verify_series_checks_each_boundary(self):
        module = repack_module()
        entries, final = module.generate(
            self.repo, self.manifest_path, self.manifest, self.groups, self.out)
        manifest = {
            "base_commit": self.base,
            "patched_tree": final,
            "patches": [dict(entry, path="../out/" + Path(entry["path"]).name)
                        for entry in entries],
        }
        self.assertEqual(module.verify_series(self.repo, self.manifest_path, manifest), final)
        broken = json.loads(json.dumps(manifest))
        broken["patches"][0]["after_tree"] = "0" * 40
        with self.assertRaises(module.RepackError):
            module.verify_series(self.repo, self.manifest_path, broken)


class RealManifestTest(unittest.TestCase):
    def test_grouped_manifest_and_archive(self):
        module = repack_module()
        manifest_path = ROOT / "backends/hipfire/upstream.json"
        manifest = module.load_manifest(manifest_path)
        self.assertTrue(module.is_grouped(manifest))
        self.assertEqual(len(manifest["patches"]), len(module.GROUPS))
        self.assertEqual(manifest["patched_tree"], manifest["patches"][-1]["after_tree"])
        archive = manifest_path.parent / manifest["previous_series"]
        self.assertTrue(archive.is_file())
        individual = module.load_manifest(archive)
        self.assertEqual(len(individual["patches"]), 73)
        self.assertEqual(individual["patched_tree"], manifest["patched_tree"])
        self.assertEqual(
            [item["id"] for item in manifest["patches"]],
            ["escha-baseline", "fork-base", "kvmem-stack"],
        )
        by_group = {item["id"]: item for item in manifest["patches"]}
        baseline = (manifest_path.parent / by_group["escha-baseline"]["path"]).read_bytes()
        for marker in (b"hipfire-kvmem", b"KvMemTaps", b"kvmem_ctl", b"PagedAttentionKv"):
            self.assertNotIn(marker, baseline)
        stack = (manifest_path.parent / by_group["kvmem-stack"]["path"]).stat().st_size
        self.assertLess(stack, 4_000_000)
        for item in manifest["patches"]:
            patch = (manifest_path.parent / item["path"]).resolve()
            self.assertTrue(patch.is_file(), patch)
            self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), item["sha256"])


if __name__ == "__main__":
    unittest.main()
