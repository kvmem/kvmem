# hipfire upstream merge drill

The hipfire backend is an external checkout plus a grouped patch series, not a
fork that can be fast-forwarded. Upstream moves independently; a conflict found
while preparing a feature branch costs much more than a scheduled probe. The
merge drill answers one question without touching a developer checkout:

> If the current pinned series were rebased onto upstream today, which files
> conflict, and does the host build still compile?

It does not resolve conflicts, rewrite patches or update the pin.

## Source and series

Read the current values from the metadata; do not copy them into automation:

- canonical upstream repository: `https://github.com/warpfront/hipfire.git`;
- pinned base commit, group order and final tree: `backends/hipfire/upstream.json`;
- backend index entry: `backends/versions.json`;
- grouped patch files: `backends/patches/hipfire-group-*.patch` beside
  `upstream.json`.

The three groups are applied in manifest order. Group 1 is the standalone Escha
baseline and group 2 is the independently applicable tool/Codex/Bonsai fork
base. Group 3 is the contiguous KVMem/feature stack: it contains the KVMem
integration, fork optimizations, full KV takeover (MTP head paging, DFlash/DN
parking, ownership/metrics) and Swift `-xt` multi-slot Lloyd admission. Those
changes rewrite each other's lines, so they are kept as one tree diff. Apply all
of them; only the first two groups are independent of KVMem. The manifest's
`repository` is the fork used by `prepare-backends.py`; the drill merges the
canonical upstream by default. The ordered 65-change archive is
`upstream-75.json`; older archives are `upstream-74.json`, `upstream-73.json`,
`upstream-72.json`, `upstream-71.json`, `upstream-70.json`, `upstream-69.json`,
`upstream-68.json`, `upstream-67.json` and `upstream-66.json` beside it. Those
archives and the individual files under `backends/patches/` are history, not
alternate heads.

## What the drill does

`scripts/hipfire_merge_drill.py` performs this sequence in a temporary clone:

1. clone upstream (or `--upstream-dir` for an offline/local source);
2. check out the pinned base and verify every group's path, SHA-256, dependency
   order and `after_tree`;
3. apply the groups, then require `git write-tree` to equal the manifest's
   `patched_tree`; a mismatch means the series, not upstream, is broken;
4. commit the reconstructed tree and merge the upstream ref with
   `--no-commit --no-ff`;
5. report conflict files and the number of `<<<<<<<` hunks;
6. when the merge is clean and checks are enabled, run the contract test, the
   kvmem pin check, and `cargo check -p hipfire-arch-qwen35` both by default
   and with `--features kvmem` (`KVMEM_ROOT` and `KVMEM_REQUIRE_PIN=1` are set
   for the build).

The merge is never committed, no patch is edited, and the temporary clone is
removed unless `--keep` is given. A conflict is a result, not a script failure.

## Running locally

Run from the KVMem checkout:

```sh
python3 scripts/hipfire_merge_drill.py \
  --skip-cargo \
  --json /tmp/hipfire-merge-drill.json \
  --summary /tmp/hipfire-merge-drill.md
```

For an offline or already-fetched upstream copy:

```sh
python3 scripts/hipfire_merge_drill.py \
  --upstream-dir /path/to/hipfire \
  --upstream-ref master \
  --skip-cargo
```

Useful switches:

| Switch | Effect |
|---|---|
| `--strict` | exit 1 on conflicts and failed host checks |
| `--skip-cargo` / `--skip-checks` | skip contract, pin and both cargo checks |
| `--keep` | keep the temporary clone and print its path |
| `--upstream-ref` | branch/tag/commit to merge (default: `master`) |
| `--upstream-repository` | override the repository URL; `--upstream-dir` is a local clone source |
| `--cargo-timeout` | per-check timeout in seconds (default 1800) |
| `--json` / `--summary` | machine-readable report / append Markdown to a job summary |

Statuses are `clean`, `conflicts`, `check-failed`, `patch-failed`,
`tree-mismatch`, `merge-failed` and `error`. By default a completed drill exits
0; `--strict` turns conflicts and host-check failures into exit 1. A failed
patch, tree reconstruction or merge is an operational error and exits 1.

## Reading a conflict report

The job summary lists the conflicting paths and hunk count. The JSON report has
the same data under `conflicts.files` and `conflicts.hunks`, plus
`upstream_commit`, `patch_tree`, `failed_patch` and check tails.

Expected conflicts are hook-shaped: the fields and dispatch points in
`serve_engine.rs`, forward/tap signatures, and config or CLI registration. The
KVMem controller, selector, shim, Rust crate and the split child modules should
not need conflict resolution; if they do, inspect the group boundary rather than
editing generated group patches. Files under `backends/patches/` are generated
diffs and must never be hand-edited.

A wide rebase surface is a warning, not an instruction to auto-resolve. The
scheduled workflow warns above eight conflicting files. Treat that as a signal
to schedule a deliberate rebase; do not merge the temporary drill result.

## Deliberate rebase and regeneration

Use a disposable worktree at the new upstream base:

```sh
git range-diff <old-base>..<old-series-tip> <new-base>..<new-series-tip>
```

`git rerere` may replay previously recorded resolutions in that disposable
worktree, but review every replayed hunk. After resolving only hook files:

```sh
python3 scripts/repack_hipfire_series.py --write --hipfire-dir /path/to/hipfire
python3 scripts/repack_hipfire_series.py --check --hipfire-dir /path/to/hipfire
python3 scripts/prepare-backends.py --backend hipfire --check
```

`--write` regenerates the group patches from the archived individual series and
updates `upstream.json`; `--check` verifies every `after_tree` using an isolated
Git index. If the ABI, shim or host-source manifest changed, regenerate the pin
and the contract expectations as described in `backends/hipfire/README.md`.
Update the base commit, group SHAs and `patched_tree` together; a partial update
fails both the drill and `prepare-backends.py --check`.

## CI

`.github/workflows/hipfire-merge-drill.yml` runs weekly (`0 3 * * 1`, Monday
03:00 UTC) and on manual dispatch. It is not a pull-request check, so a conflict
does not block normal merges. The Markdown report is appended to
`$GITHUB_STEP_SUMMARY`; the JSON report is uploaded as an artifact. The workflow
warns above eight conflict files and fails on operational errors or failed host
checks, so a silent breakage is still visible in Actions.
