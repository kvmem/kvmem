# Multi-backend repository

The `master` branch contains the 0.18.0 multi-backend KVMem framework.
The `v0.18.0` tag preserves the release source snapshot. Both engines are
Git submodules referencing existing upstream projects. The parent commit pins the
core, adapters, engine revisions and integration patches together.

| Path | Ownership |
|---|---|
| `kvmem/` | Portable memory contracts, native KV payloads, retrieval and Host storage |
| `src/adapter/`, `tools/` | llama.cpp product adapter and serving |
| `backends/llamacpp/` | Official ggml-org/llama.cpp submodule at a fixed commit |
| `backends/ninfer/` | iamwavecut/ninfer-all submodule at a fixed commit |
| `backends/patches/` | KVMem changes relative to each fixed engine revision |
| `scripts/windows/` | Build, launcher and packaging entry points |
| `backends/versions.json` | Baseline revisions, licenses, patch hashes and prepared tree hashes |

Each backend keeps its existing license and third-party notices. Source-only
upstream vocabulary and test fixtures are included; inference model weights,
builds, logs, session snapshots and personal DSH configuration are not.

The 0.18.0 source pins ninfer-all to `8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3`.
llama.cpp keeps its existing revision. Qualification progress and limits are in
[the migration record](ninfer-all-migration-validation.md); the earlier dated
qualification reports describe the Ryan baseline. New upstream releases are adopted
deliberately, with an updated integration patch and prepared-tree hash.

The 2026-10-17 development refresh pins `08fd7019925f43b4b42539400afd461b70f7144a`.
Its qualification targets Qwen3.8-27B Dense and a native-versus-KVMem IQ3_S throughput
comparison at equal Main KV capacity (native context = KVMem B+G). It does not qualify
35B MoE, Flash-Next or Gemma 4 for KVMem. Current evidence is recorded in the same
migration record; the earlier release results apply to their dated baseline.

The 2026-10-23 upstream/KV refresh pins `db8c114802a070a7903f84d01142fda0b00fa7df`.
It includes the INT8/rk8v4 and NVFP4 page-feedback specializations while preserving
master's independent Query controls and native tool constraints. The separate
`13thblade/ninfer-all-nvfp4` kernel/route/codec import is deferred and is not included.
The [migration record](ninfer-all-migration-validation.md) separates historical runtime
evidence from the static-only checks of this split and its integration with current master.

The 2026-10-25 candidate pins `63428ac249da8a6c5d891b508d7e483d8c7def6c`
and includes the NVFP4 fork from `c02c7692bd10a9b4a7effef67b60f58673985aa6`
as two author layers plus a local integration layer. The current scope is Qwen
dense, including upstream's restricted GGUF tensor-parallel route. KVMem shared
prefill batching and early first-token publication remain deferred. See the
[current integration record](ninfer-upstream-63428ac2-integration.md) for completed
checks, performance costs and remaining qualification; the records above remain
historical evidence for their pinned revisions.

## Checkout and preparation

```sh
git clone --recurse-submodules \
  https://github.com/kvmem/kvmem.git
cd kvmem
python scripts/prepare-backends.py
python scripts/prepare-backends.py --check
```

Python 3.9+ and Git are required for source preparation. An existing checkout can
run the same script: it initializes missing submodules, verifies the pins and
applies each integration patch once. `--backend llamacpp` or `--backend ninfer`
prepares only the selected engine. `--check` never fetches or changes sources; it
checks the full prepared tree, including patch-added files. The patches reconstruct
the previously tested source trees exactly, including the ninfer generated-endpoint
cache fix. They are not applied by CMake itself.

Prepared submodules normally appear modified in `git status` because the integration
is a local patch over the upstream pin. Preparation preserves developer changes
and fails on conflicts; it does not reset a backend or switch to an upstream branch.
Strict `--check` rejects additional source edits until the patch and manifest are
updated. Core-only builds need neither engine.

GitHub's automatic source ZIP omits submodule contents; use Git for development.
`scripts/rocm/export-source.py` can export a committed, fully prepared source bundle
with both engines and a per-file manifest. Such bundles can be built without Git
metadata; preparation verifies their backend files against that manifest.

## Builds

The engines run as separate workers, so they use separate CMake build directories.
This avoids sharing compiler flags, target names or device allocations between
backends. It does not require separate Git branches.

On Windows, use VS2022 C++ Build Tools, CMake/Ninja and the CUDA toolkit appropriate
for the selected engine. The locally qualified ninfer route uses CUDA 13.2 and
`120a` with native SM120 enabled for the RTX 5060 Ti. Other GPU architectures need
their own qualification. The wrapper accepts `core`, `llamacpp`, `ninfer` or `all`:

```powershell
# Run these commands in a VS2022 x64 developer PowerShell.
# Model-free portable core and host tests.
./scripts/windows/build-backends.ps1 -Backend core
ctest --test-dir build-backends/core --output-on-failure

# llama.cpp worker. Existing scripts/windows/build.ps1 remains available.
./scripts/windows/build-backends.ps1 -Backend llamacpp -CudaArchitectures 120a

# ninfer worker. Set VCPKG_ROOT to your own vcpkg checkout first.
./scripts/windows/build-backends.ps1 -Backend ninfer -CudaArchitectures 120a `
  -CudaPath 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2' `
  -CMakeArgs @("-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake")
```

The build wrapper prepares its selected backend automatically. `-Python` selects
the Python executable. Compilation defaults to eight parallel jobs; `-Jobs` overrides it.
`-DryRun` prints the resolved argument arrays without fetching
or applying patches; `-ConfigureOnly` and `-BuildOnly`
separate configure and compilation. Backend-specific CMake arguments should be
passed with the matching `-Backend`, not indiscriminately to `all`.

The wrapper configures dependencies from the checked-in ninfer vcpkg manifest.
An existing compatible dependency installation can be passed with
`-DVCPKG_INSTALLED_DIR=... -DVCPKG_MANIFEST_INSTALL=OFF`; no local prebuilt ninfer
operator libraries are needed. The backend CMake default resolves the portable
core at `../../kvmem` relative to `backends/ninfer`.

Keep shared dependency installations outside dated experiment or worktree folders.
Pin a versioned bundle and verify its headers, libraries and runtime DLLs before
building; use the same bundle when running the worker. When a cached dependency
root has disappeared or changed, reconfigure dependency discovery before resuming
compilation. Reusable CUDA objects need not be deleted. Never overwrite a bundle
still used by another build or remove it during ordinary experiment cleanup.

The equivalent portable-core commands on Linux are:

```sh
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release \
  -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF
cmake --build build-core --parallel 2
ctest --test-dir build-core --output-on-failure
```

For direct CMake use, prepare the selected backend first. For ninfer, configure
`-S backends/ninfer` in a separate directory. Its own
`docs/` and `vcpkg.json` describe the native build dependencies. Existing
llama.cpp CUDA/HIP scripts and the UI builder prepare the referenced sources before
building. They do not change the pinned upstream revisions.

The resulting workers are `build-backends/llamacpp/bin/llama-kvmem-server.exe` and
`build-backends/ninfer/apps/ninfer-serve.exe`. Pass their paths to
`scripts/windows/start-backend.ps1` or `package-backends.ps1`; the latter's
`-NinferSource` is `backends/ninfer`. No model is included in the package.

## Development and release

### README maintenance

Keep `README.md` and `README.zh-CN.md` as a concise, version-independent project
introduction, quick start and documentation index. Keep the two languages in sync.
Use `/releases/latest` for the stable release and `/releases` for previews and history.

Routine version releases and package refreshes do not require README edits. Put
version numbers, per-GPU downloads, package sizes, checksums, driver requirements,
release-specific defaults and qualification results in GitHub Releases, package
notes and the relevant documentation. Update the homepage only when the project
scope, basic startup steps or permanent navigation changes, or when correcting an
error. Do not replace its stable links with direct versioned asset URLs.

Backend changes are an ordered `patches` list in `backends/versions.json`. Each
entry records its `id`, repository-relative `path`, SHA-256, preceding `requires`,
and exact `after_tree`. The backend's `patched_tree` is the final entry's tree.
The existing `<backend>-kvmem.patch` stays first. Bonsai format support and its
optional performance changes are separate layers; see [Bonsai integration](bonsai-integration.md).

Export each binary/full-index Git diff against the preceding prepared tree (the
first patch uses `base_commit`), including newly added files. Keep patch bytes
unchanged by text redirection. Use an isolated Git index to compute tree hashes;
no backend commit or change to the developer's index is necessary. Preparation
preflights the entire series before editing sources and can resume an exact known
intermediate tree. Unknown source edits are preserved and rejected, never reset.
Validate `prepare-backends.py --check` and replay from the pinned base before publishing.

For upstream updates, deliberately update the submodule gitlink and `base_commit`,
rebase the required patch series onto that revision, update hashes and run affected
checks. Do not follow an unpinned branch. Root `patches/` remains historical material;
the active patches live under `backends/patches/`.

The current branch is a development candidate. P6-P10 and the later source
admissions are in this tree: nine KV types, text and resident/CPU vision at up
to eight requests, ngram, fast prefill on the admitted formats, and the ninfer
layer pipeline. Active KV tiering replaces the disabled ninfer disk entry with
process-local storage shared by active and idle histories; it does not support
restart persistence. See [tiering qualification](active-kv-tiering-plan.md). The release
matrix and a package that matches this source are not finished. Current patch
hashes are in `backends/versions.json`. See [validation status](multi-backend-validation.md)
and [supported backend combinations](multi-backend.md). Once accepted, merge into
`master` through normal Git history; do not force-replace the main branch.
