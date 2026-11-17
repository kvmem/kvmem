# Radiance v2 migration and RX7800M qualification

Status: **migration, RX7800M text qualification and final change audit passed for the scope below**.

Performance follow-up: the initial IQR port below was a functional baseline and omitted the RDNA4 GEMM performance paths. The [RDNA3 IQR follow-up](radiance-v2-iqr-performance-20261116.md) records their subsequent migration, final artifact identity and same-machine measurements: document prefill28.99 to328.43 tokens/s, with decode still approximately12 tokens/s. The historical timings below are retained as baseline evidence.

## Standalone gate

Upstream v2.0.0 `ddcb9c69bb6c9ff270f8436bd4c059fba88ac6b7`, ABI39, was built and tested independently before beginning this KVMem migration. Windows core/architecture DLL loading and the gfx1101 kernel plugins passed. The original installed KVMem package remains unchanged.

Hardware: RX7800M, gfx1101, approximately 12 GiB VRAM. Physical identifiers are retained in private validation evidence. Toolchain: Clang24/ROCm Core10.1, MSVC14.44, Windows SDK10.0.26100.0, Release.

Requested model: `StillDeadcode/qwen3.8-27b-iqr-v4`, revision `e5f1b3ef70741d705a64a28b852acf3fcefcead7`, `qwen3.8-27b-iq2r-l-v4.rad`. Downloaded through HF Mirror; 11,990,364,160 bytes; SHA256 `89fb6bfbd602594ed8a79b655bee7e15e7732b7c2c78e34db3d145b2e58706bd`.

Standalone evidence:

- 59 h256 admission checks and 28 real-GPU attention cases, compared with an independent FP64 reference, including paging/ragged lengths, BF16/FP8, guard regions and short physical tables under a larger declared context bound.
- IQR GEMM, embedding, vocabulary, activation quantization, norm, RoPE and gated GEMM tests passed, including device-only BF16 weights and a nondefault stream.
- Seven real-model requests passed: arithmetic, Chinese, multi-chunk prefill, separate document/question recall, longer decode and repeat recovery.
- TP1/C1, max context2048, BF16 KV, prefill256, checkpoint interval256/slots4: the document question reused768 tokens and answered the planted code correctly. Held VRAM was approximately8.80 GiB. The observed16.39 tokens/s longer decode is a single smoke measurement, not a stable performance benchmark.
- Vision, speculation, concurrency and KVMem Query/Pressure were outside this standalone gate.

Controller evidence is under `radiance-rdna3-h256-20261115/STANDALONE-VALIDATION.md` and `m16-validation/standalone/` in the KVMem experiment workspace.

## KVMem migration

The committed KVMem baseline is `0d7b4688f1e08b0334b67c2baf31a9160ed1ac98`. The existing dirty release checkout was preserved. Its eleven committed v1.17 native patches were verified and replayed against the exact upstream ancestor before a three-way migration onto the qualified v2 source.

The new consolidated native patch is pinned in `backends/radiance/upstream.json`. Private ABI1263927335 distinguishes the observer extension from stock ABI39 and the previous ABI35 extension. Previous qualification metadata is retained explicitly under `previous_integration`; it does not certify v2.

Integration changes include an optional observer factory in v2 Fleet, the existing single-model launcher contract carried as an already-parsed Config into Fleet, per-engine plugin-registry access, and merged scheduler/KV observation and retention hooks. Windows64-bit pointer carriers were preserved across the new RDNA4 source units. The JSON dependency used by KVMem monitoring is now explicit because upstream v2 no longer ships it.

The gfx1101 worker advertises only the selected text TP1/C1 kernel scope, BF16/FP8 KV and no speculation or images. Existing general/RDNA4 build capability definitions remain available; their previous GPU qualification is historical until revalidated on v2.

## Validation status

- Consolidated patch applies cleanly to the pristine pinned v2 source.
- Eight focused host interface objects compile: engine, bringup, Fleet, scheduler, batch, KV manager, adapter observer and worker entry.
- Launcher Python tests17/17; source-identity tests5/5; capability-profile tests5/5.
- Selected service, Qwen35 W8A8 architecture, libr3d and libr3diqr plugins compile and load on native Windows. No standalone ABI39 DLL is substituted for a KVMem private-ABI DLL.
- Scheduler regression: 198 cases, 549034 checks passed with GPUs hidden. A test-only archive reuses the compiled core objects so the Windows executable's fake device functions interpose correctly. The production core remains a DLL.
- Native request-monitor, independent attention-feedback and native-memory GPU suites passed. Memory tests cover native page spill/restore, protected prefixes, rollback and stale-frontier rejection, recurrent checkpoints, RAM/SSD and BF16/FP8/packed-byte fixtures. Byte fixtures do not qualify full-model packed KV or speculation.
- Final native parser: nine cases passed, covering ordinary/default names, paths with spaces, repeated options, bounded-policy options, and rejection of invalid routing names, unsupported speculation, zero budget and unknown options. The final binary's capabilities restrict the gfx1101 build to TP1/C1, BF16/FP8 KV, F32 state, no images or speculative decode.
- After the final capability-header rebuild, the198-case scheduler suite passed again. A separate three-request smoke run used the actual `rx7800m-iq2r` launch profile, including model preflight, runtime validation and child environment, and passed arithmetic,110-token generation and repeat recovery.

### KVMem whole-model result

Configuration: TP1/C1, logical context8192, resident selection budget1024, block128, generation reserve512, Query span limit256, Query replay limit512, sink128, recent256, host history2048 MiB, prefill256, BF16 KV, F32 state. Speculation and vision are off. The native pool reserves2304 resident tokens including execution/checkpoint allowances; the selected-history budget remains1024. Static weights occupy8.44 GiB. No KV precision reduction was necessary.

| Request | Prompt tokens | Actual reused tokens | Response | Wall time |
| --- | ---: | ---: | --- | ---: |
| Arithmetic | 25 | 0 | 42 | 1.390 s |
| Document-only | 2665 | 0 | ACK | 92.454 s |
| Separate question | 2695 | 2666 | 68391 | 3.171 s |
| Longer decode | 34 | 0 | Integers1–30, 110 output tokens | 10.375 s |
| Arithmetic repeat | 25 | 0 | 42 | 1.266 s |

The document was sent first, followed by a separate question in the same conversation. Both requests/responses and the unchanged document are saved. The answer was not supplied in the question. The question's actual native Query span was `[2671,2686)`, proven length15; replay was `[2666,2695)`, with fixed selection and completion before sampling. Earlier resident Pressure reduced visible history to1024 tokens. Monitor HTTP200 confirmed actual cached2666, snapshot restoration, selected1024 and the same BF16/F32/prefill256 configuration.

The longer decode measured12.13 tokens/s in this one run. It is a functional smoke measurement, not a stable benchmark or controlled comparison with standalone. The 2665-token document exceeds the resident budget and exercises bounded recall; it does not certify 256K quality or all8192 tokens of the configured logical limit.

### Findings and fixes

1. The v2 runtime operand-bounds helper now takes the declared buffer element count rather than a tensor object. The observer path was migrated to `buf_numel_[handle]`. The selected engine closure and real-model execution then passed.
2. The v2 Front discovers routes from a prototype Server without model dependencies. Conditional registration hid `/api/monitor/snapshot` and produced404 in the first integrated recall run. Registration is now unconditional; the real handler returns404 without a monitor,503 before a snapshot, and200 when available. The full recall run passed after the fix; the initial failed run is preserved.
3. Capability records from v1.17 were moved to explicit historical fields. The gfx1101 profile advertises only its selected kernel scope; the launcher rejects other model plugins and embedded speculation before runtime loading.
4. Windows workers no longer request an executable import library. The v2 core supplies its symbols from `rad_core.dll`; the worker itself exports none. Keeping `ENABLE_EXPORTS` made Ninja repeatedly relink an unchanged worker because the nonexistent import library was an expected output. Linux retains executable exports/whole-archive behavior.

These are migration fixes; no inherited KVMem behavior was silently changed to match a suspected defect.

The validation build reuses48 proven-compatible objects from the standalone build, using unchanged commands and complete Ninja transitive dependency evidence. It does not alter old timestamps or Ninja records. The selected closure is173 host compiles,24 HIP compiles and21 links; other model plugins and unused kernel families are excluded. See `radiance-rdna3-h256-20261115/KVMEM-V2-BUILD-SCOPE.md` and the worker's build-scope/reuse records.

The monitor fix required one host object and three links. The final capability-header update required two host objects and four links (including the test archive), with zero HIP recompiles. Each plan was inspected before execution; compilation used8 jobs. Host, HIP and JSON dependency bundles were checked before native builds and validation.

## Reproduce and inspect

Controller source: `$HOME/.codex/worktrees/radiance-v2-rdna3/source`. Worker: `wangl@100.124.223.74`, task root `$ARTIFACTS-Lab/radiance-rdna3-h256-20261115`. The selected build is `build-kvmem-v2-gfx1101` under that root. Its `bin/radiance-kvmem-server.exe` and `radiance_home` belong together. The separate `build-v2-gfx1101` remains the standalone gate.

On the worker, launch the tested profile with:

```powershell
$lab = '$ARTIFACTS-Lab'
$task = "$lab/radiance-rdna3-h256-20261115"
& "$lab/build-deps/tools/cpython-3.11.15-windows-x86_64/python.exe" `
  "$task/kvmem-source/backends/radiance/tools/serve.py" --platform windows --profile rx7800m-iq2r `
  --exe "$task/build-kvmem-v2-gfx1101/bin/radiance-kvmem-server.exe" `
  --home "$task/build-kvmem-v2-gfx1101/radiance_home" `
  --rocm "$lab/build-deps/rocm-core-10.1.0-poly-9070xt-v1/installed" `
  --model "$lab/models/radiance/Qwen3.8-27B-IQR-v4/qwen3.8-27b-iq2r-l-v4.rad"
```

Per-task dependency locks are `host-dependencies.json`, `sdk-dependencies.json`, `json-dependencies.json`. The JSON headers come from immutable `nlohmann-json-aaf127c0-v1`, manifest SHA256 `5a548be3ec37673e29c0b5a64ed92b5b6772670acddbb3999f9091877061addc`; pass its `installed/include` directory as `--json-include` for future native builds. Use the same pinned runtime DLL bundle as the compiler.

The initial consolidated patch SHA256 was `2c2b3aeea11a71fa71e2f5c723d0b6613209abd915f183a3335f49f696502aaa`; the current patch identity is in `backends/radiance/upstream.json` and the linked performance follow-up. Controller evidence for this initial qualification is stored under `radiance-rdna3-h256-20261115/m16-validation/kvmem/`: build plans, reuse audits, exact source/artifact hashes, full native test logs, requests, responses and monitor snapshots. `kvmem-recall-results.json` records the passing run. The harness intentionally terminates its own server in `finally`; the recorded server exit code1 is the Windows termination result, while `completed:true` and runner exit0 indicate test success.

Current limits: RX7800M Windows text is qualified for this model/configuration. FP8 has kernel/native-memory coverage, not a full-model run. Vision, embedded MTP/DFlash, full-model packed KV, multiple GPUs, concurrent sequences, other model families, RDNA4 and Linux require separate v2 qualification. Existing v1.17 results remain historical.

Final review covered the actual consolidated patch and adapter changes, ABI and source identity, Fleet observer ownership, scheduler retention/release, runtime operand bounds, Front routing, selected plugin/dependency closure, launcher rejection paths and qualification claims. The findings above were fixed and revalidated. No unresolved issue was found in the selected migration/7800M text scope; the explicit platform/feature exclusions remain qualification limits, not passing claims.

No commit, push, merge, release upload or installed-package replacement has been performed.
