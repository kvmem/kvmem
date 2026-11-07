# Paths and device identities anonymized; measurements unchanged.
# PolyStrata immutable history snapshots — 2026-11-07

This backend-only change uses the existing `SealedKvBlock`, `KvSnapshot`, `HostKvStorage`, `HostKvRecord`, and `StorageIoQueue`. It does not change the public snapshot/storage/resource-policy interfaces or contract tests. Integration initially waited for the coordinating session. After NInfer PR #272 merged, the user explicitly authorized auditing and merging this PR.

## Audit and implementation

The old PolyStrata archive already captured dirty blocks, sealed native records, shared those records on `fork`, and used common RAM/SSD storage. `KvmemHistory::fork` shared native state through a const checkpoint root while copying mutable history/selector metadata. `park` detached the host archive, versions, mean-key statistics and selector; it did not preserve GPU allocations. `enter` restored native state and the selected execution view.

The missing piece was a checkpoint-owned immutable KV manifest. Checkpoints formerly relied on the later active/parked archive. Each checkpoint now retains its exact full history record origins and an immutable shared version table. The selected view remains native `resident` metadata. Complete unchanged blocks preserve their strong origin identity across different frontiers and branches. A dirty/tail replacement gets a new physical record and origin. Identity comparison is safe because the old origin holds its allocation alive; there is no bare-pointer global index or token-hash equivalence assumption.

Restore reinstates the checkpoint's exact records and versions, clears both native resident bindings and selector GPU-slot metadata, then reloads the accepted view. GPU allocation, scheduling, KV/scale formats and native model-state save/load remain PolyStrata-owned. Before destructive restore and publication, the transfer checks its original session stamp. Existing reset/rebind transaction exclusion, leased sources and cancellation draining remain in force.

All native state images now reserve common host capacity (previously this was Gemma-only). Capturing state temporarily reserves the existing native exchange margin as well; otherwise several retained state images could admit a checkpoint but prevent the next required KV exchange. The existing optional checkpoint/entry reclamation path handles refusal. A fork may reclaim its state after becoming the sole owner of its immutable root; a published or in-flight parent prevents reclamation. Snapshot manifests, immutable version tables and origin objects reserve metadata once; origin metadata releases on last strong ownership even if a weak identity reference survives.

This is not a second generic cache policy or an expansion of the public API. No necessary public-interface extension was identified for this delivery.

## Validation

`poly_kvmem_snapshot_test MODEL.gguf [--q8]` is an explicit, `EXCLUDE_FROM_ALL` native test target. Both F16 and Q8 passed on the final source:

- Complete-block append retains the same origins; growing tails get independent origins.
- Published source bytes remain identical; old-parent and sparse-view native KV are compared byte for byte.
- SSD-cold forks, branch edits, old-parent return, native-state verification, oversized quota refusal, submitted-transfer cancellation and injected restore failure/retry pass.
- Invalidating a session with a submitted transfer rejects its stale completion; the retained source can subsequently restore.
- After all strong references are released, host and SSD account usage are zero, including while an expired weak origin handle remains.

The existing history planner and retained-session/shared-quota tests pass, including directory lease preservation and eviction. The repository backend-preparation tests pass (10 cases). The complete patch is applied to a private index over pinned native tree `7f307fd6b22832a462c25344af02fdf831c6b94a`; the manifest records the resulting tree and SHA-256.

The final real-model SSD request run passed all nine requests: document first, separate question, parent return, follow-up, changed document/question, old branch return, anonymous prefix reuse and ordinary tool continuation. Four directory evictions occurred. SSD reads were **146,883,264 B**, writes **158,935,040 B**, and reported I/O errors zero. At low quota an evicted branch can require recomputation; successful answers are not presented as proof of a surviving cache hit. Native byte tests provide the sharing proof.

Two execution/capacity defects introduced during this adaptation were reproduced and fixed before final validation: stale selector GPU slots after restoring an immutable manifest, and failure to preserve the required RAM exchange margin when admitting non-spillable model state.

## Same-device comparison

Windows/MSVC 19.44, CUDA 13.2, sm_120a, `/MD` Release; RTX 5050 Laptop, device `GPU-A`. Only this GPU was used, serially; unrelated services were not stopped. Model: Qwen3.6-35B-A3B-UD-Q4_K_XL, recorded SHA-256 `55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1`.

Both arms use context 16384, B=2048, R=512, prefill 128, sink/recent 128, host 512 MiB, Q8 KV, three MTP drafts, one request at a time and identical request JSON. The document is an ordinary first turn; the real question is separate. Native prompt rows are **3641 / 40 / 67** in both arms. Baseline is the existing implementation with only the same counters added; [the instrumentation patch](polystrata-baseline-metrics-20261107.patch) is a benchmark fixture, not a production patch layer.

| Measurement | Instrumented baseline | Candidate |
| --- | ---: | ---: |
| Document prefill | 21.1860 s | 16.1650 s |
| Separate question prefill | 1.2944 s | 0.8528 s |
| Fixed generation prompt prefill | 0.9747 s | 0.6609 s |
| 256-token decode | 27.7 tok/s | 32.0 tok/s |
| Dirty archive D2H | 40,020,992 B | 40,020,992 B |
| Execution-view D2H | 135,190,528 B | 86,361,088 B |
| Execution-view H2D | 122,169,344 B | 122,169,344 B |
| Serialized state images captured | 395,231,232 B | 395,231,232 B |
| Observed SSD read/write | 0 / 0 B | 0 / 0 B |
| Sampled GPU peak | 7140 MiB | 7140 MiB |
| Sampled process-tree RSS peak | 13846.30 MiB | 16165.68 MiB |

These are single trials with materially different Windows memory pressure (minimum available RAM 5.61 versus 333.65 MiB). MTP acceptance also varies. Timing and RSS are observations, **not a stable or causal speedup claim**. Native prompt timing includes waits. The supported structural result is less execution-view download, with unchanged archive capture and upload bytes. The baseline already had incremental archive capture; this PR does not claim to have invented that saving.

## Accounting and supported boundary

`KVMEM_NATIVE_IO` reports cumulative process-native KV DMA counters. `state_image_bytes` counts native serialized state images, not all engine/device traffic. Verification-only state round trips are outside that image counter; performance arms disable transfer verification. `KVMEM_SNAPSHOT.logical_view_bytes` sums attempted padded full-history manifest sizes, including optional captures subsequently refused; it is not physical allocation or copied bytes.

Common host/disk account usage is authoritative and counts physical KV allocations once despite aliases. Candidate host usage additionally includes reserved model state and new snapshot metadata. Do not add `physical_state_bytes` to candidate `physical_payload_bytes`: the legacy-named latter field already includes those reservations. The baseline final host counter was 38,693,804 B **plus** 263,487,488 B of uncharged state; the candidate final shared account was 302,995,860 B. This counter increase is principally corrected accounting, not that much new state allocation.

State remains a whole native immutable RAM image and is non-spillable. Restore staging remains the full selected execution view; the measured reusable pinned buffers total 61,276,160 B. Native selector/statistics/history bookkeeping, ordinary fork-wrapper metadata, allocator/control-block overhead and model/runtime workspace remain outside the new snapshot reservation and are visible in RSS. Existing native bounds still govern them; this change does not promise a process-wide RSS cap, independently budgeted metadata, or bounded streaming state restore.

Validated: one CUDA GPU, Qwen3.6 text, F16/Q8 KV byte restoration and MTP request flows. No new multimodal, Gemma, mixed-KV, HIP, multi-GPU/TP, fairness or restart-persistence claims. Data is scoped to the existing engine/cache directory; cross-engine transport is not introduced.

Build scope was audited before each compilation. Initial native class-layout changes required eleven host units; unchanged CUDA kernels and compatible core libraries were reused after source/header/toolchain checks. Later implementation-only changes compiled one host unit and relinked. Normal parallelism was eight. No existing Ninja metadata was fabricated or deleted. The inherited `LIBCMT` default-library warning was traced to CUDA's `cudadevrt.lib`; audited project objects use `/MD`.

Full settings, executable hashes, responses, counters and memory results are in [the JSON report](polystrata-immutable-snapshots-20261107.json). Local raw fixtures, logs, dry runs and build scripts are retained in `poly-snapshot-lab-20261106`; source patch identity is pinned in `backends/versions.json`.

## Pre-merge integration audit

Synced main through `7f0bc56eee5951db33ae4b7a250360ef4fcada1e` (NInfer #272), including public storage contracts #271 and Radiance #270. The merge preserves each backend's manifest entry and introduces no PolyStrata native patch changes. Reviewed origin ownership and tail replacement, exact restore bindings, stale-completion checks, cancellation draining, shared-state reclamation, and physical accounting; no new blocking defect was identified.

The added `HostKvStorage::owns/demote` API is not called by this adapter; existing eviction calls retain the unrestricted selection behavior. The `RawKvStore` packed-record API is not used for PolyStrata native KV, and its added coverage bookkeeping does not replace PolyStrata's mean-key checkpoint restoration. `SealedKvBlock/KvSnapshot` and the accepted native-state format are unchanged. Fresh integration CI compiles against the new public headers and implementation, rather than relying on the older locally linked core library. The real-model F16/Q8 and request evidence above remains evidence for the original tested core revision; this audit does not relabel it as a new GPU run against #271.
