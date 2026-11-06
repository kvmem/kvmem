# llama.cpp immutable snapshot adaptation: baseline audit

Status: backend integration pending public-interface coordination. Source audit,
five host tests and two native baseline models completed; no optimized candidate
or performance comparison is claimed.

Baseline KVMem: f0a3d51f889d0100ae97bc43a46a398937ffb4b5.
Pinned llama.cpp: cd959bae99ff4bed22bef2604f45241c09bbb8c8.
Ordered patches: kvmem, pq2, ptq1, ptq1-performance, checkpoint-policy.
Expected prepared tree: 1a8155219356073fe1f8a3528d2c7f203521554b.
Remote: XIAO / 5070TiM, GPU-940f1324-a09e-39f9-bd7f-ea87f1c12f8e.

## Existing paths

- `tools/llama-kvmem-driver.cpp::multimodal_checkpoint` drains target work,
  captures accepted tail MeanK/policy state, calls native sequence serialization
  with PARTIAL_ONLY, and separately captures speculative carry. Same live
  checkpoint/row can reuse the complete checkpoint object.
- Native `llama-kv-cache.cpp::state_write/state_read` skip non-SWA attention
  under PARTIAL_ONLY. Ordinary dense checkpoints therefore must NOT be described
  as full attention KV exports. The KVMem factory explicitly skips SWA models;
  native SWA serialization must not be confused with supported KVMem coverage. Hybrid memory
  delegates to native attention and recurrent serialization; recurrent state is
  not a collection of interchangeable attention blocks.
- `llama_memory_kvmem::harvest_gpu_v` captures missing accepted packed K/V rows.
  `RawKvStore::fork_prefix` seals and forks existing packed records, copies
  mutable statistics, and truncates only the child. New writes detach shared
  records. `fork_active_conv` drains capture and harvests resident missing rows
  before forking. Legacy CPU arenas are independently copied.
- `fork_conv`, `detach_conv`, `attach_conv`, `restore_attached` preserve separate
  logical row positions, selected blocks and fixed GPU slots. Slots are not
  compact execution positions or immutable identities. `NativeBackend` validates
  per-cell logical/spatial positions and accepted coverage before native transfer.
- `multimodal_restore` first validates tail state, materializes cold state, then
  restores native state, trims unaccepted rows, restores speculative carry and
  checkpoint tail. A failed native write can invalidate the destination; a
  retained source must remain available independently.
- `llama_driver_materialize_checkpoint` creates a private copy when the cold
  source is marked shared. `checkpoint_tier` uses StorageIoQueue and the same
  SpillFile as KV, but the recurrent/carry vectors have a separate checkpoint
  host account. This is not yet unified HostKvStorage ownership.

## Public capability request (pending coordination)

User identified `KVMEM缓存修复NInfer - M16 3080ti` as coordinator on
2026-11-07. The four-part request below was delivered to that chat
(`01a1b156-37cc-7a83-b3aa-f94c18a6a948`); integration awaits its coordinated API.
The coordinator acknowledged the request and will publish design/PR/commit in
`C:/Users/leyew/AppData/Local/KVMem/gpu-lab/shared-snapshot-coordination.md`.
Its initial direction retains full-record KvSnapshot coverage, sealed full-block
references and private tails allocated through existing HostKvStorage.

1. Call site: construct an attention KvSnapshot from an already captured
   RawKvStore without exporting GPU bytes again or copying archive payload.
   Current public RawKvStore exposes copying reads and whole-store forks, but
   not sealed packed record references and exact valid coverage. Its private
   LayerBlk owns HostKvRecord K/V planes. Access by block/token hash alone would
   not establish native identity.
   KvSnapshot currently requires every referenced record to be covered fully;
   RawKvStore packed records are capacity-sized. Partial tails therefore need
   explicit valid-range descriptors and a coordinated private-tail capture
   rule; unused record suffix bytes must not become accepted KV implicitly.
2. Suggested minimum: an engine-boundary operation returning sealed K/V record
   forks with block/layer/plane, valid token count, row bytes and record extent.
   It should reject incomplete requested coverage atomically. Prefer a neutral
   packed-record descriptor in RawKvStore; llama-specific layout/positions and
   native validation remain in this adapter. The coordinator chooses final API.
3. Ownership and threading: caller drains native/capture writers first; source
   store lock protects coverage inspection and sealing; result owns immutable
   record references. Do not perform device operations under the store lock.
   Existing HostKvStorage is the sole payload owner/account; acquiring forks
   adds no payload charge. Native metadata and restore staging must be admitted
   through the coordinated resource policy, not a new backend budget. Cancel
   before publication drops references; active I/O retains leases until drained.
   Any failure leaves source bytes/coverage usable. First subsequent source or
   branch write detaches via existing COW.
4. Required common tests: complete block identity reuse; private growing tail;
   rewrite/invalidate/truncate and block-ID reuse; missing layer/plane coverage;
   noncontiguous selections; source deletion with outstanding lease; SSD cold
   reads without whole-record RAM promotion; quota failure and cancellation;
   no duplicate physical charge. Backend tests additionally verify native cell
   positions, old parent restore and recurrent/speculative accepted boundaries.

No public files or contract tests have been changed. Do not implement this
request by a backend-owned copy of RawKvStore or an independent payload pool.

## Confirmed capacity-estimation repair

`llama_memory_kvmem::host_capacity` says dense PARTIAL_ONLY checkpoints contain
the attention working set and includes full `memory_breakdown()` in its estimate.
The pinned native non-SWA serialization skips that KV. The backend estimate now
omits this duplicate dense working-set term; KVMem rejects SWA at construction.
At 64 rows, Qwen3-0.6B's estimate falls from 223,595,072 to 17,877,568 bytes while
its actual partial image remains 8 bytes. Qwen3.5-4B's estimate is unchanged at
437,473,344 bytes and its partial image remains 52,691,548 bytes. Both models pass
all four boundary/branch/native-state cases before and after this repair. This
is an admission-estimate correction, not measured RSS savings or a throughput
claim. Evidence: `capacity-before-evidence` and `capacity-after-evidence`.

## Validation still required

The baseline results below do not qualify an optimized adapter or establish a
performance improvement. Before further compilation inspect
toolchain/dependency compatibility, existing artifacts and exact Ninja target
closure. Use eight build jobs. GPU launches require a fresh occupancy check.
Compare identical model/device/configuration; report capture bytes separately
from logical referenced bytes, native state and restore staging. Dense non-SWA,
SWA, hybrid, MTP, DFlash, other speculative modes, transposed V and multi-GPU are
distinct qualification combinations. Long-context recall uses an ordinary
document turn followed by an ordinary question turn, preserving both requests.

## Completed baseline qualification (2026-11-07, Asia/Shanghai)

- Remote checkout: `D:/KVMem-Lab/llamacpp-shared-snapshots`, branch
  `codex/llamacpp-shared-snapshots`, original f0a3d51 commit/tree imported as a
  shallow repository. Controller managed worktree has the same branch/base.
- Pinned native commit, all five patch SHA256 values and intermediate/final tree
  hashes passed the repository's `prepare-backends.py` verifier on the remote.
  Import initially contained CRLF differences from Git archive. Those were
  checked independently, then the pristine imported baseline was restored from
  exact Git blobs before patch application. No expected hashes were changed.
- Portable MSVC 19.44.35222, SDK 10.0.26100.0, CMake 3.31.6-msvc6, Ninja 1.12.1,
  CUDA 13.2.86 and Python 3.11.15. Tool manifests checked 16,341 and 11,866 files.
  Dependency bundle manifest c7684b1d96347de80ba32b99edcbcf5fd77c8e890463694dea89cf65170ef59a
  was checked before configure/build/test. Native runtime additionally uses an
  isolated MSVC/OpenMP 14.51.36247 bundle and the same CUDA bundle's bin/x64.
- Host targets: kvmem-shared-cache-test, kvmem-session-transfer-test,
  kv_snapshot_test, tiered_kv_storage_test, raw_kv_store_test: **5/5 passed**.
  Dry run: 19 host compiles, 6 links, zero CUDA. Eight jobs. Release tests use
  explicit checks, not disabled `assert` checks. Latest CTest elapsed 0.84 s.
- Localized MSVC output initially resulted in zero recorded header dependencies.
  UTF-8 console codepage plus the matching showIncludes prefix fixed this; the
  final RawKvStore object records 129 dependencies. A temporary repair config
  accidentally dropped /EHsc and caused three test failures; restoring complete
  default flags fixed them. These were task setup regressions, not legacy bugs.
- Native validation reused known matching libraries after comparing normalized
  source, identical pinned prepared native tree, toolchain/configuration, stable
  dependency bundle, command records, and transitive headers. The original
  retained build log supplied host /showIncludes evidence where its Ninja
  records were empty; 379 logged objects / 574 distinct headers had no newer or
  missing header. CUDA dependency records were retained. Disabled HBM is an
  explicitly empty translation unit. No old Ninja metadata was rewritten.
- Remote native dry run required **one link, zero compiles**. This is a focused
  validation executable, not a newly built release package. The link reported
  LNK4098 (LIBCMT versus other runtime libraries); final release linkage remains
  to be audited before shipping. The baseline runtime tests below passed.
- Existing checkpoint-tail test passed both tail sizes (1/32 and 31/32) for
  Qwen3-0.6B dense and Qwen3.5-4B hybrid. It verifies sparse selection, evicted
  tail restoration, native state byte equality after slot normalization,
  deliberate coverage-gap rejection and clean recovery. It does not qualify
  speculative draft state, concurrent branches or optimized capture sharing.
- Both tests used the recorded 5070TiM UUID, Q8_0 K/V, block size 32, budget 256,
  generation reserve 128, context 1024, batch/ubatch 128. GPU occupancy was
  checked before each launch; no foreign process was stopped. A post-dense
  utilization sample delayed hybrid launch until the GPU was idle again.
- Models downloaded from HF Mirror, immutable revision and LFS SHA256 checked:
  Qwen_Qwen3-0.6B-IQ3_XXS.gguf (345,867,680 B), revision
  60b85c0e3d8fe0f6474f406922a26d12aca4550d, SHA256
  4553e20544d2667785db4e48b5556d02e70989234be5853b8bf200d65381d701;
  Qwen_Qwen3.5-4B-IQ3_XXS.gguf (2,243,829,728 B), revision
  4168f45a16a1290d65a4ec0fa312ae917a4c15d6, SHA256
  37046243577d6b44e98137680623b807ae92b147982a91d5726e4c0c64e7508e.

Still required: coordinated sealed-record export API; backend integration and
native state accounting; requested failure/branch/speculative matrix; controlled
before/after prefill/decode/capture/I/O/VRAM/RSS measurements; final audit, latest
master synchronization, and PR. No public implementation or contract test was
modified, and no PR or merge has been performed.

## Additional native boundary qualification

The backend-specific `checkpoint-tail-test.cpp` now also exercises exact block
boundaries at 64 and 96 rows, alongside private tails at 65 and 95. It logs actual
full and PARTIAL_ONLY serialized image sizes. All four cases pass on both fixed
models, including noncontiguous selection, source state restoration, deliberate
coverage-gap rejection and recovery. This does not establish block sharing.

| Model | Rows | Full native state bytes | PARTIAL_ONLY state bytes |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B IQ3_XXS | 64 | 3,900,864 | 8 |
| Qwen3-0.6B IQ3_XXS | 65 | 3,961,804 | 8 |
| Qwen3-0.6B IQ3_XXS | 95 | 5,790,004 | 8 |
| Qwen3-0.6B IQ3_XXS | 96 | 5,850,944 | 8 |
| Qwen3.5-4B IQ3_XXS | 64 | 53,807,668 | 52,691,548 |
| Qwen3.5-4B IQ3_XXS | 65 | 53,825,104 | 52,691,548 |
| Qwen3.5-4B IQ3_XXS | 95 | 54,348,184 | 52,691,548 |
| Qwen3.5-4B IQ3_XXS | 96 | 54,365,620 | 52,691,548 |

GPU/configuration is unchanged from the baseline above. One host test object and
one link were audited and executed remotely; no CUDA recompilation. The object
records 124 transitive header dependencies. Evidence is retained remotely under
`boundary-evidence`, separately from `baseline-evidence`.

## SSD and document/question baseline

The unchanged server passed `checkpoint-coverage-regression.py` on the fixed
hybrid model with Q8_0 KV, context 8192, budget 1024/reserve 256, batch 128,
128 document lines, tool continuation, spread interval 256, 12 retained points,
hot count 1, checkpoint Host 1024 MiB, KV Host 128 MiB and SSD 1024 MiB.
Document and question are separate ordinary turns. Native reuse: question 2843,
tool continuation 2873, trailing edit 2854, return to parent 2854, middle edit
1562 rows. Cold state reads occurred on switch-away, middle edit and front edit.
All protocol/native Query/reuse assertions passed. Tracing was enabled; these
request durations are not controlled performance measurements.

An earlier, separately retained run with checkpoint Host 256 MiB, SSD 512 MiB
and count 6 failed the middle-edit coverage assertion (reuse 29 rows). Logs show
optional checkpoints skipped for Host budget exhaustion and idle owner eviction.
The larger-budget run passed without code changes, so this is currently a
capacity-dependent retention limitation, not evidence of corrupted restoration.
Do not treat the two configurations as a performance comparison or discard the
failed run. Remote evidence: `coverage-baseline-ssd` and
`coverage-baseline-ssd-room`.

## Packed coverage and restore interface follow-up

All current llama packed writers capture a block's contiguous prefix starting at
`orig_pos_start`. The single `HarvestVJob` constructor in `harvest_gpu_v` supplies
that start; asynchronous commits and synchronous fallback use the same range.
MTP `harvest_k/harvest_v` do likewise. No current llama caller was found to create
a within-block hole. This does not make the public high-water count a proof.

A temporary diagnostic linked against the baseline public library reproduced:
with block size 4 and 4-byte K/V rows, writing only `[2,3)` makes both
`has_*_gpu(0,0,3)` return true. After `invalidate_packed_from(1)`, both
`has_*_gpu(0,0,1)` still return true for the unwritten first row. The coordinator
owns the new export coverage contract; no old public read/write semantics or
contract tests were changed here.

Export alone does not provide snapshot-based restore. `multimodal_restore`
restores native model state, trims logical rows, restores MeanK and calls
`restore_checkpoint_tail`; native stage-in then reads RawKvStore. The requested
minimal import must install sealed record references and exact coverage into a
private destination atomically, including cold history, with no payload copy or
second physical charge. Native layout/positions and recurrent/carry remain the
adapter's responsibility. This request and ownership/failure tests were sent to
the coordinator, together with the existing independent checkpoint RAM-account
issue. The coordinator has confirmed compatibility of the existing separate KV
Host H and checkpoint-host Q quotas using shared HostKvStorage mechanisms and
one SpillFile D quota. Metadata/staging must remain explicitly accounted and
must not be advertised as bounded process RSS. The native state image transport
must retain existing identity/length/checksum validation before model restoration.

## Old parent and branch qualification

The backend-specific native test now preserves a parent bundle before appending,
then clears and reuses native storage and logical block IDs, appends a different
suffix, and alternates restoration of the old parent and child. Both full native
serialized states match their respective original bytes. Resetting/deleting
parked owners leaves the active child unchanged. All 64/65/95/96-row cases passed
on both fixed models, with the same GPU and Q8_0 configuration. This covers
native bundle COW and old-parent preservation; it does not yet qualify the
pending KvSnapshot import/export integration. Evidence: remote `branch-evidence`.

## Separate pending public prototype

At the coordinator's request, a public export/import prototype was made in the
independent `raw-kv-snapshot-prototype` worktree, branch
`codex/pending-raw-kv-snapshot-prototype`, commit
`eb252a655b11c9d8cf10bae0e25a1078ba271db4`. It is not in this backend branch and is
not approved for direct integration or merging. The coordinator owns its audit,
revision, public PR and merge. Its four focused host tests pass remotely (0.17 s).
See that commit's `docs/raw-kv-snapshot-prototype.md` for exact semantics and gaps.

## Capture instrumentation and baseline measurements

Backend-private process counters now record successfully stored packed K/V bytes
and published native checkpoint bytes. They include target and draft attention;
references/forks do not increment them. Failed transfers and non-archive D2H
remain separate transfer diagnostics. No new synchronization is introduced.
The native boundary test additionally asserts that a repeated unchanged fork
performs zero additional packed captures; both models pass all four cases.

`tests/llamacpp-snapshot-benchmark.py` wraps the unchanged ordinary-request
checkpoint workload with tracing disabled. It samples device framebuffer use
and the existing monitor, and reads Windows process PeakWorkingSetSize plus
native prefill/decode logs. Device peaks are sampled, not exact per-process
high-water values. These are short outputs (2-24 tokens), not a sustained decode
throughput qualification. Keep the monitor overhead/configuration identical in
the future candidate comparison. The instrumented baseline includes the dense
capacity repair, but no immutable snapshot integration.

Three fresh-server runs per model, identical SSD configuration to the adequate
budget baseline above, produced the following medians/totals:

| Model | Total prefill ms (median) | Total decode ms (median) | Packed K bytes | Packed V bytes | Native state bytes | RSS peak bytes (median) | Sampled GPU MiB (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1903.53 | 332.70 | 213918208 | 213918208 | 408 | 1084682240 | 669 |
| Qwen3.5-4B | 4321.46 | 349.96 | 61685248 | 61685248 | 2687269544 | 3987767296 | 2541 |

Capture counts were identical across all three runs. Dense SSD read/write totals
were 178,884,832 / 273,690,480 bytes each run. Hybrid reads were 210,766,384 bytes;
writes ranged from 1,666,993,940 to 1,673,285,396 bytes due to asynchronous tier
placement. Source evidence: `perf-accounted-baseline-{dense,hybrid}-ssd-{1,2,3}`.
These are baseline-only measurements; there is no optimization comparison yet.

## MTP and failed-budget qualification

The existing Qwen3.5-4B IQ3_XXS includes one native next-token layer. With draft
MTP enabled, the ordinary document/Query/tool/branch/edit SSD baseline passed,
including cold state loads on question, switch-away, middle edit and front edit.
Evidence: `coverage-baseline-mtp-ssd`. This qualifies the default MTP state mode
only; DFlash, explicit replay/snapshot alternatives and multi-GPU remain untested.

A new inherited failure was found at Q=128 MiB: prefill could succeed, but the
final native checkpoint failed admission. Native rollback succeeded, yet the
exception escaped the HTTP handler as an empty 500 and bypassed sampler release.
The server now catches commit failures through the driver, completes the existing
rollback, emits a protocol error and releases the sampler. All streaming commit
sites use the same failure boundary, and check cancellation before publication.
No capture boundary or native restore ordering was moved to hide this failure.

`llamacpp-checkpoint-failure-regression.py` passed JSON and SSE late-commit errors,
rollback and repeated rejection with less than 4 KiB checkpoint state retained.
At 128 MiB this model cannot fit even a short generation's necessary checkpoints;
the original test's expectation of successful normal generation was incorrect
and the failed evidence was preserved. At 192 MiB, the existing coverage test
separately passed Query capacity rejection, rollback and subsequent normal
generation. Evidence: `commit-failure-evidence-2` and `coverage-budget-reject-192`.
