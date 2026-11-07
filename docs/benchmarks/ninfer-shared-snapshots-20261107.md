# NInfer shared snapshots: implementation and qualification

Public paths and device identities use stable aliases; measurements are unchanged.
Full UUIDs and paths remain in the private raw manifests, bound by the JSON
`_sanitization.original_sha256` digest.

The adapter uses the public `SealedKvBlock` / `KvSnapshot` manifests introduced
by PR #267. It retains NInfer's native page transfer batches, lane scheduler,
checkpoint directory, `HostKvStorage` account and `StorageIoQueue`. No second
storage manager, GPU allocator or restore scheduler was added. Public extensions
from separately merged PR #271 are inherited from master, not carried in this
backend patch. The integration base is `9f45277` (including PR #270).

## Existing behavior and changes

NInfer already shared sealed archive allocations across branches, preserved
immutable checkpoint origins, and tracked native page versions to avoid
unnecessary transfers. Therefore this change does not claim to eliminate an
existing full-history GPU capture on every checkpoint. The controlled workload
below confirms equal GPU capture bytes before and after.

Each `KvmemPrefixCheckpoint` now owns its exact immutable KV manifest. Complete,
unchanged records share a sealed origin; the proof is a live allocation owner,
held throughout lookup and publication. Tokens, logical page numbers and version
counters are insufficient. A partial final page is copied into a private sealed
record at the accepted frontier. Modified/replayed pages use native record COW;
old checkpoints keep their original allocation. Record capacity includes native
padding, attention/scales and the supported auxiliary KV planes.

Checkpoint completeness, history lookup, fork preparation and restoration use
the selected checkpoint's manifest rather than the latest mutable archive.
Cold forks create owner handles without promotion. Layout, ordered spans, byte
stride and coverage are validated before native restore. Model state, mean-key
tail, graph/feedback state, positions and prefix identity remain attached to the
same accepted frontier. Existing Query replay boundaries are preserved; no
earlier checkpoint or extra recomputation was introduced.

The adapter deduplicates physical references and manifest metadata across active
lanes, retained groups and pending transactions. Retention admission includes old
immutable versions and private tails beyond active reservations. Metadata credit
is recalculated after admission evicts an owner. Origin compaction additionally
requires the same strongly owned manifest, preventing numeric-origin-only reuse.

## Ownership, budgets and failure boundaries

| Resource | Accounting and limit |
| --- | --- |
| KV RAM (`H`) | Existing shared `HostKvStorage`; each physical allocation charged once, including private tails and COW versions. Borrowed/running leases keep their charge. |
| SSD (`D`) | Existing process-local spill account, shared by cold KV records and cold native state images; no restart persistence. Failed reads/writes do not publish a restored checkpoint. |
| Native state (`Q`) | Existing pinned image pool, checkpoint host policy and cold image extents. Whole state images are retained/shared by checkpoint origin; this change does not split or migrate them to a new state abstraction. |
| Metadata | Object/container capacity enters existing checkpoint host/retention admission, shared manifests and sealed origins deduplicated. Allocator/control-block overhead and temporary lookup sets are process overhead; this is not a hard RSS ceiling. |
| Staging | Existing bounded GPU transfer batches, but a full selected host KV view and whole native state staging may still be required. `transfer_staging_bytes` and `statistics_staging_bytes` report these separately. `H` alone is not total host memory. |

Publication remains on the native lane's completion boundary. Async work holds
its source owners/leases until completion; cancellation discards the unpublished
candidate. Full records remain valid after other directory owners are evicted.
Budget failures preserve the published source, though ordinary storage placement
or policy eviction may already have occurred. No global metadata allocator or
new cross-backend resource policy is implemented here.

New metrics are `snapshot_capture_bytes` (completed native KV D2H payload,
including pressure capture and later publication failures), `snapshot_view_bytes`
(successfully captured logical views), `snapshot_tail_copy_bytes` (private host
tail copies), `snapshot_count` and `snapshot_metadata_bytes`. Capture bytes include
record padding, exclude native StateImage and verification transfers, and are
not PCIe bus counters. Existing physical/IO/state/staging counters remain exposed.
`checkpoint_kv_storage=common_sealed_snapshot` and
`restore_staging=bounded_gpu_batch_full_selected_host` describe the implementation.

## Validation environment and scope

All compilation and execution took place on the isolated M16 worktree:
`$ARTIFACTS-Lab/ninfer-shared-snapshots-20261106`.
GPU: RTX 3080 Ti Laptop, UUID
`GPU-A`, driver 555.97.
MSVC 19.44.35222, CUDA 12.9.86, SDK 10.0.26100, Release, SM86. Dependency
bundles were pinned and checked before build/validation. GPU occupancy and an
exclusive task marker were checked before each run; foreign processes were not
stopped. Measurements ran without concurrent compilation or model transfer.

The first remote executable build had no compatible existing SM86 objects and
its necessary runtime dependency closure was audited before compilation.
Subsequent builds preserved objects and Ninja records. Final master integration
planned 97 host compiles, one CUDA compile, 14 links and a build-ID action. The
single CUDA dependency was the changed public `raw_kv_store.hpp` included by
`page_softmax.cu`; the other CUDA objects were reused. Build parallelism was 8.
The final write-fault fixture adjustment planned four host objects/eight links,
zero CUDA. Patch export then removed Windows CRLF serialization noise from only
the 11 changed source files. Exact before/normalized SHA256 evidence establishes
that this last operation changes no source tokens or ABI; validated binaries are
reused without forging timestamps or Ninja records.

### Host tests

`ninfer_kvmem_snapshot_test` is a standalone host target compiling only its test
and the existing host storage/spill implementations. It covers complete-record
sharing, independent tails, unchanged old-source bytes, reused logical indices
with new allocations, simultaneous branch COW, SSD read without RAM promotion,
capture/COW budget failure, queued cancellation, running-owner lifetime and
eventual zero RAM/SSD/queue charges. Invalid layout/count/overflow coverage is
rejected. It does not require a model or CUDA execution.

The separately merged public PR's five focused host tests also passed on M16;
its Linux/Windows CI passed. Those public tests are not presented as NInfer
real-model coverage. Exhaustive allocator failure injection and deterministic
interruption at every possible IO instruction are not covered.

### Real models

Qwen3.5-4B-MTP-Q4_K_M, SHA256
`c8f5c3a8b9f578d07cc17b0ed01835566b0728c4b1c0cd3ec3839943fad3100e`,
int8 KV, eager execution: native content groups/eviction, MTP input reuse,
two-lane exact RAM-budget admission, concurrent MTP requests, SSD reuse and
cancel/resume, cold-read failure, blocked-read cancellation and source-preserving
fork recovery. Native verification checks selected KV bytes and full state
images; fork fault tests also compare against an independent fresh execution.

Bonsai2-T5-MTP-Q4-DFlash2-Q2_K: DFlash2 document/question, repeated question,
branch and old-parent restore with `payload_exact=1`, `state_exact=1`, including
Query replay. It requires its 2048-token draft ring: the successful run uses
`B=4096`, `R=256`, `H=1024 MiB`, `D=4096 MiB`, two lanes. Smaller test settings
were explicitly rejected at startup before cache execution. This validates the
native recurrent/conv and DFlash fixed-state image boundary for this model; it
does not establish support for every speculative mode or model family.

Qwen3.5-0.8B Q8 could not start: its native Q8 `linear_add` shape is not admitted
on the tested configuration. No cache correctness claim is made for that model.
Multi-GPU/TP, CUDA graphs, multimodal state and other model families were not
qualified by this task.

### Two inherited test expectations

The exact frozen baseline reproduces two old test failures after all preceding
restore/fault checks pass: `ssd_failure --fork` expected zero reuse when
`update_session_index=false`, and `memory_disk` expected a read-disabled request
to clear other published histories. The existing API defines the former as
publication permission and the latter as per-request read permission. The tests
now check reuse without publication/eviction and preserved source availability,
respectively. Production read/write policy is unchanged. Original baseline and
candidate failure logs are retained separately from corrected final runs.

The write-fault fixture also had a model-size assumption: fixed H=48 MiB holds
the entire 4B cold prompt, so its expected KV spill never occurs. The corrected
fixture derives a native record's size from physical records and sets H to three
512-token windows (two lanes plus one window of headroom). It still requires
the actual injected write failure, unchanged fresh-reference peer output,
successful retry, peer progress during blocked IO and complete cleanup.

## Controlled performance comparison

Baseline: root `f0a3d51`, prepared native tree
`dd34c887cb716a285f34abcab307a04a03562fce`.
Candidate for this comparison: same root/public implementation, native tree
`2e517abf4ae5452ad0ae6d034b3c300f61d9783a`, server SHA256
`1e051c8c1c36348f511d8a8cff1dda5f6ccd0d073d97f95685cea193fcc982b3`.
Final master integration was validated separately; its newly merged public
changes are not silently included in this controlled before/after attribution.

Same 4B model, GPU UUID and flags: MTP 3, int8 KV, eager, prefill chunk 128,
`B=768`, `R=256`, max context 8192, two lanes, 8 retained groups, sink/recent 64,
spread checkpoints, Q=1024 MiB per lane, count 12, hot count 1, interval 256,
extra recompute 0. RAM: H=256 MiB, D=0. SSD: H=48 MiB, D=4096 MiB.

Each mode used fresh processes in baseline/candidate/candidate/baseline order.
The unchanged 5282-token document was sent first as an ordinary request. The
real question was sent separately with a neutral `Received.` acknowledgement,
followed by correlated repeat questions, a branch and the old parent. Actual
cache reuse and Query spans were saved. Trace/byte verification were disabled
for timing and run separately for diagnostics. Two processes per arm are a
small sample; repeated questions are not independent document trials.

Means across the two fresh processes per arm:

| Mode | Arm | Document prefill tok/s | First-question elapsed ms | First-question decode tok/s | Peak RSS MiB | Sampled peak VRAM MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RAM | baseline | 2521.94 | 609.5 | 115.46 | 2344.74 | 3110 |
| RAM | candidate | 2489.32 | 632.5 | 113.29 | 2384.78 | 3110 |
| SSD | baseline | 2434.01 | 648.5 | 113.39 | 1532.91 | 3110 |
| SSD | candidate | 2408.82 | 672.0 | 113.13 | 1549.89 | 3110 |

Prefill is 1.29% lower in RAM and 1.03% lower with SSD; first-question elapsed
is 3.77%/3.62% higher. Decode changes are -1.89%/-0.23%. These observations do
not establish statistical significance. RSS is the Windows process lifetime
PeakWorkingSet; VRAM was sampled every 250 ms and can miss brief peaks.

Separate SSD transfer diagnostics produce identical baseline/candidate KV D2H
130,166,784 bytes and H2D 357,654,528 bytes. Successful native StateImage capture
logs sum to 1,803,074,560 bytes for each arm. These are different byte categories,
not interchangeable counters. The candidate additionally copies 18,247,680
host bytes into private tails and exposes 2,491,416,576 logical view bytes.

| SSD timing run | Storage read bytes | Storage written bytes |
| --- | ---: | ---: |
| baseline 1 | 163,858,512 | 2,137,298,496 |
| candidate 2 | 208,869,456 | 2,172,577,344 |
| candidate 3 | 260,385,912 | 2,069,544,432 |
| baseline 4 | 163,858,512 | 2,085,782,040 |

These storage counters include cold native state IO; async retention/placement
varies between runs. RAM runs perform zero SSD IO. Explicit private versions
increase retention and can increase cold reads. At the final RAM endpoint,
physical KV payload is 113,135,616 bytes in the baseline and 128,950,272 bytes
in the candidate; physical native state is unchanged at 1,133,361,152 bytes.
This adapter is a correctness/ownership integration, not a demonstrated capture
or storage-throughput optimization.

Machine-readable per-run measurements and final test results accompany this
report. Full requests/responses, native logs, dependency locks, binary/model
hashes and resource samples remain in the task's `evidence/` directories. Raw
diagnostic failures are retained and clearly distinguished from passing runs.
