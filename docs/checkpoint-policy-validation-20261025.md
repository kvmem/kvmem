# Paths and device identities anonymized; measurements unchanged.
# Checkpoint policy validation, 2026-10-25

Historical qualification: cache-key and owner behavior below describes the
recorded implementations. For key-independent content/native-snapshot reuse,
see [current semantics](checkpoint-policy.md#content-snapshots-across-backends).

## Content snapshot implementation and review, 2026-11-04

This later implementation starts from parent commit `8cf44cb`. It makes client
cache aliases diagnostic metadata across NInfer, llama.cpp, PolyStrata and
Radiance. Native token/media/position/checkpoint compatibility still authorizes
restore; each executing branch preserves its immutable source. Resource groups
only distribute bounded retention capacity. Tool continuation remains append
plus resident-only Pressure, without old-Query cold-history selection.

The review reran nine native host executables covering the common directory,
conversation selection, CPU-tier independence, lane admission/failure, shared
sources, PolyStrata sessions and image collisions, Radiance prefix policy and
sampling capacity. All passed. Python checks passed four actual frontend
image/template cases and seven evidence-harness self-tests. The prepared
PolyStrata frontend suite ran 42 tests: 40 passed and two CMake declaration
checks were initially skipped. Both declaration checks then passed in a focused
rerun with the existing bundled CMake, completing all42 checks without compiling
CUDA or HIP. Initial frontend discovery failed with an empty backend
placeholder and sandbox-denied localhost sockets; preparing the backend path
and allowing the mock localhost service resolved those environment failures.
No native recompilation was needed for this review.

The following GPU qualifications belong to their separately recorded binaries
and frontends. They are functional arms, not a performance comparison. Only
PolyStrata was rerun for the notification repair described below.

| Backend | Qualified scope and limits |
|---|---|
| NInfer | Final worker `b5342dd8dc6a`: strict Responses18 and real OpenCode18 each prove six native concurrent pairs; parent returns reuse 2052/2085 and 5106/5141 tokens respectively. Native SSD/MTP/cancellation checks are separate workloads. |
| llama.cpp | Final worker `d43e7e583c4f`: strict18 proves six native concurrent pairs and parent reuse 2042/2086. The earlier 40 SSD/MTP/CPU-tier variant requests belong to their earlier explicitly recorded workers. |
| PolyStrata | Worker `da268a6b8968` with the repaired production frontend passes all 18 actual-cache assertions and all six materialized concurrent pairs, with minimum hot reuse 97.69% and parent reuse 1835/1866. Before the notification repair, two runs and a preserved older-worker control passed cache checks but failed the strict concurrency gate at pair0; those failures remain historical evidence. Native restore/decode interleaving was present, but live cache observations arrived late. The older H4 six-pair PASS and H2 missing-pair FAIL belong to their older runs. |
| Radiance | Final worker `b38c6bff1ec1`: B1024/R512/C2/H4/MTP3 strict18 proves six materialized pairs. The separate low-H SSD/MTP0 arm has nine attempts: seven complete, one required EIO, one early native cancellation; healthy-peer completion and same-source retries pass. Full raw evidence was independently checked. Actual TP2/multi-GPU execution remains unqualified. |

PolyStrata's fixed-worker and pre-fix control use the same RTX 5050 Laptop
(`GPU-B`), Gemma-4-26B-A4B UD-Q2_K_XL,
B1024/R256/C2/H4096MiB, prefill128, sessions4 and MTP0. Generated answers become
later history, so the control is not an exact-output oracle. Each run independently
passes native accepted-prefix, response correlation and tool-policy checks.
These short tool fixtures do not establish physical Pressure behavior.

The pair0 failure was localized to the earlier shared-cache frontend adaptation.
`StrataEngine._control` parsed `CACHE_REUSE` without yielding, while Service only
published the observation on its next progress/token callback. In a timestamped
diagnostic, the second request had restored before the first generated seven
more native tokens and finished; monitor publication lagged parsing by 1.012s.
The old worker contained the same frontend observation path, so its failure did
not establish a native scheduling defect.

The applied repair yields one existing progress notification immediately after
parsing `CACHE_REUSE`. No native code or execution policy changed. A deterministic
test withholds all subsequent PP/token input: the old branch fails, while the
repair publishes both positive and zero actual reuse immediately and counts each
request once. All 43 frontend tests pass, including the CUDA/HIP CMake declaration
checks using the existing bundled CMake. The formal GPU arm uses the actual fixed
frontend without the diagnostic wrapper or an in-memory proposal; the binary,
GPU, budgets, sampling configuration and frozen acceptance gates stay the same.

Two audit defects are repaired. PolyStrata now checks the complete existing
FNV64 media fingerprint, grid and original position in live/idle/batch restore
selection, including snapshot-only fallbacks. Unknown proof declines reuse after
the first image. A real parser fixture with different embeddings and identical
truncated image token IDs now restores to token16 before the image; the same-image
control still restores128. Immutable proof reservations share the Host quota and
release on the last reference. This is a native parser/planner proof, not GPU
image-output qualification or cryptographic raw-image equality. Optional
history-only Jinja failures now disable that grouping hint; required generation
errors still propagate.

Radiance also includes the user-approved minimum repair of an inherited sampling
capacity defect. The old worker failed the original2038-token cold request at
B1024/R512 before cache reuse; the repaired worker traverses that same native
Query/checkpoint/replay boundary successfully. Canonical grouping separately uses
unrounded native message spans, rather than scheduler-rounded history boundaries.

All backend patch hashes and prepared trees are pinned in
[`backends/versions.json`](../backends/versions.json). NInfer and PolyStrata patches
were strictly reapplied and checked against their prepared source files. Finite
memory can still evict a source or reject optional reuse. Source-preserving forks
can increase live state costs; CPU arenas and native checkpoint quotas are
accounted separately. Throughput, general recall quality, GPU image output and
other untested model/backend combinations are not qualified by these checks.

Status: the original redesign and nearby-point experiment are complete within
the qualification limits below. Nearby thinning defaults off. The policy is
opt-in; `legacy` remains the default. These qualification runs preceded submission
and did not deploy a worker.

The current behavior and configuration are documented in
[checkpoint-policy.md](checkpoint-policy.md).

## Baseline and environment

The prepared KVMem source is based on `83650df1e5af03c4d87c069570b784bff063202e`
(PR 188). Prepared backend baselines are llama.cpp tree
`1593775ad2ac64b4ad6993ce1e04ca025953be0e` and NInfer tree
`3bb18fca6acf320b93ee0ab3fed2152a95daa2e9`. The original-master quality baseline
has not been modified or used as an exact-output oracle.

Initial GPU tests used RTX 5060 Ti,
`GPU-A`, with occupancy checked before launch.
The user subsequently selected RTX 5050 Laptop,
`GPU-B`, for the remainder of this task.
Results on the two devices are kept separate; a performance comparison must use
only one device throughout.
The Windows build uses MSVC 14.44, CUDA 13.2 and sm_120a. Build parallelism was
reduced from eight to four after available system RAM fell to about 550 MiB.
Python test tools run with the configured Python 3.14 runtime instead of the
repository's preferred Python 3.11. The full NInfer CUDA build and subsequent C++ rebuilds
now pass. Incremental builds returned to eight jobs after RAM pressure ended.

Each HTTP fixture saves requests, responses, trace segments, native restore
frontiers, GPU identity and invocation settings in its named evidence directory.
These diagnostics ran while CPU compilation could be active. Their wall times
are not a controlled performance comparison.

## Completed checks

| Check | Result |
|---|---|
| Common Host suite | 23/23 pass |
| Native llama partial KV/recurrent restore | Exact-byte checks at rows 65 and 95 pass; incomplete coverage rejected; clean recovery passes |
| Conversation selection | 77,440 arithmetic cases and explicit-owner middle edits pass; inferred/unknown-owner fork protection remains |
| Lane ownership | Failed handover and parked-owner budget reclamation pass; busy, switching, queued and resident owners remain protected |
| Image framing / storage | Exact bytes, identity, checksum, truncated image, disk/queue limits, in-flight source lease and reservation refund pass |
| NInfer changed C++ units | frontend, Program, storage, prefill and pool test compile |
| NInfer native pinned pool | Hard byte limit, idle count, alias and lifetime checks pass |

The common image tests exercise framing and queue failures directly. They do not
establish every backend's end-to-end behavior under a corrupted live SSD file.

## Llama HTTP and cancellation results

For the roughly 7K-token text fixture, the following numbers are actual native
prefix restores, not merely admitted API cache-hit counts. The document is sent
first with a short neutral acknowledgement; the question is a separate ordinary
request. Its answer is not supplied in the conversation.

| Policy, Qwen3.5 0.8B Q8_0 | Question | Tail edit | Switch back | Middle edit | Front edit |
|---|---:|---:|---:|---:|---:|
| Legacy | 6555 | 7084 | 7084 | 0 | 0 |
| Spread RAM | 6683 | 7084 | 7084 | 2074 | 26 |
| Spread SSD | 6683 | 7084 | 7084 | 2074 | 26 |

These runs use interval 512, count 12, hot target 2, checkpoint Host budget
1024 MiB, B=1024 and R=256. The SSD case performs real cold-image reads.
The middle-edit difference includes an intentional selection-policy change:
spread permits replacement of the suffix of an explicitly bound conversation;
legacy retains its prior continuation/fork gate. It is a new reuse capability,
not an acceleration of an already equivalent legacy edit path.

In these 5060 diagnostic runs the 0.8B record is about 19.31 MiB including charged
metadata. The recorded checkpoint Host peak is 328.38 MiB for RAM and 154.93 MiB
for SSD; the latter reads 77.06 MiB of native checkpoint payload over the seven
requests. The 4B MTP record is about 50.39 MiB. These are workload observations,
not fixed checkpoint sizes or process-memory limits. The 7K spread directory's
maximum initial gap is 2048 tokens after pruning; the 53K/16K fixture's is 16384.

Additional passing fixtures:

- `llama-mtp-spread-disk`: Qwen3.5 4B Q4_K_M, MTP with three draft tokens,
  document/question, tail/middle/front edits, session switch and real SSD reads.
- `llama-mtp-tool`: ordinary tool continuation uses `append_pressure`, no cached
  Query features and no cold-history replay; native reuse is 7096 tokens.
- `llama-mtp-cancel-frontier`: four long/short session switches and two prefill
  cancellations; both recoveries retain a valid 1902-token prefix.
- `llama-53k-interval16k`: 52,836-token document, context 65,536, interval 16,384,
  count 24; middle edit restores row 16,410 from SSD. The observed archive has
  seven to ten points. This validates a long-history case at the proposed
  interval, not a sweep of interval or capacity tuning.
- `llama-count1` and `llama-count2-bounded`: requests complete with the directory
  capped at one or two records. Full recomputation is a valid outcome when all
  retained points lie beyond the next request's legal restore frontier.
- `llama-host64-rejection`: a necessary snapshot cannot fit, the question reports
  capacity exhaustion, rollback succeeds, and a subsequent short request works.
  Charged peak is 60,927,692 bytes, below 64 MiB.
- `llama-host128-ram-reclaim2`: optional captures skip under pressure and an older
  parked owner is reclaimed. Charged peak is 121,541,868 bytes, below 128 MiB.
  This costs reuse: switch-back restores only 28 tokens and front edit restores
  zero. The byte limit does not guarantee a retained useful prefix.

The instrument-recall probe passes in these fixtures. This is one probe per
document configuration; related follow-ups are correlated. It does not qualify
general long-context answer quality, all MTP acceptance patterns or 27B vision.
Checkpoint accounting measures charged native images and metadata, not RSS.

Subsequent RTX 5050 Laptop checks are separate from the table above:

- `llama-5050-mtp-tool-verified` passes the 4B MTP/SSD fixture, including actual
  checkpoint reads, tool continuation, native short-question Query capture and
  edits. Its initial launch used an incorrect local model path and exited before
  loading; the verified run uses the recorded existing model.
- `llama-5050-disk-full` limits D to 1 MiB, which cannot hold a native image.
  Writes report capacity exhaustion, all seven requests still complete using
  RAM checkpoints, and middle edit restores row 2074. No cold read is claimed.
- `llama-5050-timing-parser-smoke` validates complete SSE recording, usage and
  first-text timing with diagnostics off. CPU compilation was active, so this
  is a harness check and is excluded from performance comparisons.

## Findings and disposition

- The first spread integration omitted coverage records from the outer llama
  conversation catalog. They are now included in selection and accounting.
- The inherited middle-edit gate was a branch-ownership design, not a proven
  defect. Spread intentionally changes it only for an already bound explicit ID.
- Small-budget llama admission initially failed to reclaim parked owners charged
  to the creator lane. Required admission now reclaims an eligible idle owner;
  the passing 128 MiB fixture covers the fix and its reuse cost.
- NInfer required admission now protects its admitted restore source and drains
  the single writer across active/idle checkpoint paths. RAM pressure and SSD
  execution now pass with the complete engine.
- A localized MSVC dependency prefix initially left some C++ dependency records
  empty. The prefix was corrected and affected llama objects rebuilt.
- Earlier tiny-count tests incorrectly required a cache hit, and an old
  cancellation assertion required a particular tail-reload log event. Their
  corrections assert actual budget/restore invariants; independent native
  exact-byte tail tests remain. Earlier failed evidence is retained.

## NInfer results on RTX 5050 Laptop

Qwen3.5 4B Q4_K_M, int8 KV, interval 512, count 12, hot target 2, Host 1024 MiB,
B=1024 and R=256. These are diagnostic native restores with transfer validation.
The existing Responses API with `store: false` and `prompt_cache_key` provides
the explicit session binding; no request protocol was added.

| Policy | Question | Tail edit | Switch back | Middle edit | Front edit |
|---|---:|---:|---:|---:|---:|
| Legacy | 7069 | 7079 | 7079 | 0 | 0 |
| Spread RAM | 7069 | 7079 | 7079 | 2176 | 26 |
| Spread SSD, fixed | 7069 | 7079 | 7079 | 2176 | 26 |
| Spread SSD, MTP | 7069 | 7079 | 7079 | 2176 | 26 |

SSD switch-back, middle and front edits perform actual cold-image reads. MTP
tool continuation restores row 7090 and uses `append_pressure` without Query
replay. The separate short-question turn captures nine native Query rows.
The 128 MiB RAM test completes while skipping optional captures and retaining
only two records; switch-back, middle and front edits recompute from zero. A
1 MiB SSD quota produces the expected capacity errors, retains RAM images and
completes all seven requests, including the 2176-token middle restore.
The native pool, serving-options and frontend suites pass on this build.

The MTP storage lifecycle also passes in two independent processes sharing one
spill directory: cold start, native SSD transfers, 2111-token reuse, cancellation,
resume, clear, subsequent cold request and cleanup. Neither process imports
history from its predecessor; an unrelated marker file remains intact. Checkpoint
Host charges stay below 1024 MiB (RAM sizing 464,032,292 bytes; SSD request-end
charges up to 154,933,796 bytes). These are sampled accounting observations,
not a measured process-RSS peak. Both runs end with zero SSD I/O errors.

## Authorized partial-offload repair and real-image validation

The available 27B model exceeds the 5050's dedicated VRAM. A 24-layer offload
reproduced an illegal CUDA access before image evaluation in both policies and
in the unmodified archived release. Its SHA256 matches the release manifest;
this failure predates the checkpoint redesign. The user authorized its repair.

Synchronous diagnostics on a four-layer 0.8B reproducer confirmed that the
sampler chose Q's CUDA device while borrowing CPU K. It now computes beside K
and stages the bounded current-forward Q through the existing backend-copy API.
Ordinary host buffers resolve to the CPU backend. The main scheduler finishes
before copying; the entire persistent K history is not transferred. A second
confirmed capture-lifetime defect let later backend splits overwrite the cast
mask. Marking it as an output retains it through harvest. The verification
oracle now rejects non-finite values explicitly instead of allowing NaN to be
hidden by a maximum-error reduction. No native attention math/kernel was changed.

Final-build 5050 qualification:

| Configuration | Oracle calls | Maximum absolute error | Cross-device stages |
|---|---:|---:|---:|
| 0.8B, four GPU layers, legacy | 60 | 0.0010069328 | 38 |
| 0.8B, four GPU layers, spread SSD | 51 | 0.0014899024 | 30 |
| 0.8B, all GPU, spread SSD | 51 | 1.4385125e-07 | 0 |
| 27B, 24 GPU layers, image + MTP snapshots + SSD | 135 | 0.00063167055 | 84 |

The independent dense FP64 reference decodes the represented Q/K and applies the
causal mask. The unchanged tolerances are 0.002 for quantized native matmul and
0.000002 for the CUDA page scorer. Both partial legacy and SSD tests explicitly
require that cross-device staging executes. The native partial-offload test also
passes exact KV/recurrent byte restoration at rows 65/95, missing-coverage
rejection and clean recovery. All-GPU sampling remains on the CUDA route.

The 27B fixture uses the matching Q8 projector, a 224x224 PNG, q8_0 KV,
context16384, B1024/R256, interval 512/count12/hot2, logical2048 MiB,
checkpoint Host1024 MiB and shared D2048 MiB. The image occupies native rows
[30,79); no retained checkpoint splits that group. Seven ordinary requests pass:
document/question, tail edit, switch away/back and middle/front edits. Question
reuse is 2906, tail/back 2910, middle 1615 and front 0 (safe full recomputation).
Switch-back and middle edit perform actual cold-image reads. The separate
question captures native Query [2910,2920) and answers the instrument probe.
The MTP response includes partial acceptance (5/6 draft tokens); no exact legacy
text matching is used as the correctness oracle.

The 27B current-forward staging copy is at most 3 MiB in this fixture. Charged
checkpoint Host peak is 945,732,680 bytes, below 1024 MiB; this excludes general
model/workspace/OS allocations and is not process RSS. Individual retained
records are about 149.9 MiB. Traced wall times are diagnostics, not performance
claims for 27B. The original GDN replay path still requires GPU recurrent layers;
partial offload with MTP uses the existing `--kvmem-mtp-state snapshots` option.
This qualifies image execution and checkpoint boundaries, not visual-answer
quality or every image shape/model/offload split.

Evidence: `llama-5050-27b-image-mtp-fixed`, `llama-5050-native-tail-partial`, and
the three final sampler fixtures above. Failed diagnostic iterations remain:
initial staging assumed host buffers carried device pointers (corrected), then
the retained-mask defect produced NaN until its lifetime was repaired. Temporary
intermediate-tensor diagnostics were removed before final-build qualification.

Failures retained in the evidence and their disposition:

- The first anonymous Chat Completions spread test let an unrelated short
  prompt consume the old long history through the new common-root checkpoint;
  switch-back fell to 29 tokens. Spread now requires anonymous continuation near
  the accepted input boundary. The regression verifies switch-away restores
  zero and switch-back restores 7079; deep anonymous edits start a new history.
  Chat Completions' `prompt_cache_key` is advisory, unlike the explicit binding
  on the existing Responses route.
- The first Responses SSD test fell back to full prefill on switch-back because
  newly added Host admission accessed an active continuation before it was bound.
  Admission now operates on the retained history's window. The corrected test
  proves actual cold-image reads and native restores at 7079, 2176 and 26.
- The local NInfer 0.8B artifact fails the unchanged native projection-geometry
  guard before checkpoint execution. The guard's source matches the prepared
  baseline. This fixture limitation was not changed; NInfer uses the supported
  local 4B artifact and is not compared across backends/models.
- The initial Query assertion missed a capture when a chunk crossed the Query
  end. It now verifies completed native scorer/replay rows as well as statistics
  lines. The first Responses request also used an unsupported thinking field;
  the harness now uses the existing `reasoning.effort: none` schema.
- An immediate next test was refused by the GPU occupancy check before launch.
  The harness now allows up to 15 seconds for the previous utilization sample
  to clear, and still refuses an occupied device.

## Controlled timing: NInfer

RTX 5050 Laptop throughout, Qwen3.5 4B Q4_K_M, int8 KV, no MTP, context 65536,
B=1024/R=256, chunk128, KV H=1024 MiB and two retained sessions. Spread uses
interval16384/count24/hot4/logical4096 MiB/checkpoint Host1024 MiB; SSD D=2048 MiB.
Each mode uses a fresh worker and the same seven-request sequence, with the
document sent first and the question separately. Three repetitions rotate the
legacy/RAM/SSD order. Compilation and diagnostic tracing/transfer verification
are disabled. Model startup is outside request timings; SSD reads use warm OS
file cache. These three samples do not provide a reliable per-case p95.

Median request wall time in seconds (all cases, including adverse results):

| Case | Legacy | Spread RAM | Spread SSD | RAM change | SSD change |
|---|---:|---:|---:|---:|---:|
| Document, ~53K tokens | 32.507 | 32.616 | 33.856 | +0.3% | +4.1% |
| Separate question | 0.401 | 0.459 | 0.548 | +14.5% | +36.6% |
| Tail edit | 0.328 | 0.322 | 0.385 | -2.1% | +17.3% |
| Switch away | 0.213 | 0.239 | 0.257 | +12.2% | +20.8% |
| Switch back | 0.342 | 0.372 | 0.547 | +8.7% | +59.6% |
| Middle edit | 32.382 | 22.452 | 23.158 | -30.7% | -28.5% |
| Front edit | 32.310 | 32.500 | 33.433 | +0.6% | +3.5% |

Median time to first text for document/question/middle edit is respectively
32.415/0.274/32.287 seconds in legacy, 32.526/0.322/22.353 in RAM, and
33.772/0.418/23.071 in SSD. Full samples, min/max and every case's first-text
timing are in `evidence/benchmark-ninfer-5050/summary.json` in the workspace
delivery directory. Small changes are observations, not dismissed as noise.
The middle-edit gain is a new retained-prefix capability (restore 16512 instead
of full recomputation), not a faster equivalent legacy edit algorithm. The
separate diagnostic 53K test verifies that actual native restore and nine-row
Query capture; API cache counts alone are not used as proof of an SSD hit.

The measurements support keeping spread opt-in. They do not establish gains
for all requests, cold OS-cache SSD, other models or other GPUs.

## Controlled timing: llama, sampler build before nearby thinning

RTX 5050 Laptop, Qwen3.5 0.8B Q8_0, q8_0 KV, all GPU, no MTP,
context65536, B1024/R256, chunk128, KV H512 MiB, two sessions. Spread uses
interval16384/count24/hot4/logical4096 MiB/checkpoint Host1024 MiB and D2048 MiB.
Three rotated repetitions use fresh workers, ordinary document-first then
separate-question requests, complete SSE capture, diagnostics off and no
concurrent compilation. All nine runs record the same final executable SHA256.
These replace the earlier pre-repair llama measurements for delivery claims;
the earlier `benchmark-llama-5050` files are retained as historical diagnostics.
The NInfer comparison above uses a different model and is not a cross-backend
performance comparison. SSD reads use warm OS cache; startup is excluded.

This is a shared workstation, not an exclusively reserved CPU/storage host.
Post-run inspection found another CPU unittest process starting at 16:08:13,
before the final RAM repetition finished at 16:08:31 (Asia/Shanghai). Its impact
was not isolated or measured. The subsequent other GPU test started at 16:09:05,
after these measurements. The complete observations are retained; small changes
are not dismissed as noise, and these timings do not establish isolated-host
performance. See `evidence/benchmark-host-activity.json` for the recorded times.

Median request wall time in seconds:

| Case | Legacy | Spread RAM | Spread SSD | RAM change | SSD change |
|---|---:|---:|---:|---:|---:|
| Document, ~53K tokens | 9.591 | 9.632 | 9.608 | +0.4% | +0.2% |
| Separate question | 0.561 | 0.573 | 0.580 | +2.1% | +3.4% |
| Tail edit | 0.409 | 0.375 | 0.366 | -8.3% | -10.5% |
| Switch away | 0.101 | 0.125 | 0.109 | +23.5% | +7.9% |
| Switch back | 0.474 | 0.417 | 0.419 | -12.1% | -11.5% |
| Middle edit | 9.321 | 6.527 | 6.545 | -30.0% | -29.8% |
| Front edit | 9.258 | 9.318 | 9.336 | +0.7% | +0.8% |

Median first-text times for document/question/middle edit, in seconds: legacy 9.533/0.463/9.258; spread-ram 9.574/0.478/6.465; spread-disk 9.549/0.484/6.486; full samples, min/max, first-text timing for every case and reuse counts are
in `evidence/benchmark-llama-5050-final/summary.json`. Three samples do not support
a reliable p95. Middle-edit gains come from retained-prefix reuse rather than
an equivalent optimized legacy edit path. All costs and slowdowns are retained
in the table; the policy remains opt-in.

## Completion and limits

Both builds, planned Host/native/HTTP storage checks, MTP/tool/cancellation cases,
53K coverage, the authorized partial-offload repair and final controlled timing
are complete. Default policy remains legacy. No commit, deployment or change to
the frozen original-master baseline has been made. Final exports and their
verification are recorded in the delivery directory's `export-validation.json`.

The instrument probes are limited, correlated checks, not broad quality tests.
No interval sweep, reliable p95, cold-OS-cache SSD benchmark or complete live-file
corruption campaign is claimed. The local NInfer 0.8B artifact still fails its
unchanged geometry guard, so NInfer qualification uses 4B. General model/workspace
memory is outside checkpoint Host accounting. The original MTP replay restriction
on CPU recurrent layers remains explicit; snapshots is the tested partial path.
The final llama RAM repetition overlaps another CPU unit-test process; its
effect is unmeasured, so the performance results carry that shared-host limit.

A previous GPU instruction conflict caused automatic approval review to reject
a 5060 test before launch. The user resolved it by selecting the 5050. Earlier
5060 diagnostics are kept separate from all controlled timing reported here.

## Nearby-point thinning: implementation and paired comparison

The optional rule skips constructing an ordinary coverage/reuse-anchor image
when a valid earlier archive point is less than MIN_GAP tokens away. The check
uses existing native prefix/state proofs and archive metadata; it does not read
SSD images. Historical pruning retains the earlier ordinary representative and
one latest endpoint. Required execution snapshots remain exact. Same-frontier
replacement still uses a newly published immutable payload, not position-based
aliasing. Comparisons use actual retained representatives rather than chains of
skipped neighbors. This adds one scalar and bounded directory scans, with no
cost estimator, adaptive controller or background policy loop.

The initial experimental value was 64. Complete paired results do not support
enabling it generally. The delivered startup default is 0 (disabled), including
for spread; `KVMEM_CHECKPOINT_MIN_GAP=64` explicitly enables the tested rule.
Legacy remains the overall default. No commit or deployment was made.

The before arm uses binaries frozen immediately before this optimization,
including the already-qualified sampler/offload fixes. It is the previous
spread implementation, not legacy. Both backends use RTX 5050 Laptop UUID
GPU-B throughout, with pre-launch occupancy
checks, one lane, no MTP, ~53K document first and a separate question, context
65536, chunk128, B1024/R256, interval16384/count24/hot4/logical4096 MiB and
checkpoint Host1024 MiB. Native KV H is 512 MiB for llama and 1024 MiB for NInfer;
SSD D is 2048 MiB. Llama uses 0.8B Q8_0/q8_0 KV; NInfer uses 4B Q4_K_M/int8 KV.
Each backend is compared only with its own same-model baseline.

Three repetitions change the arm order: before-RAM/after-RAM/before-SSD/after-SSD,
reverse order, then SSD pair/RAM pair. Every arm starts a fresh worker. Complete
SSE responses, first-text timings and all samples are saved. This task launches no compilation during timings, and
diagnostic tracing is off; startup is excluded; OS file cache is warm. The CPU
and storage of this shared workstation are not reserved exclusively. Three
samples do not estimate a reliable p95. Small changes remain observations and
are not dismissed as noise. Output token counts match across arms in every case.

Median request wall time, seconds, with 64 enabled versus the frozen before arm:

llama:

| Case | Before RAM | After RAM | Change | Before SSD | After SSD | Change |
|---|---:|---:|---:|---:|---:|---:|
| Document | 9.912 | 10.061 | +1.5% | 9.946 | 9.927 | -0.2% |
| Separate question | 0.733 | 0.728 | -0.8% | 0.686 | 0.673 | -1.9% |
| Tail edit | 0.491 | 0.554 | +12.8% | 0.495 | 0.534 | +7.9% |
| Switch away | 0.127 | 0.133 | +4.5% | 0.117 | 0.132 | +12.8% |
| Switch back | 0.506 | 0.616 | +21.8% | 0.559 | 0.550 | -1.6% |
| Middle edit | 6.774 | 6.831 | +0.8% | 6.946 | 6.777 | -2.4% |
| Front edit | 9.798 | 9.913 | +1.2% | 9.705 | 9.756 | +0.5% |

ninfer:

| Case | Before RAM | After RAM | Change | Before SSD | After SSD | Change |
|---|---:|---:|---:|---:|---:|---:|
| Document | 32.245 | 32.789 | +1.7% | 33.385 | 33.274 | -0.3% |
| Separate question | 0.443 | 0.423 | -4.6% | 0.516 | 0.424 | -17.8% |
| Tail edit | 0.312 | 0.377 | +20.9% | 0.366 | 0.383 | +4.8% |
| Switch away | 0.227 | 0.221 | -2.8% | 0.249 | 0.260 | +4.5% |
| Switch back | 0.366 | 0.429 | +17.2% | 0.492 | 0.574 | +16.8% |
| Middle edit | 22.000 | 22.108 | +0.5% | 22.804 | 22.851 | +0.2% |
| Front edit | 31.983 | 32.036 | +0.2% | 32.924 | 32.949 | +0.1% |

All per-case first-text values, min/max and the three raw samples are in
`nearby-20261025/{llama,ninfer}-timing/summary.json` under the delivery directory.
For example, NInfer SSD separate-question first text improves 0.392->0.287 s,
but SSD switch-back worsens 0.362->0.437 s. Llama RAM switch-back first text
worsens 0.399->0.502 s. The wall changes are not explained by output-length
differences. Their complete end-to-end attribution was not individually profiled.

The 53K middle-edit reuse stays 16410 for llama and 16512 for NInfer in all arms.
Tail-edit/switch-back reuse moves 52844->52832 for llama and 52839->52829 for
NInfer, adding 12 or 10 prefix tokens. Correctness does not imply a cheaper next
request: separate 7K traces show that the earlier restore can also lose a
same-frontier state-sharing opportunity and rebuild exact execution snapshots.
That mechanism is observed; not every millisecond of the 53K differences is
attributed to it.

Independent 7K diagnostics use interval 512/count24/hot4. They are traced and
excluded from timing conclusions. The following metrics have different native
definitions and must not be compared across backends:

| Diagnostic metric, seven requests | Before | Enabled64 |
|---|---:|---:|
| Llama maximum retained directory | 20 | 18 |
| Llama prefill native save calls | 52 | 53 |
| Llama published checkpoint payload, MiB | 674.31 | 655.05 |
| NInfer maximum retained directory, SSD | 20 | 17 |
| NInfer all native captures, RAM | 44 | 57 |
| NInfer all native captures, SSD | 57 | 57 |
| NInfer published active-coverage payload, MiB | 1670.42 | 1523.03 |

The unoptimized 7K NInfer RAM arm skips 14 optional captures under Host pressure,
whereas the enabled arm skips none under that pressure: thinning releases room
for additional optional saves. Thus a smaller directory does not imply fewer
total captures. NInfer RAM native state-copy volume increases 2161.71->2800.40 MiB;
SSD stays 2800.40 MiB. Llama's recorded prefill save count excludes generation-end
captures; NInfer's native count includes all captures. Active-coverage write
counts do not represent all NInfer idle/storage writes. These charges and
transfer counts are not a process-RSS measurement. Write counters describe
successfully published cold-image payloads; they exclude discarded writer work
and framing and do not measure physical OS/device I/O. The 512-interval diagnostic
is not substituted for the 16384-interval performance comparison.

Qualification includes the Host 23/23 suite and spacing/endpoint/identity/rewind
boundaries; native 10-row llama and 9-row NInfer Query capture; actual SSD restores;
4B MTP tool continuation with resident-only append and no Query replay; llama
MTP cancellation/switching; low Host-budget skips; and NInfer cancellation,
resume, clear, cleanup and no import in two fresh storage processes. The 27B
24-layer partial-offload image+MTP snapshots+SSD fixture passes 132 independent
FP64 oracle calls (max absolute 0.00063167055, unchanged CPU tolerance 0.002),
with no checkpoint splitting the image. All 24 timed question probes pass;
they are limited correlated checks, not broad quality qualification.

The final build changes only the startup default from 64 to 0 after measurement;
explicit 64 retains the measured algorithm. Exact measured binaries are preserved
under `nearby-20261025/bin-tested64`, and frozen before binaries under
`bin-before`, with hashes in the timing summaries. Both delivered default-off
and explicit 64 HTTP paths are requalified on the rebuilt final binaries.

Failed attempts are retained: the initial Host spill-lifecycle test hit sandbox
Temp permissions and passed using the workspace Temp; the first small diagnostic
incorrectly paired a 7K document with 16K spacing and therefore had no middle
coverage, corrected to 512 for that diagnostic only. No performance arm is
discarded because it is slow. No threshold sweep or stronger speedup claim is
made. The decision is to keep nearby thinning optional and default off.

## Audit correction: NInfer checkpoint metadata admission

The subsequent audit found that `metadata_bytes()` omitted ResidentPrefixIdentity
allocations. A native CPU reproduction copied a 53,000-token identity and truncated
it to 26 tokens; its four token/position vectors still owned 689,000 bytes. Those
bytes were previously absent from both Host admission and retained logical bytes.
Resident page IDs were also populated after the old admission check. Earlier pool
and budget tests did not establish complete metadata accounting.

The correction counts vector capacity, including nested VisionItem timestamps and
spans, rewrite frontiers, token/position identity, mean state and resident page IDs.
It also conservatively charges the checkpoint object and backend/layout identity.
Capture computes its bound from source capacities before copying attachments or
allocating the native image, and checks completed metadata against that bound.
The native accepted-prefix and restore rules, nearby spacing and startup defaults
are unchanged. This is capacity accounting, not a process-RSS measurement; shared
graph/history metadata remains conservatively charged per checkpoint. More accurate
charging can reject optional captures that an undercounted build admitted.

Evidence is in `checkpoint-policy-20261025/audit-20261025`:

- The expanded native checkpoint pool test passes its CPU-only metadata cases:
  long-prefix truncation, nested media identity, spare resident-page capacity,
  exact logical-byte admission and rejection one byte below the required size.
  The native pinned-pool lease/budget cases also pass on the RTX 5050 Laptop.
- Final NInfer 4B Q4_K_M/int8, RTX 5050 Laptop UUID
  `GPU-B`: spread SSD/MTP/min-gap64 passes eight
  document-first, question, tool-continuation, edit and session-switch requests.
  Actual cold reads and the native Query span are verified. All 60 captures fit
  their metadata admission; highest logged Host charge including the reserved SSD
  read image is 207,846,200 bytes against a 1,073,741,824-byte budget.
- RAM/default-gap0/Host128MiB passes seven requests and exercises optional skips.
  All 16 captures fit; highest logged charge is 103,371,190 bytes against
  134,217,728. Switch-back and middle/front edits recompute from zero under this
  small budget. Both question recall checks pass; these are limited probes.
- The relinked native SSD/MTP lifecycle test passes cancellation, resume, clear,
  subsequent requests, startup without imported state and cleanup preserving an
  unrelated file. No GPU math/kernel was changed. These traced runs are correctness
  checks, not a new performance comparison or broad quality qualification.

Build caveat: sandboxed Ninja regeneration failed with a permission error and the
existing incremental command index was unavailable. The ensuing full CUDA rebuild
was stopped. Using CMake's exact compile/link commands, 79 affected C++ units were
rebuilt with eight jobs, and the runtime library, service, query-policy test bundle
and SSD lifecycle test were linked against unchanged existing CUDA objects/libraries.
Commands and logs are retained in the evidence directory. The new CPU test is also
registered as `ninfer_kvmem_checkpoint_metadata_test` for subsequent CMake builds.
The previous measured performance binaries remain preserved; the corrected delivered
binary has not been rebenchmarked. Default policy remains legacy and min-gap remains
zero. No commit or deployment was performed during qualification.
