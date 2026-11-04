# Checkpoint coverage and RAM/SSD placement

The experimental `spread` policy adds historical coverage to the llama.cpp and
NInfer Qwen3.5 KVMem paths. The default remains `legacy`. PolyStrata keeps its own
native checkpoint policy. This feature does not add a request mode or persistence
across process restarts.

## Content snapshots across backends

Full-context requests use their evaluated token/media/position prefixes to find
native checkpoints. `kvmem.conversation_id`, `prompt_cache_key`, the SDK spelling
`promptCacheKey`, and Radiance's legacy `session_id` do not own a mutable cache
slot. Giving a parent and child the same key does not authorize replacement,
serialize them, or suppress either publication. Missing, different and shared
keys use the same content policy. Existing field/alias validation is retained
for protocol compatibility; display labels are independent of native identity.
Explicit stateful session handles and internal incarnation/generation stamps
still have their separate lifecycle contracts.

`kvmem/content_cache.hpp` supplies shared candidate ordering, publication and
capacity-group policies, plus an optional typed `ContentSnapshotDirectory`.
Native catalogs can call those policies without changing their payload formats.
The typed directory holds immutable objects through reader references, performs
native proof callbacks outside its metadata lock, and commits prepared metadata
atomically. Its returned retired objects must be released at the backend's safe
boundary or through a native deferred deleter. It is not a second allocator.
The current adapters retain native catalog wrappers; using the common policy
does not mean that every adapter uses the typed directory's hash index.

| Backend | Source and independent branch | Native boundaries |
|---|---|---|
| NInfer Qwen3.5 KVMem | Content-selected retained histories; sealed archive references, copied checkpoint wrappers and private selectors/Query/rollback | `legacy` and `spread`; exact recurrent checkpoint and original positions; GPU working sets stay per lane |
| llama.cpp KVMem | Exact checkpoints and source-preserving bundles; private execution branch with shared sealed packed KV/state where supported | Legacy pinned CPU arenas require independent arena/table copies; they receive no shared-payload credit |
| PolyStrata KVMem | Content-selected accepted checkpoints; preserved parked source and independent branch metadata | Solo MTP, batch and yielded prefill retain their native accepted-state rules; CUDA/HIP qualification is separate |
| Radiance KVMem | Content-selected rank-prefix snapshots with native leases and private memory policy | Capture, restore and destruction obey scheduler/rank safe boundaries; CPU tests do not qualify AMD or multi-GPU execution |

Snapshot reuse is confined to one compatible loaded backend/model instance.
The proof must cover model/layout, exact tokens and media, original positions,
accepted frontier, KV/statistics coverage and native recurrent/draft state.
Hashes and LCP are candidate filters. An arbitrary LCP is not a restorable
recurrent state. Equal token prefixes may have different sparse/replayed native
states and must remain separate unless the adapter proves equivalence.

A new user Query can discover a compatible content prefix. Tool continuation
requires the native continuation proof and then uses append plus resident-only
Pressure; it does not acquire general prefix discovery or cached-old-Q cold
history reselection. Reading cold bytes needed to restore a proven checkpoint
is separate from selecting cold history with a Query. No ingestion mode or new
document-ingestion endpoint is introduced.

Restoring does not consume the retained source. Sealed Host/SSD payloads and
read-only native images may share allocations; selectors, MeanK updates, LRFU,
Query buffers, coverage wrappers, graph feedback, rollback and live recurrent
state are private to the executing branch. A later write detaches a shared
tail. Independent copies of equal bytes are charged independently. A resident
execution binding may be reused exclusively after native validation, while a
recoverable source snapshot remains available to another request.

Only complete accepted states may publish. Publication order is recency for
retention, not a per-client winner rule: an older-started request may publish
after a newer-started sibling. Incomplete captures and failed preparation leave
the old source valid. A request that disables cache use must not erase other
snapshots by matching its client key. Explicit management/reset operations keep
their own protocol meaning rather than being inferred from key equality.

Native group IDs only allocate retention fairly. They are not client identities,
candidate filters, or permission to overwrite a previous version. The common
group planner limits both entries and groups, uses the most recent access in a
group for group LRU, and first recycles old entries from groups with multiple
versions when the physical directory is full. A new group exceeding the group
limit can evict an idle group as a capacity event. Pinned native bindings remain
protected. Unknown group IDs do not merge anonymous requests into one group.
Group classification uses a complete, accepted canonical input boundary with
exact tokens, media and positions. A short template-tail continuation hint does
not prove that two different user questions belong to one resource group.
Publication rechecks the group's completed inputs: each must lie on the same
exact prefix path as the incoming input. Among compatible groups it chooses the
deepest match; otherwise it creates an internal group. This also separates
siblings that prepared concurrently from an older common ancestor. Unknown or
incompletely accepted canonical boundaries do not authorize inheritance.
The four adapters use the common group retention policy while retaining native
admission and configured physical-entry limits. Checkpoint count/byte limits
still apply per native history version, not as a new aggregate per-client quota.
Native KV/checkpoint pressure also uses the common idle-entry eviction order:
extra versions yield before singleton groups, while active restore sources and
other native pins stay protected. Physical pressure may evict a snapshot before
the directory count is full; an outstanding lease can keep its bytes charged.

Removing a covered version requires a native dominance proof, not a larger
frontier. `content_snapshot_covers` checks nonempty, nonzero immutable-origin
sets; the adapter must also preserve all relevant KV/position/state/policy
inputs. NInfer additionally checks archive allocation identities before merging
covered groups. A copied or rewritten tail conservatively prevents that merge.
Ordinary LRU capacity eviction remains possible even without a dominance proof.

Capacity admission uses the existing physical H/D account plus new growth and
outstanding reservations. A deferred content reference keeps bytes alive without
pinning their current RAM tier; actual readers take short leases. A directory
eviction releases only its reference, and physical quota returns after the last
reader/branch releases the allocation. Checkpoint pools, private metadata,
temporary transfers, model weights, RSS and GPU VRAM have distinct scopes. The
typed directory's metadata limit is a retention bound, not a total RSS limit.

The llama legacy CPU arena is copied into an independently allocated destination
arena, including its private tier table; it is not a sealed shared payload.
Before creating or forking a store, the lane catalog reserves one configured
arena commitment. The cap is `(snapshot_slots + lanes) * cpu_bytes`, charged for
active, parked and in-flight stores; failure refunds the reservation and keeps
the source available. This conservative ledger is reported separately from raw
H/D bytes and RSS. Copying into the already allocated destination avoids a third
temporary arena. Native recurrent checkpoint pools also keep their own limits:
retaining a source while executing a private branch can need more checkpoint
space than the former consuming restore, and insufficient space can still evict
an idle source or require prefill.
PolyStrata and Radiance retain shared immutable canonical token/media metadata
with a Host reservation taken before allocation. That reservation lasts until
the last metadata reference is destroyed. Their optional CPU identity capture
can decline to an unknown group after a storage/allocation failure, recording
the error; it does not turn a new fairness hint into a required I/O dependency.
Required native KV/checkpoint operations and device failures keep their existing
error semantics. Directory-level retained-byte totals may conservatively count
a shared metadata object more than once; those totals are distinct from the
physical reservation and RSS.

The monitor retains `planned_cached_tokens`, `actual_cached_tokens` and the
legacy native-path labels `own`, `shared`, `miss`. These labels must not be used
as evidence of client ownership or physical byte deduplication. Missing native
observations stay absent/null. NInfer's legacy `cached_tokens` is an admission
quote and may remain positive when an optional restore falls back to prefill;
the independent actual observation reports the completed native boundary. A
created branch is not an actual cache hit until native restore succeeds.
PolyStrata credits the first successful restore per HTTP request, not input
already computed before an internal yield. Those counters are not net compute
saved across all later replay.

Revision-2 observations add `cache_semantics_revision=2` and `restore_path`
(`continuation`, `snapshot`, `miss`). Continuation describes the native content
path; it does not promise that GPU state stayed resident. Older workers may omit
these fields, which must remain unknown rather than being synthesized as a miss.
NInfer publishes native actual reuse at completion and in streamed prompt
progress when available. Its aggregate consumer has no live progress contract;
an active HTTP request or an admission quote alone does not prove a restore.

Radiance retains deep input and conservative pre-assistant points only at legal
native boundaries. Its sharing counters deduplicate actual sealed allocation
identities, not equal contents; pending parking retains its reservation until
complete. Extra prefill cuts can change accumulation, so text equality alone
does not establish native correctness.

This policy is process-local. It does not transfer KV between backend formats,
persist caches across restart, or promise shared physical GPU pages. Ordered
application conversations must still serialize dependent submissions. Historical
GPU reports qualify their recorded binaries and settings; they do not qualify
the new content-cache implementation or untested hardware combinations.

## Configuration

Set these environment variables before starting the worker:

| Variable | Default in spread mode | Meaning |
|---|---:|---|
| `KVMEM_CHECKPOINT_POLICY` | `legacy` | `legacy` or `spread` |
| `KVMEM_CHECKPOINT_INTERVAL` | 16384 | Approximate token spacing of optional captures |
| `KVMEM_CHECKPOINT_MIN_GAP` | 0 | Minimum spacing of ordinary retained points; 0 disables nearby thinning; 64 is the tested experimental setting |
| `KVMEM_CHECKPOINT_COUNT` | 24 | Retained coverage records per native history version, 1–256 |
| `KVMEM_CHECKPOINT_MIB` | 4096 | Logical retained payload and native metadata budget per native history version |
| `KVMEM_CHECKPOINT_HOT` | min(4, count) | Target number of images in RAM; at most count |
| `KVMEM_CHECKPOINT_HOST_MIB` | 1024 | Checkpoint Host budget per lane |

Numeric values in the table must be positive except MIN_GAP, which also accepts zero.
The table's extra spread variables are ignored in legacy mode. These defaults are
experimental starting points, not tuned recommendations.
The interval is not a bound on recomputation after pruning or skipped captures.

### NInfer checkpoint recompute limit

NInfer Qwen3.5 adds `KVMEM_CHECKPOINT_EXTRA_RECOMPUTE_TOKENS` in both `legacy`
and `spread` modes. Its range is 0–2147483647; the default 0 disables this
optimization. The console's NInfer-only **Checkpoint extra recompute limit**
(`CheckpointExtraRecomputeTokens` in saved JSON) and Windows
`-CheckpointExtraRecomputeTokens` set the same worker environment variable.
The launcher writes an explicit 0 when omitted, so an inherited environment
cannot silently enable the feature. Linux exports preserve that explicit value.
A positive setting requires a worker reporting `checkpoint_recompute_revision`
at least 1. The value 16 remains an experimental candidate, not a tuned default.

The limit permits replacing a low-value checkpoint with an already complete,
legal earlier state at most that many tokens away. For an ordinary prompt-tail
capture, the decision happens before its extra prefill split, allocation and
state copy; the earlier point must already cover the current full Query.
An actual Query replay checkpoint still captures the exact pre-Query state and
remains protected while executing. On the successful request-finish path it may
participate in nearby cleanup; its substitute can be before the Query. The
`query_replayed` flag is not a completion signal. Failed or cancelled requests
retain their valid recovery states without this nearby cleanup. Cleanup compares against
points that will actually remain, so a chain of deleted neighbors cannot silently
increase the distance. A newer output endpoint does not replace an input-side
point beyond a repeated request's legal restore frontier.

This limit bounds extra tokens for the nearby decision, not latency, total future
recomputation or memory. A later Query retry may need to recapture the exact
pre-Query image; a cold substitute may require SSD reads. Budget eviction, edited
input and read failures can require more recomputation. Native prefix, state,
session and KV proofs still govern every restore. Active restore/transfer leases
and shared owners delay physical release; an idle pool can retain a freed buffer.

Keep `KVMEM_CHECKPOINT_MIN_GAP=0` when enabling this NInfer setting. Launchers reject
the two positive settings together. A directly launched worker rejects simultaneous
active thresholds in spread mode; in legacy mode the old MIN_GAP setting is ignored.
Neither path applies both thinning rules. Query replay's
`--kvmem-query-replay-max-tokens` remains an independent admission cap. Existing
count, bytes and hot/Host budgets retain their meanings. This setting does not
enable spread mode or change llama.cpp/PolyStrata checkpoint behavior.

Use the existing `--kvmem-host-mib`, `--kvmem-disk-mib` and `--kvmem-disk-path`
options for native KV storage and SSD capacity. Checkpoints and KV share D;
checkpoint payloads do not get a second disk allowance. H remains a KV budget.
See [active KV tiering](active-kv-tiering.md) for quota and file lifetime rules.

The checkpoint Host account includes live native images, captured attachments,
read/capture buffers, outstanding writer leases and idle pooled images. NInfer
uses one worker account of `HOST_MIB × max_concurrency`, including a reserved
single-image read buffer when SSD is enabled; its idle pinned pool retains at
most one image. Llama charges each image to the lane account that created it,
including after conversation migration. Neither limit is a process RSS limit:
GPU state, model/workspace, allocator overhead, unrelated metadata and OS page
cache have separate lifetimes.
Native image bytes are exact. Aliases of one checkpoint payload are charged
once, while metadata shared by distinct snapshots is charged conservatively
per snapshot; the checkpoint counter is not an exclusive-allocation RSS meter.
NInfer charges retained vector capacity, including token types, all three
position axes, nested media identity, rewrite frontiers and resident page IDs.
Truncating an identity does not release its capacity. Capture admits a conservative
bound from the source capacities before copying metadata or allocating the native
image; the completed metadata must fit that admission. Small Host budgets can
therefore skip more optional captures than builds that omitted these allocations.

## Capture, retention and selection

The capture planner uses the structured system-prefix boundary, observed LCP
anchor, periodic completed prefill chunks, the boundary before the last mutable
message, and native committed endpoints. Exact semantic boundaries can split a
text batch. Periodic captures follow actual completed chunks. A checkpoint never
splits an indivisible media group or names a frontier not yet evaluated. Optional
long-generation sampling is not enabled in this version.

### NInfer input reuse and template-safe fallback

NInfer Qwen3.5 KVMem distinguishes two input-side uses. The existing template
rewrite boundary remains the conservative fallback for a later real-user turn:
adding that user can change how earlier assistant reasoning is rendered. The
existing `memory_input_end` identifies the end of the accepted input before the
current generation opener. The separate `InputReuse` role can preserve a complete
state there for deep reuse when the actual next request only appends assistant/tool
history. It does not reuse the `PromptTail` role or its Query-retirement rules.
At T=0, an exact Query image can therefore remain alongside `InputReuse` where
the previous ordinary-tail path could retire it; this repair does not promise
unchanged peak or retained checkpoint memory.
Capturing the deeper state does not overwrite the conservative fallback merely
because it is newer. It does not change the template, request fields or non-KVMem
execution.

The deeper point is considered only when
`0 < rewrite_frontier < memory_input_end < prompt_tokens`, and is captured only
at an evaluated, legal boundary covering the full current Query, if any. For
these requests it replaces the ordinary `prompt_tokens - 1` capture target;
it does not add that target as another copy. If its frontier already has a valid complete
state, the roles share that image rather than copying it again. The NInfer
extra-recompute limit uses the same actual-state and shared-image rules before
an optional nearby capture; a planned point or a failed capture is not a
substitute. For the ordinary 499-token example, `memory_input_end` and the
existing InputFallback are both at 492, so this distinction adds no capture and
leaves the existing policy intact: T=0 retains the ordinary 498 tail, while
T=16 may use the completed 492 state under its usual skip rules.
When the conservative point is 34 and the deep input ends much later, keeping
both can require another full recurrent-state image and prefill split. That
cost is intentional and is separate from the benefit of a later deep restore.

The next request still selects the deepest candidate that passes its actual
token/position/media prefix, accepted-state, KV and session checks. A tool
continuation may match the deeper input; a new user that changes the template
must reject that candidate and can use the conservative point or an earlier
legal one. Neither the label nor the message type authorizes reuse. Ordinary
tool continuation remains append plus resident-only Pressure, without a new
Query recall. Existing retention and resource budgets still apply.

This input-boundary repair was authorized and directly validated on 2026-10-31
with Qwen3.5-4B-MTP-Q4_K_M, INT8 KV, FP16 GDN, MTP4 and one RTX 5060 Ti.
The unchanged `input-fallback` fixture passes at T=0 and T=16: the first direct
tool append restores the true input boundary 1447 instead of 34. The earlier
1450 estimate and the observed LCP 1451 are not that input boundary. The expanded
native test also passes repeat, tool continuation and cancellation recovery;
a new real-user turn has LCP 37 and correctly restores fallback 34.
Evidence: [original fixture T0](../../input-reuse-20261031/original-fixture-T0/results.txt),
[T16](../../input-reuse-20261031/original-fixture-T16/results.txt), and
[native MTP report](../../input-reuse-20261031/native-mtp/report.json).

The [paired performance report](../../input-reuse-20261031/perf-4b/report.json)
compares the preceding checkpoint-cleanup worker with this repair on the same
card and settings: one warmup excluded, three measured cold-input/tool pairs
per profile, 96 generated tokens per request. Median times in milliseconds:

| T | Cold prefill, old → repaired | Direct tool native TTFT, old → repaired | Two-request wall time, old → repaired |
|---:|---:|---:|---:|
| 0 | 557.0 → 582.9 (+4.65%) | 661.3 → 195.0 | 2249 → 1813 |
| 16 | 552.2 → 576.1 (+4.32%) | 654.6 → 192.8 | 2219 → 1781 |

The extra capture increases cold-prefill time while preserving 1447 tokens for
the immediate tool request. These are 4B single-card, small-sample results for
fixed synthetic assistant/tool history with the generated answer discarded;
they do not establish a 27B or general throughput gain.

The completed repair regression set also passes T=0 and T=16 for
[ordinary decode](../../input-reuse-20261031/native-ordinary/report.json) and
[long Query, MTP, shared reuse and SSD](../../input-reuse-20261031/query-mtp-ssd/report.json),
including cancellation recovery and an actual cold read of the InputReuse image
at frontier 6443. The [27B single-card short-prompt check](../../input-reuse-20261031/short-27b-single/report.json)
confirms that the ordinary 499-token case adds no InputReuse capture: T=0 keeps
492 and 498, while T=16 skips 498 using 492. This 27B check establishes capture
and restore behavior, not TP correctness or performance. These runs use native
transfer verification for restore claims; they do not inject native allocation
or copy failures. Host contracts separately cover incomplete-source rejection
and publication ownership. The earlier checkpoint-cleanup package remains a
separate baseline, not evidence for this repair.

### Nearby coverage and retention

Before constructing optional coverage or reuse-anchor images, a metadata-only
lookup checks for a valid earlier archive point less than MIN_GAP tokens away.
If found, capture is skipped; restoring that earlier point later performs the
extra prefill normally. Backend prefix/state proofs still govern reuse. No SSD
image is loaded for this check, and equal frontiers never prove state identity.

The historical directory also thins close points even below capacity: retain the
earlier representative and one latest endpoint. Older endpoints follow ordinary
spacing when a new endpoint arrives. Comparisons use actual retained ordinary
points, so skipped neighbors cannot move the representative forward indefinitely.
Required Query/replay, rollback and MTP snapshots remain exact execution leases.
When those leases end, their images compete under the same historical spacing.
MIN_GAP=0 disables this MIN_GAP-based capture gate and directory thinning; it does
not disable NInfer's separate extra-recompute setting described above. This
simple rule has no cost predictor, usage estimator or background control loop.
It defaults off: the 64-token experiment reduced some retention/write work but
regressed several tail-edit and switch-back requests. The validation record
reports the complete paired results. Enabling it trades retained images for
extra prefill and can also require rebuilding exact execution snapshots.

Native Query replay, reconstruction and cancellation snapshots remain execution
leases. Optional captures reserve room before allocating their native image and
skip when there is no room. Required captures first reclaim idle buffers, demote
eligible images, or evict optional retention; if admission still fails, they
report capacity exhaustion before allocating the image. Required snapshots are
never silently replaced by an approximate periodic point.
When retained branches consume the creator lane's remaining Host quota, required
admission can evict an older idle directory entry. Busy, switching, queued and
resident native bindings remain protected. A request whose reusable entry was
evicted recomputes its prefix. A request rejected after prefill began uses its existing rollback lease.

After a turn, reusable execution snapshots enter the bounded archive and release
their execution slots. Shared payload identities are counted once. Within a branch-local coverage catalog, a new payload
at the same frontier may replace that catalog's old version. Across request
branches, equal frontiers neither replace entries nor alias recurrent or
Query-replay state.

Retention softly protects the root, latest anchor and newest frontier. Among
other records, remove the smallest
`(next_frontier - previous_frontier) / max(interval, newest - previous_frontier)`;
ties prefer the least recently used record. This keeps denser recent coverage
and coarser old coverage. Under tiny budgets the soft protections yield to the
count/byte limits. For example, two recent retained points can both lie beyond a
future prompt's legal restore frontier, requiring a full prefill.

Selection considers metadata from RAM and SSD together, chooses the deepest
legal frontier and preserves native prefix, accepted-token, media, KV-coverage,
recurrent, MTP, mean-K and retrieval-history checks. It does not read every image
to find a candidate. Restoring a prefix discards the edited suffix only
from the new private execution branch; it leaves retained source snapshots valid.

Full-context Chat Completions and non-stateful Responses use content selection,
including for edits, regenerate and rollback. Cache keys do not select a single
mutable lineage in either checkpoint policy. OpenCode parent/child sessions may
share a key or omit it; no client-derived key or parent/fork protocol is needed.
This does not remove explicit server-side response/session handles from actual
stateful protocols.

The bounded native directories, digests and placement metadata remain in RAM.
After shortlist ordering, each adapter still verifies its exact native restore
proof before use. NInfer bounds the ranked restore candidates; a filtered digest
does not prove state completeness. The common typed directory can perform native
validation before limiting returned candidates, while native adapters retain
their own bounded validation loops. Neither path reads all SSD images to search.

The executing branch owns mutable selectors/statistics and checkpoint wrappers.
Sealed KV and immutable images retain their existing physical accounts. Once
other branches/readers release a page, a sole writer may update it without a
second copy. Holding a deferred source or active reader requires COW until that
reference ends. RAM/SSD capacity limits apply after publication regardless of
whether client keys are present.

After capacity checks, a restore binds the source generation and reads only the
selected cold image and required KV pages. Workers cannot mutate the source or
publish engine state. Cancellation releases unpublished branch references after
running work drains; preparation failure follows the native clean-prefill or
rollback path. Native restore success, rather than the admission quote, records
actual prefix reuse.

Ordinary tool continuation still appends with resident-only Pressure. A saved Q
does not authorize cold-history recall. Checkpoint policy does not change the
Query/Pressure algorithm or add document-ingestion protocol fields.

## SSD publication and failure

One checkpoint writer per owning llama lane, or one for the NInfer Program,
borrows a retained immutable source. It reserves D, writes native segments with
identity, sizes and checksum, and publishes only after the engine revalidates
the owner and source. The worker does not call CUDA or mutate the directory.
Cancelled, replaced or failed writes keep a valid source until the task drains;
unpublished destinations release their quota. Cold records share the existing
process-local spill-file lifetime and cannot be restored after a restart.

Before a cold restore, reserve Host space and validate the complete image before
publishing its bytes to native state. Llama can reject an unreadable candidate
and try an earlier legal point. NInfer's already-admitted materialization
transaction falls back to a clean full prefill on an image-read failure; it does
not change frontier midway through that transaction. Partial native restore
failure follows the backend's clean reset/rollback path.

RAM-only and full/unavailable SSD use the same bounded directory. A hot-record
target can be temporarily exceeded by execution leases or a pending write;
the Host byte limit governs admission. Optional captures can be skipped, so
larger directories do not guarantee more points under a small Host budget.

## Verification

The [historical validation record](checkpoint-policy-validation-20261025.md)
contains results and limits for its recorded checkpoint-policy binaries. New
content-cache backend combinations require their own validation records.
`content_cache_test` covers reverse-order concurrent publication, native proof
rejection, source lifetime, metadata limits, group fairness, origin coverage and
publication/restore failure without loading a GPU.
`tests/content-cache-regression.py` tests an already configured dedicated HTTP
endpoint: document and acknowledgement first, separate questions on three
branches, six overlapping pairs cycling shared/different/missing keys, old-parent
return, and ordinary tool continuation. It requires correlated native actual
observations and two materialized active requests for every pair. Use `--stream`
when the backend exposes live actual observations only through streaming, as
NInfer does. Launcher/GPU/budget arguments record settings; the script does not
configure the server. Its `--self-test` and replay transport fixtures do not
qualify a native backend or GPU. Save the launch configuration and native logs
with each real run; a small-budget miss remains adverse capacity evidence.
The [NInfer cross-owner validation](ninfer-cross-owner-validation-20261029.md)
records Chat rollback, OpenCode forks, shared storage and cold-read failure tests.

`kvmem-checkpoint-policy-test` checks retention, identity aliases, tiny budgets,
fixed leases, truncation and exact capture boundaries. `kvmem-checkpoint-image-test`
checks framing, checksum, partial reads, full disk/queue, cancellation, source
lifetime and reservation refund. The native checkpoint-tail test compares exact
KV/recurrent bytes and rejects incomplete coverage.

NInfer's `ninfer_ssd_failure <artifact> --fork` checks two-stage document/question
reuse against fresh prefill, a failed cold read, cancellation during cold I/O,
parent survival, changed prefixes and requests that forbid index updates.
Host KV tests cover shared RAM/SSD records, copy-on-write, allocation failure,
owner deletion and physical quota release; mean-K tests cover independent tails.

`tests/checkpoint-coverage-regression.py` exercises ordinary document-first and
separate-question turns, tail/middle/front edits, session switches and actual SSD
reads. `tests/session-restore-regression.py --checkpoint-policy spread` covers
MTP switching and cancellation; `--checkpoint-disk-mib` enables its SSD case.
`--gpu-layers 4 --attention-verify --expect-attention-stage` qualifies the mixed
CPU/GPU sampler on the small fixture against dense FP64. Partial-offload MTP
uses `--mtp-state snapshots`; the default replay path requires GPU recurrent
layers. These options do not change the ordinary request protocol.
These diagnostic runs retain requests, responses, logs, GPU UUID and policy
settings. Their traced timings are not controlled throughput measurements.
NInfer may retain its admitted, quoted reuse count after an optional restore
fails and the prefix is recomputed. Actual restore evidence is
`KVMEM_HISTORY_RESTORE`; an API cache-hit counter alone is insufficient to prove
that a cold checkpoint was used.

The implementation borrows coverage ideas from Strata and the local PolyStrata
v0.1.45-poly.2 source review (release commit
`974552f562aecb3e86cdd2679e76e4515b52b9d2`, `conv_cache.hpp` and `generate.cpp`);
payloads and restore proofs remain native to each
KVMem backend. Changing the default requires separate representative performance
and quality qualification, including 27B and longer contexts.
