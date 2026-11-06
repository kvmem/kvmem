# Checkpoint coverage and RAM/SSD placement

The experimental `spread` policy adds historical coverage to the llama.cpp and
NInfer Qwen3.5 KVMem paths. The default remains `legacy`. PolyStrata keeps its own
native checkpoint policy. This feature does not add a request mode or persistence
across process restarts.

## Shared owners across backends

For Chat Completions, `kvmem.conversation_id` names the mutable lineage;
`prompt_cache_key` and the SDK spelling `promptCacheKey` are aliases with lower
priority. Conflicting nonempty aliases are rejected, including when an explicit
conversation ID is present. Keys preserve all bytes up to the 256-byte limit.
Radiance also accepts its existing `session_id` as a legacy fallback; the
128-byte dashboard label is separate from cache identity. Longer legacy labels
remain accepted for display without becoming an owner key.

The same owner contract applies to llama.cpp and PolyStrata. An explicit owner
selects a mutable lineage, not a bypass for token, media, position, accepted-state
or model/layout checks. An anonymous request may discover a reusable native
prefix; it does not acquire a durable named conversation. Keys are bounded by
UTF-8 bytes, not dashboard label length. Invalid types, empty explicit owners and
conflicting aliases return an input error.

| Backend | Native source and independent branch | Boundaries |
|---|---|---|
| NInfer Qwen3.5 KVMem | Owner-first native history lookup; sealed archive references and immutable state images, private selectors/Query/rollback | `legacy` and `spread`; active GPU pages and mutable recurrent state remain per lane |
| llama.cpp KVMem | Exact native checkpoints; source-preserving bundles, sealed packed KV and read-only recurrent/draft checkpoint references | Modern host/SSD tier or multi-conversation mode; old pinned CPU arenas disable foreign forks; SSD requires Flash Attention |
| PolyStrata KVMem | Native accepted stamp and recurrent checkpoint; parked source preserved while child enters, shared sealed archive/state | CUDA qualification; solo MTP and ordinary batch decode keep their native execution modes; HIP hardware validation is separate |
| Radiance KVMem | Native rank-prefix state, accepted token/media proof and independent rank memory policy | CPU/portable validation is separate from AMD execution; remote GPU qualification requires the machine to be running |

These are host/SSD and read-only checkpoint references within one loaded
backend/model configuration. They are not a portable KV format, a cross-process
cache, or a promise that separate GPU working sets share their physical pages.
Different backends keep their native restore rules. No second cache manager or
document-ingestion endpoint is introduced.

No key is required for sharing. A new user Query may discover a complete exact
prefix from another owner, including an anonymous owner, in either NInfer
`legacy` or `spread` mode. The selected checkpoint must still prove model/layout,
token and media identity, original positions, accepted frontier and native
recurrent state. Lookup examines RAM metadata and reads only selected cold
payloads. Hashes and longest common prefixes are candidate filters, never a
substitute for those proofs. Tool continuation does not enable foreign-owner
discovery or cold-history Query reselection; it uses append and resident-only
Pressure. Existing proven continuations remain eligible for native restore.

llama.cpp clones an active source without switching its GPU working set first;
normal lane handover then restores the child. A partial archive tail is private,
and restored checkpoint statistics supply its exact state. PolyStrata preserves
the selected parked source until the branch is published; failed enter/restore
does not consume it. Checkpoint wrappers, coverage catalogs, MeanK, LRFU, graph
feedback, Query buffers and draft rollback are mutable per owner even when
their sealed byte allocations are shared.

Capacity admission charges the native physical H/D account once, plus new
growth and outstanding reservations. Logical bytes summed across owners can
exceed physical bytes. Metadata, statistics, model weights, OS file cache,
temporary transfers, process RSS and GPU VRAM have separate scopes; H is not an
RSS limit. An LRU directory eviction releases only its own references. The last
reader or owner releases the allocation and its quota.

The monitor distinguishes `planned_cached_tokens`, `actual_cached_tokens` and
`cache_source` (`own`, `shared`, `miss`). Missing native observations stay absent
or null. NInfer's pre-existing `cached_tokens` wire accounting is an admission
quote: a best-effort failed restore can silently recompute that prefix. Its new
independent actual observation reports zero in that case, and the monitor uses
the actual value when available. This does not rewrite the original wire
timings or Prometheus admission accounting. A created fork is not an actual
cache hit until native restore succeeds. llama.cpp's physical checkpoint bytes
also include private retrieval metadata and are labelled separately from pure
recurrent-state bytes.

PolyStrata reports the first successful native restore boundary per HTTP request;
input computed before an internal yield is not newly credited on resume. This
is not net compute saved across every later replay. Its scope-version-2 memory
snapshot includes all native sequence archives, slot checkpoint references and
parked histories. Foreign discovery requires more than one retained history;
with one history, different named owners start fresh. Internal solo/batch/MTP
continuations after decode retain the owner with an empty Query span. A prefill
yielded before its new Query retains that still-pending bounded span.

A foreign hit forks the native history. Historical sealed RAM/SSD records and
read-only state may share physical allocations; mutable selectors, feedback,
catalogs and live recurrent state belong to each branch. Releasing or editing one
owner cannot change bytes still leased by another. Normal capacity eviction may
remove an idle directory entry, and physical release waits for its last lease.
For NInfer anonymous requests, the existing native continuation proof still
permits dropped output and a re-templated input tail of at most 64 tokens; deeper
shared prefixes now use a fork. An anonymous request is not a durable named
conversation. Clients that need reliable lineage selection should send a key.

Radiance retains both a deep input point and a conservative point before the
first assistant message, when those are safe native chunk boundaries. An input
edit or template rewrite may require falling back or recomputation. It cannot
restore recurrent state at an arbitrary LCP. DFlash recent selection covers its
declared window; accepted prefix and Query checkpoints include the matching
native window bytes, including RAM/SSD parking and forks. Restoring these bytes
keeps the saved frontier instead of subtracting a fixed window. The scheduler
still respects native prefill quantum, Query boundaries and uncached logits;
its separate native-cache refill gate remains for hits without window state. Extra prefill cuts can
change floating-point accumulation; text equality is not the sole correctness
test.

Radiance admission counts repeated sealed allocation identities once across
active histories, input points and parked sessions. Independent captures of
equal bytes and per-branch metadata remain charged separately. The monitor
exposes logical bytes, proven sharing credit and accounted cache bytes, alongside
the storage account's actual RAM/SSD payload charges. Pending parking retains a
conservative full reservation until publication. These counters do not describe
total process RSS, model weights or transient runtime workspace.

This scope shares physical host/SSD records inside one worker and compatible
native model instance. It does not add a common GPU page pool, transfer KV
between backends or persist caches across restart. Radiance retains its existing
native prefix references. Owner keys are routing hints, not a tenant
security boundary. Concurrent requests are native snapshot branches; applications
requiring a single ordered conversation must serialize submissions themselves.

## Configuration

Set these environment variables before starting the worker:

| Variable | Default in spread mode | Meaning |
|---|---:|---|
| `KVMEM_CHECKPOINT_POLICY` | `legacy` | `legacy` or `spread` |
| `KVMEM_CHECKPOINT_INTERVAL` | 16384 | Approximate token spacing of optional captures |
| `KVMEM_CHECKPOINT_MIN_GAP` | 0 | Minimum spacing of ordinary retained points; 0 disables nearby thinning; 64 is the tested experimental setting |
| `KVMEM_CHECKPOINT_COUNT` | 24 | Retained records per conversation, 1–256 |
| `KVMEM_CHECKPOINT_MIB` | 4096 | Logical retained payload and native metadata budget per conversation |
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
When parked owners consume the creator lane's remaining Host quota, required
admission can evict an older idle owner. Busy, switching, queued and resident
owners remain protected. A later request for an evicted owner recomputes its
prefix. A request rejected after prefill began uses its existing rollback lease.

After a turn, reusable execution snapshots enter the bounded archive and release
their execution slots. Shared payload identities are counted once. A new payload
at the same frontier replaces the directory entry; equality of frontier alone
never aliases recurrent or Query-replay state.

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
to find a candidate. Restoring a prefix invalidates the edited suffix.

In llama multi-conversation selection, spread mode permits an explicit, already
bound `kvmem.conversation_id` to edit that owner's earlier history using a valid
checkpoint. This replaces that owner's suffix, rather than retaining an extra
branch. Inferred/unknown conversation IDs retain the existing continuation gate;
legacy mode also keeps that gate for explicit IDs. This is an intentional policy
change required for middle-edit reuse, not a correction of the old fork policy.

NInfer's Responses route binds `prompt_cache_key` to a session key
(`store: false`, without a previous response); that explicit owner can also replace an earlier
suffix. Chat Completions now binds a non-empty `prompt_cache_key` to the same live-session
semantics, including `store:false`. An explicit `kvmem.conversation_id` takes precedence.
The OpenCode compatible SDK spelling `promptCacheKey` is also accepted on Chat;
conflicting non-empty values under both spellings are rejected.
For OpenCode's `@ai-sdk/openai-compatible` provider, enable
`provider.<id>.options.setCacheKey: true`; OpenCode then sends its session ID as
`promptCacheKey`. Reverting messages within that session preserves the key.
Keep the identity stable through append, regenerate and rollback. Separate conversations
should use distinct identities; sharing a key shares one owner, so a suffix edit can discard
checkpoints another branch would otherwise reuse. A key identifies an owner; token/media,
native-state and KV proofs still govern every restore. Missing/null/empty cache keys remain
anonymous. Both NInfer policies protect anonymous histories from consumptive
restores by unrelated requests sharing only the system prefix: a continuation
must reach the previous accepted input boundary, allowing a 64-token template
tail margin. Earlier edits without a session key start a separate history or fork
a proven foreign prefix. This
keeps a short unrelated conversation from truncating the retained long history.

NInfer first searches the current owner's checkpoints. If none is
valid, a request with a current user Query may discover another named or anonymous owner's
matching prefix. The bounded retained directory, token/media/position digests,
coverage and placement metadata live in RAM. Digest filtering reads no SSD
payload; at most 16 foreign candidates receive exact prefix checks. Candidates
prefer the deepest legal frontier, then the fewest cold state/KV bytes, then
recent publication. A hash match never replaces the exact identity checks.

The new owner copies mutable selector/statistics and checkpoint metadata through
the selected frontier. Sealed KV records and immutable native images share their
physical storage; a later KV write detaches that owner's record. Physical payload
quota is charged once while shared. Owner metadata and mean-K copies still cost
RAM, and the first divergent page costs a new KV allocation. The source owner
remains intact and both branches can independently append, roll back and return.
Once the other owners and all outstanding readers release a KV page, its sole
owner can write it without allocating a second copy. A deferred read source or
active read lease still requires copy-on-write until that reader releases it.
Ordinary session/count/byte eviction limits still apply after publication.

RAM and SSD candidates are equally eligible. After selection and capacity checks,
the transaction binds the source generation and reads only the selected cold
image and required native KV pages. Read workers do not mutate the parent.
Cancellation releases unpublished child references; preparation failure falls
back to the admitted full prefill. The native restore verifies state before a
history hit is recorded. Previously quoted API reuse counters can still include
the planned prefix on that fallback; use the native restore log to distinguish
completed reuse. OpenCode needs no parent ID or fork protocol change: its new
session/cache key selects a new owner with an identical serialized prefix.
The payload format and restore proof remain specific to the NInfer Qwen3.5 KVMem
path; the shared owner contract above also applies to the other backends.

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

See the [validation record](checkpoint-policy-validation-20261025.md) for current
results, fixed failures and qualification limits.
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
