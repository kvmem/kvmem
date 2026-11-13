# hipfire backend

hipfire is a Rust + HIP engine whose Qwen3.5 serve path keeps KV in paged Q8
arenas (one FullAttention K/V arena per layer, 128-token pages). This backend
lets KVMem own *residency* for those pages: a full page is either resident in
the slot's block table or replaced by the `SPILLED_PAGE` sentinel while its
packed Q8 bytes live in the KVMem host store. The masked attention kernels
skip sentinel pages, so an evicted page is invisible until a selection
restores it.

## Pieces

| Piece | Where |
|---|---|
| C ABI shim over the portable KVMem host library | `hipfire_kvmem_shim.{h,cpp}` (this directory) |
| Rust FFI crate + shim build | hipfire `crates/hipfire-kvmem` (`build.rs`, `KVMEM_ROOT`) |
| Paged spill primitives (sentinel, page ops, kernel masks, paged split-K verify kernel) | `../patches/hipfire-paged-spill.patch` |
| Statistics taps, controller, serve wiring; escha-dense baseline, Bonsai TQ2/BQ1 slots admission | `../patches/hipfire-kvmem.patch` |
| RDNA3 (gfx1100) operator pass: MQ3G256V2 q8x verify decode, real escha grouped-WMMA batched prefill (replaces the `kvmem` patch's `prefill_chunk=1` stopgap), escha-pro/xt F16 in_proj handling | `../patches/hipfire-rdna3-opt.patch` |
| Multi-slot spec-decode admission fix: `--spec mtp`/`dflash` engage under the router's default penalized+sampled requests (exact-AR penalty prepass + greedy/point-mass sampled verify, shared `SpecRetire`) | `../patches/hipfire-spec-accept.patch` |
| Batched spill/restore: pinned-slab stream-ordered async page copies with one sync per chunk (byte-neutral; `HIPFIRE_KVMEM_BATCH_XFER=0` restores per-page blocking) | `../patches/hipfire-batch-restore.patch` |
| Opt-in residency-selection policy knobs: `SELECT_METHOD` (retrieval/recency), LRFU decaying heavy-hitter history (round lifecycle wired through the shim + crate FFI), GPU high/low watermark hysteresis band; defaults byte-neutral | `../patches/hipfire-policy-knobs.patch` |
| Query-span multi-row Q scoring: `QUERY_SPAN` (1 = mean-K cosine, byte-neutral; >1 = page-softmax over a per-slot host Q ring), `query_score_frame` + `page_softmax_scores` wrapped through the shim + crate FFI; zero VRAM growth | `../patches/hipfire-query-span.patch` |
| Host page-softmax SIMD dot reduction: crate build adds `-fopenmp-simd` so the core dot product's `#pragma omp simd` is honoured (float32 accumulation over FP16-rounded operands); per-selection latency ~3.4× lower, default byte-neutral | `../patches/hipfire-softmax-simd.patch` |
| Layerwise recall foundation (Phase 3.2 sub-patch ①): crate-agnostic `LayerwisePagePool` composite-slot pool + `PackedArenaLayout` packed-page geometry (rdna-compute), Q8 packed-page adapter bridging the engine stride authority to the core generic `PayloadLayout` manifest (hipfire-arch-qwen35), and the core layerwise-API C ABI shim entries + crate FFI (payload builder / catalog `plan_layer` / `NativeKvLayerRegions` / `LayerwiseQueryRequest` state machine / native scorer); additive, NOT wired into live serve, Legacy byte-neutral | `../patches/hipfire-layerwise-foundation.patch` |

The shim is host-only C++17: it wraps `RawKvStore` (block statistics),
`KvMemStore` (block bookkeeping / selection) and the Mean-K retrieval scorer.
No kvmem symbol reaches a HIP kernel; device transfers stay in hipfire.

## Building

`scripts/prepare-backends.py --backend hipfire` clones the pinned base commit
into `../hipfire` beside this checkout (override with `--hipfire-dir`) and
applies the verified patch series below; `--backend all` includes it when a
ROCm toolchain is detected (`HIPFIRE_ROCM_PATH`, `/opt/rocm` or `hipcc`).
`cargo build` refreshes `Cargo.lock` in the checkout, so `--check` verifies
the pre-build prepared tree; setup resets the checkout to the pinned base and
re-prepares it on every build.

Point `KVMEM_ROOT` at this checkout (default is the sibling `../kvmem` of the
hipfire workspace) and build the CLI **and the daemon** with the `kvmem`
feature — the multi-slot serve engine lives in the `daemon` binary, so building
only the CLI leaves a stale engine:

```sh
HIPFIRE_ROCM_PATH=/opt/rocm cargo build -p hipfire-cli -p hipfire-daemon --features kvmem --release
```

The patch series applies to hipfire `89e32856d25d6f6e6645319f2ec2c641c46be525`
(v0.4.6, fork `smashdog/hipfire`); see `upstream.json` for hashes and the
verified tree. That release still predates the escha-dense merge, so the
`kvmem` patch also carries the escha-dense baseline (escha trellis
projections, quantize/dispatch support) it builds on, plus the Bonsai
TQ2G128/BQ1G128 flat-lowbit slots admission. On the v0.4.5 -> v0.4.6 rebase
the Escha shared-weight DType tags were renumbered 65/66 -> 68/69 (upstream
took 65..67 for MQ4G256V2P/MQ4G128V2P/HC8E4M3Row); the HFQ wire qt=42/43 is
unchanged, and the fp16-x scratch marker invalidation the old patch added is
now enforced by upstream's `fp16_x_scratch_writable`.

The third `rdna3-opt` patch is a gfx1100 (RDNA3) operator pass on top of that
integration, verified byte-exact (`89e32856d` + the first three patches reproduce
its `after_tree`). It removes the `kvmem` patch's escha prefill
stopgap — that patch forced `prefill_chunk=1` so every prefill row ran through
the lowered paged decode path — and replaces it with a real grouped-WMMA
batched-prefill trellis arm in `forward_slots` (one weight read per
(layer,batch) instead of per slot). Escha prefill is ~10.6-10.9x faster and its
greedy needle argmax matches the per-token reference token-for-token;
`HIPFIRE_PREFILL_CHUNK_OVERRIDE=1` still forces the serial reference for the A/B.
The arm also handles the escha-pro/xt checkpoints that store DeltaNet
`in_proj_a/b` as F16/F32 instead of MQ6G256V2 (`slots_plain_gemm_key` gains
F16->GemmF16WmmaMb8 / F32->GemmF32Batched; need_rot/anchor are derived from the
actual in_proj dtypes). For decode it adds a 3-bit MQ3G256V2 arm to the gfx1100
q8x verify kernels (bit-exact vs the reference GEMV), lifting MQ3 MTP decode
36.8 -> 45.9 tok/s (+24.7%); q8x verify only activates at batch>=2 (MTP spec
decode), so SPEC=off is unaffected.

The fourth `spec-accept` patch fixes multi-slot speculative-decode admission so
`--spec mtp`/`dflash` actually engage under the router's default sampling,
verified byte-exact (`89e32856d` + all four patches reproduce `upstream.json`'s
`patched_tree`). The cold-admit gates previously required
`!request_penalized && !request_sampled`, so the engine default
`repeat_penalty=1.05` plus a router-injected `presence_penalty` and `temp>0`
silently forced every slot back to plain AR (DFlash throughput == AR). The patch
drops those gates and runs an exact-AR penalty prepass
(`spec_apply_verify_penalties`) before greedy argmax OR point-mass sampled
acceptance (`mtp_sampled_accept`, `point_mass=true` — DFlash2 proposals are
deterministic q=δ(c)); MTP and DFlash share `SpecRetire`. A/B at SLOTS=2 on
gfx1100: DFlash greedy-neutral prose 40→54.8 tok/s and code 40→93.7, MTP sampled
prose 40→57.1. Sampled+penalty DFlash can still net-lose on prose (the greedy
selector's proposals mismatch the penalized nucleus), so MTP is the better
default for a sampled config; the multi-slot DFlash VRAM ceiling (per-slot
persistent rings, KVMem skips idle-session reclaim) is unchanged — use SLOTS=2 on
a 20 GiB card.

The fifth `batch-restore` patch batches the KVMem spill/restore transfers,
verified byte-exact (`89e32856d` + all five patches reproduce `upstream.json`'s
`patched_tree`). The controller previously copied each spilled/restored page with
a blocking per-page, per-layer `hipMemcpy`; the patch gathers a chunk of pages
into one reusable pinned slab (`PinnedTransferBuffer`, 64 MiB cap, allocated once
so a larger chunk never re-page-locks mid-prefill) and moves it with
stream-ordered async offset copies, draining the whole chunk at a single sync
point before the host reads it (spill) or before `forward_step` reads it
(restore). The copies ride the steady-copy stream when one is installed, else the
legacy null stream — the same stream prefill compute uses, so ordering holds
either way; this matters because the capture stream is only installed inside
decode-graph capture, *after* `prepare_step`, so a stream-required batch would
fall back to per-page exactly during the TTFT-critical prefill/restore. Spill
still drains every read before any `stage_out_page` frees a physical page, and
the whole batch completes inside Pass 1 before Pass 2 reuses a freed page.
`HIPFIRE_KVMEM_BATCH_XFER=0` restores the exact per-page blocking path. Byte
neutrality was gated on gfx1100 (N-gate): `HIPFIRE_KVMEM_VERIFY=1` reported 0
readback mismatch over a 5k–17.5k-token over-budget needle sweep, greedy decode
was bitwise-identical (sha256) on vs off, and recall was identical. The transfer
component of `prepare` is ~11% faster batched (spill 31 pages 75.2 vs 86.5 ms,
restore 30 pages 45.5 vs 49.7 ms); wall TTFT is parity at these host-RAM-spill
scales because prefill compute dominates (prepare is ~1% of an 11.5 s prefill),
and the transfer win grows with the restored page count.

The sixth `policy-knobs` patch wires the opt-in residency-selection policy knobs,
verified byte-exact (`89e32856d` + all six patches reproduce `upstream.json`'s
`patched_tree`). Every knob defaults to the historical behavior, so the patch is
byte-neutral unless a knob is set. `SELECT_METHOD` chooses the executor-level
signal that feeds the shared `pick_topk` ranking: `retrieval` (the default) scores
the resident set against the tapped mean-K query, `recency` feeds no learned
signal (the core keeps sink + recent and fills the rest by its deterministic id
tie-break); `h2o` is rejected at construction until the attention-heat tap lands.
`LRFU` adds a decaying heavy-hitter history on top of retrieval: each round
reinforces the top `LRFU_TOP` fraction of blocks and decays the rest by
`LRFU_LAMBDA`. The core requires an explicit round lifecycle
(`begin_retrieval_round` → `set_retrieval_scores` preview → `pick_topk_blocks` →
`commit_retrieval_round`); the patch adds the two round entries to the C ABI shim
(`hipfire_kvmem_shim.{h,cpp}`, a kvmem-repo change) and the crate FFI
(`crates/hipfire-kvmem/src/lib.rs`), and the controller opens a round before
previewing scores and commits it after the pick. Both entries are no-ops when
`LRFU` is off, so the default Retrieval path is unchanged. The GPU
high/low watermark ratios derive a hysteresis band from the core watermark
functions (`keep ≤ trigger ≤ budget_blocks`): the trigger replaces the single
budget spill threshold and the keep target narrows the core selection budget via
`set_runtime_select_budget`, so the band can only make residency more
conservative, never overshoot the physical pool. Setting either watermark enables
the band (the unset ratio defaults to 0.95/0.85). The knobs are exposed through
three layers — engine env, router parameter spec, wrapper docs — not the embedded
`/ui` chat surface.

Byte neutrality was gated on gfx1100 (N-gate): with no policy env the greedy
decode sha256 still equals the batch-restore baseline `29e8a27d…` (cross-build
bit-identity), setting the non-watermark defaults explicitly (`expdefault`)
reproduces it exactly, recall is identical over a 5k–17.5k-token over-budget
needle sweep, and `HIPFIRE_KVMEM_VERIFY=1` reports 0 readback mismatch. Each
non-default value passed its own B-gate (0 mismatch, recall and decode parity
reported vs default): `lrfu` keeps recall `True/True/False` (== default) with the
round lifecycle active (223 spills / 63 restores, no "probe outside a round"
error across four sequential rounds); `watermark` (`Some((28,22))` band) keeps
recall `True/True/False` with more conservative residency (237 spills / 47
restores); `recency` trades the 16k middle needle by design (`True/False/False`,
160 spills / 0 restores — the needle is never selected back) while staying
byte-exact. All knobs leave the under-budget greedy decode bitwise-identical.
Sustained decode parity (256-token generation) confirms no throughput regression:
over-budget with the policy active 14.1–14.3 tok/s vs default 14.2, and
under-budget kernel-neutral 40.18–40.27 tok/s sustained vs default 40.2 — every
knob within 10% on both regimes. Two-slot concurrency (`SLOTS=2`, two simultaneous
over-budget requests with distinct needles, both slots running) keeps each slot's
LRFU round isolated: both requests recall their own passphrase with 0 round
errors and 0 readback mismatch.

The seventh `query-span` patch adds multi-row Q scoring for retrieval selection,
verified byte-exact (`89e32856d` + all seven patches reproduce `upstream.json`'s
`patched_tree`). `HIPFIRE_KVMEM_QUERY_SPAN` (default `1`, range `[1,512]`) chooses
how many recent Q rows feed the resident-set ranking. At `1` the controller keeps
the historical path: each layer contributes its single tapped last Q row
(post-RMSNorm, pre-RoPE) scored by `mean_key_retrieval_scores` (mean-K cosine),
so the default is byte-identical. At `>1` a per-slot host Q ring
(`[n_fa_layers × span × q_width]`, layer-major) accumulates the recent Q rows —
fed each step from the *existing* `q_tap` via `read_q_slot`, so there is zero VRAM
growth (the ring lives in host RAM) — and selection scores resident pages by
`page_softmax_scores` (FP16-rounded operands, per-row softmax mass, scale
`1/√head_dim`, Mean GQA aggregation) over the span. The ring is reset on slot
reset and at each request boundary so a new turn never inherits the previous Q
span. The C ABI entry `hk_page_softmax_scores` wraps `query_score_frame` +
`page_softmax_scores` in the kvmem-repo shim (`hipfire_kvmem_shim.{h,cpp}`, a
separate kvmem commit); the crate FFI (`crates/hipfire-kvmem/src/lib.rs`) and the
controller ring + branch (`crates/hipfire-arch-qwen35/src/kvmem_ctl.rs`) are the
hipfire patch. The two scoring functions are mathematically distinct (cosine vs
softmax); `span=1` keeps the old path entirely, so there is no bit-identity
concern between them. The knob only affects the scored `SELECT_METHOD=retrieval`
branch and is exposed through the same three layers — engine env, router
parameter spec, wrapper docs — not the embedded `/ui`.

Byte neutrality was gated on gfx1100 (N-gate): with no `QUERY_SPAN` env (and with
`QUERY_SPAN=1` explicit) the greedy decode sha256 still equals the policy-knobs
baseline `29e8a27d…` (cross-build bit-identity), recall is identical
(`True/True/False` over the 8k/16k/28k over-budget needle sweep), and
`HIPFIRE_KVMEM_VERIFY=1` reports 0 readback mismatch (225 spill checksums). The
under-budget greedy decode is bitwise-identical across *every* span (selection
never fires without spill), confirming the kernel path is untouched. Each
`span>1` passed its own B-gate: `span16/64/256` keep recall `True/True/False`
(== default) with 0 mismatch and decode parity within 10% on all three targets.
Sustained decode (256-token generation) confirms no throughput regression:
over-budget with the span active 13.9 tok/s vs default 14.3, and under-budget
kernel-neutral 39.85–39.93 tok/s sustained vs default 40.44 — the ring D2H read +
page-softmax overhead is negligible. A `TRACE=1` activity probe proves `span>1`
genuinely computes multi-row page-softmax rather than silently falling back:
`span=1` emits zero query-span lines, while `span=64` emits `rows>1` selections
whose `score_sum` equals the row count exactly (11.0000 / 12.0000 for 11 / 12
rows) — the fingerprint of per-row softmax normalization (each Q row contributes
total probability mass 1.0), distinct from mean-K cosine. Two-slot concurrency
(`SLOTS=2`, two simultaneous over-budget requests with distinct needles) keeps
each slot's Q ring isolated: both requests recall their own passphrase with 0
readback mismatch and no page-softmax/controller error. A ten-SKU recall sweep
under `QUERY_SPAN=64` (escha / escha-pro / escha-xt, Bonsai tq2 / bq1, mq3 /
mq3-pro / mq3-xt, mq4-xt / mq4-xts) confirms the ring geometry adapts per model
(each SKU's `n_fa_layers × q_width`): all ten recall their over-budget needle
(10/10) with 0 readback mismatch and no controller or admission error.

A note on discrimination: a two-needle compound probe (vault at 25%, backup at
75%, one compound question) shows identical recall across `span1/64/256`. This is
a homogeneous-filler limitation, not an inactive span: when the filler is uniform
both mean-K and page-softmax rank the needle block highest, so selection
coincides. The `TRACE=1` score-sum fingerprint above is the definitive activity
evidence; recall discrimination between the two scoring functions requires
heterogeneous filler and is left to the long-needle sweep rather than gated here.

The eighth `softmax-simd` patch optimizes the host page-softmax dot reduction that
`query-span` introduced. Profiling an over-budget selection on gfx1100 (28k-token
prompt, `span=64`) isolated the per-page dot product — a serial `double` reduction
over `head_dim=256` that `-O2` cannot auto-vectorize without reassociation — as
~96% of the cost (means 6.1 ms / dot 153 ms / ~167 ms per selection). The core
change (`kvmem/src/host/graph_recall.cpp`, a separate kvmem commit) accumulates the
dot in `float32` under a `#pragma omp simd reduction`; both operands are
FP16-rounded, so each product is exact in float32 and the float reduction is a
principled refinement aligned with the device float math, not a precision loss.
The hipfire patch adds `-fopenmp-simd` to the crate build
(`crates/hipfire-kvmem/build.rs`) so the pragma is honoured — SIMD-only, no OpenMP
runtime linked, and no other pragma in the host library is activated.
Per-selection latency falls ~167→49 ms (3.4×; the pure dot ~7×, the residual now
co-dominated by the softmax `exp` loop).

The edit is confined to `page_softmax_scores`, which runs only for `span>1` over
budget, so the default `span=1` mean-K path is untouched: the N-gate greedy decode
sha256 still equals the policy-knobs baseline `29e8a27d…`, recall is identical
(`True/True/False` over the 8k/16k/28k sweep), and VERIFY reports 0 mismatch (225
spill checksums). Each `span>1` keeps its B-gate — VERIFY 0, decode parity within
10% on all three targets, and recall `True/True/False` preserved (the float
perturbation flips no needle selection in the sweep); the under-budget greedy
decode is bitwise-identical across every span. Selection is interval-gated (~2
actual selections per 256-token decode), so the amortized throughput impact is
~1–2% (14.0→14.2 tok/s). A HIP GPU page-softmax kernel is therefore **deferred**:
the residual ~49 ms per selection does not justify the port + wiring for an opt-in
mode whose steady-state throughput cost is already negligible.

The ninth `layerwise-foundation` patch lays the additive groundwork for per-layer
(`RecallStrategy::Layerwise`) KV recall — Phase 3.2 sub-patch ① of three. It is
purely additive, NOT wired into the live serve allocation, and adds no env toggle,
so Legacy remains the only path and is byte-neutral. Three pieces land:
(1) `crates/rdna-compute/src/layerwise_pool.rs` — a crate-agnostic pure-host
`PackedArenaLayout` (one composite slot = 128 tokens × all fa layers × {K,V},
layer-major, Q8 scales inline) plus `LayerwisePagePool`: per-layer independent
stripe selection into shared composite slots, generation handles, per-layer
compact views with a core-validated equal `visible_tokens` extent, page residency,
and an intrinsic drop-evict refcount (a slot frees only when every layer stripe
drops). (2) `crates/hipfire-arch-qwen35/src/layerwise_layout.rs` — the Q8
packed-page adapter that walks the engine's OWN stride authority
(`SlotKvTierPlan::resolve`) into the core generic `PayloadLayout` manifest
(layer-major {Key,Value} per fa layer) and bridges to `PackedArenaLayout` so a
composite slot's byte layout is provably identical to the core record
(`slot_bytes == record_bytes`, per-layer plane offsets aligned) — the coherence
guarantee that lets a sealed archive layer region map onto a slot stripe with no
repacking in sub-patch ②. (3) the core layerwise-API C ABI shim entries
(`hipfire_kvmem_shim.{h,cpp}`, a separate kvmem commit) exposing the EXISTING core
`PayloadLayout` builder / `LayerwiseMemoryCatalog::plan_layer` /
`NativeKvLayerRegions` / `LayerwiseQueryRequest` state machine / native scorer,
plus the crate FFI and three smoke tests. The core library is strategy-agnostic
and unchanged (core logic zero-change); the legacy `MemoryBackend`
whole-working-set coordinator is NOT consumed by layerwise (direct-drive, design
§3.5).

The shim smoke (14 crate tests: catalog 3 + payload 4 + request 6 + base shim 1)
and the new host tests (pool 12 + adapter 4) pass. The N-gate on the rebuilt engine
reproduces the sealed chain byte-for-byte: greedy decode sha256 == `29e8a27d…`,
VERIFY 0 mismatch (65 restore readback lines, 225 spill checksums, 0 block-table
dups), and recall identical to the policy-knobs/query-span baseline
(`True/True/False` over 8k/16k/28k — the 28k miss is the pre-existing mean-K
retrieval limit under the 45-page budget, the recall-benefit target for ②③, not a
regression). A pre-existing `qwen35/weights.rs` `dense_fixture` (missing the
`biases`/`escha` fields the escha chain added) is repaired test-only (`None/None`,
matching production) so the crate's `--lib` test target compiles; release N-gate
builds never compiled `cfg(test)`, so the breakage had stayed latent.

Sub-patch ② adds the GPU HIP scorer + per-layer driver into the three paged paths
+ write-through feed + drop-evict policy + stripe COW restore
(`RECALL_STRATEGY=Layerwise` opt-in); sub-patch ③ adds the pressure attention-mass
tap + quota accounts + env exposure + fail-closed guards and seals the
recall-benefit gate. Each sub-patch is Legacy byte-neutral and independently
re-verified; rollback granularity = sub-patch.

## Enabling

`HIPFIRE_KVMEM=1` on `hipfire serve`. Requires the paged Q8 KV pool (the only
KV quantisation supported by this first version) and the batched VL path. The
prefix cache is disabled while KVMem is active: KVMem owns page residency, so
sessions always cold-start and every block's statistics are fed from its own
forward.

| Env | Default | Meaning |
|---|---|---|
| `HIPFIRE_KVMEM` | off | enable the controller |
| `HIPFIRE_KVMEM_BUDGET_TOKENS` | min(cap, pages/slot − headroom) | resident tokens per slot before pages spill |
| `HIPFIRE_SLOTS_PAGED_PAGES` | 64 × n_slots | total physical pages (overrides pool sizing) |
| `HIPFIRE_KVMEM_SINK_BLOCKS` | 1 | oldest blocks pinned for every selection |
| `HIPFIRE_KVMEM_RECENT_BLOCKS` | 1 | newest blocks pinned for every selection |
| `HIPFIRE_KVMEM_SELECT_INTERVAL` | 4 | pages between re-selections (in full pages) |
| `HIPFIRE_KVMEM_HOST_BYTES` | 16 GiB | RAM tier capacity |
| `HIPFIRE_KVMEM_SPILL_DIR` | — | set to enable the disk tier |
| `HIPFIRE_KVMEM_SPILL_BYTES` | 64 GiB | disk tier capacity |
| `HIPFIRE_KVMEM_TRACE` | off | per-step residency diagnostics (`HIPFIRE_KVMEM_TRACE_PAGES=1` adds picked/restored page ids) |
| `HIPFIRE_KVMEM_TIME` | off | per-step forward/tap timings + controller call breakdowns |
| `HIPFIRE_KVMEM_VERIFY` | off | checksum every spilled/restored page and read the device block table back against the host table; 0 mismatch proves the spill/restore round is byte-exact |
| `HIPFIRE_KVMEM_BATCH_XFER` | on | batch spill/restore through a pinned slab with stream-ordered async copies + one sync per chunk (byte-neutral); `0`/`off`/`false` restores the per-page blocking path |
| `HIPFIRE_KVMEM_SELECT_METHOD` | `retrieval` | executor selection signal feeding the shared `pick_topk` ranking: `retrieval`/`0` scores the resident set against the tapped mean-K query; `recency`/`2` feeds no learned signal (sink + recent + the core's id tie-break). `h2o`/`1` is rejected until the attention-heat tap lands |
| `HIPFIRE_KVMEM_LRFU` | off | decaying heavy-hitter retrieval history: each round reinforces the top-scoring blocks and decays the rest by `lambda`. Requires `SELECT_METHOD=retrieval`; off by default so the Retrieval path is byte-neutral |
| `HIPFIRE_KVMEM_LRFU_LAMBDA` | 0.5 | LRFU decay weight on the prior committed score, in `[0,1]` (0 = no memory, 1 = no decay) |
| `HIPFIRE_KVMEM_LRFU_TOP` | 0.9 | LRFU hit fraction gating positive reinforcement, in `(0,1]`; scores below the `(1−top)` quantile are clamped to ≤0 so an unselected block is never rewarded |
| `HIPFIRE_KVMEM_GPU_HIGH_WATERMARK` | — (0.95) | spill-trigger ratio of the resident budget. Setting **either** watermark enables the hysteresis band; the unset one defaults to 0.95/0.85. Must satisfy `0 < low ≤ high ≤ 1` |
| `HIPFIRE_KVMEM_GPU_LOW_WATERMARK` | — (0.85) | keep-target ratio of the resident budget; narrows the core selection budget so residency stays below the trigger. The band can only make residency more conservative, never overshoot the physical pool |
| `HIPFIRE_KVMEM_QUERY_SPAN` | 1 | retrieval query span in `[1,512]`. `1` scores the single tapped last Q row by mean-K cosine (byte-identical to the historical path); `>1` accumulates a per-slot host Q ring of recent post-RMSNorm/pre-RoPE Q rows and scores resident pages by `page_softmax_scores` (FP16-rounded operands, per-row softmax mass, scale `1/√head_dim`). Zero VRAM growth (the ring lives in host RAM); only affects `SELECT_METHOD=retrieval` |

With `cap ≤ budget` nothing ever spills, the tap kernels only copy, and the
forward is bit-identical to an unpatched hipfire. Spill/restore moves arena
bytes verbatim, so a spilled-and-returned page is bit-exact too. Multirow
spec-verify steps (MTP, 4..=8 rows past 4096 tokens) run the reference
split-K WMMA verifier through a block-table twin: a bench-verified bitwise
copy of the descriptor-less route (`diff floats 0`), with absent pages
masked out and skipped wholesale instead of sweeping the logical context.
