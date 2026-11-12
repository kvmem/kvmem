# PolyStrata integration

The opt-in `polystrata` backend pins upstream `v0.1.45-poly.9` at
`83a7a33272413bf87b8d2742dc557c165e2cb2d2`. It supports qwen35moe GGUF models
with F16/Q8_0 KV and Gemma 4 26B-A4B with F16/int8/int8k KV on one CUDA GPU.
This revision also carries the opt-in HIP
integration, marked experimental and disabled by default. Its compile/link CI
is a merge gate; AMD inference qualification is required before promoting it
to validated hardware support, rather than blocking the validated CUDA update.
The source submodule stays at the upstream commit; `backends/versions.json`
identifies all twenty-five ordered patches and their resulting trees.

The poly.9 integration sizes each Qwen35 batched attention launch from that
sequence's compact position, including its decode window. Sequence-buffer lookup
uses the allocated KV capacity rather than the logical context length, preventing
overlapping address ranges from selecting another buffer when KVMem bounds KV.

The KVMem Python launcher preserves the pre-poly.9 mapped-expert default by
passing `--ram-experts-mib -1` when no explicit value is supplied. This does not
lock all expert pages in RAM; the operating system still manages the file
mapping. An explicit MiB limit enables the native RAM expert pool; `-2` selects
poly.9's automatic budget. The automatic pool showed variable decode performance
on the RTX 5050 qualification host, so the launcher does not enable it implicitly.
Native CPU thread selection is unchanged by the launcher.
See the [poly.9 qualification record](benchmarks/polystrata-poly9-qualification-20261111.json)
for the fixes, controlled RTX 5050 comparisons, retained failed investigations,
native acceptance coverage and untested hardware paths.

## Gemma 4 26B-A4B

Gemma support is carried by the final `gemma4-kvmem` patch. It covers bounded
Query/replay, resident-only Pressure, native RAM/SSD history, retained sessions,
concurrent ordinary decode, solo assistant MTP and image input. This source
change requires a rebuilt worker; older release binaries do not gain these
capabilities from a console update alone.

Select **PolyStrata · Gemma 4 26B-A4B** in the console. Its initial settings use
F16 KV and ordinary decoding. Gemma also accepts `int8` and `int8k`; `q8_0` is
the Qwen adapter's format. To enable MTP, select the matching independent
`gemma4-assistant` GGUF in **Draft model** and choose a one-to-three-token draft
ceiling. Native batches use ordinary decode. QAT and non-QAT releases need
their own matching assistant and vision projection files.

Direct launch, using the same model's tokenizer and chat template exported
automatically from its GGUF:

```powershell
python tools/polystrata/server.py --source backends/polystrata --worker PATH/strata-poly.exe --model PATH/gemma-4-26B-A4B-it.gguf --gpu GPU-YOUR-UUID --model-alias kvmem-polystrata-gemma4 --context 262144 --budget 32768 --reserve 2048 --prefill 512 --host-mib 16384 --kv-type f16
```

Add `--mtp-drafts 3 --draft-model PATH/mtp-gemma-4-26B-A4B-it-Q8_0.gguf` for
the assistant. Gemma's exported merges are JSON because token pieces can
contain newlines; automatic export and an explicit tokenizer directory both
accept `merges.json`. Images require a matching mmproj and `strata-vision`;
CPU encoding is the validated route in this change.

The five global layers participate in block recall. The 25 local layers keep
their continuous 1024-token rings, saved with native checkpoints. Rotary
positions stay logical when global rows move to compact physical positions.
PageSoftmax and native attention feedback use Gemma's scale of 1. Checkpoints
at image ends and the assistant turn boundary preserve safe prefixes across
Gemma's generated and reconstructed assistant templates. Tool continuation
keeps the ordinary `(0, 0)` no-Query contract.

At this model's geometry, a full-window checkpoint occupies 200 MiB with F16,
106.25 MiB with int8, or 153.125 MiB with int8k, plus 11 KiB when the assistant
hidden state is present. These state images share the configured native RAM
quota with KV records. Request admission includes one required state image
and the KV replacement margin; with SSD, the state and resident staging must
still fit RAM. Optional history marks are reclaimed before allocating their
replacement, and an optional capture may be skipped under pressure. The live
accepted prefix and an in-flight Query rollback image remain valid. A quota
below the required state size is rejected before prompt evaluation. The full F16
global KV is 20 KiB/token; local rings and execution workspaces are additional.
These are layout sizes, not a measurement of total process memory.

Windows CUDA 13.2 / MSVC / sm120a qualification used fixed UD-Q2_K_XL weights
and matching assistant/mmproj. Necessary checks cover native F16/int8/int8k
records, 32/64-token blocks, graphs, sparse Query/replay, parked histories,
two active sequences, forced SSD reads, elastic experts, assistant acceptance
boundaries and cancellation/reset. Separate document/question HTTP requests over a 12,508-token
document reused 12,501 tokens, captured Query `[12511, 12527)` and answered
correctly. The same document and question also passed an unchanged released
PolyStrata worker with KVMem disabled and a fully resident KVMem reference.
A retained-image follow-up reused 252 tokens and repeated the code correctly.
The references are functional checks, not a throughput comparison. GPU choice
and native evidence are recorded with the task's validation artifacts.

Gemma validation selected idle RTX 5050 Laptop or RTX 5060 Ti; the shared Qwen
Q8/MTP regression used RTX 5060 Ti. The two-slot HTTP check
observed two active slots and correct isolated follow-ups with prefix reuse.
No new AMD, SYCL or Linux GPU qualification is claimed. The 256K configuration
probe reused the same 12.5K document; it is not a 256K-document quality test.
The [validation record](benchmarks/polystrata-gemma4-26ba4b-20261102.json)
distinguishes the initial native matrix from the later Host-quota checks.
Model-free benchmark guards preserve failed HTTP turns and bounded server error
bodies, require question-scoped Query/replay evidence, and reject smoke-only or
incomplete measurements from performance summaries.

The [Gemma performance comparison](benchmarks/polystrata-gemma4-performance-20261102.json)
uses three interleaved rounds on one RTX 5060 Ti, F16 KV, graphs, prompt batch 512,
256 output tokens, and all 3840 experts resident. At 4K input the prefill/decode
gap to the fixed native worker fell from 7.1%/4.7% to 4.7%/1.8%; at 12K it fell
from 8.4%/10.9% to 2.9%/2.3%. The change specializes 512-dimension sampled attention
by page and GQA group, and reuses dead attention scratch for pre-RoPE statistics.
The same 4K CUDA diagnostic reduced feedback-kernel time from 65.1 to 16.8 ms.
Sampling, recall and checkpoint guarantees remain the same. Two-turn HTTP checks
retained the correct answer and 12501 cached tokens in both resident and sparse
cases; sparse question latency still includes state restoration and Query replay.

The backend provides bounded Query/replay, resident-only Pressure on tool
continuations, process-shared RAM/SSD history storage, native concurrent sequences,
solo MTP and CPU/GPU vision. Batch execution uses ordinary decode.
See [launch and package instructions](multi-backend.md#polystrata-qwen35moe-接入2026-10-22-源码).

The mainline adaptation preserves independent scorer, graph and LRFU controls.
The console checks the native worker's capability response before launching the
Python frontend; configured replay limits reach the worker without an alias mismatch.
The frontend forwards native scoring confirmation records to its supervising console.
This change includes the existing launcher and monitor integration. The later unified
chat and console shell work is separate.

The poly.4 update retains the seven ordered KVMem patches, including elastic
workspace/KV/expert management, managed MTP and 32/64-token blocks. It includes
upstream's Windows expert-file read-ahead, tiled router product and MMQ half-sum
overflow protection. The new half-sum scratch buffer keeps its full lifetime
while the larger local elastic workspaces shrink and grow. Native test targets
link the new Flint source required by the upstream engine factory; that historical
qualification covered qwen35moe. Current Gemma coverage is described above.

`STRATA_GDN_MMA=1` is upstream's optional CUDA GDN prefill path and stays off by
default. Its snapshot and graph-recording fallbacks are retained. It requires
separate state/replay validation before claiming a speed or quality benefit.
The web chat's send and regenerate paths share the same explicit request
overrides, client-owned reasoning, native timing and incomplete-stream detection.
The poly.5 qualification below is historical; the current pin is in
`backends/versions.json`. The current pin is the poly.9 release. The poly.9
update carries upstream's multi-row step with optional block-token guesses,
the elastic workspace layer and configurable 32/64-token KVMem blocks; the
rebase keeps KVMem's per-sequence capture, Query/replay and expert staging
beside the new batch and elastic paths. It also retains upstream's
`gemma4-kvmem` layer, rebased onto poly.9, so Gemma 4 26B-A4B support is not
dropped by the version bump.
The poly.5 layer had rebased the integration, added HIP KVMem integration,
linked Kolibri sources into model test targets, and preserved the `(0, 0)`
no-Query sentinel when image tokens are expanded. A continuation must not
reinterpret that sentinel as a request to recall old history.

### Direct HIP launch

After building a HIP worker against this checkout's portable KVMem core,
select it explicitly in the Python launcher:

```sh
python3 tools/polystrata/server.py --backend hip --gpu 0 \
  --source backends/polystrata --worker /path/to/strata-poly \
  --model /path/to/model.gguf
```

`--gpu` is the HIP runtime ordinal reported by `strata-device --list-devices`.
This path does not query `nvidia-smi`. CUDA remains the default and requires a
GPU UUID. Existing console/launcher GPU selection is unchanged; this option is
for direct Python launches. Use a vision encoder built for the selected device
and a projector matching the language model; the filename alone does not prove
that their embedding widths match. HIP startup prints an experimental-status
notice. Deployments still require inference checks on the intended AMD
architecture; build success alone is not a correctness or performance claim.

### Poly.7 HIP validation (2026-11-02)

On Linux x64 with ROCm 10.1 / hipcc clang 24 and one gfx1100 RX 7900 XT, the
worker was built with `-DSTRATA_ENABLE_HIP=ON` against the KVMem core, and the
encoder with the local `STRATA_VISION_HIP=ON` layer (`gfx1100`). The native
history, sessions and KVMem kernel targets passed; upstream's `poly_fattn_test`
does not compile under HIP because it still uses `CUBLAS_TENSOR_OP_MATH`, so
the build only targets the worker and the KVMem tests. With
Ornith-1.5-35B-A3B Q6_K, q8_0 KV, a 65,536-token budget and a 512 MiB reserve,
a 128,924-token prompt prefilled in 3m30.7s (612.4 tok/s) and answered
correctly while parking 1.31 GiB of history. The follow-up reused 128,403
tokens, moved 521 prompt tokens and answered in 3.74s, capturing Query
`[128403, 128915)` with the page-softmax scorer and one bounded replay. Vision
ran on the GPU (1.38 GiB); the worker and encoder together used 19.53 GiB of
20.0 GiB usable VRAM. These are functional checks on one machine, not a
performance claim.

### Poly.5 candidate validation (2026-10-28)

On Windows x64, CUDA 13.2 / MSVC and architecture 120a, the worker, GDN state
target and KVMem regression targets were built with eight parallel jobs. The
prepared twelve-layer chain reproduced its recorded tree. The frontend suite
passed 354 tests with three skips. Adapter tests cover CUDA/HIP selection,
link dependency ownership, image Query boundaries, assets and UI contracts.
The link tests fail on the original PR's prepared tree and pass after correction.

RTX 5050 Laptop checks passed native kernels, history/session policy, F16/Q8
MTP accepted state, elastic experts and KV reclaim. Full F16/32-token and
Q8/64-token runs each passed twelve forced accepted-prefix cases, plus
cancellation/reset and native-transfer checks. Real HTTP checks passed
Unicode text, separate document/question turns, CPU image encoding with GPU
inference, retained-image follow-up, tool continuation with no Query, and
recovery after a streaming disconnect. The 6,651-token document was sent first;
the subsequent question reused 6,652 tokens, captured Query `[6657, 6673)` and
returned the expected key. These are functional checks, not a performance claim.

On RTX 5060 Ti, GDN state/rollback/continuation/residency/cancellation passed
with `STRATA_GDN_MMA=0` and `1`, using separate identical empty profile seeds.
Two concurrent native sequences also passed with two slots active together;
both retained follow-ups reused prefixes (362 and 386 tokens) without label
cross-contamination.
The optional GDN mode remains off by default. No new AMD hardware, native HIP
compiler, Linux CUDA runtime, SYCL or GPU vision result is claimed here.

### Poly.4 update validation (2026-10-25)

On Windows x64 with CUDA 13.2, architecture 120a and eight parallel build jobs,
the worker and native regression targets build successfully. The server suite
completes 981 tests with 15 skips; the nine adapter tests, ten patch-preparation
tests, browser contracts and shared streaming fixtures pass.

On an RTX 5050 Laptop with Cyber-Tiel-Coder-35B-A3B-MTP UD-Q4_K_XL, the native
checks cover 32/64-token blocks, F16/Q8, graphs on/off, SSD acceptance faults,
accepted KV/recurrent/Mean-K state, bounded replay, elastic workspace lifetimes
and managed MTP. The opt-in GDN checks include an independent double-precision
reference and snapshot/rollback checks; GDN remains off by default.

Separate document/question HTTP turns pass with both block sizes. The document
is identical in both runs; each question reuses 12,684 prefix tokens and reports
the native Query span `[12689, 12705)`. These are correlated functional probes,
not an independent performance comparison. Two concurrent sequences also pass,
with two native slots active together. No poly.2/poly.4 speed comparison or
Linux/HIP/SYCL build is claimed by this Windows CUDA validation.

## Configurable KV management blocks

PolyStrata accepts `--kvmem-block-tokens 32|64` in the native worker and
`tools/polystrata/server.py`. The Windows launcher exposes `-BlockTokens 32|64`,
and the console JSON field is the integer `BlockTokens`. Omission selects 32.
The console's PolyStrata resource settings provide the same choice and verify
the worker capability before launch. Native `KVMEM_BLOCKS` and `/props` report
the effective value. Changing the value requires a service restart; all native
slots, MTP state and retained histories in that process share it.

B, R, sink and recent remain token counts. B must be at least 128; B/R/sink/recent
round down to the selected block size with a configuration notice. R/recent must contain at least one block,
sink + recent must fit B, B + R must fit context, and prefill is capped at the effective R with a configuration notice.
For example, B=160, R=32, sink=32, recent=32, prefill=32 is valid with 32-token
blocks. Native Q8 channel groups and the attention sampling cadence stay at 32;
they are independent of KV management blocks.

Smaller blocks refine recall and eviction boundaries. At fixed token budgets
they do not halve the native KV payload. Mean-K entries, feedback page columns,
PageSoftmax workspace and host catalogs grow with the block count; host/SSD
records become smaller. Checkpoint layout identity includes the block size, so
incompatible views are rejected. Existing SSD storage remains process-local.

For the 35B Qwen3.5 MoE shape (10 full-attention layers, 2 KV heads, head dimension
256), 200,000 tokens require 3,125 blocks at 64 or 6,250 at 32. With a resident
B+R of 34,816 tokens, one active slot and an exactly 200,000-token configured
context, the extra persistent GPU workspaces are approximately 28.2 MiB without
MTP, or 29.2 MiB with MTP. If the configured context is 262,144 while the actual
conversation is 200,000, the estimate becomes approximately 32.9/34.0 MiB.
If all 200,000 tokens are resident (B+R=200,000), the estimate is approximately
150.9 MiB without MTP or 157.0 MiB with MTP. Logical context and resident KV
capacity therefore need to be distinguished when comparing memory use.
These are allocation estimates for this shape/settings, excluding CPU catalogs;
expert caching can consume released VRAM and hide the delta in total GPU usage.
They are not quality or end-to-end throughput guarantees.

The implementation is carried as the ordered `block-tokens` patch after
`elastic-mtp`. Verification covers both kernel specializations against an
independent causal softmax reference, partial Mean-K blocks, CUDA Graph reuse,
accepted MTP prefixes, native KV archives, layout rejection and launch/UI paths.
The model HTTP probe in `tools/polystrata/block_tokens_api_test.py` sends the
document first and the question separately, saves requests/responses and per-turn
timings, and requires real prefix reuse and a bounded native Query replay.

The 2026-10-25 RTX 5050 Laptop allocation audit used Q8 KV, context 262,144,
B=32,768, R=2,048 and no MTP. Both block sizes reserved 400 MiB of native KV
and mapped the same 80 MiB initially / 400 MiB at full resident capacity.
The 32-token configuration left 3,698 MiB for the initial expert cache versus
3,732 MiB at 64; both held the same 255 MiB free-memory target. This approximately
34 MiB shift into other allocations agrees with the workspace estimate, and
demonstrates why total GPU usage alone can hide the block-size overhead.

For strict MTP numerical tests, expert placement and scheduling are held fixed
for both prefix evaluations; ordinary production routing is unchanged. Automatic
CPU/GPU scheduling made the original repeated-prefix byte comparison unstable
at both block sizes. Fixed routing produced identical accepted KV, recurrent
state and Mean-K, including the additional 31- and 159-token prefix cases.

## Attention acceleration

The Python launcher, also used by the Windows console, enables upstream
`STRATA_FATTN=1` and `STRATA_FDEC=1` by default. These select flash attention
for prefill and tensor-core attention for decode when the worker supports them.
The native worker checks hardware and scratch-space compatibility and retains
its ordinary attention path when an accelerated kernel cannot run.
Explicit environment values are preserved; set either variable to `0` before
starting the console or Python launcher to disable that path for comparison.
The native startup log confirms whether each accelerated kernel was selected.
These switches do not change the KV budget, generation reserve, KV type or MTP.

The KVMem attention-feedback path splits sampled QK work by page and shares key
loads across grouped query heads. It still normalizes over the complete causal
window and publishes only accepted samples. Its temporary reductions reuse the
existing attention score workspace, so they do not reserve additional expert VRAM.
Unsupported shapes or insufficient scratch retain the original capture kernel.
For Qwen35 GQA8, warps reduce pages independently. Sampled and unsampled decode
use separate CUDA-graph variants so unsampled steps omit feedback kernels and
redundant coordinate updates; sampling and accepted-token boundaries are unchanged.

Host transfer buffers now grow on demand. History checkpoints archive dirty KV
in slices of at most 4096 rows; only a view exchange requires a full-view buffer.
The restore buffer is allocated only when KV is moved back to the GPU. Bounded Query staging
is likewise allocated only for a Query that needs capture. This reduces pinned
RAM pressure without changing the KV budget, history quota, sampling or replay
policy. The declared Host MiB remains a KV payload quota, not a total RAM limit.
See the [RTX 5050 performance report](polystrata-performance-20261024.md) for
paired timings, correctness checks, remaining gaps and runtime provenance.
The [32K follow-up](polystrata-32k-performance-20261024.md) records the next
feedback-kernel and graph optimization, all paired reruns and their variability.

## Block-size validation on 2026-10-25

The final GPU matrix used RTX 5050 Laptop (device UUID retained in local evidence) with
Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL. CUDA 13.2 / MSVC builds used eight
parallel compilation jobs. Earlier RTX 5060 Ti diagnostics are kept separately.

- Both block sizes passed independent F16/Q8 causal-softmax references, Mean-K
  boundary cases and CUDA Graph coordinate replay. The native history/session
  policy tests also passed.
- Full 32/64 Q8 MTP accepted-prefix matrices passed with graphs; 32 F16 passed
  without graphs. Checks include native KV bytes, recurrent state, Mean-K,
  incompatible checkpoint layout rejection, rejected-token sampling, reset and
  bounded Query/replay. The 32 Q8 run also passed SSD restore and acceptance-fault
  invalidation/rebuild.
- 32-token elastic workspace cycles, rollback/retry, cache growth, Pressure and
  Query/replay passed. The MTP/elastic combination preserved logits, draft
  probabilities and accepted state across full/narrow workspace mappings, then
  completed 48 growth cycles. A 4,161-token archive test verified full native
  bytes and a dirty partial tail across the 4,096-row staging boundary.
- The 32-token Q8 HTTP concurrency test reached two simultaneous active slots;
  distinct conversation labels stayed isolated and both follow-ups reused prefixes
  (362 and 386 cached tokens).
- The console, frontend, browser and Windows launcher checks passed, including
  numeric draft persistence, native capability/actual-value agreement, old-worker
  rejection and default-64 behavior. All seven native patch layers reproduced
  their recorded tree hashes through the repository's preparation checker.

The two-stage HTTP probe used identical 12,682-token reference prompts, B=2,048,
R=512, Q8 KV, two MTP drafts and eight CPU threads. Both document replies were
`Received.`; both question replies returned the expected reference key. Each
question reused 12,684 prefix tokens, captured native Query `[12689,12705)` and
replayed `[12689,12714)`. Requests, responses, document text and native logs were
saved separately for each run.

| Block tokens | Document HTTP time | Question HTTP time |
| --- | ---: | ---: |
| 32 | 90.875 s | 3.140 s |
| 64 | 87.625 s | 1.656 s |

These are single-run smoke-test timings, not a throughput benchmark or a general
quality comparison. The 200K figures above are memory estimates, not a 200K recall
test. HTTP concurrency was rerun at C2; C4/C8 and image requests were not rerun.

## Validation on 2026-10-23

- Reconstructed the native patch against the pinned base and verified the resulting tree.
- Built the CUDA worker and native regression targets with eight compilation jobs.
- Ran the Python console, packaging, portability, public-evidence and adapter suites;
  ran the Windows backend launcher contract and the complete PolyStrata launch-plan chain.
- Verified independent scorer/LRFU combinations and explicit Query/replay limits.
- Passed the PolyStrata browser fixtures and JavaScript reasoning/history contract.
- On RTX 5050 Laptop, passed native normalized capture, Query spans, graph positions,
  F16/Q8 page-mass references, history/session planning and shared quota checks.
- On that card, both F16 and Q8 passed separate document and question turns: the
  document reply was `Received.`, the question reused 3438 prefix tokens and returned
  the expected reference key. Requests, responses and native spans are retained locally.
- Q8 MTP accepted-prefix and cancellation checks matched serial native KV bytes,
  recurrent/hidden state and MeanK statistics in the selected one-case regression.

Prior integration validation also covered C2/C4/C8, SSD, acceptance faults and CPU/GPU
vision. Those full matrices were not all rerun for mainline adaptation. No performance
comparison was run as part of this merge validation.

## Native CI coverage

The `polystrata-cuda-build-linux` job prepares and verifies the pinned patch
series, then builds `strata-poly`, `poly_gdn_mma_state_test`, `poly_elastic_test`
and all four `poly_kvmem_*_test` targets against
the same checkout's `kvmem/` core. It uses CUDA 13.2 with an explicit Blackwell
architecture, and the GGML commit pinned by PolyStrata. It does not reuse a
previously installed worker. Compiler caching speeds up repeated builds; build
and test diagnostics are saved as a CI artifact.

The hosted runner has no GPU or model. It executes only the history and session
policy tests; the kernel, MTP, GDN state and elastic targets are compiled and
linked, not executed. Host CI also checks image Query/no-Query boundaries and
the vision log's unload/shutdown lifetime. The CMake dependency-contract test
evaluates both backend branches without compiling HIP; it is not a HIP build.
The separate `polystrata-hip-build-linux` matrix compiles and links the worker
and five regression executables for `gfx1100` and `gfx1201` with ROCm 7.2.4.
It pins the official container digest and BLAS development package versions
and does not execute HIP binaries or access a GPU. Failure in either
architecture fails the aggregate merge gate.
GPU kernel correctness, F16/Q8 MTP boundaries, chunked archive integrity and
two-session model isolation still require local GPU qualification. A successful
CI build is not evidence of inference correctness or performance on its own.

Native code, patches, version pins and CUDA recipe changes trigger this job.
Documentation-only changes may skip it. Its result is included in the existing
`host-and-server-units-linux` aggregate gate, so a failed or unexpectedly skipped
PolyStrata build cannot be reported as a successful native check.

## Known limits

- `truncate_middle: true` is rejected when KVMem is enabled, even for short or image
  prompts. Middle truncation breaks the original history and Query boundaries.
  Remove the setting or set it to `false`; requests exceeding the logical context
  must be shortened. Startup, model switching and config edits validate this
  combination, and request preparation also rejects a stale enabled setting.
  PolyStrata without KVMem retains its native middle-truncation option.

A prior two-stage, 7131-token code-document probe answered both questions correctly
at B2048 and with original full context. B1024 answered the first correctly but
returned `Received.` for the second. Query/replay and prefix boundaries were valid;
this is evidence of a small-budget quality limitation, not a universal B2048 guarantee.
The two questions share a document and are correlated probes.

In the earlier RTX 5060 Ti fixed-output, fresh-process, no-CUDA-graph F16 comparison,
median prefill throughput was 279.8 tokens/s upstream and 259.3 tokens/s integrated
(-7.3%). Decode was 37.96 versus 37.70 tokens/s (-0.7%). The prefill difference remains
unresolved and does not establish a general throughput regression across configurations.

Multi-GPU and batch MTP are unsupported in this integration. HIP and SYCL were not
validated. SSD history is process-local. Host limits account for native KV payload
and Gemma window-state images, not total process memory or every auxiliary
checkpoint/workspace allocation.
