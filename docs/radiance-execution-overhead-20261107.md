# Radiance integration execution overhead

This change stays on Radiance 1.17 (`19d4e8fe0e18071b9b6300f15fc4ccf38dc4fa3d`).
It addresses overhead introduced by installing KVMem's observer; it does not
require Radiance to use NInfer's execution or KV representation.

## Implementation

### Recorded passes

`KVMEM_RAD_TAPE_MODE=off|draft|all` selects the control path, draft-only tapes, or
tapes with live observer steps. The default is `off`; see the qualified scope below.
Ordinary inference without an observer retains upstream behavior.

A tape stores the layer/group, anchor and logical Q/K operand descriptions.
Recording pauses around each observer invocation. Replay resolves the current
buffer and KV bindings and calls the current observer at that point in the
producer stream. Request-owned destinations, capture events and attention scratch
are never embedded in the device tape. Observer descriptors participate in the
existing first-replay and periodic audit. Callback errors invalidate the step.
Calibration, profiling, unsupported geometry and native staging retain their
existing fallbacks.

### Transfer and attention completion

Successful page transfers wait on their own completion event. A completed poll or
drain also proves destruction safe, avoiding repeated full-stream waits. Copies
using the producer stream do not enqueue a redundant event/wait pair on that
same stream. Partial enqueue failures still drain the queues that may reference
the staging memory or source-ready event.

Attention capture records completion after the last required layer copy.
Collection waits for that event, not for subsequently queued draft passes. Empty
sample frames submit no copies and need no wait. Missing layers and event errors
fail closed; cancellation still drains partially submitted work.

The spill transaction also borrows its completed pinned capture for the CPU
storage job, eliminating the whole-window pinned-to-vector copy. Transaction
cancellation/destruction joins the storage ticket before releasing the transfer.
The worker performs no device calls. Per-record staging and the shared storage
quota remain unchanged. `KVMEM_RAD_BORROW_SPILL=0` restores the copying control
path; borrowing is the default. `KVMEM_SPILL_COPY` reports avoided CPU copy bytes
and the largest removed whole-window staging allocation for that native owner.

For new RAM records this removes one of the spill path's three host copies; GPU transfer bytes are
unchanged. It does **not** eliminate full selected-view RAM/pinned restore staging
or the separate prefix-checkpoint capture vector. Do not equate the removed
allocation's size with measured process RSS reduction.

### CPU statistics overlapping draft execution

`KVMEM_RAD_EARLY_COLLECT=1` opts into early accepted-token readback and enables
the statistics producer's early completion event. The engine lets the observer
consume the accepted rows while the drafter runs. CPU mean-K, query and attention
bookkeeping precede an explicit all-rank fence. KV rollback, selection, checkpoint
publication and sequence retirement remain after that fence. The engine also
waits before scheduler commit and terminal callbacks. Failed early collection
also fences before a per-request capacity/storage refusal can clear the session.

This is within-step overlap. Cross-step run-ahead remains disabled: it would
require independent observer frames and safe ownership of two mutable native
views. The native architecture's prefill pipeline remains unchanged.

The common storage/provider interfaces and cache-hit policy are unchanged.
The optional early-collection contract is internal to Radiance's StepObserver.

## Measurement and acceptance

Use the ordinary `qwen3.8-27b-iq3r-v4.rad`, DFlash7, TP1 on the same RX9060XT.
Do not substitute IQ3R-S or upgrade the upstream revision between runs.

| Experiment | Intended mechanism | Target, not a promised gain |
| --- | --- | --- |
| Tape | Avoid repeated architecture traversal and kernel submission setup | CPU submission time -20%; decode +5% |
| Spill copy | Remove one whole-window host memcpy and allocation | New RAM-record spill host-copy bytes -33%; measure wall time and RSS independently |
| Synchronization | Stop waiting after the relevant producer has completed | Target-stage time -10%; lower TTFT/wall time where that wait is on the critical path |
| Early collection | Overlap accepted-row CPU processing with draft kernels | Decode/throughput +5%, with no unacceptable TTFT or token-gap regression |

Compare original worker, sync-only candidate, draft-only tapes, full tapes,
early collection without tapes, and the combined candidate. Run three repetitions
with reversed ordering on the middle repetition, exclude warmups, fix prompt and
generated length, and compare output/acceptance as well as timing. A non-target
regression exceeding 3% beyond measured noise requires investigation or gating.

Final coverage includes the canonical 256K Task2 with budget 36864 and final
generation cap 4096, SSD/Query/cancellation and concurrent branches. Model answer
quality and runtime/cache correctness are separate outcomes.

The final candidate passed 155 checks in the 12K RAM/SSD/Query/tool/cancellation
suite, the C2 branch suite (18 requests), six native/portable test executables,
and the native-memory GPU suite again with borrowing disabled. Local Python
coverage passed 89 tests. The complete pinned native patch chain was reapplied
to a clean checkout and its resulting file inventory/hashes matched the prepared
build tree. The replay producer test runs 1,035 changing frames across the
periodic audit boundary. The C2 suite checks cache behavior, not C2 performance.

## Regression coverage

- Replay changes capture destination, query slice, request identity and input
  values while holding the same pass key; it crosses the 1024-play audit boundary.
- Rejected target rows do not enter accepted statistics. A live callback failure
  must fail a replay as well as an ordinary pass.
- Fake-queue tests prove that collection/destruction does not execute unrelated
  work after a completion event; failure tests prove partial work is drained.
- Native GPU tests cover Q/K ordering, attention numerics, selected-page restore,
  RAM/SSD snapshots and repeated transfers beyond the native queue-ID limit.

## Measured results

Hardware: RX 9060 XT (`gfx1200`, same device in every run), Ryzen 9 3950X,
ROCm HIP 7.16, Linux, TP1. The model is ordinary IQ3R v4, SHA-256
`99f306975bf5f48d6b023a177893fa85c6469ee080f2d7955caa3eaa16b84542`.
Both workers use the same native kernel binary. This is not an upstream or model
upgrade. Machine-readable measurements are in the
[companion benchmark report](benchmarks/radiance-execution-overhead-20261107.json).

### Fixed-output decode ablations

Each mode had three fresh worker loads, one warmup and three measured requests
per load (nine measured samples). The middle repetition reversed mode order.
The prompt asks for consecutive integers: 47 input tokens and exactly 256 output
tokens, greedy sampling, seed 42, thinking off, DFlash7, prefill 256. This is a
high-acceptance workload, not representative of every Task2 answer.

| Mode | Decode mean ± sample SD, tok/s | Mean TTFT, ms | Mean HTTP wall, s |
| --- | ---: | ---: | ---: |
| Original worker | 135.564 ± 0.360 | 209.170 | 2.1092 |
| Completion-event changes | 135.824 ± 0.564 | 209.934 | 2.1066 |
| Above + draft-only tapes | 136.290 ± 1.092 | 208.860 | 2.0990 |
| Above + full tapes, early collection off | 139.015 ± 0.572 | 209.238 | 2.0625 |
| Completion events + early collection, tapes off | 138.259 ± 0.362 | 206.037 | 2.0697 |
| Completion events + full tapes + early collection | 140.043 ± 0.377 | 206.801 | 2.0467 |

All 72 outputs, including warmups, have the same hash and accepted 223 of 231
offered draft tokens. Combined decode improved **3.30%**, and HTTP wall time fell
**2.96%** against the original. Relative to completion events alone, full tapes
improved decode 2.35% and early collection improved it 1.79%; these are separate
ablations and must not be added. Completion events alone had no established
speed benefit. The 5% decode target was **not met**.

This matrix used candidate v1, before the borrowed-spill optimization and the
early-failure cleanup fence were added. It exercises neither spilling nor that
failure path. Final-candidate GPU/functional tests, host traces and long-context
tests qualify the later code separately.

### Host submission and copies

Separate final-candidate CPU traces exclude the warmup and cover 102 target,
102 draft-history and 102 drafter submissions per worker, with no dropped spans.
Their summed `forward_submit` elapsed time fell from **114.201 ms to 19.860 ms
(-82.61%)**. Target submissions alone fell from 96.601 ms to 18.617 ms.
These are host submission durations, not CPU utilization or GPU kernel times.
The monitor recorded 396 tape replays, four recordings, four audits and zero
refusals. The host-submission target was met; the smaller end-to-end improvement
shows that removing this overhead does not remove the GPU workload.

The final 12K RAM/SSD/Query suite avoided **276,824,064 CPU copy bytes (264 MiB)**,
with a largest removed staging allocation of **16 MiB**. GPU transfer bytes,
per-record storage copies and common resource reservations are unchanged.
RAM/SSD native-memory GPU tests pass with borrowing both enabled and disabled.
The 14-round Task2 copy controls completed but performed zero transaction spill
copies in either mode: checkpoint retention had already archived those blocks.
Those timings cannot demonstrate a spill-copy speedup. A capture/restore wall-time
or RSS reduction is therefore **not established** by this result.

### Defaults and limits

Completion-event waiting and borrowed spill captures are enabled by default.
Tape replay and early statistics collection remain opt-in: the measured decode
gain is below the acceptance target, and Windows GPU, TP2, MTP and streaming
tail-latency qualification are not covered here. To reproduce the tested Linux
configuration, set these variables in the **worker's environment** before launch:

```sh
export KVMEM_RAD_TAPE_MODE=all
export KVMEM_RAD_EARLY_COLLECT=1
# KVMEM_RAD_BORROW_SPILL=1 is already the default.
```

Check `/api/monitor/snapshot` → `kvmem.execution` for the enabled controls and
actual replay/audit counters after a request has run: this metadata is refreshed
by engine steps, so the initial startup snapshot can precede tape initialization.
Setting tape mode to `off` and early collection to
`0` restores the conservative execution path; `KVMEM_RAD_BORROW_SPILL=0` restores
the host-copy control. These variables are read at worker initialization, so
changing them requires a worker restart.

This delivers safe **within-step** overlap, not cross-step run-ahead or a new
prefill pipeline. No claim is made that the synchronization target (-10% target
stage time) or a p95 streaming-gap target has been measured or met.

### Paired 256K Task2

One run per worker, combined candidate first and original worker second. Both
use ordinary IQ3R v4, DFlash7, B36864/R4096, prefill 256, H4096 MiB/D8192 MiB,
C1, one retained session, context 262144, seed 42 and temperature 1.0. The fixed
fixture retains its 128-token thinking budget, 512 intermediate generation cap
and 4096 final generation cap. These benchmark caps do not change the unlimited
chat-service settings. Outputs stop naturally rather than being padded to 4096.

| Metric | Original | Combined | Change |
| --- | ---: | ---: | ---: |
| Total native prefill time, s | 419.389 | 414.618 | -1.14% |
| Effective prefill, tok/s | 615.079 | 622.155 | +1.15% |
| Followup decode, native token count, tok/s | 55.378 | 54.377 | -1.81% |
| Final code decode, native token count, tok/s | 55.380 | 56.495 | +2.01% |
| Total HTTP request wall, s | 617.371 | 615.525 | -0.30% |
| Sampled board VRAM peak, MiB | 14603.520 | 14609.434 | +5.914 MiB |
| Sampled running process RSS peak, MiB | 8587.039 | 8589.723 | +2.684 MiB |
| Process swap peak, MiB | 0 | 0 | unchanged |

Effective prefill uses `(final_prompt_tokens - 1) / total_prefill_seconds`,
matching the existing Task2 report. It is not a separately measured first-pass
kernel throughput. The followup decode aggregate includes the final request and
subtracts one first token per request from the decoded token count. Board VRAM
is device-global, sampled about every 0.4 seconds; RSS about every 0.2 seconds.

Both runs completed 33/33 requests, with 32/32 cached followups, a final prompt
of 257,958 tokens and 2,305 final output tokens. **Every round's content and
reasoning output matches**, as do draft acceptance counts (7,016/17,752,
39.52%). Combined replay counters show 8,229 plays, 334 recordings, 267 audits
and zero refusals. Both passed all runtime/cache/memory gates.

There is **no established overall long-task speedup** from this single pair.
The small prefill/final-decode improvements coexist with slower aggregate
followup decode, and total wall time is effectively unchanged. No measured
regression exceeds the 3% investigation threshold, but this is not a statistical
confidence interval or evidence of a universal non-regression.

Quality is unchanged, including existing failures: only **11/31** intermediate
batch acknowledgements contain both the correct ID and checksum. Both final
answers have valid Python syntax and an argparse reference; semantic correctness
has not been established. Runtime completion must not be presented as a quality
pass.

## Integration with the concurrent storage change

During qualification, master advanced to `f3134a084ccdcd045886799ff470e62edf378301`
(PR #271: shared packed-record/state-storage contracts). Those common files were
combined with this adapter, rebuilt, and passed all six Radiance test executables
plus five common tests (checkpoint images, packed snapshots, tiered storage,
raw KV and KV snapshots). The real-model RAM/SSD/Query/tool/cancellation suite
also passed again on that integrated worker.

The performance tables intentionally retain the original paired v2 measurements;
they do not silently substitute the later common-storage implementation. The
machine-readable report separately identifies the integrated v3 binary and its
compatibility evidence. Raw synthetic test archives are retained with SHA-256
and file counts in that report.
