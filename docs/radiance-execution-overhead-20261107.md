# Radiance integration execution overhead

This change stays on Radiance 1.17 (`19d4e8fe0e18071b9b6300f15fc4ccf38dc4fa3d`).
It addresses overhead introduced by installing KVMem's observer; it does not
require Radiance to use NInfer's execution or KV representation.

## Implementation

### Recorded passes

`KVMEM_RAD_TAPE_MODE=off|draft|all` selects the control path, draft-only tapes, or
tapes with live observer steps. The default remains off during qualification.
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

## Regression coverage

- Replay changes capture destination, query slice, request identity and input
  values while holding the same pass key; it crosses the 1024-play audit boundary.
- Rejected target rows do not enter accepted statistics. A live callback failure
  must fail a replay as well as an ordinary pass.
- Fake-queue tests prove that collection/destruction does not execute unrelated
  work after a completion event; failure tests prove partial work is drained.
- Native GPU tests cover Q/K ordering, attention numerics, selected-page restore,
  RAM/SSD snapshots and repeated transfers beyond the native queue-ID limit.

Performance results and final default selection are pending qualification.
