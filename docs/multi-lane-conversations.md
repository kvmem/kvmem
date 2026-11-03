# Multi-lane conversations

`--parallel P` is the number of independent GPU execution lanes (1 through 8).
`--kvmem-conversations N` is the global number of host conversation stores.
After environment/CLI parsing and before validation, N becomes `max(N, P)`.
The server logs an adjustment once and exposes requested/effective values.

For example:

```sh
llama-kvmem-server -m model.gguf --device CUDA0 --flash-attn on \
  --parallel 2 --kvmem-conversations 3 --threads-http 8 \
  --kvmem-budget 2048 --kvmem-gen-reserve 1024 --kvmem-cpu-gb 0.5
```

Choose budgets for the model and card. Each of the P contexts gets its own
`budget + gen_reserve` GPU KV window, recurrent buffers and optional MTP follower.
Model weights are shared. `--kvmem-cpu-gb` is per host store, so 3 stores at
0.5 GiB can allocate 1.5 GiB of arenas, plus raw KV and checkpoints. P stores
exist at startup; additional stores are lazy, with a hard global N limit.
`--kvmem-conversations-gb` is a soft cap on accounted retained host bytes, not
process RSS. Idle parked, unreferenced stores are evicted by LRU; resident or
queued stores can keep the total above the cap, with a warning.

For four lanes, use `--parallel 4 --kvmem-conversations 8 --threads-http 16`.
Multiple lanes require at least `2 * P` HTTP workers; automatic worker selection
also enforces that floor. More lanes increase GPU working-set/scratch memory
and permit more simultaneous execution, but do not guarantee higher throughput.

## Scheduling

A conversation has host state and, optionally, one lane residency. No lane is
assigned permanently. Completion leaves a warm residency; returning requests
prefer it when free. A parked conversation can resume on any free lane.

Ready request heads are considered in arrival order, skipping heads whose store
or lane is busy. A request still preparing media is not ready. With an explicit
`kvmem.conversation_id`, requests serialize in arrival order before preparation,
so a second turn never reads a half-written first turn. Different IDs can use
other lanes while that turn waits. Queued references protect host caches from
soft byte-cap eviction, but cannot reserve the last count-limited slots or pin
their GPU residency. Without IDs, the existing
media-aware prefix and recurrent-checkpoint policy infers cache identity;
explicit IDs are recommended when overlapping requests belong to one conversation.
An ID never bypasses prefix/checkpoint validation.

The modern host/SSD tier and multi-conversation path also allow a different
owner to fork an exact native prefix. Full owner keys and their aliases follow
the [shared-owner contract](checkpoint-policy.md#shared-owners-across-backends).
The source catalog entry and sealed allocations remain valid when the child
rewinds, generates, fails or is released. A live source is captured read-only;
only the subsequent normal handover moves the lane working set. Old pinned CPU
arenas do not support these foreign forks; the monitor reports
`shared_prefix_fork_supported=false` for that configuration.

| Event (P=2, N=3) | Lane 0 | Lane 1 | Parked in RAM |
|---|---|---|---|
| A and B begin | A running | B running | none |
| A finishes; C arrives | C running | B running | A |
| B finishes; A returns while C runs | C running | A running | B |
| A finishes; queued A2 is ready | C running | A2 running, warm | B |

At N=P, a new conversation recycles an eligible idle store instead of allocating
an N+1 arena. If all stores are busy or switching, it waits. Queued cache hints
alone cannot prevent admission; their prefix and media needs are rechecked.
An invalid request is rejected before any reservation, eviction or KV switch.
Disconnected waiters release queue/cache references. Completion, disconnect and
errors release lane reservations and commit a usable prefix or clear a failed
restore into a cache miss.

## Switching a lane

The outgoing entry is marked as transitioning while the incoming entry is
reserved. Inference and GPU copies execute outside the scheduler mutex.

1. Synchronize outgoing writers, harvest partial KV/mean state and drain the
   GPU working set to the host mirror.
2. Invalidate all outgoing target/MTP cell mappings and positions, reset the
   lane's slot allocator and invalidate attention-view epochs. The buffers stay.
3. Park raw KV, runtime arena, row positions and query accumulators as one
   independently mutable bundle. Its sealed KV and read-only checkpoints may
   retain references shared with another bundle. A parked runtime has no lane backend pointer.
4. Bind the incoming bundle to this lane, stage its valid working set and
   restore recurrent/MTP checkpoints through the existing prefill path.
5. Publish matching metadata only after quiescence. Scheduling reads committed
   snapshots, never a busy lane's changing token/checkpoint containers.

Transfer/restore failures keep ownership consistent and invalidate affected
cached content for subsequent requests; the current request can return an error.
This does not promise to roll back every old cache byte after a GPU/storage error. Startup and swaps
check the model, KV types, MTP presence and pool geometry.

## Shared projector

Add `--mmproj projector.gguf --mmproj-device none` for CPU, or a device reported
by `--list-devices`, such as `CUDA0` or `Vulkan0` when built with Vulkan.
The projector is loaded once and its placement is independent of target lanes.
It may share the target GPU; size for model weights, both lanes, projector
weights and the encoder's peak scratch memory together.

Tokenization and encoding use a shared mutex. Encoding happens before acquiring
a lane, including when both lanes are decoding. Request-owned immutable
embeddings and native chunk metadata are then consumed without holding the
encoder mutex during target decode. Identical media reuse the bounded embedding
cache. Cache plus in-flight embeddings share a 128 MiB ceiling; an individual
prepared request exceeding it is rejected, and other preparers wait with
cancellation support when live references occupy the ceiling. Native video/audio
concurrency is outside this multi-lane implementation.

## Status and validation

Shared-prefix admission keeps the catalog-to-lane mapping one-to-one. A new
or detached branch can run on the source lane; a slot resident on another lane
cannot be rebound there. When the legacy total-store limit leaves no room for
a detached branch, a resident target runs fresh on its own lane. This preserves
the source when a separate target exists, at the cost of recomputing that request.
If the only usable target is the request's own source, fresh execution replaces
that old history. Queued preview pins are hints, not hard count reservations:
selection replans and rechecks media preparation after reuse is lost. Active or
switching entries remain protected. Inactive-limit mode can still allocate a
detached shared branch while another lane is empty.

`/props` reports physical P, effective/requested N, per-store CPU budget, soft
host byte cap and projector placement. `/slots` always has P entries. Each
entry reports the same global store count/counters, its resident store and
transition state. `X-KVMem-Lane` identifies the selected physical lane.

`tests/lane-pool-test.cpp` covers eligible FIFO, same-ID serialization,
cancellation and media retry. `kvmem/tests/backend_rebind_test.cpp` verifies that
resident/pending runtimes refuse migration and drained runtimes allocate/free
on the new backend. `scripts/test_server_lane_conversations.py` checks real KV
hits and secret/image isolation across a lane migration, capacity normalization,
invalid input, disconnect recovery and mixed projector/decode work; use
`--conversations 1` for N<P and `--mmproj ... --mmproj-device ... --mtp` for vision/MTP.
Add `--parallel 4 --conversations 5` to exercise four lanes and a parked store.

Local validation on 2026-10-01 used Windows/MSVC, CUDA 13.2.86 and Vulkan:

| Check | Result |
|---|---|
| Model-free Windows CTest | 16/16 passed |
| CLI compatibility | 78/78 passed |
| Existing single-lane conversation regression (0.8B) | 137/137 passed |
| PR #94 sustained decode/disconnect regression (27B + MTP) | 14/14 passed |
| Same-GPU projector + two lanes + MTP (27B) | 38/38 passed |
| CPU projector + two lanes + MTP (27B) | 37/37 passed |
| Intel Arc 140T projector + two CUDA lanes + MTP (27B) | 37/37 passed |
| N=1 normalized to P=2, full-capacity recycling (0.8B) | 26/26 passed |
| Legacy query policy, no CPU arena, lane migration (0.8B) | 32/32 passed |
| Global soft-byte cap and parked-store reclamation (0.8B) | 27/27 passed |

The GPU tests used an RTX 5060 Ti 16 GiB for the 27B target and synthetic color
images; they verify correctness and overlap, not a throughput improvement.
Linux builds and host tests remain covered by the repository's GitHub workflow.

The [RTX 5050 stress-test report](multi-lane-stress-5050-2026-10-01.md) records
2/4/8/16-client ramps and ten minutes of cache churn for P=2 and P=4, the
decode-graph bug found under load, its correction and subsequent recovery/
regression checks. The four-lane run completed 743 requests with zero errors;
P=4/N=5 migration and P=4/N=1 normalization also passed. It observed no
throughput gain on this small-model workload and used about 1.25 GiB more VRAM.

This builds on [qzshch's PR #94](https://github.com/kvmem/kvmem/pull/94).
Its original commit `3cc1c651d5ea64c36e3880ea71c7713823730423` is retained as an
ancestor, with its original author, so GitHub shows the contributor's work.

## Public SSD storage

Public `--kvmem-host-mib H --kvmem-disk-path PATH --kvmem-disk-mib D` works with
multiple lanes; H must hold all concurrent working windows. It cannot be combined
with the optional legacy `--kvmem-cpu-gb` arena. The old NVMe and separate session
disk interfaces have been retired; see [migration and validation](legacy-ssd-retirement-20261019.md).
The existing lane runner accepts `--host-mib`, `--disk-mib`, `--disk-path` and an
optional DFlash2 `--draft-model`/`--draft-tokens` pair. For eight lanes at its default
B2048/R1024, choose a larger H (for example 512 MiB for the small-model fixture)
than the B512/R256 qualification; its final checks require actual SSD reads/writes.
