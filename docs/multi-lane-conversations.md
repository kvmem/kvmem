# Multi-lane conversations

`--parallel P` selects independent GPU execution lanes (1 through 8). Model weights are shared; every lane has its own KV working set, recurrent buffers and optional MTP follower. More lanes permit overlap but require more working-set/scratch memory and do not guarantee higher throughput.

The current [content-snapshot contract](checkpoint-policy.md#content-snapshots-across-backends) governs reuse. `--kvmem-sessions N` configures internal retained content groups. The deprecated explicit `--kvmem-conversations N` path is normalized to at least P and cannot be combined with `--kvmem-sessions`. The current catalog retains at most `C=max(2*N,P)` physical versions under its publication policy. This differs from the historical N-total-store limit; report group and physical-version limits separately.

For example:

```sh
llama-kvmem-server -m model.gguf --device CUDA0 --flash-attn on \
  --parallel 2 --kvmem-sessions 3 --threads-http 8 \
  --kvmem-budget 2048 --kvmem-gen-reserve 1024 --kvmem-cpu-gb 0.5
```

Choose budgets for the model and GPU. In this example P=2, C=6, and the separate legacy arena commitment cap is `(C+P)*0.5GiB=4GiB`, plus raw KV, checkpoints and other allocations. Arenas are lazy, not all allocated at startup. This is not the former three-store 1.5GiB layout. Default CPU bytes are zero; public SSD and the legacy CPU arena remain mutually exclusive. [Multi-conversation KV cache](multi-conversation-kv-cache.md#legacy-cpu-arena-ledger) describes its copy and reservation ledger.

Multiple lanes require at least `2*P` HTTP workers; automatic worker selection enforces that floor. `/slots` always has P lane entries, independently of how many content versions are parked.

## Scheduling and cache hints

Ready requests are considered in arrival order while skipping unavailable execution resources. Media preparation happens before taking a lane; a request can retry preparation when its cache hint was evicted. Preview results are hints, and admission rechecks content/native state. Queued previews do not pin all retained metadata or reserve GPU residency. Active, switching and admitted source bindings are protected during native work.

`kvmem.conversation_id`, `prompt_cache_key` and `promptCacheKey` are validated compatibility metadata, not mutable cache owners. The same, different or missing key follows the same content-selection policy. Applications must order dependent turns themselves; cache aliases do not provide an explicit stateful-session contract. A returned protocol resource handle, where supported, has its own lifecycle.

A completed native prefix can seed a source-preserving fork on an available lane. Sealed KV and immutable images retain references; selectors, MeanK, Query, feedback, rollback and checkpoint wrappers remain private. A native-successful continuation can inherit its source's internal capacity group. Group equality alone cannot select or overwrite a source. Tool continuation still requires content/native continuation proof and uses resident-only Pressure rather than rediscovering cold history.

A warm resident continuation can leave the GPU working set in place, but first creates the required independent host source. This is still a source-preserving branch and can incur CPU-arena/state cost. `restore_path=continuation` does not imply zero copies or physical GPU residency.

| Event with two lanes | Lane 0 | Lane 1 | Retained sources |
|---|---|---|---|
| A and B begin | A running | B running | Existing complete versions stay independent |
| A finishes; C is admitted | C running | B running | A can remain as an immutable host source |
| A returns while C runs | C running | A branch running | B and older eligible versions can remain |
| A continuation becomes ready | C running | A continuation | Native proof and budgets are rechecked |

A physical-slot or Host/checkpoint-pressure victim is selected through the common group-fair policy: old redundant versions precede another group's sole version, while pinned execution bindings remain protected. Leased bytes may remain charged after a directory entry is retired. If no eligible resource can be released, admission or optional reuse can fail; the source must not be consumed to manufacture room. A new request does not acquire an unbounded extra arena merely because it wants the same prefix.

Disconnect, completion and error paths release operation/source reservations. A partially repaired native binding can be cleared into a cache miss, and a storage/GPU failure can return an error. This does not promise to restore every old cache byte after arbitrary device failure.

## Switching a lane

The outgoing entry is marked as transitioning while the incoming entry is reserved. Inference and copies execute outside the catalog mutex.

1. Reserve any additional legacy CPU arena before constructing a runtime or source-preserving fork.
2. Synchronize outgoing writers and harvest partial KV/MeanK; drain the outgoing working set when a handover needs it.
3. Invalidate outgoing target/MTP cell mappings, positions and attention-view epochs while preserving the engine-sized buffers.
4. Move raw KV, runtime, row positions and Query accumulators as one native bundle. A parked runtime has no lane backend pointer; shared sealed payload references survive independently.
5. Bind the incoming bundle, restore its native working set and recurrent/MTP state, then publish matching committed metadata.

The arena ledger counts active lanes, parked bundles and in-flight reservations under the catalog mutex. Normal and exceptional attach paths account installed handles before returning the reservation. Its configured-commitment charge is independent of raw H/D and checkpoint pools and is not an RSS limit. Runtime geometry, Flash Attention/KV support and accepted-state checks still apply to every restore.

## Shared projector

`--mmproj projector.gguf --mmproj-device none` selects CPU encoding; `--list-devices` lists other available placements. The projector is loaded once independently of target lanes. Size shared GPU placement for model weights, all lane working sets, projector weights and encoder peak scratch together.

Tokenization/encoding use a shared mutex before lane admission. Requests consume immutable embeddings/native chunk metadata without holding the encoder mutex during target decode. The bounded media cache and in-flight embeddings share their configured ceiling; live references can make preparers wait with cancellation support. Native video/audio concurrency is outside this lane implementation.

## Status and current validation scope

Catalog-to-lane mappings remain one-to-one. An entry resident on another lane is not rebound onto a source lane. A detached branch can use the source lane while the old host source stays preserved; source protection does not imply unlimited capacity. Publication may decline a version when all admissible victims are protected.

`/props` reports P, retention configuration, per-store CPU bytes and the legacy soft byte target. `/slots` includes group/version limits, physical payload counters, independent CPU arena live/reserved/charged/budget fields, and actual hit observations. `X-KVMem-Lane` identifies the physical lane. Revision2 adds `cache_semantics_revision=2` and `restore_path=continuation|snapshot|miss`; unknown native observations stay null.

`tests/lane-conversations-test.cpp` covers controlled source protection, handover failures, checkpoint admission, group-fair reclamation and CPU-arena reservations. `tests/lane-pool-test.cpp` tests the generic queue primitive, including its keyed-queue mode; that alone does not prove current cache aliases serialize or overlap in the HTTP server. `kvmem/tests/backend_rebind_test.cpp` checks runtime migration constraints. GPU qualification additionally needs real simultaneous native work, not just two overlapping HTTP wall-clock intervals or a peak admitted counter.

Source preservation can keep additional full checkpoint images alive. A functional test with a larger checkpoint pool, or a 2N-version catalog compared with the former N-store catalog, must record that resource change. It is not evidence of unchanged-budget performance improvement.

## Historical validation

The following record belongs to its 2026-10-01 binaries and configurations. Its counts are preserved as history; they do not qualify the new content-cache code, arena ledger, current group/version capacities or untested hardware. Old exact-output/store-count expectations are not the sole correctness oracle.

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

`--kvmem-host-mib H --kvmem-disk-path PATH --kvmem-disk-mib D` supplies the shared raw payload account across lanes. H must accommodate concurrent working windows. It cannot be combined with the optional legacy CPU arena. The old NVMe and separate session-disk interfaces are retired; see [migration and historical validation](legacy-ssd-retirement-20261019.md).

The lane runner accepts `--host-mib`, `--disk-mib`, `--disk-path` and an optional DFlash2 `--draft-model`/`--draft-tokens` pair. A storage qualification must show actual SSD reads/writes and preserve source state through failure/cancellation; enabling flags alone is insufficient. Model, KV/MTP type, H/D, checkpoint and arena scope, GPU UUID and native overlap evidence must accompany the result. Different-budget or different-GPU arms are reported separately.
