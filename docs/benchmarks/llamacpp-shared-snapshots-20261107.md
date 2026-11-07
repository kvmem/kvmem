# llama.cpp immutable KV checkpoints on RTX 5070 Ti Laptop, 2026-11-07

The [hot native state follow-up](llamacpp-hot-native-state-20261107.md) supersedes the native-state transport and performance results below. The original measurements describe the earlier worker.

The candidate restores immutable parent snapshots through dense and hybrid branch, replay, cancellation and capacity-failure paths. It does **not** improve the performance of this workload. Actual packed K/V capture and prefill time increased on both models. The instrumented baseline already skips ordinary attention KV in native PARTIAL_ONLY checkpoints; eliminating a dense full-KV export was never the opportunity here.

## Controlled performance comparison

One idle Windows RTX 5070 Ti Laptop (12 GiB class, SM120a), CUDA 13.2.86, MSVC 19.44; one fresh server for each run. A is the instrumented pre-integration server with corrected dense capacity, error handling and actual capture counters. B includes immutable attention snapshots, native state storage and branch protection (code through `c538cc7`). Both use the same pinned upstream sources, CUDA kernels, model file, GPU and runner settings. The candidate relinks the changed adapter and host storage objects. Native llama.cpp commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8`, prepared tree `1a8155219356073fe1f8a3528d2c7f203521554b`; no changes to upstream patches or build lock.

For each model we ran A, B, B, A in order. Each run performed eight requests: document, question, tool continuation, tail edit, switch away, switch back, middle edit and front edit. Temperature zero, trace off, one lane, Q8_0 KV, no speculative decoding, context 8192, working budget 1024 tokens, generation reserve 256, batch/ubatch 128. Retention: spread interval 256, count 12, hot 1; checkpoint RAM Q 1024 MiB, packed KV RAM H 128 MiB, shared SSD D 1024 MiB. The public JSON contains both binary and model hashes, normalized argv, per-request timings, raw file hashes and all eight runs.

Medians of two fresh runs per arm. Time is the sum of the eight server-reported prompt/eval phases; it includes runtime work and is not a kernel-only timer. There were only 75 dense and 26 hybrid decode tokens per run: the decode column measures these short responses, not sustained generation throughput.

| Model | Arm | Prefill s | Decode s | Prefill / decode tokens | Actual reused rows | Peak RSS MiB | Peak commit MiB | Sampled GPU peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dense | A | 1.846 | 0.325 | 6842 / 75 | 12397 | 1034.9 | 1586.3 | 671 |
| dense | B | 2.686 | 0.328 | 6842 / 75 | 12397 | 1326.2 | 1869.3 | 671 |
| hybrid | A | 4.235 | 0.303 | 7077 / 26 | 13038 | 3803.1 | 4715.2 | 2541 |
| hybrid | B | 8.981 | 0.338 | 7077 / 26 | 13038 | 3713.6 | 4605.5 | 2541 |

Prefill took 45.5% longer on dense and 112.1% longer on hybrid. Every run returned the expected instrument in its answer and the same number of reused rows within each model. Peak RSS is the Windows process working-set high-water mark. GPU memory is device-wide framebuffer usage sampled at intervals of at least 200 ms, so a shorter peak may be missed. Neither number is the checkpoint RAM charge.

| Model | Arm | Packed K MiB | Packed V MiB | Native state MiB | SSD read MiB | SSD written MiB |
|---|---|---:|---:|---:|---:|---:|
| dense | A | 204.01 | 204.01 | 0.0004 | 170.60 | 261.01 |
| dense | B | 226.87 | 226.87 | 0.0004 | 359.56 | 329.81 |
| hybrid | A | 58.83 | 58.83 | 2562.78 | 201.00 | 1605.89 |
| hybrid | B | 66.86 | 66.86 | 2562.78 | 201.00 | 1663.38 |

Capture counts successful packed transfers and published native payload bytes, excluding alias-only forks. Dense native state was exactly 408 bytes per run; hybrid native state was 2,687,269,544 bytes per run in both arms. K and V totals include archive captures; they are not logical bytes referenced by snapshots. SSD totals are observed before shutdown and include native records on the shared spill file. Native records use Q while hot; packed attention uses H. Neither budget bounds process RSS.

On hybrid the candidate spent about 4.19 s writing framed native images and 0.89 s reading them across the eight requests. Those timers combine copy, framing and checksum. The available API requires an additional simultaneously admitted capture buffer and a private checked read before invoking the native setter. The timings do not isolate checksum cost or explain the entire difference. The coordinator deferred a direct-producer/leased-view interface; this PR does not recreate it in the backend. Two runs per arm support a comparison on these fixtures, not a population confidence interval.

## Why prefill regressed: attention records versus native state

The Radiance fixes target two different copies. [PR 267's measurements](radiance-shared-snapshots-20261106.md) compare newly captured attention bytes with the logical size of full checkpoint views: immutable full blocks can be reused instead of exported repeatedly. [PR 274's staging ablations](../radiance-adapter-staging-20261107.md) separately remove an intermediate pinned-to-vector copy and bound pinned KV restore staging. Neither experiment measures llama.cpp's recurrent sequence state.

**Dense (+0.840 s per eight-request run).** All native state captures together were 408 bytes in both arms. In B, median process time for framed state write plus read was 0.36 ms. This rules out the native state checksum as the explanation. The candidate spent 0.634 s capturing attention snapshots and 0.300 s restoring them. Those timers include useful work, can include storage waits and span the requests; their sum is not a before/after subtraction. Final checkpoint RAM grew from 2.42 to 170.01 MiB, of which 170.01 MiB in B is metadata/staging and just 336 bytes hot native records. Packed K+V capture grew by 45.73 MiB and shared SSD reads by 188.96 MiB. These are different counters: a packed capture measures device-to-host transfers; a CPU metadata copy or a read of an already sealed record does not increment it.

The code does not export the full packed prefix again at every checkpoint. In `harvest_gpu_v`, covered resident layers skip capture. `RawKvStore::export_packed` forks the same sealed allocation for complete proven records and copies the valid rows of a partial tail into a private allocation. However, `llama_kvmem_checkpoint_capture` calls `fork_active_conv` at each new frontier: it checks resident coverage, materializes missing rows, and clones a conversation and its raw statistics. After export, importing the empty manifest builds another metadata index and copies the retained raw row/MeanK vectors to strip packed bindings. Normal restore forks this metadata and imports the selected packed manifest, again rebuilding the index before restaging the native layout. The B phase timings, RAM footprint and I/O locate the expensive path; they do **not** partition its time between metadata copying, tail recapture and SSD waits. There is no inference of identity from token hashes or native cell IDs.

**Hybrid (+4.746 s).** The same 2,687,269,544 native bytes were captured in each arm. A kept the native serializer's vectors as its hot checkpoint, wrote a checksummed image on the storage queue only for selected demotions, and returned the existing hot data directly on materialization. B serializes into private vectors, immediately copies them into a framed record and hashes every byte on the calling lane. A non-live restore also validates/copies the record before the native setter, including when it is in RAM. The real B process spent 4.187 s in image writes and 0.888 s in reads, versus 0.238/0.107 s in attention capture/restore. SSD read bytes were identical and packed K+V capture increased by only 16.07 MiB. The timing scopes are not additive with request latency. These figures implicate moving full-state framing onto the hot path, not an increase in native model-state serialization volume.

**Same-host CPU control (no GPU).** [Source](../../tests/llamacpp-snapshot-image-benchmark.cpp) and [all observations](llamacpp-snapshot-image-cpu-20261107.csv). Build a release executable with `/O2 /MD /std:c++17` and the merged host library. Each operation uses 52,691,548 payload bytes, two warmup rounds, then twelve rotating/reversed rounds with preallocated hot buffers. Copy controls use the public record/lease in the same 1 MiB chunks, without framing; the hash-only control runs the actual SnapshotWriter with a no-op sink. Complete read/write use the production checkpoint image API. No checksum is disabled in the server.

| Operation | Median ms |
|---|---:|
| Chunked write, copy only | 3.494 |
| Framed record write | 49.587 |
| Chunked hot read, copy only | 3.473 |
| Validated framed hot read | 49.166 |
| Existing bytewise FNV-1a, hash only | 47.002 |

In this hot-RAM fixture a serial hash dominates the cost of one image; eliminating only the memcpy cannot remove that scan. The experiment excludes allocation, GPU serialization, disk, and concurrent inference. It is not a measurement of how many seconds changing a hash would save in production.

**Follow-up for the coordinator.** For native state, evaluate direct production into a private admitted record plus an immutable validated hot lease. Preserve cold identity/length/checksum before use; never substitute an unchecked pointer for a validated record. Benchmark synchronous validation separately from one-time sealing, and distinguish hot/cold restores. For dense, first measure the private metadata clone/import and SSD waits separately, then evaluate a public operation that forks native statistics without cloning and immediately dropping packed bindings. Keep unchanged-block allocation identity, tail privacy, row-coverage proof, failure atomicity and budgets unchanged in any ablation. Neither optimization is part of this PR.

## Correctness, failure cases and scope

- Native dense Qwen3-0.6B and hybrid Qwen3.5-4B, both IQ3_XXS: 64/65/95/96 rows, shared complete block allocation, private tails, unchanged old bytes after native cell reuse, discontiguous selection, coverage-hole rejection, repeated branch/parent restore and release. At 65 rows the H=32 MiB test reaches real allocation failure and recovers the original native bytes without another forward pass. Restoring an empty accepted root preserves its lineage.
- Hybrid HTTP with no speculation and with native MTP (default replay and explicit state snapshots): edit/branch/old parent, tools, Query and SSD cold reads. For the full traced workload H=128 MiB, D=1024 MiB without MTP, D=2048 MiB with MTP. Separately D=64 MiB, H=512 MiB, Q=2048 MiB reaches an actual SSD demotion refusal and completes using retained RAM.
- Q=192 MiB: late JSON/SSE commit rejection, rollback, repeat rejection and release. With MTP, Q=256 MiB and D=64 MiB: a rejected child cannot pay for itself by evicting its parent. The parent restored 2,856 rows with **zero recomputed rows**; publishing a further generation endpoint at this budget correctly returned 500. Real TCP disconnect after output with and without MTP released branch staging and reproduced the parent output.
- Backend lane host tests cover distinct reader Q charges, concurrent cold reads, damaged image validation before publication, rejected staging, independent parent pins and exception cleanup. Native content equality tests compare model state bytes; the HTTP correctness checks additionally verify reuse and outputs. GPU qualification was serial; parallel GPU executions are not established by the host tests.
- Unqualified here: DFlash, vision/projector input, multi-GPU, other KV types/models, SWA or pure recurrent KVMem paths, sustained decode, injected mid-prefill GPU/DMA faults. Do not generalize this measurement to those combinations.

## Reproduction and artifacts

On a dedicated idle GPU, run `tests/llamacpp-snapshot-benchmark.py` with `--exe SERVER --model GGUF --gpu DEVICE --out NEW_DIR --mode spread-disk --context 8192 --lines 128 --host-mib 1024 --kv-host-mib 128 --disk-mib 1024 --count 12 --hot 1 --interval 256 --tool-continuation`. Use a new output directory and fresh process each time. Repeat with both binary hashes in ABBA order and each model hash from the [machine-readable evidence](llamacpp-shared-snapshots-20261107.json). For the source-pressure regression run `tests/llamacpp-checkpoint-failure-regression.py` with the same executable/model/GPU selectors, a fresh output directory, `--ssd-budget --mtp`.

The [implementation audit](../llamacpp-snapshot-audit.md) documents the baseline ownership and restore paths, the public dependency, and the native patch provenance.
