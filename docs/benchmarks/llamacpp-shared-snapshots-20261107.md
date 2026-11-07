# llama.cpp immutable KV checkpoints on RTX 5070 Ti Laptop, 2026-11-07

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

## Correctness, failure cases and scope

- Native dense Qwen3-0.6B and hybrid Qwen3.5-4B, both IQ3_XXS: 64/65/95/96 rows, shared complete block allocation, private tails, unchanged old bytes after native cell reuse, discontiguous selection, coverage-hole rejection, repeated branch/parent restore and release. At 65 rows the H=32 MiB test reaches real allocation failure and recovers the original native bytes without another forward pass. Restoring an empty accepted root preserves its lineage.
- Hybrid HTTP with no speculation and with native MTP (default replay and explicit state snapshots): edit/branch/old parent, tools, Query and SSD cold reads. For the full traced workload H=128 MiB, D=1024 MiB without MTP, D=2048 MiB with MTP. Separately D=64 MiB, H=512 MiB, Q=2048 MiB reaches an actual SSD demotion refusal and completes using retained RAM.
- Q=192 MiB: late JSON/SSE commit rejection, rollback, repeat rejection and release. With MTP, Q=256 MiB and D=64 MiB: a rejected child cannot pay for itself by evicting its parent. The parent restored 2,856 rows with **zero recomputed rows**; publishing a further generation endpoint at this budget correctly returned 500. Real TCP disconnect after output with and without MTP released branch staging and reproduced the parent output.
- Backend lane host tests cover distinct reader Q charges, concurrent cold reads, damaged image validation before publication, rejected staging, independent parent pins and exception cleanup. Native content equality tests compare model state bytes; the HTTP correctness checks additionally verify reuse and outputs. GPU qualification was serial; parallel GPU executions are not established by the host tests.
- Unqualified here: DFlash, vision/projector input, multi-GPU, other KV types/models, SWA or pure recurrent KVMem paths, sustained decode, injected mid-prefill GPU/DMA faults. Do not generalize this measurement to those combinations.

## Reproduction and artifacts

On a dedicated idle GPU, run `tests/llamacpp-snapshot-benchmark.py` with `--exe SERVER --model GGUF --gpu DEVICE --out NEW_DIR --mode spread-disk --context 8192 --lines 128 --host-mib 1024 --kv-host-mib 128 --disk-mib 1024 --count 12 --hot 1 --interval 256 --tool-continuation`. Use a new output directory and fresh process each time. Repeat with both binary hashes in ABBA order and each model hash from the [machine-readable evidence](llamacpp-shared-snapshots-20261107.json). For the source-pressure regression run `tests/llamacpp-checkpoint-failure-regression.py` with the same executable/model/GPU selectors, a fresh output directory, `--ssd-budget --mtp`.

The [implementation audit](../llamacpp-snapshot-audit.md) documents the baseline ownership and restore paths, the public dependency, and the native patch provenance.
