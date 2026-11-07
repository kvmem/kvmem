# PolyStrata checkpoint retention and restore costs

On a single RTX 5050 Laptop, retaining Qwen recurrent checkpoints on SSD let an old branch resume with 49 new prompt rows instead of 3,217. A separate change reduced pinned KV transfer buffers by 21.23 MiB. Sustained generation throughput varied in both directions. The [measurement data](polystrata-checkpoint-costs-20261107.json) include request fixtures, native counts, per-request latency, worker hashes, checkpoint costs and peak memory for every run.

## Changes and attribution

| Stage | Comparison | Observed effect |
|---|---|---|
| 1. Baseline and instrumentation | Master `a8cca288` with diagnostic counters | Identical request bodies across builds. Version arrays occupied 24–792 B in the three-request RAM workload. No performance gain attributed to tracing. |
| 2A. Tier native state | Baseline vs state-only, 256 MiB RAM + 2 GiB SSD | In both repeats, directory evictions fell from 4 to 0 and parked entries rose from 0 to 4. Requests 7 and 8 read 49 and 23 new native prompt rows, down from 3,217 and 3,127. The pair took 2.14–2.22 s with state tiering alone, down from 19.41–21.47 s. SSD writes increased from 151.57 to 829.08 MiB. |
| 2B. Incremental recurrent state | Compare each 4 KiB range in native checkpoints at frontiers 64 and 81 | Across 65,863,680 B, identical ranges covered 0 B with both F16 and Q8 KV. The native GDN updates recurrent state on accepted tokens. This append trace provides no reusable chunks or producer dirty contract. The adapter continues capturing the full image. |
| 3. Bound KV restore input | State-only vs state + slices, same pressure workload | Peak input fell from 30,638,080 to 8,377,600 B. Output stayed at 30,638,080 B. Combined pinned allocation fell from 58.44 to 37.21 MiB (36.3%). Old-branch timings overlap the state-only runs. No separate latency gain established. |
| 4. Reserve manifest capacity | Same frontier 3,208, same 101 records | Measured manifest size fell from 11,864 to 10,584 B. Version storage stayed at 808 B. At 512 and 4,096 blocks the isolated savings were 6,368 and 41,568 B per manifest. Timings varied with allocator and machine load. No version-table redesign or throughput gain attributed to this change. |

## End-to-end results

With 512 MiB RAM, the streaming workload used a cold 3,129-token document, a fork, and a separate 256-token generation. Prefill is the first request's prompt throughput. Decode is the long generation's reported throughput. Values within a row follow run order.

| Build | Cold prefill (tok/s) | Long decode (tok/s) | Cold client TTFT (s) | Long-generation TTFT (s) | Peak process-tree RSS (MiB) |
|---|---|---|---|---|---|
| Baseline | 146.9 / 161.3 / 185.3 | 29.7 / 30.4 / 33.7 | 21.344 / 19.438 / 16.937 | 1.015 / 0.688 / 0.843 | 18,295 / 17,753 / 17,566 |
| Final | 133.6 / 189.6 | 26.2 / 33.2 | 23.438 / 16.516 | 1.125 / 0.688 | 16,067 / 18,238 |

The adjacent baseline/final runs gave 185.3 → 189.6 tok/s prefill and 33.7 → 33.2 tok/s decode. Earlier runs had larger swings. Multi-token draft acceptance and available host RAM varied. These samples establish neither a general speedup nor an RSS reduction. All runs peaked at 7,140 MiB sampled VRAM. RSS and VRAM are one-second samples.

The pressure workload used nine serial, non-streaming requests with branch creation, edits, old-parent return and anonymous reuse. The baseline and final totals varied from 50.83–59.75 s and 29.23–59.41 s respectively. In the slowest final run, request 1 alone took 34.84 s. The wall time for requests 7 and 8 together was 19.41–21.47 s initially and 1.92–2.42 s finally. Both final runs processed the same counts of new native rows. The pressure fixture did not measure client TTFT.

| Pressure workload, cumulative unless indicated | Baseline | Final |
|---|---:|---:|
| Native KV archive D2H | 114.96 MiB | 66.56 MiB |
| View D2H, including verification | 678.23 MiB | 514.24 MiB |
| View H2D | 396.52 MiB | 349.40 MiB |
| Captured native state | 1,382.05 MiB | 1,382.05 MiB |
| SSD reads | 140.08 MiB | 315.38–315.75 MiB |
| SSD writes | 151.57 MiB | 829.08 MiB |
| Retained physical model state at last request | 188.46 MiB | 879.48 MiB (188.46 resident, 691.02 cold) |
| Peak process-tree RSS | 17,033 / 19,526 MiB | 15,995 / 18,264 MiB |

Native state capture counts image bytes once per successful capture. Verification independently exports native state and reads back GPU views. The intermediate sliced build omitted verification downloads from its view-D2H counter. The baseline and final columns above both include them. Physical state is a subset of storage usage. Metadata, model workspaces, verification scratch and pinned transfer buffers have different lifetimes. The legacy aggregate physical-payload counter includes the storage account's charges. It cannot measure KV-only deduplication. A cold state read needs private admitted RAM for one complete image.

## Correctness, scope and reproduction

The dedicated Q8 MTP test (`--window 3 --one-case --q8`) passed its serial/speculative accepted-state comparison, sampled/unsampled graph checks, bounded Query replay, native transfer verification and cancellation/reset checks on the 5050.

The native history and sessions tests passed. The F16 and Q8 snapshot tests checked complete-block origin identity, independent partial tails, byte-exact sparse and old-parent GPU views, native state round trips, SSD-cold forks, branch edits, admission exhaustion and zero RAM/SSD charge after release. The tests injected cold-read failure, exhausted the cold restore buffer budget and cancelled after a partial device upload across a non-block-aligned slice. A retained source restored on retry in each case. Invalidated session stamps rejected stale completion. The HTTP pressure run also executed bounded Query replay.

The tiered state path uses the existing Qwen GDN schema with SSD enabled. Capture fills a private mutable record, fences the device stream and seals the record. A fork retains the published root. Each slice keeps the source leases until completion. SSD reads fill the input buffer on the I/O worker and GPU uploads execute on the engine thread after a stamp check. Cancellation drains pending I/O before releasing the buffers. The output spill buffer remains sized for the selected view. A native row larger than the input target requires one whole row. The tests cover single-device text inference. They do not qualify Gemma window state, multimodal state, HIP runtime or multi-GPU restore.

Use Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf with Q8 KV, context 16,384, budget 2,048, reserve 512, prefill 128, sink/recent 128, eight retained sessions, three MTP drafts, concurrency one and thinking off. Elastic expert and physical headroom were 128 and 512 MiB. Enable PolyStrata verification, session metrics, storage metrics and tracing on both builds. Start a fresh worker per run. The JSON `fixtures` section stores the exact HTTP bodies with a shared message table. Expand each `message_ids` list into `body.messages` and submit each series in order to the chat completions endpoint. Steady-generation throughput uses 256 output tokens. The raw benchmark logs used the native reported reused/read split to verify the pressure result.

Run `poly_kvmem_snapshot_test MODEL.gguf` and repeat with `--q8`. For the [CPU microbenchmark](polystrata-manifest-cost-20261107.cpp), compile against the same release KVMem headers and library. It holds three manifests, runs reserve off/on/on/off at each size, warms up once and times 200 constructions per sample. The live checkpoint timing includes metadata admission and the temporary exchange headroom reservation, which may trigger storage I/O. The CPU microbenchmark isolates construction from those operations.
