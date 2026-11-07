# llama.cpp checkpointing with one MeanK owner

PR #275 uses the adapter measured at `41a54d0` and the public API in master `35fe8e8`. Fresh checkpoints use `fork_mean_prefix` and packed export/import. PR #280 supplied the comparison worker for this experiment. Full session-file restore has a separate provenance limit.

## State ownership and accepted boundary

The dense target writes prefill MeanK rows and decode sums only to its statistics store. Query scoring, prefix probes and tail repair read the same store. The target's raw store carries native packed K/V. The MTP mirror only writes packed K/V and contributes proved draft records. The recurrent serializer and speculative carry remain in their existing native owners.

The execution owner drains harvest and DMA at the existing checkpoint frontier. A statistics fork owns its private sums. Packed export retains sealed full records and copies accepted tail bytes into zero-padded private records. Restore prepares a private candidate, imports packed records, and reinstalls the saved logical and native state. A partial prefix comes from its actual accepted checkpoint. A naked shorter fork of a grown aggregate cannot recover its earlier sum. The adapter rejects aggregates that exceed the requested tail, including layers restored through a mean-only setter.

The changes in the two production adapter files relative to the #280 preview are 74 added and 25 removed lines. Each live or detached target has one additional index for statistics. Allocation accounting, truncation, transfer, freeze, release and the optional statistics field in the full session stream follow that owner. Legacy combined streams use their original owner and decode path. Full-session flags 0/1 mean combined source with optional MTP. Flags 2/3 carry the separate statistics section. Reading a new split stream with an older binary is outside this experiment.

## Same-device measurements

A is the validated #280 worker from the [previous study](llamacpp-dense-checkpoint-save-20261107.md). B is the unified worker. Both used the RTX 5070 Ti Laptop, the same IQ3_XXS files, CUDA 13.2 SM120a objects, Q8_0 KV, context 8192, batch 128, Q/D of 1024/1024 MiB, eight requests, spread interval 256 and twelve checkpoints. All common host C++ objects and the adapter were compiled for each worker. The five prepared llama.cpp patches and the CUDA objects were reused. Each table row is the mean of two fresh processes in ABBA order. Timings are cumulative across the eight requests. The GPU peak is sampled total framebuffer. RSS is the process high-water mark.

| Model / H MiB | Arm | Attention capture ms | Prefill s | Decode s | Packed K+V captured MiB | SSD read / write MiB | Peak RSS / GPU MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| dense / 128 | A | 514.8 | 2.666 | 0.336 | 453.75 | 357.697 / 329.808 | 1310.3 / 670 |
| dense / 128 | B | 520.6 | 2.646 | 0.341 | 453.75 | 357.697 / 329.808 | 1311.7 / 671 |
| dense / 1024 | A | 450.5 | 2.467 | 0.322 | 453.75 | 0.000 / 0.002 | 1630.6 / 669 |
| dense / 1024 | B | 466.4 | 2.471 | 0.321 | 453.75 | 0.000 / 0.002 | 1629.8 / 670 |
| hybrid / 128 | A | 182.9 | 4.474 | 0.318 | 133.73 | 201.002 / 1582.883 | 3758.7 / 2541 |
| hybrid / 128 | B | 189.4 | 4.468 | 0.336 | 133.73 | 175.877 / 1588.008 | 3760.5 / 2541 |

At H=1024 MiB cumulative dense attention capture increased from 450.453 to 466.443 ms (+15.990 ms, +3.55%). Prefill moved from 2.467 to 2.471 s. Both workers read 224 B and wrote 1,904 B. Dense reused 12,397 tokens and hybrid reused 13,038. Every message hash and per-request reuse count matched. Decode covered 75 dense or 26 hybrid tokens per run. The small prefill differences do not establish a speedup across workloads.

Six separate instrumented runs counted data at successful copy sites in packed export, raw import and the prefix fork functions. Each dense arm copied **48.40 MiB** of private packed tail data. Copied raw statistics/legacy vectors across forks and imports fell from **475.125 to 432.469 MiB**, with the 42.656 MiB import-side copy eliminated. Both H settings agreed. Hybrid copied **15.79 MiB** of packed tails per arm, with vector copies falling from **143.156 to 130.344 MiB**. The production timing runs used neither copy logging nor device readback probes. These counts exclude wrapper allocations, native recurrent bytes and unrelated process memory traffic.

## Correctness and file compatibility

Both models passed native 64/65/95/96-row tests for exact old state, full-record sharing, private tails, native cell reuse, sparse selection, wrong-frontier rejection and empty rollback. The two four-row packed parents with source-specific suffixes produced byte-identical serialized two-row children. Appending only K at row three left V unavailable and packed export at length three failed. The original parents retained their own V bytes. Service tests passed SSD cold reads and pressure, MTP replay and saved MTP state, cancellation and JSON/SSE late budget rejection and admission. A rejected branch recovered 2,856 old-parent rows with zero recomputation. External dense speculative drafts, multi-GPU layouts and other KV quantizations were outside this run.

The old-library fixture wrote actual 65-row full session files. For both dense and hybrid, the candidate rejected a truncated stream, retried the intact file on the same cold owner, and restored byte-identical native state and exact accepted tail statistics. The split writer's files passed the same read test. Subsequent immutable capture rejected all four reopened files because their raw sections lacked persisted coverage. The ordinary live/SSD checkpoint matrix uses retained sealed records and passed. In this source tree the standalone full-stream entry points have no server call sites.

The [host reproducer](llamacpp-unified-raw-probe-20261107.cpp) gives three independent results.

- A complete zero-valued KV page and one missing its third V row serialize to identical RKV1 bytes. Packed export accepts only the complete original. Reopening even that original loses the positive proof.
- A four-row mean-only store returns count 4 and sum 88 from `fork_mean_prefix(2)`. The earlier accepted checkpoint has count 2 and sum 5.
- Generic legacy FP rows retain discarded bytes through `fork_prefix(2)`. Two serialized children differ. After a K-only third write, the legacy V reader returns the discarded value 41 or 91. The pinned adapter's fresh dense and MTP paths never produce these legacy arrays. General legacy partial forks remain unqualified.

## Minimum shared work for broader support

For future self-contained raw files, persist per-plane row coverage in a versioned format. Decode into a private destination under the owning execution, checking account, row/layout bounds, bitmap geometry and payload lengths before publication. Keep RKV1 readable with unknown coverage. An old file cannot recover a missing positive proof from tokens, slot IDs, high-water counts or zero padding.

For legacy CPU rows, a prefix fork must trim or scrub discarded bytes before serialization and preserve independent K/V read validity through asymmetric rewrites. Statistics shortening requires a real accepted tail checkpoint or retained rows that can reconstruct its sum. Keep cancellation checks and metadata/payload admission on the existing owner, release partial candidates on failure, and preserve the source. The cross-version, hole-versus-zero, divergent suffix, K-only rewrite and truncated-read retry fixtures provide the acceptance cases. #280 supplies a statistics-copy operation and does not encode the missing evidence.

[Per-run data, response hashes and actual copy-site counts](llamacpp-unified-checkpoints-20261107.json)
