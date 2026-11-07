# llama.cpp hot native state on RTX 5070 Ti Laptop

The hybrid server now writes the native recurrent serializer's output directly into a private HostKvRecord under the execution's checkpoint RAM allowance Q. The producer seals it before publishing the checkpoint. A hot restore holds a read lease through the native setter, including when another owner drops the record. MTP carry still passes through the speculative API's vector and is charged to the producing or reading lane during the copy.

The existing victim policy reserves an extent on the same shared SSD file D. A StorageIoQueue worker holds the source, frames and checksums the two native spans, and writes the extent. Only successful completion allows the lane to publish the new backing. Readers already using RAM finish under their leases. A queue refusal, cancelled job, full disk or failed write leaves the hot source usable. Cold restore validates identity, lengths and checksum into private reader staging before the native setter runs. The full scan runs on the spill worker, so the image-write timer now measures worker time and can overlap inference.

## Same-device comparison

Two separate ABBA sequences per model. Each row is the median of two fresh server processes running the same eight requests. A is either the immediate #275 predecessor at `6c2f29f` or the instrumented pre-integration baseline. B is the optimized worker built from `7daa8e4`. Both A binaries match the hashes in the [original experiment](llamacpp-shared-snapshots-20261107.json). All runs use the same device, two IQ3_XXS model files, pinned native tree, Q8_0 KV, batch 128, context 8192, working budget 1024 tokens, Q/H/D of 1024/128/1024 MiB, spread interval 256, 12 checkpoints, one hot entry, and speculation off. Prefill and decode are the sum of the server's eight phase timings. Only 75 dense and 26 hybrid decode tokens were measured per run.

| A compared with B | Model | Arm | Prefill s | Decode s | Peak RSS MiB | Sampled GPU MiB | SSD read MiB | SSD written MiB | Actual K+V captured MiB | Native state captured MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous #275 | dense | A | 2.704 | 0.336 | 1322.9 | 670 | 359.56 | 329.81 | 453.75 | 0.0004 |
| Previous #275 | dense | B | 2.737 | 0.331 | 1321.9 | 669 | 359.56 | 329.81 | 453.75 | 0.0004 |
| Previous #275 | hybrid | A | 9.175 | 0.359 | 3714.4 | 2541 | 201.00 | 1663.38 | 133.73 | 2562.7799 |
| Previous #275 | hybrid | B | 4.525 | 0.324 | 3788.9 | 2541 | 175.88 | 1588.01 | 133.73 | 2562.7799 |
| Pre-integration | dense | A | 1.785 | 0.319 | 1034.6 | 671 | 170.60 | 261.01 | 408.02 | 0.0004 |
| Pre-integration | dense | B | 2.752 | 0.343 | 1322.1 | 669 | 359.56 | 329.81 | 453.75 | 0.0004 |
| Pre-integration | hybrid | A | 4.215 | 0.327 | 3853.8 | 2541 | 175.88 | 1532.64 | 117.66 | 2562.7799 |
| Pre-integration | hybrid | B | 4.530 | 0.320 | 3766.4 | 2541 | 201.00 | 1599.88 | 133.73 | 2562.7799 |

Relative to the previous #275 worker, hybrid prefill fell from 9.175 to 4.525 s (50.7% shorter). Hybrid output hashes and all 13,038 reused rows matched. In each candidate run of this pair 52 records were created, with 30 or 31 observed completed spills. The ten reads comprised six or seven RAM borrows and three or four full cold validations. The pre-integration pair measured 4.215 versus 4.530 s, leaving a 7.5% hybrid prefill gap. Total native capture was 2,687,269,544 bytes in every arm. Spill completion and placement can vary at the final monitor sample.

Dense had only 408 bytes of native state per eight requests. Against the previous #275 worker its prefill moved from 2.704 to 2.737 s (+1.2%, ranges overlap), with identical messages, packed capture, disk traffic and all 12,397 reused rows. Against pre-integration it took 2.752 versus 1.785 s (+54.2%). Attention snapshot metadata, raw record imports and growing tails remain separate work. Against pre-integration, dense message hashes differed on question, switch-back and middle-edit in both the previous #275 worker and the hot-state worker. The question and switch-back returned the expected instrument with a different article. All eight hybrid message hashes matched across both comparisons. These two samples per arm describe these fixtures.

## Recovery and qualification

- The lane host fixture exercises hot address identity, account ownership through demotion and release, concurrent private cold reads, corrupted SSD bytes and low-reader-budget rejection. Every CPU and GPU test ran against the same optimized server sources.
- The hybrid traced suite passed with SSD, native MTP replay and explicit MTP snapshots, including disk pressure with RAM fallback. Real socket cancellation with and without MTP reproduced the saved parent's output.
- At Q=128 MiB both JSON and SSE rejected a late generation commit and released branch state. Q=192 MiB now admitted both protocols, verified by an explicit positive fixture. The former 192 MiB negative expectation was observed to fail because the JSON request committed successfully. At Q=256 MiB and D=64 MiB the rejected MTP branch restored the parent's 2,856 rows with zero recomputation. A new generation endpoint at this limit returned 500.
- The binary hashes, request hashes, counters, physical I/O and process memory samples summarized per run, raw manifest/log hashes and qualification results are in the [measurement data](llamacpp-hot-native-state-20261107.json). Run the same benchmark and native fixtures described in the [original report](llamacpp-shared-snapshots-20261107.md), substituting the matched binaries in each sequence. Run the failure fixture with defaults for the 128 MiB case and with `--expect-commit --host-mib 192` for the positive case.

The optimization changes the llama.cpp adapter only. Existing limitations include other KV formats, DFlash, multimodal projector input, multi-GPU and sustained decode.
