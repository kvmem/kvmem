# llama.cpp checkpoint save on RTX 5070 Ti Laptop

A checkpoint now forks private retrieval statistics and legacy rows once. In the same drained execution it exports the active target and available draft packed records. The ordinary conversation-bundle fork keeps its original semantics. A wrong-frontier request fails before capturing packed KV. Checkpoint publication still requires both exports and the native model state to succeed.

## Paired native measurements

Both workers used the llama.cpp adapter from #275 at `5076cd6`, main at `35fe8e8`, and the same coordinated raw statistics API from #280 at `bbe32a3`. B includes the new checkpoint save path. On Windows, MSVC 19.44 rebuilt all common host sources, native adapter translation units and the server driver. Both used the identical pinned llama.cpp and SM120a GPU objects. A fresh CUDA configure failed at nvcc host detection with the portable MSVC layout. The CUDA objects were reused from the validated CUDA 13.2 build. The five upstream patches passed the repository's tree-hash check.

Each row summarizes two fresh processes in ABBA order. Each process handled eight requests. Both arms used the same IQ3_XXS model file, Q8_0 KV, context 8192, batch 128, 1024-token working budget, Q/D 1024/1024 MiB, 12 checkpoints, one hot checkpoint and spread interval 256. Speculation was off for timing. Prefill and decode are sums of the eight native phase times. Capture is cumulative native attention checkpoint time. RSS is the server's Windows peak working set. GPU is sampled total device memory. Byte counters come from the final service monitor.

| Model, KV RAM H | Arm | Prefill s | Decode s | Capture ms | Packed K+V captured MiB | SSD read / written MiB | Peak RSS MiB | Peak GPU MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dense, 128 MiB | A | 2.832 | 0.344 | 643.8 | 453.75 | 357.697 / 329.808 | 1323.7 | 669 |
| dense, 128 MiB | B | 2.584 | 0.329 | 479.2 | 453.75 | 357.697 / 329.808 | 1311.5 | 670 |
| dense, 1024 MiB | A | 2.734 | 0.336 | 630.5 | 453.75 | 0.000 / 0.002 | 1640.8 | 670 |
| dense, 1024 MiB | B | 2.541 | 0.335 | 468.9 | 453.75 | 0.000 / 0.002 | 1631.5 | 671 |
| hybrid, 128 MiB | A | 4.504 | 0.321 | 240.2 | 133.73 | 150.752 / 1561.757 | 3788.7 | 2541 |
| hybrid, 128 MiB | B | 4.449 | 0.327 | 187.3 | 133.73 | 150.752 / 1537.132 | 3835.3 | 2541 |

Dense prefill fell 8.7% at H=128 MiB and 7.1% at H=1024 MiB. At the higher H both arms read 224 bytes and wrote 1,904 bytes. Dense captured the same 453.75 MiB and reused the same 12,397 tokens. The improvement came from removing work on checkpoint indexes and private statistics. At H=128 MiB the hybrid capture phase fell from 240.2 to 187.3 ms. Its prefill means differ by 1.2%. Only 75 dense and 26 hybrid tokens were decoded per process. These timings give limited evidence for decode or small hybrid throughput changes.

All eight message hashes and per-request reuse counts matched between arms for each model. The comparison measures the increment over the previous #275 adapter. The [earlier pre-integration comparison](llamacpp-hot-native-state-20261107.md) used a different base and recorded three dense wording differences.

## Resident KV at restore

A separate diagnostic worker, built from the previous #275 adapter, sampled the quiescent restore boundary before clearing outgoing cells. For each desired canonical slot it checked the block's valid rows, native cell membership, logical and execution positions, spatial fields and token metadata. It compared every live K and V tensor row for every target layer with the corresponding saved packed bytes. The entire-view check also required the native view to be valid and free of extra occupied cells. The probe covered single-GPU dense Q8_0 without a draft. Its device reads were excluded from all timing runs.

At both H settings, **0 of 7** restores had a whole matching view. Per run, 41 of 177 requested blocks had the required placement, row metadata and exact bytes (76.23 of 324.29 MiB). One restore matched 32 of 33 blocks. These are observations at specific fences. A production partial-reuse path would need native mutation, placement and ownership proofs plus failure-safe publication. This change retains the native restore transaction for every request.

## Correctness and limits

Both models passed the native 64/65/95/96-row cases. They covered full-record sharing, private tails, the new wrong-frontier rejection, immutable old checkpoints, recycled native cells, sparse selections, old-parent and branch restoration, allocation pressure and empty rollback. The lane test passed. The service matrix passed SSD cold restore, SSD pressure, hybrid MTP replay and saved MTP state, JSON/SSE budget rejection and admission, and disconnect rollback. A capacity rejection restored 2,856 old-parent rows with zero recomputation. Dense external-draft speculation, multi-GPU layouts and other KV types were outside this run.

The [run data](llamacpp-dense-checkpoint-save-20261107.json) retain the phase and byte counters, model and binary hashes, per-response hashes and restore observations. The [diagnostic patch](llamacpp-dense-restore-probe-20261107.patch) applies to the earlier #275 adapter for reproducing the restore observations.
