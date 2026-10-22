# NInfer checkpoint host-buffer residency

KVMem checkpoints keep immutable state images in pinned host buffers. The
default adapter caches two released buffers. Concurrent Prefill/Decode can
keep more images live and cause a subsequent capture to allocate and pin
another image on the executor thread before GPU work is submitted.

Set `NINFER_KVMEM_CHECKPOINT_RESIDENT=1` before starting NInfer to preallocate
and touch the checkpoint working set during model initialization. The default
is `0`; other values are rejected when KVMem is enabled. This option changes
host allocation reuse, with the original direct D2H/H2D copies, synchronization
and last-shared-reference ownership preserved. There is no additional full-image
CPU staging copy, CUDA kernel change or retrieval-policy change.

The reservation in images is:

```
5 * (max_concurrency + retained_sessions) + 2 + MTP_backup
```

Each active or retained window has at most five checkpoint records. The two
extra images cover one replacement capture and one idle-I/O source surviving
history eviction. MTP needs one additional backup because reconstruction visits
lanes serially and completes its copies before releasing that backup. Other
speculative backends do not use this MTP backup. Shared aliases do not reserve
another image. Unexpected extra live leases allocate additional pinned memory
at runtime; the startup reservation is not a hard limit on live allocations.
The pool never reclaims a referenced image.

For three active lanes and one retained session, MTP reserves 23 images.
The `KVMEM_CHECKPOINT_RESIDENT` startup line gives their exact byte size, which
depends on model state layout. This is the total pool reservation, not incremental
memory over existing live checkpoints. It consumes page-locked host memory
separately from the KVMem KV payload budget, and increases startup work. It adds
no device KV or workspace allocation; platform-specific pinned-memory limits
still apply. Allocation failure fails model initialization
before requests are served. The option stays opt-in so callers can choose this
host-memory tradeoff.

The standalone CPU test does not need a GPU or CUDA-linked libraries:

```sh
c++ -std=c++20 -O2 -DNDEBUG -pthread -I backends/ninfer/src \
  backends/ninfer/tests/host/test_checkpoint_residency.cpp -o checkpoint-residency-test
./checkpoint-residency-test
```

It exercises disabled-mode two-buffer reuse, layout matching, shared aliases,
23 independent warmed leases, bounded overflow, shutdown, checked sizing,
allocation/touch failure cleanup, concurrent returns and last-owner lifetime.
The CMake/CTest target is `ninfer_checkpoint_residency_test`.

CPU ownership checks do not qualify GPU transfers, model quality, cache reuse,
SSD loading, or an end-to-end speedup. Those need execution and profiling of
the exact build under the intended recipe. This change does not alter the
existing cold-storage materialization call sites.
