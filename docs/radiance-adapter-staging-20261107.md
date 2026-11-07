# Radiance adapter statistics and checkpoint staging

This follows the execution-overhead work on the same Radiance 1.17 pin,
`19d4e8fe0e18071b9b6300f15fc4ccf38dc4fa3d`. The native scheduler, kernels,
checkpoint boundaries, cache matching and common RAM/SSD policy are unchanged.

## Changes and expected benefit

| Change | Expected, directly measurable benefit | Control |
| --- | --- | --- |
| Consume single-rank statistics | Avoid rebuilding already-correct row/head arrays; reduce CPU merge time | `KVMEM_RAD_MOVE_STATISTICS=0` |
| Borrow completed checkpoint capture | Remove one full pinned-to-vector CPU copy and temporary vector for newly captured immutable blocks | `KVMEM_RAD_BORROW_PREFIX=0` |
| Chunk checkpoint KV restore | Cap KV pinned staging at 16 MiB and reuse one allocation/event pair across chunks | `KVMEM_RAD_BOUNDED_RESTORE=0` |

All three are enabled by default. Controls are read once per process. Existing
tape and early-collection defaults remain unchanged. Qualification fixes tape
mode to `all`, early collection to `1`, and ordinary spill borrowing to `1` in
both sides of every comparison.

Single-rank moves retain the existing shape, identity, finite-value and attention
normalization checks. Validation finishes before ownership moves. Multiple ranks
retain their row-wise concatenation and head-weighted attention merge. Attention
negative zero is canonicalized as in the original weighted merge.

For shared prefix snapshots, the scheduler drains the D2H transfer and gives the
CPU storage callback an immutable view of its pinned bytes. Pending parking owns
the prefix, and cancellation/destruction joins its storage ticket. The scheduler
releases the transfer only after the callback and its captures have finished;
the CPU worker performs no device operations. Input, fallback and terminal cache
publication all follow this rule. Existing I/O admission charges include the
retained staging. Legacy non-snapshot capture keeps its copying path.

Restore still loads immutable RAM/SSD bytes asynchronously before reserving GPU
pages. It then copies through a single pinned allocation of at most 16 MiB,
waiting for completion before filling that allocation again. Chunks can split
native pages and layers; the last chunk may be shorter. Native page tables are
published only after all chunks complete. Failure drains outstanding copies
before releasing unpublished pages. A reused transfer clears its old completion
event state before submitting the next chunk, including failure paths.

The 16 MiB bound applies to **checkpoint KV H2D staging**, per sequential rank
restore. It does not bound recurrent/draft state buffers or remove the full
CPU restore image. This change reuses staging within one restore; it does not
introduce a global pinned pool, chunked SSD read pipeline, or cross-step execution.
Capture retains pinned memory until parking completes, rather than retaining an
ordinary vector during that interval. Both remain subject to existing admission
and history limits.

## Measurements

Hardware: RX 9060 XT (`gfx1200`), Ryzen 9 3950X, HIP 7.16, TP1. Model:
ordinary `qwen3.8-27b-iq3r-v4.rad`, DFlash7. Neither IQ3R-S nor a different upstream
revision is used as a comparison.

The standalone merge benchmark uses 16 layers, key width 512, query width 4096,
eight Query rows and either eight or 256 accepted key rows. It excludes input
construction, alternates mode ordering, discards 40 warmups and averages 360
samples per mode. Source: `backends/radiance/tools/benchmark_rank_statistics.cpp`.
Compile with C++20/O3, the adapter and common include directories, and the built
common `libkvmem.a` plus pthreads.

| Accepted rows | Copy merge | Consuming merge | Reduction |
| --- | ---: | ---: | ---: |
| 8 | 335.750 us | 236.099 us | 29.68% |
| 256 | 1953.560 us | 1002.160 us | 48.70% |

The real-model statistics ablation uses the same new executable, three loads per
mode, reversed ordering on the middle pair, one warmup plus three measured
256-token counting requests per load. All 24 outputs have the same hash and the
same 223/231 accepted/offered draft count. Mean decode is 139.797 versus 139.623
tok/s (standard deviations 0.418 and 0.691); mean HTTP wall time is 2.049685 versus
2.049215 seconds. This establishes **no measurable end-to-end speedup** for that
short workload despite the lower isolated CPU merge cost.

The checkpoint capture A/B runs use the same executable and RAM/SSD/Query/tool/
cancellation suite. Each captures 30 snapshots, 2,481,979,392 logical bytes and
338,690,048 D2H bytes. Borrowing avoids exactly 338,690,048 bytes (323 MiB) of
intermediate CPU copying. GPU transfer bytes are unchanged. Removed copy bytes
are not a measured RSS or latency reduction.

The restore A/B runs each restore 1,359,478,784 KV bytes across 15 cache resumes.
The largest pinned KV allocation falls from 138,412,032 bytes (132 MiB) to
16,777,216 bytes (16 MiB), an 87.88% reduction. The bounded mode completes 88
chunks versus 15 whole-image batches. Each restore still uses just one staging
allocation and reuses its two events. These are allocation/copy observations;
the extra chunk completion waits can affect latency and must be judged with the
end-to-end measurements.

## Validation and limits

Tests cover allocation-preserving TP1 moves, malformed/nonfinite statistics,
attention normalization and negative zero, independent CPU parking, premature
staging release refusal, native sparse/page/state restores, and reused transfer
failure cleanup. A 37-byte test staging limit deliberately splits native pages;
an independent GPU recapture checks the restored bytes against the original.

Local Python coverage passed 89 tests. The six native/portable executable checks
passed with both controls for each cache change (24 invocations). Each of the
four real-model RAM/SSD/Query/tool/cancellation suite runs passed 155 checks.
Compiled adapter/common source was compared with commit
`e0e57eb2dcb84eec7ea16bdece37a6c8978ae5da`: 155 C++/HIP/header files match after
line-ending normalization, with one existing trailing-newline-only difference
in an unrelated attention test.

The qualification evidence also records bounded-staging controls, concurrent
branches and paired 33-round 256K Task2 runs with a 36,864-token resident budget
and 4,096-token final generation cap. The C2 branch suite passed all 18 requests.

## Paired 256K Task2 result

Both sides use the same final executable with all three new controls off, then
on. There is one load per side; this pair does not establish a confidence
interval or isolate each optimization's contribution to total time. Ordinary
IQ3R, DFlash7, TP1, prefill batch 256, RAM 4096 MiB and SSD 8192 MiB are fixed.
Intermediate requests cap generation at 512; the final cap is 4096, with 2305
tokens actually generated. Both final prompts contain 257,958 tokens.

| Metric | Controls off | Controls on | Observed change |
| --- | ---: | ---: | ---: |
| Effective prefill, tok/s | 630.329 | 650.453 | +3.19% |
| Followup decode, native token count, tok/s | 55.209 | 59.819 | +8.35% |
| Final decode, native token count, tok/s | 56.635 | 65.431 | +15.53% |
| Total HTTP request time, seconds | 607.355 | 581.168 | -4.31% |
| Running process RSS peak, MiB | 8735.230 | 8219.129 | -516.102 MiB / -5.91% |
| Device VRAM peak, MiB | 14603.555 | 14603.473 | Effectively unchanged |
| Process swap peak, MiB | 0 | 0 | Unchanged |
| Largest checkpoint KV pinned allocation, MiB | 1268 | 16 | -98.74% |

All 33 output pairs match, including reasoning text. Both sides have 32/32 cached
followups and the same 7,016 accepted draft tokens out of 17,752 offered. All
runtime/cache checks pass. Model-quality evidence is unchanged: 11/31 intermediate
batch acknowledgements and a syntactically valid final Python answer; semantic
correctness is not established by those checks.

The 66 checkpoint captures have identical logical and GPU transfer bytes;
borrowing removes 8,879,865,856 bytes (8.27 GiB) of intermediate CPU copying.
The 32 restores move the same 36,874,223,616 KV bytes; bounded restore uses 2,216
chunks, with one reusable staging allocation per restore. RSS is sampled from
the process and VRAM from device-global AMD sysfs, so neither should be inferred
solely from an allocation size or copy counter.

Machine-readable measurements, controls, checks and executable identities are
in [the benchmark report](benchmarks/radiance-adapter-staging-20261107.json).
The locally retained raw evidence archive contains 1054 files, 1,863,364 compressed
bytes, SHA-256 `72bdcbaee1daa143d89c514bce9dbcf154b375f5ac363bf8eb702f0f316b917b`.
