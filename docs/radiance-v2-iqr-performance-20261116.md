# Radiance v2 RDNA3 IQR performance follow-up

The first RX7800M port reproduced the packed layout, decoder and numerical behavior but replaced the RDNA4 matrix paths with a per-token dot-product kernel. It established functional compatibility; it did **not** complete the requested migration of the RDNA4 performance paths. That omission was the main cause of the slow document prefill reported in the [initial migration report](radiance-v2-migration-20261116.md).

The consolidated patch now includes RDNA4-derived GEMV, narrow WMMA and tiled prefill paths adapted to gfx1101. On the same RX7800M and requested IQ2R model, document prefill increased from **28.99 to 328.43 tokens/s** in the controlled pair. The final artifact confirmed **329.34 tokens/s**. Decode remains approximately **12 tokens/s**; this is not a broad decode-speed improvement or a claim that every RDNA4 optimization has been ported.

## Implemented paths and provenance

Reference source is pinned Radiance v2.0.0 `ddcb9c69bb6c9ff270f8436bd4c059fba88ac6b7`, particularly `libr4d/r4d_gemm_iqr.hip`, `r4d_gemm_8a8_prefill.h`, `r4d_iqr_dec.h` and `r4d_fp8_wmma.h`.

| Dispatch | Migrated behavior | gfx1101 implementation |
| --- | --- | --- |
| M=1 | MAD/row-F32 GEMV: packed-code prefetch, direct hash-byte dot products, joint reduction of 16 rows | `libr3diqr/r3di_gemv.h` |
| M=2..64 | Decode a 16x128 weight tile once, reuse across up to 16 activation rows; four waves split K | `r3di_narrow.h`, native int8 WMMA |
| M>64, contiguous output | 128x128 output tile, eight waves, double-buffered activation LDS, weight-fragment prefetch and grouped raster | `r3di_prefill_body.h`, adapted from the upstream prefill body |
| Prefill weight preparation | Cooperative compressed-to-fragment decode, once per 128-row weight band per invocation; shared across token tiles | `r3di_prefill_decode.h`, `r3di_frag_decode.h` |
| WMMA interface | Preserve the fragment-plane layout while adapting input replication and accumulator row mapping to RDNA3 | `r3di_wmma.h` |

The narrow path also handles noncontiguous output and ragged token tails. Persistent weights remain compressed. Large prefill uses caller-owned temporary fragment storage plus counters, reported through the plugin scratch callback. Counter initialization uses the supplied stream; the launch path adds no allocation or device synchronization. The qualified runtime remains one sequence.

RDNA3's WMMA operands and accumulator mapping differ from RDNA4. The adapter gathers both fragment halves into the gfx11 operands and keeps accumulators in native interleaved row order until the output store. See AMD's [RDNA3 WMMA explanation](https://gpuopen.com/learn/wmma_on_rdna3/) and [RDNA3 compiler builtin documentation](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna3-dense-wmma-builtins.html). This uses native int8 WMMA, not a transplanted RDNA4 FP8 instruction.

The upstream biased-integer-as-float shortcut failed the existing independent long-K error bound in the first GEMV candidate: B3/L16/run256, M1/K5120, output124 was -0.621094 versus reference -0.615510, absolute error0.00558415. The final GEMV uses exact int32-to-float conversion and prefill disables the analogous shortcut. Original tolerances were retained. The initial failure log is preserved.

Still unported: specialized vocabulary-head acceleration, fused gated epilogues, chained execution and the full upstream format family. The gated operation remains the previously validated GEMM-plus-quantizer composition. The selected five MAD row-F32 formats are the scope of this port. The copied prefill design notes describe upstream RDNA4 tuning, not measured RX7800M tuning.

## Controlled measurements

Hardware: Windows RX7800M/gfx1101, HIP device0, PCI `0000:0c:00.0`, UUID bytes `30303030303031323030303030303030`, total VRAM12,868,124,672 bytes. Build: Release, Clang24/ROCm Core10.1, MSVC14.44, Windows SDK10.0.26100.0. The worker executable is unchanged between variants. The baseline plugin directory was preserved before optimization.

Model: `qwen3.8-27b-iq2r-l-v4.rad`, model revision `e5f1b3ef70741d705a64a28b852acf3fcefcead7`, downloaded through HF Mirror. SHA256 `89fb6bfbd602594ed8a79b655bee7e15e7732b7c2c78e34db3d145b2e58706bd`, 11,990,364,160 bytes.

Whole-model settings and request payloads are identical across variants: TP1/C1, logical context8192, budget1024, prefill256, replay512, Query limit256/replay512, BF16 KV, F32 state, sink128, recent256, host history2048 MiB, deterministic sampling, no speculation or vision. The 2665-token document is sent as an ordinary turn; the question follows separately and does not contain the answer.

| Measurement | Original baseline | First optimized run | Final artifact confirmation |
| --- | ---: | ---: | ---: |
| Document prefill, tokens/s | 28.99 | 328.43 | 329.34 |
| Document request wall time | 92.171 s | 8.359 s | 8.328 s |
| Separate question wall time | 3.235 s | 1.828 s | 1.750 s |
| Longer response decode, tokens/s | 11.97 | 12.05 | 12.38 |
| Longer response output tokens | 110 | 110 | 110 |
| Reused tokens in question | 2666 | 2666 | 2666 |

The controlled pair gives about11.33x prefill throughput. These are short runs, not sustained-load or broad workload benchmarks. The two candidate runs are reported separately; the best value is not substituted into the original pair. The small decode variation is insufficient to establish a meaningful decode improvement.

The final kernel microbenchmark uses identical packed weights, buffers and stream, two warmups, seven interleaved rounds with three launches per round, and median HIP-event time. N=K=5120; each candidate result is compared numerically with the baseline.

| Format | M | Baseline ms | Candidate ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| B2 | 1 | 0.050570 | 0.045933 | 1.101x |
| B2 | 16 | 0.580783 | 0.066677 | 8.710x |
| B2 | 256 | 6.661193 | 0.441330 | 15.093x |
| B4 | 1 | 0.050440 | 0.049283 | 1.023x |
| B4 | 16 | 0.602693 | 0.054570 | 11.044x |
| B4 | 256 | 6.860610 | 0.473603 | 14.486x |

M1/M16 matched baseline outputs exactly in these fixtures. M256 relative L2 error was1.25e-5/B2 and9.90e-6/B4, below the0.004 bound. Microbenchmark speedups are not substituted for whole-model results.

## Correctness and memory

- Final native suite:144 passing records, including74 GEMM fixtures across B2..B6, run256/run512, M1/2/16/17/64/65/129 and longer-K cases. GEMM uses an independent canonical-trellis double-precision reference.
- GEMM fixtures execute three consecutive calls on a nondefault stream, exercising scratch-counter reuse; missing/undersized scratch is rejected and the scratch guard remains intact. Existing BF16, embedding, vocabulary, quantization, norm, RoPE and gated-composition tests also pass.
- All five requests pass for baseline and both candidates: arithmetic, document ACK, separate recall, longer decode and recovery. The question answers68391, reuses2666 tokens, records native Query span `[2671,2686)`, and completes replay `[2666,2695)`. The monitor confirms selected1024 tokens.
- Static weights stay8.44 GiB and KV stays292 MiB. The activation arena changes from118.21 to257.98 MiB, including scratch from30.23 to170.00 MiB. Engine device allocations rise from8.84 to8.97 GiB. Settled card-free VRAM changes from2.97 to2.83 GiB, on a card reporting11.98 GiB total. These are initialization/settled readings, not a continuously sampled peak. No budget or KV precision reduction was used.
- Token embedding312.61 MiB remains in pinned host memory under the existing default placement. This is unchanged from baseline.

## Artifact identity and reproduction

The reviewed native patch remains `backends/patches/radiance-v2-kvmem.patch`; upstream/private ABI remains39/1263927335. Plugin version is `0.2.0-iqr-tiled`. The patch now has216 modified and77 added files relative to pristine upstream; this follow-up changes10 files relative to the initial migration. Local and tested worker native sources match all1845 file hashes.

| Artifact | SHA256 |
| --- | --- |
| Consolidated patch | `6a3fb7e33031b9682d1547d1a9847bd1ca83681d4d36a4053e50ae53f5f144f8` |
| Source manifest | `19b1522e0c4a57cb132d8d2a3ef61461747c4668c8a18702cfdec6e551f5ed4c` |
| Worker executable, unchanged | `9d50b040ac14781d8f5fabdfb929eb156bbad69b6816c5bfe7a41831dfcbc01e` |
| Baseline IQR plugin | `ec3e2ba56d34fa4eca6b5022607ea1bb993ecd7385fa22324cb507b8d01f7a7f` |
| Final IQR plugin | `1ab611e1738ce36cd01b36a8b58f882eba5b1994632018d2a829bddfb016219b` |

The final focused build compiled two host objects and one HIP object, then linked the plugin and GPU test. Final Ninja dry-run: zero jobs, no missing command records and no regeneration. Full dependency-bundle verification preceded this optimization; subsequent focused retries verified immutable manifest hashes and critical compiler/runtime/header bytes without repeating the full unchanged bundle scan. Source-identity tests pass5/5; the actual source identity and clean patch applicability also pass. The local test runner initially failed because its default sandbox temporary directory was unwritable; rerunning with a workspace-local temporary directory passed. No test expectation was changed for that environment issue.

Machine-readable results, exact timing fields, source/artifact identities and evidence hashes: [performance evidence](benchmarks/radiance-v2-rx7800m-iqr-20261116.json).

Controller evidence: `build-rdna3-iqr-opt/evidence/` under this checkout, including `numeric.log`, `numeric-biased-failed.log`, `bench.log`, all three model JSON/log pairs, final build records and `final-identity.json`. Full requests, responses, command lines and monitor snapshots are preserved. The one-shot `port_paths.py` scaffold refuses to overwrite the finalized source; reconstruction uses the consolidated patch.

Worker optimization directory: `$ARTIFACTS-Lab/rdna3-iqr-opt-20261116`. The candidate plugin is in the existing `radiance-rdna3-h256-20261115/build-kvmem-v2-gfx1101/radiance_home`; preserved baseline is `rdna3-iqr-opt-20261116/baseline/radiance_home`. To repeat the kernel tests, use the matching HIP runtime environment and:

```powershell
$lab = '$ARTIFACTS-Lab'
$build = "$lab/radiance-rdna3-h256-20261115/build-kvmem-v2-gfx1101"
$opt = "$lab/rdna3-iqr-opt-20261116"
& "$build/bin/r3di_iqr_gpu_test.exe" "$build/radiance_home/kernels/libr3diqr.dll"
& "$build/bin/r3di_iqr_gpu_test.exe" --bench `
  "$opt/baseline/radiance_home/kernels/libr3diqr.dll" `
  "$build/radiance_home/kernels/libr3diqr.dll"
```

`model_baseline.py` and `model_candidate.py` preserve the exact whole-model recipe. Each checks GPU identity and idle VRAM before starting and terminates its own server afterward. Server exit code1 is that intentional Windows termination; `completed:true` and runner exit0 indicate success.

The initial report's platform/feature exclusions still apply. Its scheduler/core qualification is inherited from the unchanged core; it was not rerun or counted as new optimization testing. No commit, push, merge, release upload or installed-package replacement was performed.
