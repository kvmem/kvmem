# Radiance v2 DFlash2 on RX7800M

Follow-up: the [paired rerun with GPU power telemetry](radiance-v2-dflash2-power-rx7800m-20261116.md) records four unchanged-build runs, raw power samples, and separately identified GPU versus board-power availability. The measurements below remain the original qualification record.

The requested Qwen3.8-27B IQ2R-L v4 container now runs its embedded DFlash2 drafter on the native Windows RX7800M build. In the final same-build comparison, three 256-token replies decode at a combined **25.55 tokens/s with DFlash2 versus14.23 without it:1.80x**. The Windows process counter sampled **11.29 GiB dedicated GPU memory** in a separate smoke run. BF16 KV and F32 recurrent state were retained.

This is a short qualification of this model, GPU and profile, including bounded cross-turn Query/Pressure recall. It does not establish sustained throughput, general model quality, identical output tokens, or support for all RDNA3 configurations. Full requests, responses, timings and artifact hashes are in the [benchmark record](benchmarks/radiance-v2-rx7800m-dflash2-20261116.json).

## What was required to enable it

The model already contains a5-layer DFlash2 drafter, trained block8/depth7,2048-token attention window, head128/GQA4 and grouped two-tap dynamic convolution. Its draft linear weights are MXFP4 and its draft vocabulary head is affine U2. Enabling the command-line flag alone was insufficient for the previous focused build.

- Added existing upstream RDNA3 MXFP4, U2 head, head128 attention and DFlash selector units to the focused `libr3d` closure. There are now20 HIP units and83 kernel rows in this closure.
- Migrated the pinned RDNA4 fused/vectorized DFlash convolution body into `libr3diqr`, using the same two-tap/group16 math with a gfx11 launch. Coefficient row pitch, block boundaries, alignment and output guards are covered by tests.
- Corrected the U2 head tile choice for K5120 at small row counts. On Windows the joined scale/zero descriptor lives in device-only memory, so its small header is copied on the caller's stream before dispatch. This currently adds a stream fence; it is a remaining optimization opportunity.
- Restricted Windows draft MXFP4 to the exact lookup decoder. The optional upstream ALU decoder failed the independent numerical bound on a zero-code fixture. Its approximation maps FP4 zero to a small nonzero value. The tolerance was not relaxed, and the qualified rows do not advertise the ALU tuning alternative.
- Migrated `rad_gdn_materialize_v1` from the pinned RDNA4 producer into the RDNA3 recurrent producer. It folds only the accepted draft prefix into an exclusively owned two-slot clone, preserving live state. Empty replay no longer dereferences absent gating inputs. KVMem still requires a single supported recurrent producer, matching ABI/layout and the materializer export across all relevant bands.
- Added explicit DFlash2 support to the RX7800M profile and capability payload. MTP, images and concurrent execution remain outside this profile.

The upstream pin remains Radiance v2.0.0 `ddcb9c69bb6c9ff270f8436bd4c059fba88ac6b7`, public/private ABI39/1263927335. `libr3diqr` is now `0.3.0-dflash2`. The earlier [IQR GEMV/WMMA/prefill optimization](radiance-v2-iqr-performance-20261116.md) is preserved. Specialized main-model vocabulary acceleration, fused gated epilogues and chained execution remain unported.

## Same-build measurements

Hardware: Windows RX7800M/gfx1101, HIP device0, PCI `0000:0c:00.0`, UUID bytes `30303030303031323030303030303030`, total12,868,124,672 bytes (11.98 GiB). Toolchain: Release Clang24/ROCm Core10.1, MSVC14.44, Windows SDK10.0.26100. Model SHA256: `89fb6bfbd602594ed8a79b655bee7e15e7732b7c2c78e34db3d145b2e58706bd`; the already downloaded HF Mirror file was reused.

Both final runs use the same worker/core/kernel binaries, requests, maximum output length and environment (`OMP_NUM_THREADS=2`, `KVMEM_CAPTURE_EARLY_SEAL=1`). TP1/C1, logical context8192, prefill256, budget3072, sink128/recent2176, BF16 KV, F32 state, replay512, Query256/replay512, host history2048 MiB and deterministic temperature0 are identical. Only drafter/depth differ. The off comparison deliberately uses the on configuration's larger retention budget; the ordinary off profile still defaults to1024/256.

| Reply | DFlash2 off, tok/s | DFlash2 depth7, tok/s | Speedup | Accepted/proposed drafts |
| --- | ---: | ---: | ---: | ---: |
| Hash-table explanation,256 tokens | 14.35 | 25.25 | 1.76x | 189/469 (40.3%) |
| LRU code/explanation,256 tokens | 14.23 | 24.32 | 1.71x | 187/476 (39.3%) |
| BFS maze explanation,256 tokens | 14.11 | 27.27 | 1.93x | 195/427 (45.7%) |
| Three-reply aggregate | **14.23** | **25.55** | **1.80x** | 571/1372 (41.6%) |
| Integers1–30,110 tokens | 14.29 | 53.89 | 3.77x | 95/98 (96.9%) |

Decode rates exclude prefill and the first output token, following the server's timing definition. The aggregate is total decoded tokens divided by total decode time, not the mean of per-request rates. The regular integer sequence is unusually easy to draft and is not included in the three-reply aggregate. The code/explanation replies are capped at256 tokens; they are not complete programs or executed code-quality tests.

The three longer replies are **not byte-identical** between modes. Arithmetic, the integer sequence and the recall answer agree. Numerical fixtures and these functional checks passed, but no claim of token-identical or statistical-quality equivalence is made. Multi-row verification and single-row decode take different arithmetic paths; that is a possible source of divergence, not a proven explanation of every differing token.

An earlier pair before the final launch profile's environment was applied measured22.08–24.70 tok/s on versus12.45–12.63 off. Those records are retained separately. The final table uses the final pair exclusively; it does not mix the faster on run with the earlier slower baseline.

DFlash2 adds overhead where little text is produced. For the3985-token document, prefill was337.10 tok/s on versus365.37 off; document request wall time was12.63 versus11.13 seconds. The six-token recall answer took2.70 versus1.78 seconds. The 256-token reply wall times improved from18.30/18.42/18.58 seconds to11.11/11.50/10.33 seconds.

## Correctness and memory

The graph declares zero unresolved required execution bands and logs DFlash2 proposing seven tokens in one eight-row pass. The final model run verifies actual nonzero draft counts rather than inferring activation from the flag.

- 53 DFlash2 numerical records pass against independent CPU fixtures:18 fused convolution,6 MXFP4 GEMM,6 selector,3 U2 head and20 accepted-state materialization cases. State tests cover heads1/3, FP32/FP16 and accepted prefixes0/1/4/8/16, including untouched factor data/guards and invalid-prefix rejection. FP16 is tested at the operator level; whole-model state remains F32.
- 144 existing IQR regression records pass on the final DLLs, including74 canonical-trellis GEMM fixtures and the existing quantization/embedding/vocabulary/norm/RoPE/gated-composition checks.
- Both full runs pass arithmetic, the110-token integer response, a3985-token document exceeding budget3072, its separate recall question, and another arithmetic request. The answer is68391,3986 tokens are reused, the monitor reports3072 selected tokens, and Query replay completes `[3986,4015)` before sampling. The first document also triggers bounded replay `[3584,3985)`.
- Local profile/capability/platform/source-identity tests pass27/27. Clean patch applicability and all1848 local/remote native file hashes pass. Final Ninja dry-run reports zero jobs, no missing command records and no regeneration.

With DFlash2, static GPU weights are10.03 GiB, KV allocation617 MiB and the activation arena271.02 MiB. Engine device allocations total10.90 GiB; the worker reports932 MiB free after loading. The additional draft weights and state fit without lowering KV precision. Budget3072/recent2176 preserves the2048-token draft window plus the partial block margin; reducing the budget to the old1024 would not be an equivalent test.

A separate final smoke run samples Windows `GPUProcessMemory` for the actual worker PID:14 readings at a nominal one-second interval, with query overhead between readings, no sampling errors. Highest sampled dedicated usage is12,121,776,128 bytes (**11.29 GiB**). Highest sampled shared usage is2,101,198,848 bytes (**1.96 GiB of system memory**, separately counted). The profile uses host history and mapped token embeddings; shared system memory must not be added to dedicated usage and called VRAM consumption. These samples do not prove an instantaneous card-wide peak or rule out every driver migration event.

The initial external HIP-process sampler stayed at the idle free-memory value while another process held the model. Those readings are retained for audit but **invalid for model occupancy** and excluded from the memory conclusion. Memory-instrumented smoke timings are also excluded from the speed comparison.

## Reproduce and identify

Build the focused targets as documented in the [backend README](../backends/radiance/README.md). To launch the tested mode, add `--profile rx7800m-iq2r --drafter dflash2` to the supervised `backends/radiance/tools/serve.py` command with the built `--exe`, `--home`, matching `--rocm`, requested `--model`, and `--platform windows`. It selects depth7, budget3072/recent2176 automatically; the default remains speculation off.

| Final artifact | SHA256 |
| --- | --- |
| Consolidated native patch | `c97e3b26051c8f6899485141b5cc1238b5c91c5f01156a64481827f0be68ec11` |
| Source manifest | `9ce00fc75d1078a94dd43acfa1e132e4903e9b2453be8dae2919acd5b60b1ad3` |
| Worker | `f1d4ee1f987485bd5d0bfbb04b93bd906f309f1411a026a3a26ac330198c179a` |
| libr3d | `a1d372591188f37fc92441a6a949d5499f221960d30df2847695dfd8795c0c31` |
| libr3diqr | `8895055be968507e91215ad0f9ba3deb9869bbe8b7c954408ab27e3edd9ac221` |

The consolidated patch has216 modified and80 added files against pristine upstream. This follow-up changes11 native files relative to the preceding IQR optimization, including three additions. The worker's pre-DFlash binary$HOME/adapter backup remains under the remote task's `build-rdna3-dflash2/baseline` directory.

Raw evidence and runners are in local `build-rdna3-dflash2/evidence` and the remote `$ARTIFACTS-Lab/build-rdna3-dflash2` directory. The benchmark JSON records their hashes. Failed MXFP4 numerical and missing-materializer startup attempts are retained, along with an incomplete run caused by GBK console output in the harness; none is counted as a passing measurement. Final source and tests are in the checkout and tested lab build. No commit, merge, release or installed-package replacement was performed.
