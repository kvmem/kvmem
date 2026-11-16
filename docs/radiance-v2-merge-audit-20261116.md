# Radiance v2 RDNA3 pre-merge audit

Status: **scoped local and RX 7800M validation passed**. The audit found and
fixed three boundary defects. The final kernel plugin passes ordinary,
additional audit and DFlash2 numerical suites plus short IQ3R/IQ4R-S model tests.
Required pull-request CI remains a separate merge gate.
The earlier preview ZIPs predate these audit fixes and are not the current
candidate's binaries.

## Findings and changes

1. `libr3diqr/r3di_args.h`: flattening a tensor whose leading batch dimension
   includes padding incorrectly returned the innermost row pitch. Operators
   could then read the wrong rows. The helper now verifies all non-singleton
   leading strides. Dense rows and uniformly padded rows remain accepted.
   A standalone C++ regression reproduces the failure with the previous header
   and passes with the new header; the regression also covers singleton axes.
   The first repair lost the explicit rank-2 single-row pitch check; the full
   GPU suite caught it before merge. The final helper preserves that check,
   and both the portable regression and complete GPU suite pass.
2. `libr3diqr/r3di_iqr_embed.hip`: subtracting a positive vocabulary offset from
   an extreme negative token in `int32_t` could overflow and turn an invalid
   token into a valid row. Subtraction now uses `int64_t`. New GPU tests use
   `INT32_MIN`/`INT32_MAX` and require out-of-range rows to remain zero. The old
   plugin reproduced a value of 0.984375 instead of zero; all three tested
   embedding encodings return zero with the final plugin.
3. `libr3diqr/r3di_gemm.hip`: the BF16-group prefill path loads a vector of row
   scales, but admission checked only four-byte scale alignment. It now requires
   sixteen-byte scale alignment for that vectorized path. Ordinary native
   allocations are already 256-byte aligned. Row-F32's scalar loads retain
   their existing alignment requirement.
4. Patch and package provenance: the monolithic patch obscured temporary
   kernel ownership and the source packager assumed a one-element patch chain.
   The series is now split into core integration, general RDNA3, RDNA3 IQR, and
   offload-budget patches. Shared validation checks all ordered hashes;
   packagers record the complete series and validate the source inventory.
   See [upstream replacement procedure](radiance-upstream-maintenance.md).
5. Mainline compatibility: preserved the newly merged IQ4R U2 MTP fix as a
   fifth patch. Native v2 architecture compilation caught its dense-only `ffn`
   field access in the shared MoE translation unit; the port now explicitly
   limits the new vocabulary-class MTP behavior to dense Qwen.
6. Test isolation: the dry-run launcher test mocked input validation but still
   detected the actual WSL environment, causing a missing-runtime-directory
   error only on WSL. It now mocks environment construction too; production
   WSL runtime validation remains intact.

## Numerical review

Reviewed canonical trellis reconstruction, 256/512/1024-row wrap boundaries,
MAD integer decoding, BF16 group vs row-F32 scale application, GEMV reduction,
gfx11 WMMA accumulator mapping, fused gate rounding, embedding inverse FWHT,
prefill publication/counters, scratch ownership and vocabulary quantization.
The gfx11 per-row BF16 scale lookup uses row `2*t + lane_half`, matching the
native accumulator layout; row-F32 scales are applied after reduction and
column factors within each K128 block. Per-block int32 dots cannot overflow
for these int8 inputs. Group scales are applied before combining K blocks.

Existing GPU tests use independently retained canonical codes and mathematical
tail-biting trellis decoding, not the packed GPU decoder as their oracle.
The earlier suite covered M=1/2/16/17/64/65/129/256, scale forms, embedding,
vocabulary, guards and fused/nonfused state comparisons. It did not fully cover
the large-output GEMV dispatch or more than four prefill token bands.

The new `r3di_iqr_gpu_test <plugin> --audit` adds N=4096/K=5120 GEMV for u2-u6,
long-K IQ3R/IQ4R-required forms in narrow/prefill paths, M=641 grouped-grid
prefill with a partial final group, repeated scratch reuse, and extreme
embedding indices. All 15 additional cases pass on the RX 7800M, as do 222
ordinary checks and 76 DFlash2 checks. This is not a numerical proof for every
geometry, model or platform.

## Efficiency review

- Single-token decode retains packed weights, prefetches codes and uses integer
  dot operations; small multi-token verification reuses a decoded tile with
  gfx11 WMMA. BF16-group scales require per-row loads and folds, unlike the
  shared column-factor optimization available to row-F32 weights.
- Prefill cooperatively decodes one temporary int8 fragment plane per call and
  reuses it across M bands. Its `N*K` workspace is 170 MiB for the 34816x5120
  gate/up shape, plus counters. This is a per-operation workspace, not an
  expanded permanent model copy; it still needs memory-budget accounting.
- The fused gate path avoids a separate full gate/up intermediate and reuses
  last-arriver counters. Counter allocation includes a device synchronization
  on first use for a stream/output key, rather than on every token. The plugin
  releases keyed allocations through its free callback.
- Chained execution remains unsupported by the IQR launch wrapper. This leaves
  launch/scheduling overhead compared with upstream paths that can chain.
  Some draft operations still use the existing exact fallback implementation.
  No claim of peak bandwidth, peak power or RDNA4 throughput parity is made.
- IQ4R-S RAM-offload measurements include transferring several GiB of weights
  across the external GPU link per step. They cannot isolate kernel efficiency.
  Likewise the previous short DFlash2 and non-DFlash2 runs use different memory
  budgets and are not a controlled speed comparison.

## Final GPU and model evidence

[Machine-readable measurements](benchmarks/radiance-v2-merge-audit-20261116.json)
include source/log fingerprints, per-request results, kernel timings and power
summaries. The final plugin SHA-256 is
`4dbe9cd1373314882e3c8398eb4915c404b29e524010f7c69bc37d8f0edfc600`.
All 142 `libr3d`/`libr3diqr` source files match the local candidate. The binary's
six GPU bundles each contain gfx1100, gfx1101, gfx1102 and gfx1103 code objects.
Only gfx1101 was executed on hardware.

Six fixed GEMV/narrow/prefill shapes compare the previous and final plugin on
the same card, interleaved over seven rounds of three launches. Outputs match
exactly (`rel_l2=0`, `max_abs=0`). Timing ratios range from 0.981 to 1.020;
this short sample shows no material regression, not peak-performance proof.

Each model/mode completed two requests: `17+25` returned `42`, and the sequence
request returned `1, 2, 3, 4`. Both DFlash2 runs issued 14 draft tokens and
accepted eight on the sequence request. BF16 KV/F32 state, prefill128,
TP1/C1, RAM embeddings and layer offload were used throughout.

| Model | Mode | Budget / context | Short decode tokens/s | Sequence request chip W mean / peak | Run peak VRAM GiB |
| --- | --- | --- | ---: | ---: | ---: |
| IQ3R | Off | 512 / 2048 | 12.683 | 50.00 / 70 | 11.368 |
| IQ3R | DFlash2-7 | 3072 / 4096 | 1.121 | 47.28 / 55 | 11.636 |
| IQ4R-S | Off | 512 / 2048 | 0.231 | 50.84 / 56 | 11.727 |
| IQ4R-S | DFlash2-7 | 3072 / 4096 | 0.786 | 49.55 / 56 | 11.636 |

These are 11-output-token correctness probes with different cache budgets,
not a controlled DFlash2 speed comparison. ADLX chip-power samples cover the
whole HTTP request, including prefill; they are not board watts or isolated
decode-kernel power. The very short IQ3R off request has only three power samples.
Offloaded weight traffic limits these runs and prevents drawing conclusions
about peak GPU throughput or the earlier all-resident IQ2R performance.

The real-model tests used the final kernel plugin with the existing qualified
Windows worker and architecture plugin. The separately preserved v2 IQ4R U2
MTP architecture port has native host compilation/test coverage, but was not
part of this new GPU model campaign. The five authorized kernel/test files were
the only sources transferred for this campaign. The harness terminates each
server after successful requests; recorded Windows exit code 1 is that cleanup,
not a request failure. No campaign server or power sampler remained running.

## Completed checks and merge gate

- The final five-patch series reconstructed all 1,857 candidate native files byte for
  byte, with independent reverse-apply checks for each temporary component.
- 106 adapter Python tests; 24 Radiance launcher and 27 launch-plan tests.
- 73 portable CTest entries after syncing current master, including the new tensor-stride regression.
- Native `arch_test`, `mem_test` and `sched_test` compile and pass after the
  v2 MTP port (225/200/198 cases), using CTest-provided plugin paths. The initial direct invocation
  of `mem_test` lacked that test environment; it was rerun through CTest.
- 67 local console tests, rerun outside the sandbox after local loopback and
  hardlink permissions blocked the sandboxed run.
- 19 CI-classification tests and 13 portability tests (two platform skips).
- Public evidence scan and JavaScript syntax check pass. Raw historical
  experiment paths are retained privately; exported measurements are unchanged.
- Final GPU evidence: 222 IQR checks, 15 additional audit checks and 76 DFlash2
  checks, plus all four short model/mode runs above.

The working branch incorporates KVMem master `8ef684a`. Required pull-request
CI must pass before merging. Linux ROCm/7900 XTX, the v2 MTP architecture change
on a GPU, RDNA4, full multimodal and concurrent workloads remain explicit
qualification gaps. Earlier preview ZIPs have not been rebuilt with this audit.
