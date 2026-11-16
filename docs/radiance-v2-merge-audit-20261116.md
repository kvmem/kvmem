# Radiance v2 RDNA3 pre-merge audit

Status: **not approved for merge yet**. The audit found and fixed boundary
defects, and portable checks pass. The new GPU changes and additional numerical
tests still require compilation and execution on the authorized RX 7800M.
The earlier preview ZIPs predate these audit fixes and are not the current
candidate's binaries.

## Findings and changes

1. `libr3diqr/r3di_args.h`: flattening a tensor whose leading batch dimension
   includes padding incorrectly returned the innermost row pitch. Operators
   could then read the wrong rows. The helper now verifies all non-singleton
   leading strides. Dense rows and uniformly padded rows remain accepted.
   A standalone C++ regression reproduces the failure with the previous header
   and passes with the new header; the regression also covers singleton axes.
2. `libr3diqr/r3di_iqr_embed.hip`: subtracting a positive vocabulary offset from
   an extreme negative token in `int32_t` could overflow and turn an invalid
   token into a valid row. Subtraction now uses `int64_t`. New GPU tests use
   `INT32_MIN`/`INT32_MAX` and require out-of-range rows to remain zero.
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
5. Test isolation: the dry-run launcher test mocked input validation but still
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
embedding indices. **These additional GPU cases are pending**, so the review
is not a numerical proof for every geometry, model or platform.

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

## Completed checks and remaining gate

- The four-patch series reconstructed all 1,857 candidate native files byte for
  byte, with independent reverse-apply checks for each temporary component.
- 102 adapter Python tests; 23 Radiance launcher and 27 launch-plan tests.
- 72 portable CTest entries, including the new tensor-stride regression.
- 67 local console tests, rerun outside the sandbox after local loopback and
  hardlink permissions blocked the sandboxed run.
- Public evidence scan and JavaScript syntax check pass. Raw historical
  experiment paths are retained privately; exported measurements are unchanged.
- Previous GPU evidence: 222 IQR checks and DFlash2 tests plus short IQ3R and
  IQ4R-S model runs on Windows gfx1101. This evidence predates the audit changes.

Before merging: compile the changed plugin and test executable; run both the
ordinary suite and `--audit`, rerun DFlash2 and model smoke checks, verify binary
and source provenance, integrate the current KVMem master and pass required CI.
Linux ROCm/7900 XTX remains an explicit hardware qualification gap.
