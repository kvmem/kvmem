# GGUF tensor-parallel row alignment

Reviewed on 2026-11-15 against mainline `23bb988`.

NInfer's GGUF integer matrix route rejects output row counts that are not a
multiple of 128. The tensor-parallel loader intersects stored-layout boundaries
with `ops::linear_share_units()`, but GGUF previously fell through to `{1, 1}`.
Manual or automatic uneven splits could therefore produce a shard that storage
accepts and compute rejects. A larger prefill chunk or resident vision tower can
change the automatic split; neither is required to trigger the defect.

The fix returns `{128, 1}` for GGUF and documents that contract. Column constraints
remain supplied by the stored block layout. No kernels, weight bytes, arithmetic,
public interfaces, or non-GGUF share units change in this patch.

## Validation and audit

- Reviewed the actual loader boundary calculation, integer matrix guard, final
  patch, test registration and ordered backend manifest. The final prepared tree
  is `fcf81cfe947eb3e6b0639c39c54532b272ac7252`.
- Both complete `linear.cpp` translation units, before and after the fix, compiled
  with MSVC 14.44, CUDA 13.2 headers and C++20 Release settings.
- The focused host regression fails before the fix on the alignment invariant
  and passes after it. It checks 19 GGUF formats, three dimensions (248320, 17408,
  6144), eight split patterns including extreme and three/four-rank allocations,
  complete contiguous coverage, unchanged equal halves and seven non-GGUF
  contracts. The reported 129641-row shard becomes 129664 rows, with a complement
  of 118656, for the exact supplied row proportions.
- Directly linking the complete linear object without its GPU libraries failed
  on unused GPU references, including with link-time optimization. The focused
  executable instead compiles the production `linear_share_units()` definition
  verbatim with its real headers and links the actual `tensor_split.cpp` and
  regression source. No unresolved-symbol suppression or arithmetic stubs are
  used. This validates the boundary contract, not a complete engine link.
- All 64 backend patches and the exact prepared source tree pass verification;
  the existing patch-preparation suite passes all 10 tests.
- The audit corrected the initial standalone test registration to use the native
  `ninfer_tests` bundle, avoiding a duplicate GPU kernel image and following the
  project's normal test dispatch and runtime staging. Run it in a native build
  as `ninfer_tests ninfer_linear_share_units_test` or through CTest.
- A focused host project using the real bundle-generation helpers and final test
  registration compiled four C++ objects and linked two libraries and the test
  dispatcher, with no CUDA compilation. CTest passed (1/1). It retains the
  verbatim function extraction described above, rather than linking the full engine.
- Windows sandbox temporary-directory permissions and a compiler PDB probe error
  prevented the first validation attempts. The same tests and configuration passed
  with normal local permissions. The localized compiler dependency prefix was
  taken from actual compiler output before building; no test condition was relaxed.

No remaining code defect was found in the final scoped review. Real-model
multi-GPU startup with an 8192-token chunk, perplexity and performance were not
rerun. The reported perplexity of 2.598610 is user-supplied evidence, not a result
of this validation. Unchanged arithmetic source does not alone establish bitwise
end-to-end equivalence across different shard geometries. Hosted pinned-source
CI verifies patch preparation; it does not compile the complete NInfer CUDA engine.

The supplied ratios 0.478/0.522 and 0.493/0.507 are distinct examples. The exact
reported row fraction is 129641/248320, approximately 0.522072. The regression
tests both stated ratios and the exact row-count example separately.
