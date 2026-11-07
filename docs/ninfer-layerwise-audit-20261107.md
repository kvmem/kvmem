# NInfer layerwise integration audit — 2026-11-07

Candidate reviewed over `a8cca288b5106870b3388f674c5dbdbbf11cd67a`. The NInfer recipe is based on native tree
`56e5e7e6a1240249ba1d7569f67bbf7cb26d23c7` and reconstructs
`082a7fe73dc848c6d0d9b0029dc62a43a690692a` exactly. The patch roundtrip passed.
Algorithm reference: kvmem-vllm `72a81cc99aa78d85004ecf1364fc9f3094db8816`,
still its main branch head when rechecked on November 7.

## Result and scope

Dense Qwen 27B, NInfer only. Target Query uses fresh current-layer Q, without
target Query replay. MTP defaults to first-layer; explicit upstream-global retains
its isolated resident-view probe. DFlash2 keeps its independent native window.
Existing ordinary prefix and native recurrent-state checkpoints remain necessary.
Shared configuration, capability checks and resource interfaces stay backend-neutral;
an unimplemented worker cannot opt into Layerwise merely through launcher fields.

No scoring, S/M/N, sampling, eligible-token clock or selection formula was changed
by this integration audit. Mainline adapter/control-vector edits are explicitly
rejected for Layerwise until their layer-major execution is implemented.

## Corrections made during integration

- Preserve mainline FP32 target logits, current TP interfaces and native load planning.
- Price the auxiliary MTP scratch allocation on each rank that actually owns MTP KV.
- Preserve PR #272 immutable checkpoint ownership: Legacy uses its sealed manifest;
  Layerwise uses its checkpoint-owned exact canonical KV plus Mean-K index/provenance.
  Lookup, cold accounting, forking and restore dispatch to that immutable owner.
- Preserve other newly merged PolyStrata/Radiance changes in their own recipes.
- Correct stale tests that required explicit MTP selection despite the approved
  first-layer default, and update launcher fixtures to the actual worker capability contract.

## Validation

The integration runs below used RTX 5060 Ti
`GPU-A` (public alias), after occupancy checks.
Model SHA-256 `46bb5c800b0284377c47c97f3d334331b984f2fa9c55624b9eb4d65d5aeb9b03`;
worker SHA-256 `9e6a99432429afcdda0edf653f4fc2c6e5b9c70bebfcb2ddb242bd2037d3fe32`. Source/image/model pins were checked
before and after the public functional runs.

| Check | Result |
|---|---|
| Shared core and server-option host tests | 20 passed |
| Layerwise product Python tests | 39 passed |
| Model inspection regression | 27 passed |
| Local console regression | 64 passed |
| Actual-worker browser contract / launcher DOM | 7 / 9 checks passed |
| Native immutable source, snapshot, physical-layout CPU, serve options, frontend, checkpoint metadata suites | 6 passed |
| Real 27B physical tests | first-layer positive/cancel/post-forward; upstream-global positive; DFlash2 positive/cancel passed |
| Ordinary document → separate question → correlated follow-up | Both MTP policies passed; answers, native Query spans and actual prefix reuse checked |

The real-model physical tests exercise C=256/K=192/G=64 with physical draft width
16 and one valid row at H=L=255, then exact full/partial checkpoint restoration.
They check native target/draft KV, GDN/ring state, Raw FP32 statistics, canonical
history and per-layer provenance. They are boundary/rollback checks, not quality
or acceptance-rate benchmarks.

The public tests use K=8192/G=1024/C=9216, Int8 KV, prefill 128, MTP draft 3,
NGram 0 and CUDA graphs. The document has 12,226 prompt tokens; the question
actually reuses 12,160 tokens and the follow-up reuses 12,224. First-layer runs
with the MTP history flag omitted and has zero probes. Upstream-global explicitly
selects that policy; probe costs are 257.124 ms and 328.201 ms, both restored.
These two functional observations do not replace the earlier controlled 30-run
policy comparison and are not a new speed/acceptance ranking.

## Follow-up validation

Before the endpoint eviction fix, the follow-up patch differed from the model-qualified tree only under `tests/`.
Four additional CPU and five single-GPU fixture executables passed. They cover
budget ownership, graph feedback layout, physical output, attention against an
independent reference, KV transfer/write, DFlash features and the Program Query
lifecycle. Prefix feedback and Query feedback are counted separately; retired
immutable feedback is charged to the retained-source account. Those fixture changes preserved the public model worker hash. The shared HIP source uses the ROCm runtime names;
the CUDA branch remains identical to the source used for the real-model runs.
The standalone conditioning host build and Windows browser launcher also passed.

## Limits and bounded behavior

The initial three-turn runs skipped the third endpoint because the shared source
account could reject image growth without reclaiming idle histories. On that
specific quota failure the worker now trims free buffers, uses the existing
group-aware eviction order for idle histories, and retries after sufficient
bytes are released. Restore sources and pending transfers are protected.
The native regression checks that an all-protected cache preserves its source
and that releasing the protection allows the endpoint to be published.

The fixed worker repeated both policies on the same 27B and 5060 Ti with the
original 1 GiB metadata limit. All four turns published endpoints in each run.
The third and fourth turns each evicted one history. Turn four reused 12,275
tokens. Its aligned Query boundary preceded the third endpoint, so the worker
restored the matching older endpoint. Results and worker identity are in the
[endpoint eviction regression](benchmarks/ninfer-endpoint-eviction-20261107.json).

The Windows launcher aggregate has 31 passes, 43 platform skips and one
Linux-only `fcntl` import error; it is not reported as an all-platform pass.
The first public GPU attempt could not connect to localhost under the socket
sandbox, sent no document, and was retried with local socket access. No result
from that attempt is treated as model validation.

Build scope was audited before each launch. Reused objects were checked against
compiler commands and transitive source/header dependencies, normalizing only
line endings for C/CUDA text comparison. Initial integration reused 738 objects
and required 174 C++ objects plus one shared scorer CUDA object. PR #272's shared
structure changes then required 102 C++ objects, no CUDA. Final repairs rebuilt
only affected C++ objects/tests; Ninja logs and dependency databases were retained.

Portable metrics, identities and log hashes are in
[the audit evidence](benchmarks/ninfer-layerwise-audit-20261107.json).
Other backend native implementations, MoE and multi-GPU execution are outside
this audit's GPU qualification. Merge still requires the user's explicit confirmation.
