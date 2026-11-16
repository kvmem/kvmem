# Radiance upstream updates and retirement of local RDNA3 kernels

`backends/radiance/upstream.json` pins the upstream commit, ABI and **ordered**
SHA-256 patch series. Build from a fresh directory using `prepare_native.py`;
do not copy files over a newer Radiance tree or mix plugin DLLs from another ABI.

| Patch | Ownership and replacement boundary |
| --- | --- |
| `radiance-v2-kvmem.patch` | KVMem engine/observer integration, platform portability and existing RDNA4 integration. Retain and rebase when updating upstream. |
| `radiance-v2-rdna3.patch` | Changes within `libr3d`: attention H256/GQA6, device support and the required general Qwen/DFlash2 operators. Retire matching hunks as upstream acquires equivalent support. |
| `radiance-v2-rdna3-iqr.patch` | The separate `libr3diqr` extension and its root CMake registration. Can be removed as one component when upstream RDNA3 IQR coverage is qualified. |
| `radiance-v2-offload-budget.patch` | Layer-offload KV-budget rounding fix and native regression test. Retire independently when fixed upstream. |

| `radiance-v2-iq4r-mtp-u2.patch` | Preserve the current KVMem mainline fix for a dedicated U2 MTP draft head with a vocabulary-class target; retire independently when upstream covers it. |

Every temporary component also records its retirement condition in `upstream.json`.
The IQR extension depends on the patched `libr3d` host shape/device helpers.
Do not retire the general RDNA3 patch while retaining the extension unless those
helpers and required operators are present in upstream.

## Updating the upstream pin

1. Fetch the intended upstream release into a separate clean checkout. Record its
   immutable commit, ABI and archive checksum. A release/version number alone is
   insufficient proof of supported operations or numerics.
2. Rebase the core integration first. Compare upstream's declarations, layout
   hooks and kernels against the required model encodings, not only op names.
   Required IQR cases include 1024-row trellis runs, u5 BF16 per-128 group scales,
   u2/u3/u4 rotated embeddings and width-class vocabulary heads.
3. Remove superseded hunks from the component patches. If all IQR coverage is
   upstream, remove the entire IQR entry from `additional_native_patches` and
   its CMake registration, then update build target lists, package plugin lists
   and launch/profile capability metadata to use the upstream plugin. Retain
   the numerical oracle tests as an independent test harness and update their
   symbol/ABI bindings. Do not leave the old higher-priority plugin installed:
   it could silently win selection over upstream.
4. Regenerate hashes and prepare a fresh tree with `prepare_native.py`. Its
   manifest must enumerate every remaining patch and every source file.
   `source_identity.py` rejects old/missing/reordered patches. Source and binary
   packagers record the complete series, not just its first element.
5. Run the portable adapter, native architecture and memory tests. On RDNA3,
   run the independent canonical-trellis GPU suite, `--audit` (long reductions,
   large output width and grouped prefill), attention and DFlash2/fusion tests.
   Include normal and padded rows, guards, rejected shapes, zero groups,
   256/512/1024 trellis runs and both BF16-group/row-F32 scales.
6. Compare old and upstream providers on the same hardware, model, placement,
   prompt/output length, KV precision and draft budget. Test IQ2R, IQ3R and
   IQ4R-S, DFlash2 on/off, and separate all-VRAM from layer-offload results.
   Check outputs and accepted-draft counts before comparing HIP-event kernel
   times and model throughput. A short RAM-offloaded test is not a kernel
   performance benchmark; chip watts are not total board power.
7. Qualify Windows and Linux separately. Record which GPUs ran the tests and
   which were only compiled. Start a clean runtime directory and rebuild the
   engine, architecture plugin and kernel plugins together, clearing stale
   tuning/kernel metadata. Merge only the reviewed, passing candidate.

The current hardware evidence is Windows RX 7800M/gfx1101. Four gfx110x code
objects in a binary do not establish Linux/gfx1100 runtime qualification.
Historical v1.17 RDNA4 qualification does not automatically apply to v2.

## Efficiency constraints retained intentionally

Decode keeps compressed weights and uses the native gfx11 integer dot/WMMA
paths; small multi-token verification reuses a decoded weight tile across token
rows. Prefill uses caller-owned `N*K` bytes of temporary int8 fragments plus
small counters, decoded once per invocation and shared across token bands.
For the Qwen gate/up shape `N=34816,K=5120` that plane is 170 MiB; it is workspace,
not a persistent expanded copy of all model weights. This is still material on
a 12 GiB card and must be included in placement budgets.

The gated last-arriver counters allocate and synchronize once for a new
stream/output owner; subsequent launches reuse them. They are released via
`rad_plugin_freed`. No per-token host weight readback is required. This does not
establish that decode saturates the GPU: small kernels, cache traffic and
CPU/driver scheduling still matter. RAM offload additionally transfers weights
over the host link each step; a higher power limit cannot remove that cost.
