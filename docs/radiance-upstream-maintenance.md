# Radiance upstream updates

`backends/radiance/upstream.json` pins the upstream commit, ABI and **ordered**
SHA-256 patch series. Build from a fresh directory using `prepare_native.py`;
do not copy files over a newer Radiance tree or mix plugin DLLs from another ABI.

| Patch | Ownership and replacement boundary |
| --- | --- |
| `radiance-v2.2-kvmem.patch` | ABI43 integration, KVMem observer and checkpoint hooks, platform portability, and RDNA4 kernels. Includes the remaining KVMem U2/width-class head policy on top of upstream MTP. |
| `radiance-v2-offload-budget.patch` | Layer-offload KV-budget rounding fix and native regression test. Retire independently when fixed upstream. |

The ABI39 RDNA3 patches, including `libr3diqr` and the RX 7800M Windows
profile, are kept together on `maint/radiance-v2-rdna3` at controller commit
`ecfcc82f75348f453982a21352397e2d39c45a25`.

## Updating the upstream pin

1. Fetch the intended upstream release into a separate clean checkout. Record its
   immutable commit, ABI and archive checksum. A release/version number alone is
   insufficient proof of supported operations or numerics.
2. Rebase the core integration first. Compare upstream's declarations, layout
   hooks and kernels against the required model encodings, not only op names.
   Required IQR cases include 1024-row trellis runs, u5 BF16 per-128 group scales,
   u2/u3/u4 rotated embeddings and width-class vocabulary heads.
3. Remove superseded hunks from the patches after checking matching coverage.
   Keep the relevant regression tests and update the build targets, package
   plugins and launcher capability metadata when changing a kernel provider.
4. Regenerate hashes and prepare a fresh tree with `prepare_native.py`. Its
   manifest must enumerate every remaining patch and every source file.
   `source_identity.py` rejects old/missing/reordered patches. Source and binary
   packagers record the complete series, not just its first element.
5. Run portable adapter, native architecture and memory tests. Check that
   document ingestion followed by a separate question proves prefix reuse and
   Query replay in the runtime trace. For RDNA3 procedures, see the maintenance
   branch and its historical qualification.
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

The Windows RX 7800M/gfx1101 evidence applies to the maintenance branch.
The [RDNA4 ABI43 Windows qualification](radiance/evidence/windows-gfx1201-abi43-text.json) covers the single-GPU RX 9070 XT text configuration.

## RDNA3 maintenance branch measurements

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
