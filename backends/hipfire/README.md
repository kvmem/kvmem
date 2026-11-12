# hipfire backend

hipfire is a Rust + HIP engine whose Qwen3.5 serve path keeps KV in paged Q8
arenas (one FullAttention K/V arena per layer, 128-token pages). This backend
lets KVMem own *residency* for those pages: a full page is either resident in
the slot's block table or replaced by the `SPILLED_PAGE` sentinel while its
packed Q8 bytes live in the KVMem host store. The masked attention kernels
skip sentinel pages, so an evicted page is invisible until a selection
restores it.

## Pieces

| Piece | Where |
|---|---|
| C ABI shim over the portable KVMem host library | `hipfire_kvmem_shim.{h,cpp}` (this directory) |
| Rust FFI crate + shim build | hipfire `crates/hipfire-kvmem` (`build.rs`, `KVMEM_ROOT`) |
| Paged spill primitives (sentinel, page ops, kernel masks, paged split-K verify kernel) | `../patches/hipfire-paged-spill.patch` |
| Statistics taps, controller, serve wiring; escha-dense baseline, Bonsai TQ2/BQ1 slots admission | `../patches/hipfire-kvmem.patch` |
| RDNA3 (gfx1100) operator pass: MQ3G256V2 q8x verify decode, real escha grouped-WMMA batched prefill (replaces the `kvmem` patch's `prefill_chunk=1` stopgap), escha-pro/xt F16 in_proj handling | `../patches/hipfire-rdna3-opt.patch` |

The shim is host-only C++17: it wraps `RawKvStore` (block statistics),
`KvMemStore` (block bookkeeping / selection) and the Mean-K retrieval scorer.
No kvmem symbol reaches a HIP kernel; device transfers stay in hipfire.

## Building

`scripts/prepare-backends.py --backend hipfire` clones the pinned base commit
into `../hipfire` beside this checkout (override with `--hipfire-dir`) and
applies the verified patch series below; `--backend all` includes it when a
ROCm toolchain is detected (`HIPFIRE_ROCM_PATH`, `/opt/rocm` or `hipcc`).

Point `KVMEM_ROOT` at this checkout (default is the sibling `../kvmem` of the
hipfire workspace) and build the CLI **and the daemon** with the `kvmem`
feature — the multi-slot serve engine lives in the `daemon` binary, so building
only the CLI leaves a stale engine:

```sh
HIPFIRE_ROCM_PATH=/opt/rocm cargo build -p hipfire-cli -p hipfire-daemon --features kvmem --release
```

The patch series applies to hipfire `89e32856d25d6f6e6645319f2ec2c641c46be525`
(v0.4.6, fork `smashdog/hipfire`); see `upstream.json` for hashes and the
verified tree. That release still predates the escha-dense merge, so the
`kvmem` patch also carries the escha-dense baseline (escha trellis
projections, quantize/dispatch support) it builds on, plus the Bonsai
TQ2G128/BQ1G128 flat-lowbit slots admission. On the v0.4.5 -> v0.4.6 rebase
the Escha shared-weight DType tags were renumbered 65/66 -> 68/69 (upstream
took 65..67 for MQ4G256V2P/MQ4G128V2P/HC8E4M3Row); the HFQ wire qt=42/43 is
unchanged, and the fp16-x scratch marker invalidation the old patch added is
now enforced by upstream's `fp16_x_scratch_writable`.

The third `rdna3-opt` patch is a gfx1100 (RDNA3) operator pass on top of that
integration, verified byte-exact (`89e32856d` + the three patches reproduce
`upstream.json`'s `patched_tree`). It removes the `kvmem` patch's escha prefill
stopgap — that patch forced `prefill_chunk=1` so every prefill row ran through
the lowered paged decode path — and replaces it with a real grouped-WMMA
batched-prefill trellis arm in `forward_slots` (one weight read per
(layer,batch) instead of per slot). Escha prefill is ~10.6-10.9x faster and its
greedy needle argmax matches the per-token reference token-for-token;
`HIPFIRE_PREFILL_CHUNK_OVERRIDE=1` still forces the serial reference for the A/B.
The arm also handles the escha-pro/xt checkpoints that store DeltaNet
`in_proj_a/b` as F16/F32 instead of MQ6G256V2 (`slots_plain_gemm_key` gains
F16->GemmF16WmmaMb8 / F32->GemmF32Batched; need_rot/anchor are derived from the
actual in_proj dtypes). For decode it adds a 3-bit MQ3G256V2 arm to the gfx1100
q8x verify kernels (bit-exact vs the reference GEMV), lifting MQ3 MTP decode
36.8 -> 45.9 tok/s (+24.7%); q8x verify only activates at batch>=2 (MTP spec
decode), so SPEC=off is unaffected.

## Enabling

`HIPFIRE_KVMEM=1` on `hipfire serve`. Requires the paged Q8 KV pool (the only
KV quantisation supported by this first version) and the batched VL path. The
prefix cache is disabled while KVMem is active: KVMem owns page residency, so
sessions always cold-start and every block's statistics are fed from its own
forward.

| Env | Default | Meaning |
|---|---|---|
| `HIPFIRE_KVMEM` | off | enable the controller |
| `HIPFIRE_KVMEM_BUDGET_TOKENS` | min(cap, pages/slot − headroom) | resident tokens per slot before pages spill |
| `HIPFIRE_SLOTS_PAGED_PAGES` | 64 × n_slots | total physical pages (overrides pool sizing) |
| `HIPFIRE_KVMEM_SINK_BLOCKS` | 1 | oldest blocks pinned for every selection |
| `HIPFIRE_KVMEM_RECENT_BLOCKS` | 1 | newest blocks pinned for every selection |
| `HIPFIRE_KVMEM_SELECT_INTERVAL` | 4 | pages between re-selections (in full pages) |
| `HIPFIRE_KVMEM_HOST_BYTES` | 16 GiB | RAM tier capacity |
| `HIPFIRE_KVMEM_SPILL_DIR` | — | set to enable the disk tier |
| `HIPFIRE_KVMEM_SPILL_BYTES` | 64 GiB | disk tier capacity |
| `HIPFIRE_KVMEM_TRACE` | off | per-step residency diagnostics |

With `cap ≤ budget` nothing ever spills, the tap kernels only copy, and the
forward is bit-identical to an unpatched hipfire. Spill/restore moves arena
bytes verbatim, so a spilled-and-returned page is bit-exact too. Multirow
spec-verify steps (MTP, 4..=8 rows past 4096 tokens) run the reference
split-K WMMA verifier through a block-table twin: a bench-verified bitwise
copy of the descriptor-less route (`diff floats 0`), with absent pages
masked out and skipped wholesale instead of sweeping the logical context.
