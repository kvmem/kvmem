# Radiance Windows / Linux AMD backend

Radiance 1.17.0 / IQ3R-S v4 is qualified on Linux RX 9060 XT gfx1200 TP1/C1 with FP8 KV/F32 state for text, bounded RAM/SSD history and DFlash2 depth 7, including the 12K two-stage test. CPU PNG request/cache/placement checks pass; the original image suite is 21/22 because low-resolution OCR reads 5829 as 5029, also reproduced by ordinary native without the KVMem observer. The separate higher-resolution probe passes on both paths. JPEG model requests, Windows, TP2 and general-profile workloads remain unqualified. See the [integration record](../../docs/radiance-iqr-v4-integration-20261105.md) and [startup runbook](../../docs/radiance-iq3v4-runbook-20261105.md). The dated sections below describe earlier model/platform runs.

Build and run with `tools/build.py --platform auto|windows|linux` and
`tools/serve.py --platform auto|windows|linux` on the target OS.

The adapter implements native page ownership, shared RAM/SSD spill and restore,
recurrent checkpoints, multiple retained conversations and concurrent prefix
branches. Query uses the common page-softmax or meanq-cosine scorer, bounded
replay, graph recall and LRFU. Prefill/decode Pressure selects resident history;
ordinary tool continuation does not recall cold history.

Cache ownership follows the common [checkpoint contract](../../docs/checkpoint-policy.md#shared-owners-in-ninfer-and-radiance).
New user requests may discover exact prefixes across named or anonymous owners;
native branches share sealed host/SSD records while keeping mutable state
independent. Admission deducts only proven physical aliases. GPU page sharing
and cache persistence are outside this implementation.

The shared launcher and standalone profile default to a 256-token prefill chunk. Existing
shared-launcher configurations with an explicit `Prefill` retain that value.

The shared console launcher supports one or two AMD GPUs, up to eight concurrent text/image
requests, fixed or native cost-adaptive MTP (up to seven drafts, default three), and process-local
SSD tiering. SSD files are temporary: a server restart does not restore sessions.
CPU storage work yields per request; failed optional cache parking drops that
entry, while a failed required read fails only the affected request. GPU transfer
failures remain fatal. RAM/SSD quotas are shared by active and retained sessions;
bounded I/O staging, statistics, model weights and runtime workspaces are separate.

Two GPUs use native tensor parallelism with exact collective transport, not a
layer split. Query heads and KV heads must divide evenly across the devices.
Each rank owns its KV/state bytes; accepted statistics are combined by head,
and a working set becomes executable only after every rank is ready. Prefix
branches, checkpoints and RAM/SSD quotas cover the complete set of rank shards.
The Linux dual-R9700 qualification includes a host-staged collective path with
peer-to-peer access disabled; it is not a throughput comparison with one GPU.

On systems where HSA cannot confirm CPU access to the VRAM pool, AQL kernel
argument rings and recorded tapes use the existing coherent host allocation.
Model weights and KV payloads remain on their selected GPU devices.

The native worker caps explicit and model-resolved draft depths at seven. The
shared launcher checks embedded MTP/DFlash2 metadata and the trained depth before
launch. The current profile uses a verify window of up to eight positions.

Integration testing uses Linux, Radeon AI PRO R9700 (gfx1201), Qwen3.8-27B MXFP4
and FP8 KV. Native still-image input requires a qwen35 container with vision
weights and an FFmpeg-enabled build. Enable `Vision`
in the shared launcher; `VisionTokens` limits image tokens after spatial merge.
Image content and native rotary layout participate in retained-prefix identity.
Complete image pages remain required through Query and Prefill/Decode Pressure.
KV/state reuse does not currently skip native image preprocessing/encoding.

BF16/FP8 native memory contracts are tested separately. Windows AMD,
other models/devices, configurable n-gram drafting and DFlash combinations
still require qualification; their shared-launcher gates remain explicit.
Video/audio input is not enabled by this adapter.

`VisionResidency=resident|cpu` selects the complete Qwen vision encoder's execution
site. CPU mode maps its weights from the container and keeps its activations and
encoder inputs on the host. `EmbeddingPlacement=ram|vram` independently places the
text embedding table; its default remains `vram`. The CPU validation on RX9060XT
used `ram` to fit the 27B model and two sessions in 16GB.

The CPU implementation includes an approved correction to both libavx and libref:
GEMM must preserve a residual before writing an overlapping output. Independent
arithmetic tests establish correctness rather than treating libref as an oracle
for the defect it shared. The 2026-11-03 Linux/i7-8700/RX9060XT run passed PNG
tests at 64/256 image tokens with MTP3 and two overlapping native requests.
Windows CPU vision, other CPU ISA, natural-image/OCR quality and CPU/GPU
performance parity are outside that run. See
[CPU image validation](../../docs/radiance-cpu-image-validation-20261103.md).

Original standalone profile: RX9060XT/gfx1200, BUDGET36864, generation reserve16384,
batch512, context262144, FP8 K/V, MTP3. Service: localhost18380, model radiance-local.

- [Build and deployment](../../docs/radiance-integration.md)
- [Client examples](../../docs/radiance/CLIENT-QUICKSTART.md)
- [Performance results](../../docs/radiance/PERFORMANCE-COMPARISON.md)
- [WSL runtime preparation](wsl-runtime/README.md)

Pinned source and patch metadata are in upstream.json. Preserve upstream
LICENSE/NOTICE with redistributed sources and binaries.

## Packed KV and recurrent state (2026-11-01)

The launcher accepts `KvType=fp8|bf16|fp8_v4|k6_v4` and
`StateCacheType=f32|f16`. Defaults remain `fp8` and `f32`. Native flags are
`--kv-cache-dtype` and `--state-cache-dtype`. For the qualified lower-memory
profile, use these fields in the ordinary launcher configuration:

```json
{"KvType":"k6_v4","StateCacheType":"f16","Budget":36864,"Reserve":4096}
```

`fp8_v4` stores FP8 K and group-32 INT4 V; `k6_v4` stores group-32 INT6 K
and INT4 V, with FP16 scales. Physical row widths are 400 and 352 bytes per
KV head/token at logical head dimension 256, versus 512 bytes for FP8 K/V.
The native formats apply to the shared MTP attention group too. F16 selects
the GDN recurrent state; convolution state retains its existing format.
Pressure decodes the packed K layout, and retained-state identities include
both format selections. Storage/restore copies exact packed payload bytes.

The selective backport comes from upstream `203064b4877f65fdbc70eca1077e65f4681bad61`.
It keeps the existing scheduler, prefix fix and FP8/BF16 attention kernels.
The extended build uses custom ABI `1263927297`; deploy its server and native
plugins together. This is not the complete upstream ABI32 release.

Qualification: Linux RX9060XT gfx1200, Qwen3.8-27B MXFP4, head_dim=256,
GQA=6, block_size=16. GPU numerical, fused store and RAM/SSD checkpoint tests
pass. Four format/state combinations pass 68 real-model checks, including
document-first separate-question recall. FP8/F32 and K6_V4/F16 each complete
the canonical 33-request 256K Task2 at B36K/R4K with 31/31 eligible cache hits.
Generated Task2 output quality is evaluated separately from execution/cache
correctness. Windows, RDNA3, other geometries, packed image reuse and packed
multi-request concurrency are not qualified by these runs.

For this model, the MTP projections and shared output head already use MXFP4.
The remaining BF16 `mtp.fc` payload is 100MiB; MXFP4 would reduce it to
26.5625MiB before alignment/workspaces. That FC kernel/conversion path is not
implemented here, and its speed/acceptance impact has not been measured.
