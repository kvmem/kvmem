# Radiance Windows / Linux AMD backend

For native Windows RX 9070 XT (gfx1201), start with the [console and chat quickstart](../../docs/radiance/CLIENT-QUICKSTART.md). The qualified text configuration uses Radiance 1.17.0, ordinary Qwen3.8-27B IQ3R v4, DFlash7, one GPU and one sequence, FP8 KV and F32 state. The full 256K Task 2 run and same-build comparisons are linked below.

The Linux RX 9060 XT (gfx1200) IQ3R-S v4 qualification covers single-GPU text, bounded RAM/SSD history and DFlash7, including a 12K two-stage test. CPU PNG request/cache/placement checks pass. The original image suite scored 21/22 on low-resolution OCR; the separate higher-resolution probe passed. See the [Linux integration record](../../docs/radiance-iqr-v4-integration-20261105.md) for tested profiles and exclusions.

Build and run with `tools/build.py --platform auto|windows|linux` and
`tools/serve.py --platform auto|windows|linux` on the target OS.

For the native Windows RX 9070 XT text worker, use an x64 Visual Studio
developer PowerShell with CMake, Ninja and the Windows HIP SDK. From the
repository root, set `ROCM_PATH` to the SDK directory and run:

```powershell
python backends/radiance/tools/prepare_native.py --destination build/radiance-native
python backends/radiance/tools/build.py --platform windows --source build/radiance-native --build build/radiance-win-gfx1201 --rocm "$env:ROCM_PATH" --gpu-target gfx1201 --ffmpeg OFF --target radiance-kvmem-server --target qwen35_w8a8 --target libr4d --target libavx --target libref
```

The worker is in `build/radiance-win-gfx1201/bin` and its plugins are in
`build/radiance-win-gfx1201/radiance_home`. Keep the SDK runtime DLLs
beside the executable. If the CPU plugins link the Visual Studio LLVM
OpenMP runtime, make the matching `libomp140.x86_64.dll` available to the
Windows plugin loader on the development machine. The RX 9060 XT uses
`--gpu-target gfx1200` (the default).

On native Windows gfx1201, the single-rank, single-sequence Qwen35
`w8a8_fwht128` text profile with DFlash7 and all-VRAM placement issues both
logical compute lanes on one HIP stream. Stream order handles the GDN
projection dependencies. The launcher records the statistics completion event
after the last layer copy, so the CPU can collect the frame as soon as those
bytes are ready. Set `KVMEM_RAD_HIP_SINGLE_STREAM=0` before launching the
worker or the Windows supervisor to restore separate compute streams.
The [RX 9070 XT measurements](../../docs/radiance/evidence/windows-gfx1201-dflash7-decode.json)
include a fixed 256-token response and cached 32K and 49K Task 2 requests.
On [the full 33-request Task 2 run](../../docs/radiance/evidence/windows-gfx1201-dflash7-task2-256k.json),
the final request reached 257,958 prompt tokens and decoded 1,542 tokens at
109.18 tok/s. The same-host historical run reached 43.39 tok/s. All 33 response
bodies, reasoning texts and speculative token counts matched.

The adapter implements native page ownership, shared RAM/SSD spill and restore,
recurrent checkpoints, multiple retained conversations and concurrent prefix
branches. Query uses the common page-softmax or meanq-cosine scorer, bounded
replay, graph recall and LRFU. Prefill/decode Pressure selects resident history;
ordinary tool continuation does not recall cold history.

Cache ownership follows the common [checkpoint contract](../../docs/checkpoint-policy.md#shared-owners-in-ninfer-and-radiance).

DFlash keeps at least its declared window in the recent selection. Accepted
prefix and Query checkpoints carry an immutable copy of the native draft-window
KV alongside recurrent state; RAM/SSD parking and forks include these bytes.
Restoration publishes both at the saved frontier, without a fixed window-sized
checkpoint setback. Native prefill alignment, Query boundaries and the final
uncached token still constrain reuse. See the [conditioning validation](../../docs/radiance-dflash-conditioning-20261106.md).
Recent also reserves partial-tail slack: a 2048-token draft window with
128-token selection pages needs at least 2176 effective recent tokens. See the
[merge audit](../../docs/radiance-v117-merge-audit-20261106.md) for the coverage,
cache admission and DFlash7/MTP3 regression results.
New user requests may discover exact prefixes across named or anonymous owners;
native branches share sealed host/SSD records while keeping mutable state
independent. Admission deducts only proven physical aliases. GPU page sharing
and cache persistence are outside this implementation.

Radiance checkpoints now reference immutable blocks in the common host/SSD
store. New frontiers capture only missing blocks and a private partial tail;
unchanged full blocks also alias the historical archive. Rewind invalidates
changed origins immediately, even when a later block reuses its logical ID.
Recurrent state and accepted draft windows keep their existing native snapshots.
Restoration still stages the selected execution window; the reported capability
is `full_selected_view`, not full-history RAM promotion.

`--kvmem-cache-sessions` bounds content groups. `--kvmem-cache-records N` separately
bounds published records; zero (default) follows the group limit, replacing the
previous hidden minimum of 16 records. A record can carry endpoint, input and
fallback recovery points. Active readers and bounded in-flight parking retain
their own leases. The monitor distinguishes groups, records, recovery points,
logical view bytes and actual captured KV bytes. Use an explicit record limit
when comparing versions.

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

BF16/FP8 native memory contracts are tested separately. The native Windows
hardware evidence above covers the specified text profile. Other Windows
model, device, multimodal and multi-GPU combinations require separate qualification.
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
