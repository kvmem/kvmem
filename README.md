**English** | [简体中文](README.zh-CN.md)

# KVMem 0.18.0 · llama.cpp & NInfer

KVMem manages long conversations and agent tool histories within a bounded GPU working set. Native KV history lives in Host RAM, with optional SSD tiering; retrieval brings selected history onto the GPU. **`master` contains the 0.18.0 multi-backend source**, and **stable NInfer 0.18.0 packages** are available for Windows RTX 30/40/50 GPUs.

**RTX 20 / TITAN RTX and Tesla V100:** the new **[NInfer v0.18.1 RC1](https://github.com/kvmem/kvmem/releases/tag/v0.18.1-ninfer-rc1)** hardware-trial packages are available now, with the Web UI built in. [Download RTX 20 / sm75](https://github.com/kvmem/kvmem/releases/download/v0.18.1-ninfer-rc1/kvmem-v0.18.1-ninfer-rc1-windows-rtx20-sm75.zip) · [Download V100 / sm70](https://github.com/kvmem/kvmem/releases/download/v0.18.1-ninfer-rc1/kvmem-v0.18.1-ninfer-rc1-windows-v100-sm70.zip)

The repository is now **[kvmem/kvmem](https://github.com/kvmem/kvmem)**. GitHub redirects the former [kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp) URL, including existing release and download links. Do not reuse the old repository name, which would replace the redirect.

**QQ community: 1040777853**

[Downloads](#downloads) · [Quick start](#quick-start) · [Defaults](#defaults) · [Performance](#performance) · [Source](#source) · [Docs](#documentation)

<a id="downloads"></a>

## Downloads

**Use English paths for the package and model. The full path, including the username, parent directories and filename, must not contain Chinese characters.**

### RTX 20 / TITAN RTX and Tesla V100 · v0.18.1 RC1

| GPU | Windows x64 NInfer v0.18.1 RC1 | Build | Size |
|---|---|---|---:|
| RTX 20 / TITAN RTX | [Download ZIP](https://github.com/kvmem/kvmem/releases/download/v0.18.1-ninfer-rc1/kvmem-v0.18.1-ninfer-rc1-windows-rtx20-sm75.zip) | sm75 / CUDA 13.2 | 1161.3 MiB |
| Tesla V100 | [Download ZIP](https://github.com/kvmem/kvmem/releases/download/v0.18.1-ninfer-rc1/kvmem-v0.18.1-ninfer-rc1-windows-v100-sm70.zip) | sm70 / CUDA 12.9 | 939.4 MiB |

[RC1 release notes](https://github.com/kvmem/kvmem/releases/tag/v0.18.1-ninfer-rc1) · [ZIP SHA256](https://github.com/kvmem/kvmem/releases/download/v0.18.1-ninfer-rc1/SHA256SUMS-ninfer-old-gpu.txt) · [Package validation](https://github.com/kvmem/kvmem/releases/download/v0.18.1-ninfer-rc1/PACKAGE-VALIDATION.json)

These packages include the embedded Web UI and are intended for hardware trials. Their build targets, dependencies and archives have been checked, but RTX 20, TITAN RTX and V100 physical-GPU qualification is still pending.

### RTX 30 / 40 / 50 · v0.18.0 stable

| GPU | Windows x64 NInfer 0.18.0 | Size |
|---|---|---:|
| RTX 30 / sm86 | [Download ZIP](https://github.com/kvmem/kvmem/releases/download/v0.18.0/kvmem-v0.18.0-ninfer-windows-rtx30-sm86.zip) | 971.6 MiB |
| RTX 40 / sm89 | [Download ZIP](https://github.com/kvmem/kvmem/releases/download/v0.18.0/kvmem-v0.18.0-ninfer-windows-rtx40-sm89.zip) | 964.5 MiB |
| RTX 50 / sm120a | [Download ZIP](https://github.com/kvmem/kvmem/releases/download/v0.18.0/kvmem-v0.18.0-ninfer-windows-rtx50-sm120a.zip) | 859.0 MiB |

[Full release notes](https://github.com/kvmem/kvmem/releases/tag/v0.18.0) · [ZIP SHA256](https://github.com/kvmem/kvmem/releases/download/v0.18.0/SHA256SUMS-ninfer-only.txt) · [Package guide](https://github.com/kvmem/kvmem/releases/download/v0.18.0/README.md)

Packages include the NInfer server, embedded Web UI, launchers and CUDA runtime dependencies. **Models and the llama.cpp worker are supplied separately.** A CUDA 13.2-compatible NVIDIA driver and the Microsoft Visual C++ 2015–2022 x64 runtime are required; running a package does not require the CUDA Toolkit or Visual Studio. NVFP4 **model weights** require the RTX 50 package.

<details>
<summary>Earlier packages</summary>

| Platform | Backend | Model | Release |
|---|---|---|---|
| Windows / Linux / WSL2 NVIDIA | llama.cpp | GGUF | [CUDA v0.16.0-rc3](https://github.com/kvmem/kvmem/releases/tag/v0.16.0-rc3) |
| Windows / Linux / WSL2 AMD | llama.cpp | GGUF | [ROCm Beta2](https://github.com/kvmem/kvmem/releases/tag/rc3-rocm-beta2) |
| Windows Bonsai 2 | Dedicated | PTQ1 GGUF | [prism.3](https://github.com/kvmem/kvmem/releases/tag/v0.16.0-rc3-prism.3) |

These packages retain their own feature and parameter scope; mainline updates do not add 0.18.0 features to them. The 0.18.0 binary release covers Windows NInfer only. Source builds for Linux, ROCm and llama.cpp are described below.

</details>

## What's in 0.18.0

- **Multi-backend source**: Shared KVMem memory, retrieval and storage contracts, with separate native inference and serving implementations for llama.cpp and NInfer. Upstream versions, integration patches and full provenance are pinned in the [backend manifest](backends/versions.json).
- **History and state reuse**: Query-based KV selection preserves original logical positions. Multi-turn requests, tool continuations, edits and cancellations restore state according to each backend's exact-prefix and checkpoint rules.
- **RAM/SSD tiering**: Active and idle histories share quotas. Complete native K/V, quantization metadata and MTP KV can move to SSD and are read back according to the selected history, without requantization.
- **LRFU and GQA max**: Windows launchers enable history weighting with `lambda=0.5`, `top-frac=0.9` and GQA `max`; sink=12288 and recent=1024.
- **Tools and clients**: NInfer enables tool choice and strict schema constraints by default, with Chat Completions, Responses and Anthropic Messages support. Only complete tool calls are published; unsupported schemas produce explicit errors. Responses reasoning returns `summary=[]` while retaining raw reasoning text. Image detail accepts low/high/original and maps them to auto. Native tool constraints compose with thinking, JSON output and speculative decoding.
- **NInfer execution options**: Nine KV formats, fixed or adaptive MTP, ngram, concurrent text and CPU/resident image requests, and a layer pipeline across up to eight GPUs of the same compute capability. Consult the [capability and qualification guide](docs/multi-backend.md) for supported combinations.

The stable release retains the RC3 engine binaries and updates version metadata, launch defaults, documentation and checksums. **Sink defaults to 12288 tokens (12K), recent to 1024 tokens (1K)**. The NInfer minimum remains 64; values 0–63 are rejected. `BUILD-INFO.json` records binary provenance. `v0.18.0` preserves the initial source snapshot; launcher revisions ship through `master` and updated packages. `master` also retains mainline Responses and build fixes.

<a id="quick-start"></a>

## Windows package quick start

1. Download the ZIP matching your GPU and extract it completely into an English path.
2. Supply a **NInfer v3 `.ninfer`** model, either as `model.ninfer` in the package root or selected with `-Model`.
3. Open PowerShell in the extracted directory, list GPUs and start the service. GPU indices or full UUIDs are accepted; UUIDs are recommended for multiple GPUs.

```powershell
nvidia-smi -L

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-ninfer.ps1 `
    -Model 'D:\models\model.ninfer' -Gpu '0'
```

Open the Web UI at `http://127.0.0.1:18200/`; the OpenAI-compatible API base is `http://127.0.0.1:18200/v1`. Keep the terminal open; Ctrl+C stops the service. Add `-DryRun` to inspect the resolved arguments.

CPU vision and MTP are enabled by default, so the model needs the corresponding weights and preprocessing resources. Add `-NoVision` if vision weights are absent; add `-MtpDrafts 0` if MTP weights are absent, which also disables adaptive MTP and ngram in the packaged shortcut. NInfer does not need a separate mmproj. If you only have GGUF, see the pinned [NInfer conversion guide](https://github.com/iamwavecut/ninfer-all/blob/8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3/docs/weight-conversion.md). llama.cpp uses GGUF and needs a matching external mmproj for images.

<a id="defaults"></a>

## Default recipe and budgets

These are the defaults of **`start-ninfer.ps1` in the 0.18.0 package**, with 1K=1024 tokens. The unified `start-backend.ps1` launcher and direct worker have different defaults.

| Setting | Default |
|---|---|
| Context | 204800 / 200K |
| GPU history budget B / generation reserve R | 36864 / 16384 |
| Default output limit | 16384 |
| Main / MTP KV | INT8 / INT8 |
| MTP / ngram | Adaptive MTP4 / ngram31, min-match=12 |
| Vision | CPU, 1024 tokens/image |
| Thinking budget | 8192 |
| Host KV payload | 12288 MiB / 12 GiB |
| Prefill chunk | 256 tokens |
| Sink / recent | **12288 / 1024 tokens (12K / 1K)** |
| Retained idle sessions / concurrency | 1 / 1 |
| LRFU / GQA | Enabled, lambda=0.5, top-frac=0.9 / max |
| SSD / device profile | Disabled / off |
| Listen | 127.0.0.1:18200 |

This recipe was checked for model loading and text requests on an RTX 5060 Ti 16GB with Qwen3.8-27B GSQ-RCO IQ3_S vision/MTP weights. Other model/GPU combinations need their own capacity checks. Context sets a logical limit, not a memory capacity guarantee.

| Budget | Meaning |
|---|---|
| B | Selected GPU history tokens per active request, including sink, recent and mandatory image regions. |
| R | GPU append workspace. NInfer can archive KV, refresh its selection and continue generating when space runs out, so output need not fit within R. The current llama.cpp serving path caps per-turn output, including reasoning, at R. |
| H / HostMiB | The process's RAM quota for managed native KV, including Main/MTP. This is not a limit on total process memory. |
| D / DiskMiB | Shared SSD quota for active KV and idle state attachments. |

For NInfer, B/R align to 64-token pages, prefill is a multiple of 128 and no larger than R, and B+R must not exceed context. Recent is at least 64 and rounds down to whole pages. The default recent=1024 retains eight llama.cpp retrieval blocks (128 tokens each) or sixteen NInfer pages. Identical parameter names do not guarantee identical behavior; see the [backend differences](docs/multi-backend-differences.md) before transferring configurations.

NInfer KVMem KV formats: `bf16`, `int8`, `fp8`, `nvfp4`, `k8v4`, `rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8`. Main/MTP use the same format. llama.cpp accepts `f16`, `f32`, `q8_0`, `q5_0`, `q4_0` and can configure Main/MTP separately. Weight quantization and KV format are independent settings. NInfer `int8` and llama.cpp `q8_0` are different native layouts and are not interchangeable.

### Retrieval and SSD

LRFU updates historical scores from the current query once per request. `top-frac` controls which hits may accumulate scores; **it is not a fraction of KV retained**. GQA max takes the maximum relevance across query heads sharing a KV head. Sink, recent and retained media still consume B.

Run these examples in the **extracted release package**:

```powershell
# Disable history weighting and use mean aggregation
& .\scripts\windows\start-ninfer.ps1 -Gpu '0' -NoLrfu -ScoreAggregation mean

# Optional SSD tiering; use an English directory path
& .\scripts\windows\start-ninfer.ps1 -Gpu '0' `
    -DiskPath 'D:\kvmem-cache' -DiskMiB 4096
```

SSD is disabled by default; specifying only `-DiskPath` sets D to 4096 MiB. H/D exclude some mean-K statistics, checkpoints, staging and OS file cache. RAM must still fit active work and paging headroom. Files are reusable only **within the current service process**; restart persistence and cross-backend KV migration are unsupported. Do not combine shared SSD flags with legacy llama.cpp online NVMe/session-disk flags. See [RAM/SSD tiering](docs/active-kv-tiering.md) and its [qualification record](docs/active-kv-tiering-validation-20261010.md).

### Multiple GPUs and concurrency

```powershell
# Package launcher; distribute layers automatically
& .\scripts\windows\start-ninfer.ps1 -Gpu '0,1'

# Manual layer allocation for a 64-layer model
& .\scripts\windows\start-ninfer.ps1 -Gpu '0,1' -StageLayers 40,24
```

NInfer supports a layer pipeline across up to eight GPUs of the same CUDA compute capability. VRAM sizes may differ; use the package matching the architecture. Layer counts must be positive and sum to the model's layer count. GPU list order defines stage order; the first GPU holds embedding/head/MTP and resident vision. Default CPU vision runs on the Host. Comma-separated UUIDs are also accepted.

Text and resident/CPU image requests support up to eight active lanes. B/R are per request and H/D are per process, so increasing concurrency increases resource requirements. Concurrent ngram widths above 15 are clamped to 15; CPU image encoding remains serial. Fast prefill is admitted only for INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8. NInfer KVMem rejects video, overlay vision, DFlash, parallel prefill and legacy `--lookup-ngram`. See [multi-backend runtime](docs/multi-backend.md) for parameter ranges, model boundaries and qualified combinations.

<a id="performance"></a>

## Completed RC3 performance results

These are the completed **RC3** measurements from 2026-10-13 on **Windows, RTX 5060 Ti 16GB, Qwen3.8-27B GSQ-RCO IQ3_S vision/MTP**. Stable 0.18.0 retains the same engine binaries. Results describe this benchmark configuration, not every model/GPU or the default recipe.

| Measurement | Result | Scope |
|---|---:|---|
| Daily text decode | 50.08 token/s | 3 measured rounds |
| Daily image decode | 55.90 token/s | 3 measured rounds |
| Daily HTML/SVG decode | 64.54 token/s | 3 rounds, 512 output tokens each |
| Complete daily task | 30.10 s | Median total time |
| Long-history tool rounds | 53.24 token/s | 32 rounds including final code |
| Final long-history code | 44.87 token/s | Final server-native decode rate |
| Final time to first token | 16.87 s | First reasoning or content fragment |
| Long-history request total | 855.10 s | Sum of 33 requests, excludes loading |
| Final input / output | 261522 / 512 tokens | Near 256K |
| Requests completed / tool-prefix hits | 33/33 / 32/32 | One complete long-history run |

Settings: context=262144, B/R=36864/16384, Main/MTP INT8, GDN FP16; fixed MTP3, adaptive off, ngram=0; prefill=512, thinking=128, output limit=512; GPU resident vision=1024 tokens/image; LRFU(.5/.9), GQA max; Host=12288 MiB, SSD off, sessions/concurrency=1/1, device profile=off; temperature=1, top_p=.95, top_k=20, min_p=0, seed=42; streaming Chat Completions.

The daily test uses one warm-up and three measured rounds with roughly 12K text, an 896×896 image and HTML/SVG generation. The long-history run injects simulated read_file results. Aggregate decode rates include reasoning and divide total output tokens by total server generation time. Tool execution is excluded; code completeness and general retrieval quality were not established. Whole-GPU VRAM peaked near **14.12 GiB**. Available system RAM fell to **8.78 MiB**, so memory pressure may affect latency and total time.

[Full report, configuration, timings and logs](https://github.com/kvmem/kvmem/releases/download/v0.18.0/RC3-performance-results-20261013.zip). Historical llama.cpp results are in the [16 GiB report](docs/recommended-config-performance.md) and [earlier 256K tool test](docs/long-context-benchmark-2026-09-14.md); they use different configurations from this table.

## API and Web UI

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness |
| `GET /v1/models` | Loaded model IDs |
| `GET /props` | Backend, budgets and capabilities |
| `POST /v1/chat/completions` | OpenAI-compatible chat, streaming, tools and supported images |
| `POST /v1/responses` | OpenAI Responses, including SSE streaming |

NInfer also serves native Anthropic Messages. Protocols, tool constraints and schema combinations follow the worker/package documentation. llama.cpp accepts the `/responses` alias; do not assume the same alias for NInfer. NInfer supports images but rejects video; llama.cpp needs mmproj for images and additional build options plus ffmpeg/ffprobe for source video support.

This non-streaming example queries the loaded model ID first, avoiding a placeholder model name:

```powershell
$models = Invoke-RestMethod 'http://127.0.0.1:18200/v1/models'
$body = @{
    model = $models.data[0].id
    messages = @(@{ role = 'user'; content = 'Hello' })
    max_tokens = 128
    stream = $false
} | ConvertTo-Json -Depth 8
Invoke-RestMethod 'http://127.0.0.1:18200/v1/chat/completions' `
    -Method Post -ContentType 'application/json; charset=utf-8' -Body $body
```

The default listener is local-only with no API key. For external access, the unified launcher accepts `-ListenHost` and `-WorkerArgs @('--api-key', 'your-key')`; clients send a Bearer key. Native TLS is unavailable; configure an HTTPS proxy as needed. See the [Windows guide](scripts/windows/README.md) for llama.cpp full/lightweight UI builds and shortcuts.

<a id="source"></a>

## Source preparation and builds

`master` is the current mainline; `v0.18.0` pins the release snapshot. Engines are submodules with versioned KVMem patches that must be prepared before building. GitHub-generated source archives omit submodule contents; use Git for development:

```sh
git clone --recurse-submodules https://github.com/kvmem/kvmem.git
cd kvmem
python scripts/prepare-backends.py
python scripts/prepare-backends.py --check
```

Preparation requires Python 3.9+ and Git. Select one backend with `--backend llamacpp` or `--backend ninfer`. `--check` verifies pins, patches and the complete prepared tree without changing source files. Prepared submodules normally appear modified because of the integration patches; conflicts are reported without resetting developer changes.

### Windows

Install VS2022 C++ Build Tools, CMake/Ninja and an appropriate CUDA Toolkit. This RTX 50 NInfer example uses CUDA 13.2, `120a` and vcpkg. Adjust for other architectures. Builds default to **8 parallel jobs** with separate directories for the two workers.

```powershell
# Portable core; no model or GPU required
./scripts/windows/build-backends.ps1 -Backend core
ctest --test-dir build-backends/core -C Release --output-on-failure

# llama.cpp CUDA worker
./scripts/windows/build-backends.ps1 -Backend llamacpp -CudaArchitectures 120a

# NInfer worker; set VCPKG_ROOT to your vcpkg checkout
./scripts/windows/build-backends.ps1 -Backend ninfer -CudaArchitectures 120a `
    -CudaPath 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2' `
    -CMakeArgs @("-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake")
```

Source unified launcher examples:

```powershell
./scripts/windows/start-backend.ps1 -Backend ninfer `
    -Worker './build-backends/ninfer/apps/ninfer-serve.exe' `
    -Model 'D:/models/model.ninfer' -Gpu '0'

./scripts/windows/start-backend.ps1 -Backend llamacpp `
    -Worker './build-backends/llamacpp/bin/llama-kvmem-server.exe' `
    -Model 'D:/models/model.gguf' -Gpu '0'
```

The unified launcher defaults to context=262144, B=32768, R=1024, prefill=128 and output limit=1024. NInfer uses INT8 with MTP/ngram/vision disabled; LRFU is enabled with sink=12288 and recent=1024. Enable NInfer vision/MTP explicitly with options such as `-Vision` and `-MtpDrafts`; pass llama.cpp-specific native options through `-WorkerArgs`.

**The source-tree `start-ninfer.ps1` is a fixed-recipe shortcut reading `model.ninfer` from the repository root. It does not expose the packaged shortcut's extra `-Model`/`-NoVision` options.** Use the source unified launcher for custom models and settings; the earlier package quick start applies to extracted release ZIPs.

### Linux / ROCm

```sh
# Portable core only
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release \
  -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF
cmake --build build-core --parallel 8
ctest --test-dir build-core --output-on-failure
```

Prepare the chosen backend before GPU builds. Configure llama.cpp from the repository root and NInfer from `backends/ninfer` using its native dependency manifest. See the [ROCm guide](docs/rocm.md) for HIP builds. Linux Host tests and CUDA compile checks do not establish Linux GPU inference qualification; the current stable binary packages still cover Windows NInfer only. See the [multi-backend repository workflow](docs/multi-backend-repository.md) for detailed builds, export and packaging.

## Qualification and limitations

- All three stable packages passed Release builds, dependency and architecture checks, help commands with a clean PATH, PowerShell 5.1 argument checks and extraction verification. The RTX 50 package passed model loading, health and short Responses requests on an RTX 5060 Ti; RTX 30/40 lack inference qualification on matching hardware.
- Mainline integration passed 17 local core tests and nine CI checks, covering Linux/Windows Host, CUDA compilation, full/lightweight UI, pinned backend source and server units. Successful compilation does not establish runtime qualification for every model, GPU or feature combination.
- Existing RC3 performance measurements are reused. Recipes, concurrency, cold SSD access, memory pressure and models can change performance. Full image, multi-GPU performance and SSD stress matrices were not rerun for the stable promotion.
- KVMem NInfer currently targets Qwen3.5/3.8 Dense. Supported weight formats do not establish support for additional upstream model architectures or suspend/graft/router modes.
- Device faults may require a worker restart; reuse of the original session after a fault is not guaranteed. Model weights, vision, MTP, workspace, checkpoints and system cache consume memory beyond the KV budgets.

Include package version, GPU, driver, model, changed settings and error logs when reporting issues; remove private conversation content from logs.

<a id="documentation"></a>

## Documentation and layout

| Guide | Coverage |
|---|---|
| [0.18.0 release notes](https://github.com/kvmem/kvmem/releases/tag/v0.18.0) | Packages, performance and provenance |
| [Multi-backend repository](docs/multi-backend-repository.md) | Submodules, preparation, builds and packaging |
| [Multi-backend runtime](docs/multi-backend.md) | Startup, capabilities and supported combinations |
| [Backend differences](docs/multi-backend-differences.md) | Defaults, budgets, R, KV and vision |
| [Active KV tiering](docs/active-kv-tiering.md) | Shared RAM/SSD quotas and lifetime |
| [Tiering validation](docs/active-kv-tiering-validation-20261010.md) | Qualification evidence and limits |
| [Memory backend contract](docs/memory-backend-contract.md) | Backend integration contracts |
| [Migration validation](docs/ninfer-all-migration-validation.md) | Pinned NInfer baseline |
| [Documentation-only CI](docs/ci.md) | Build and test skipping for documentation changes |
| [Windows guide](scripts/windows/README.md) | llama.cpp builds and launcher details |
| [ROCm guide](docs/rocm.md) | HIP/ROCm builds |
| [llama.cpp multimodal report](docs/multimodal-implementation-report-2026-09-14.md) | Earlier image implementation |

```text
kvmem/                  Shared memory contracts, retrieval and Host storage
src/adapter/, tools/    llama.cpp adapter, CLI and serving
backends/llamacpp/      Pinned llama.cpp submodule
backends/ninfer/        Pinned NInfer submodule
backends/patches/       Current backend integration patches
backends/versions.json  Pins, patch hashes, prepared tree hashes and licenses
scripts/               Preparation, builds, launchers and packaging
docs/                  Design, usage, releases and dated validation records
patches/               Historical patches
```

Dated reports preserve their original test context and candidate terminology. Use this README, the current release notes and the capability tables to select current entry points and package scope.

## Credits and licenses

Thanks to the maintainers and contributors of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), the original [Neroued/ninfer](https://github.com/Neroued/ninfer) and community engine branches, including the [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) integration, Ryan-gsq's Windows/GSQ-RCO work, and contributors to 3090/4090 adaptations, adaptive MTP, fast prefill and ngram. Full NInfer attribution is in the [release notes](https://github.com/kvmem/kvmem/releases/tag/v0.18.0#%E8%87%B4%E8%B0%A2). LRFU/GQA max references [melis2023/kvmem-actkv](https://github.com/melis2023/kvmem-actkv). Thanks also to Bilibili testers including melis and redsnow23.

Each backend retains its upstream license and third-party notices, identified in the [backend manifest](backends/versions.json). Binary packages include the corresponding licenses; cuBLAS DLLs follow the included CUDA EULA rather than NInfer's Apache-2.0 license. Models are distributed separately under their own terms.

## Paper and citation

[KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU](https://arxiv.org/abs/2609.04852)

```bibtex
@misc{chai2026kvmem,
  title         = {{KVMem}: Virtualizing Million-Token Agent Workspaces on a Consumer {GPU}},
  author        = {Di Chai and Leye Wang and Zeshen Su and Zhiguo Xia and Zhihang Yu},
  year          = {2026},
  eprint        = {2609.04852},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.04852}
}
```
