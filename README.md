# KVMem 0.18.0 · llama.cpp & NInfer

KVMem 在有限显存中管理长对话与 Agent 工具历史：完整原生 KV 保存在 Host RAM，可选下沉到 SSD；GPU 按当前查询保留所需历史和生成工作空间。`master` 已包含 **0.18.0 多后端源码**，Windows RTX 30/40/50 的 **NInfer 0.18.0 正式运行包**已发布。

KVMem manages long conversations and agent tool histories within a bounded GPU working set. Native KV history lives in Host RAM, with optional SSD tiering; retrieval brings selected history onto the GPU. **`master` contains the 0.18.0 multi-backend source**, and **stable NInfer 0.18.0 packages** are available for Windows RTX 30/40/50 GPUs.

**QQ community / QQ 交流群：1040777853**

[下载 / Downloads](#downloads) · [快速开始 / Quick start](#quick-start) · [默认配方 / Defaults](#defaults) · [性能 / Performance](#performance) · [源码 / Source](#source) · [文档 / Docs](#documentation)

<a id="downloads"></a>

## 下载 / Downloads

**运行包和模型使用英文路径，完整路径不能含中文，包括用户名、上级目录和文件名。 / Use English paths for the package and model; the full path must not contain Chinese characters.**

| GPU / 显卡 | Windows x64 NInfer 0.18.0 | Size / 大小 |
|---|---|---:|
| RTX 30 / sm86 | [Download ZIP](https://github.com/kvmem/kvmem-llama.cpp/releases/download/v0.18.0/kvmem-v0.18.0-ninfer-windows-rtx30-sm86.zip) | 971.6 MiB |
| RTX 40 / sm89 | [Download ZIP](https://github.com/kvmem/kvmem-llama.cpp/releases/download/v0.18.0/kvmem-v0.18.0-ninfer-windows-rtx40-sm89.zip) | 964.5 MiB |
| RTX 50 / sm120a | [Download ZIP](https://github.com/kvmem/kvmem-llama.cpp/releases/download/v0.18.0/kvmem-v0.18.0-ninfer-windows-rtx50-sm120a.zip) | 859.0 MiB |

[完整中英文 Release Note / Full release notes](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/v0.18.0) · [ZIP SHA256](https://github.com/kvmem/kvmem-llama.cpp/releases/download/v0.18.0/SHA256SUMS-ninfer-only.txt) · [包内使用说明 / Package guide](https://github.com/kvmem/kvmem-llama.cpp/releases/download/v0.18.0/README.md)

运行包包含 NInfer 服务端、内嵌 Web UI、启动脚本与 CUDA 运行依赖，**不包含模型或 llama.cpp worker**。需要支持 CUDA 13.2 的 NVIDIA 驱动与 Microsoft Visual C++ 2015–2022 x64 运行库；运行预编译包无需安装 CUDA Toolkit 或 Visual Studio。NVFP4 **模型权重**仅适用于 RTX 50 系包。

Packages include the NInfer server, embedded Web UI, launchers and CUDA runtime dependencies. **Models and the llama.cpp worker are supplied separately.** A CUDA 13.2-compatible NVIDIA driver and the Microsoft Visual C++ 2015–2022 x64 runtime are required; running a package does not require the CUDA Toolkit or Visual Studio. NVFP4 **model weights** require the RTX 50 package.

<details>
<summary>历史运行包 / Earlier packages</summary>

| Platform / 平台 | Backend / 后端 | Model / 模型 | Release / 版本 |
|---|---|---|---|
| Windows / Linux / WSL2 NVIDIA | llama.cpp | GGUF | [CUDA v0.16.0-rc3](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/v0.16.0-rc3) |
| Windows / Linux / WSL2 AMD | llama.cpp | GGUF | [ROCm Beta2](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/rc3-rocm-beta2) |
| Windows Bonsai 2 | Dedicated / 专用 | PTQ1 GGUF | [prism.3](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/v0.16.0-rc3-prism.3) |

这些包保留各自的功能和参数范围，不会因主线更新而获得 0.18.0 功能。Linux、ROCm 和 llama.cpp 的源码构建见下文；本次 0.18.0 正式运行包仅发布 Windows NInfer。

These packages retain their own feature and parameter scope. The 0.18.0 binary release covers Windows NInfer; source builds for the other routes are described below.

</details>

## 0.18.0 功能 / What's in 0.18.0

- **多后端源码 / Multi-backend source**：共享 KVMem 内存、检索和存储契约；llama.cpp 与 NInfer 分别保留原生推理和服务实现。上游版本、集成补丁与完整来源由 [backend manifest](backends/versions.json) 固定。 / Shared memory, retrieval and storage contracts with separate native engine workers and pinned integration patches.
- **长历史与状态复用 / History and state reuse**：按查询选择 KV，保持逻辑位置；多轮请求、工具续接、编辑和取消按后端的精确前缀与 checkpoint 规则恢复。 / Query-based KV selection with original logical positions and backend-specific prefix/checkpoint validation.
- **RAM/SSD 分层 / RAM/SSD tiering**：活动和闲置历史共用配额；完整原生 K/V、量化附属数据与 MTP KV 可下沉，按实际选中的内容读取。 / Active and idle histories share quotas; selected native KV is restored without requantization.
- **LRFU 与 GQA max / LRFU and GQA max**：Windows 启动入口默认开启历史加权，`lambda=0.5`、`top-frac=0.9`、GQA `max`；sink=5120，recent=64。 / Windows launchers enable history weighting and max aggregation with these defaults.
- **工具与客户端 / Tools and clients**：NInfer 默认接入工具选择与 strict schema 约束，支持 Chat Completions、Responses 和 Anthropic Messages；仅发布完整工具调用，不支持的 schema 明确报错。Responses reasoning 返回 `summary=[]` 并保留原始思考文本；图片 detail 接受 low/high/original 并映射到 auto。 / Native tool constraints compose with thinking, JSON output and speculative decoding; client compatibility includes Responses reasoning and image detail handling.
- **NInfer 执行选项 / NInfer execution options**：九种 KV 格式、固定或自适应 MTP、ngram、文本及 CPU/resident 图片并发，以及最多八张同计算能力 GPU 的按层分配。具体组合受 [能力与验收范围](docs/multi-backend.md)约束。 / Nine KV formats, MTP, ngram, concurrent text/image requests and a layer pipeline across compatible GPUs; consult the supported combinations before changing the recipe.

正式版基于 RC3 引擎，更新版本、启动默认值、说明及校验文件。recent 默认值为 **64 tokens**，这是 NInfer 支持的最小值，0–63 会被拒绝。引擎二进制与 RC3 相同；版本来源见包内 `BUILD-INFO.json`，发布源码固定在 `v0.18.0`。`master` 在此基础上保留主线 Responses 修复和构建修复，并继续接收后续更新。

The stable release retains the RC3 engine binaries and updates version metadata, launch defaults, documentation and checksums. **Recent defaults to 64 tokens**, the NInfer minimum; values 0–63 are rejected. `BUILD-INFO.json` records binary provenance, and `v0.18.0` pins the release source. `master` also retains mainline Responses and build fixes and continues to receive updates.

<a id="quick-start"></a>

## Windows 正式包快速开始 / Windows package quick start

1. 下载匹配显卡的 ZIP，完整解压到英文路径。 / Extract the matching ZIP completely into an English path.
2. 准备 **NInfer v3 `.ninfer`** 模型。可放在包根目录命名为 `model.ninfer`，也可用 `-Model` 指定。 / Supply a NInfer v3 model as `model.ninfer` in the package root or select it with `-Model`.
3. 在解压目录打开 PowerShell，列出显卡并启动。可用显卡编号或完整 UUID；多个显卡建议使用 UUID。 / Open PowerShell in the package directory, list GPUs and start the service. GPU indices or full UUIDs are accepted.

```powershell
nvidia-smi -L

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-ninfer.ps1 `
    -Model 'D:\models\model.ninfer' -Gpu '0'
```

Web UI：`http://127.0.0.1:18200/`；OpenAI API base：`http://127.0.0.1:18200/v1`。保持终端开启，Ctrl+C 停止。添加 `-DryRun` 可预览完整参数。

Open the Web UI at `http://127.0.0.1:18200/`; the OpenAI-compatible API base is `http://127.0.0.1:18200/v1`. Keep the terminal open; Ctrl+C stops the service. Add `-DryRun` to inspect the resolved arguments.

默认启用 CPU 视觉和 MTP，因此模型须包含对应权重及预处理资源。缺少视觉权重时追加 `-NoVision`；缺少 MTP 权重时追加 `-MtpDrafts 0`，包内快捷入口会同时关闭自适应 MTP 与 ngram。NInfer 模型无需单独的 mmproj。只有 GGUF 时参考固定版本的 [NInfer 转换指南](https://github.com/iamwavecut/ninfer-all/blob/8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3/docs/weight-conversion.md)。llama.cpp 则使用 GGUF，图像另需匹配的 mmproj。

The default recipe needs a model containing vision resources and MTP weights. Add `-NoVision` if vision weights are absent; add `-MtpDrafts 0` if MTP weights are absent, which also disables adaptive MTP and ngram in the packaged shortcut. NInfer embeds its vision resources; llama.cpp uses GGUF plus a matching external mmproj. See the pinned conversion guide above for GGUF-to-NInfer preparation.

<a id="defaults"></a>

## 默认配方与预算 / Default recipe and budgets

下表对应 **0.18.0 运行包内 `start-ninfer.ps1`**，1K=1024 tokens。统一 `start-backend.ps1` 和直接运行 worker 使用不同默认值。

These are the defaults of **`start-ninfer.ps1` in the 0.18.0 package**, with 1K=1024 tokens. The unified launcher and direct worker have different defaults.

| Setting / 参数 | Default / 默认 |
|---|---|
| Context | 204800 / 200K |
| GPU history budget B / generation reserve R | 36864 / 16384 |
| Default output limit / 默认输出上限 | 16384 |
| Main / MTP KV | INT8 / INT8 |
| MTP / ngram | Adaptive MTP4 / ngram31, min-match=12 |
| Vision / 视觉 | CPU, 1024 tokens/image |
| Thinking budget / 思考预算 | 8192 |
| Host KV payload | 12288 MiB / 12 GiB |
| Prefill chunk | 256 tokens |
| Sink / recent | 5120 / **64 tokens** |
| Retained idle sessions / concurrency | 1 / 1 |
| LRFU / GQA | Enabled, lambda=0.5, top-frac=0.9 / max |
| SSD / device profile | Disabled / off |
| Listen / 监听 | 127.0.0.1:18200 |

这套配方已在 RTX 5060 Ti 16GB、Qwen3.8-27B GSQ-RCO IQ3_S 图文/MTP 模型上完成加载和文本请求验证；其他模型、显卡或显存容量需自行核对。Context 是逻辑上限，不能保证任意 RAM/VRAM 条件下都可达到。

This recipe was checked for model loading and text requests on an RTX 5060 Ti 16GB with Qwen3.8-27B GSQ-RCO IQ3_S vision/MTP weights. Other model/GPU combinations need their own capacity checks. Context sets a logical limit, not a memory capacity guarantee.

| Budget / 预算 | Meaning / 含义 |
|---|---|
| B | 每条活动请求选中的 GPU 历史 tokens；sink、recent 和必选图片区域计入 B。 / Selected GPU history per active request, including mandatory regions. |
| R | GPU 追加工作空间；NInfer 可在空间不足时归档、重选并继续生成，输出不必限制在 R 内。llama.cpp 当前服务路径仍以 R 限制单次输出，包含思考。 / Append workspace; NInfer can refresh its selection beyond R, while llama.cpp still caps per-turn output at R. |
| H / HostMiB | 本进程受管理原生 KV 的 RAM 配额；Main/MTP 都计入。不是进程总内存上限。 / Managed native KV RAM quota, including Main/MTP; not a process RSS limit. |
| D / DiskMiB | 活动 KV 和闲置状态附件共用的 SSD 配额。 / Shared SSD quota for active KV and idle state attachments. |

NInfer B/R 按 64-token 页对齐，prefill 为 128 的倍数且不超过 R；B+R 不超过 context。recent 至少 64，按整页向下取整。llama.cpp 默认检索 block 为 128，recent 按完整 block 计算，因此脚本传入 64 不等于保留一个 llama.cpp block。两后端参数名相同也不保证行为相同，详见 [实现差异](docs/multi-backend-differences.md)。

For NInfer, B/R align to 64 tokens, prefill is a multiple of 128 and no larger than R, and B+R fits context. Recent is at least 64 and rounds down to whole pages. llama.cpp's default retrieval block is 128, so a launcher value of 64 does not retain one llama.cpp block. See the backend differences before transferring configurations.

NInfer KVMem KV 格式 / formats：`bf16`、`int8`、`fp8`、`nvfp4`、`k8v4`、`rk8v4`、`rk4v4`、`rk4v4-e8`、`rk2v4-e8`；Main/MTP 使用相同格式。llama.cpp 的 KV 参数为 `f16`、`f32`、`q8_0`、`q5_0`、`q4_0`，Main/MTP 可分别设置。模型权重量化与 KV 格式是独立配置，NInfer `int8` 与 llama.cpp `q8_0` 也不能直接互换。

NInfer Main/MTP use the same KV format; llama.cpp can configure Main/MTP separately. Weight quantization and KV format are independent settings. NInfer int8 and llama.cpp q8_0 are different native layouts, not interchangeable KV representations.

### 检索与 SSD / Retrieval and SSD

LRFU 用当前 query 的相关性更新历史分数，每个请求提交一次历史更新。`top-frac` 控制允许累加本轮分数的命中范围，**不是 KV 保留比例**；GQA max 对同一 KV head 对应的 query heads 取最大相关性。sink、recent、媒体保留仍占用 B。

LRFU updates historical scores from the current query once per request. `top-frac` controls which hits may accumulate scores; **it is not a fraction of KV retained**. GQA max takes the maximum relevance across query heads sharing a KV head. Mandatory history still consumes B.

以下示例在**正式包解压目录**运行： / Run these examples in the **extracted release package**:

```powershell
# Plain retrieval / 关闭历史加权，使用 mean 聚合
& .\scripts\windows\start-ninfer.ps1 -Gpu '0' -NoLrfu -ScoreAggregation mean

# Optional SSD / 可选 SSD 分层，目录须使用英文路径
& .\scripts\windows\start-ninfer.ps1 -Gpu '0' `
    -DiskPath 'D:\kvmem-cache' -DiskMiB 4096
```

SSD 默认关闭。只指定 `-DiskPath` 时 D 默认 4096 MiB。H/D 不包含全部 mean-K、checkpoint、暂存和系统文件缓存；RAM 仍必须容纳活动工作集与换页余量。文件只供**当前服务进程**复用，不支持重启恢复或跨后端 KV 迁移。公共 SSD 参数不能与 llama.cpp 旧在线 NVMe/旧会话磁盘参数混用。详见 [RAM/SSD 分层](docs/active-kv-tiering.md)及其 [验收记录](docs/active-kv-tiering-validation-20261010.md)。

SSD is disabled by default; specifying only `-DiskPath` uses 4096 MiB. H/D exclude additional statistics, checkpoints, staging and OS cache. RAM must still fit active work and swap headroom. Files are reusable only **within the current process**; restart persistence and cross-backend KV migration are unsupported. Do not combine shared SSD flags with legacy llama.cpp NVMe/session-disk flags.

### 多卡与并发 / Multiple GPUs and concurrency

```powershell
# Package launcher / 包内启动器；自动分配层数
& .\scripts\windows\start-ninfer.ps1 -Gpu '0,1'

# Example for a 64-layer model / 64 层模型的手动分配示例
& .\scripts\windows\start-ninfer.ps1 -Gpu '0,1' -StageLayers 40,24
```

NInfer 使用按层 pipeline，最多八张 GPU，要求相同 CUDA 计算能力；显存容量可不同，使用匹配架构的包。每卡层数为正数，合计等于模型层数；GPU 列表顺序即 stage 顺序，第一张卡承担 embedding/head/MTP 和 resident 视觉，默认 CPU 视觉在 Host 执行。也可用逗号连接 UUID。

NInfer supports a layer pipeline across up to eight GPUs of the same compute capability. VRAM sizes may differ. Layer counts must be positive and sum to the model's layer count. Device order defines stage order; the first GPU holds embedding/head/MTP and resident vision. CPU vision stays on the Host. Comma-separated UUIDs are also accepted.

文本和 resident/CPU 图片最多八条活动请求，B/R 为每请求预算，H/D 为每进程预算；增加并发会增加资源需求。多路 ngram 宽度大于 15 时自动降至 15；CPU 图片编码仍串行。快速 prefill 只开放 INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8。NInfer KVMem 不支持视频、overlay 视觉、DFlash、并行 prefill 或旧 `--lookup-ngram`。参数范围、模型边界和已验收组合见 [多后端运行](docs/multi-backend.md)。

Text and resident/CPU image requests support up to eight active lanes. B/R are per request and H/D are per process. Concurrent ngram widths above 15 are clamped to 15; CPU image encoding remains serial. Fast prefill is admitted only for INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8. NInfer KVMem rejects video, overlay vision, DFlash, parallel prefill and legacy lookup-ngram. Check the capability guide for architecture and combination limits.

<a id="performance"></a>

## RC3 性能实测 / Completed RC3 performance results

直接使用 2026-10-13 已完成的 **RC3** 结果：Windows、**RTX 5060 Ti 16GB**、**Qwen3.8-27B GSQ-RCO IQ3_S 图文/MTP 模型**。正式版保留相同引擎二进制；下表为该测试配置的观测值，不代表默认配方、所有显卡或模型的性能。

These are the completed **RC3** measurements from 2026-10-13 on **Windows, RTX 5060 Ti 16GB, Qwen3.8-27B GSQ-RCO IQ3_S vision/MTP**. Stable 0.18.0 retains the same engine binaries. Results describe this benchmark configuration, not every model/GPU or the default recipe.

| Measurement / 项目 | Result / 结果 | Scope / 范围 |
|---|---:|---|
| Daily text decode / 日常文本生成 | 50.08 token/s | 3 measured rounds / 3 轮聚合 |
| Daily image decode / 日常图片生成 | 55.90 token/s | 3 measured rounds / 3 轮聚合 |
| Daily HTML/SVG decode / 日常代码生成 | 64.54 token/s | 3 rounds, 512 output tokens each |
| Complete daily task / 日常完整任务 | 30.10 s | Median total time / 总耗时中位数 |
| Long-history tool rounds / 长历史工具轮次生成 | 53.24 token/s | 32 rounds including final code |
| Final long-history code / 长历史最终代码生成 | 44.87 token/s | Final server-native decode rate |
| Final time to first token / 末轮首字等待 | 16.87 s | First reasoning or content fragment |
| Long-history request total / 长历史请求总耗时 | 855.10 s | Sum of 33 requests, excludes loading |
| Final input / output | 261522 / 512 tokens | Near 256K / 接近 256K |
| Requests completed / tool-prefix hits | 33/33 / 32/32 | One complete long-history run |

测试设置 / Settings：context=262144，B/R=36864/16384，Main/MTP INT8，GDN FP16；固定 / fixed MTP3，adaptive off，ngram=0；prefill=512，thinking=128，output limit=512；GPU resident vision=1024 tokens/image；LRFU(.5/.9)，GQA max；Host=12288 MiB，SSD off，sessions/concurrency=1/1，device profile=off；temperature=1，top_p=.95，top_k=20，min_p=0，seed=42；streaming Chat Completions。

日常测试预热一轮后测三轮，覆盖约 12K 文本、896×896 图片和 HTML/SVG 生成；长历史注入模拟 `read_file` 结果。聚合速度按总输出 tokens / 总服务端生成耗时计算，包含思考；不计工具执行时间，也未据此验证代码完整性或通用召回率。整卡显存峰值约 **14.12 GiB**；长历史阶段系统可用 RAM 最低 **8.78 MiB**，明显内存压力可能影响首字等待和总耗时。

The daily test uses one warm-up and three measured rounds with roughly 12K text, an 896×896 image and HTML/SVG generation. The long-history run injects simulated read_file results. Aggregate decode rates include reasoning and divide total output tokens by total server generation time. Tool execution is excluded; code completeness and general retrieval quality were not established. Whole-GPU VRAM peaked near **14.12 GiB**. Available system RAM fell to **8.78 MiB**, so memory pressure may affect latency and total time.

[完整报告、原始配置、计时与服务日志 / Full report, configuration, timings and logs](https://github.com/kvmem/kvmem-llama.cpp/releases/download/v0.18.0/RC3-performance-results-20261013.zip)。历史 llama.cpp 性能结果见 [16 GiB 报告](docs/recommended-config-performance.md)和 [旧版 256K 工具测试](docs/long-context-benchmark-2026-09-14.md)，不与本表混为同一配置。

## API 与 Web UI / API and Web UI

| Endpoint | Purpose / 用途 |
|---|---|
| `GET /health` | Readiness / 就绪状态 |
| `GET /v1/models` | Loaded model IDs / 已加载模型标识 |
| `GET /props` | Backend, budgets and capabilities / 后端、预算与能力 |
| `POST /v1/chat/completions` | OpenAI-compatible chat, streaming, tools and supported images |
| `POST /v1/responses` | OpenAI Responses, including SSE streaming |

NInfer 另提供原生 Anthropic Messages 接口；具体协议、工具约束和 schema 组合以 worker 及随包说明为准。llama.cpp 支持 `/responses` 别名，NInfer 不保证相同别名。图片与视频能力按后端区分：NInfer 支持图片、不支持视频；llama.cpp 图片需 mmproj，源码视频还需 ffmpeg/ffprobe 及相应构建选项。

NInfer also serves native Anthropic Messages. Protocol and schema support follow the worker/package documentation. llama.cpp accepts the `/responses` alias; do not assume the same alias for NInfer. NInfer supports images but rejects video; llama.cpp needs mmproj for images and additional build options plus ffmpeg/ffprobe for source video support.

以下非流式示例先查询真实 model ID，避免把占位名称作为模型： / This non-streaming example queries the loaded model ID first:

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

默认仅监听本机，不设置 API key。对外服务可用统一入口的 `-ListenHost` 和 `-WorkerArgs @('--api-key', 'your-key')`；客户端使用 Bearer key。原生服务不提供 TLS，远程访问应按部署需要配置 HTTPS 代理。llama.cpp 的 full/lightweight UI 构建与快捷脚本见 [Windows 使用说明](scripts/windows/README.md)。

The default listener is local-only with no API key. For external access, the unified launcher accepts `-ListenHost` and `-WorkerArgs @('--api-key', 'your-key')`; clients send a Bearer key. Native TLS is unavailable; configure an HTTPS proxy as needed. See the Windows guide for llama.cpp full/lightweight UI builds and shortcuts.

<a id="source"></a>

## 源码准备与构建 / Source preparation and builds

`master` 是当前主线；`v0.18.0` 固定发布快照。两个引擎是 Git submodule，KVMem 改动位于 `backends/patches/`，构建前须应用补丁。GitHub 自动生成的 Source ZIP 不含完整子模块，开发请使用 Git：

`master` is the current mainline; `v0.18.0` pins the release snapshot. Engines are submodules with versioned KVMem patches that must be prepared before building. GitHub-generated source archives omit submodule contents; use Git for development:

```sh
git clone --recurse-submodules https://github.com/kvmem/kvmem-llama.cpp.git
cd kvmem-llama.cpp
python scripts/prepare-backends.py
python scripts/prepare-backends.py --check
```

Python 3.9+ 和 Git 用于准备源码；`--backend llamacpp` / `--backend ninfer` 可只准备一个后端。`--check` 只检查 pin、patch 与完整树，不修改源码。准备后的子模块显示 modified 属于应用集成补丁的正常状态；脚本保留开发者修改，冲突会报错。

Preparation requires Python 3.9+ and Git. Select one backend with `--backend llamacpp` or `--backend ninfer`. `--check` verifies the prepared tree without changing it. Prepared submodules normally appear modified because of the integration patches; conflicts are reported without resetting developer changes.

### Windows

需要 VS2022 C++ Build Tools、CMake/Ninja，以及选定后端适用的 CUDA Toolkit。以下 RTX 50 NInfer 路径使用 CUDA 13.2、`120a` 和 vcpkg；其他架构需选择匹配配置。默认使用 **8 个并行编译任务**，两个 worker 使用独立构建目录。

Install VS2022 C++ Build Tools, CMake/Ninja and an appropriate CUDA Toolkit. This RTX 50 NInfer example uses CUDA 13.2, `120a` and vcpkg. Adjust for other architectures. Builds default to **8 parallel jobs** with separate directories for the two workers.

```powershell
# Portable core; no model or GPU required / 公共核心，无需模型或 GPU
./scripts/windows/build-backends.ps1 -Backend core
ctest --test-dir build-backends/core -C Release --output-on-failure

# llama.cpp CUDA worker
./scripts/windows/build-backends.ps1 -Backend llamacpp -CudaArchitectures 120a

# NInfer worker; set VCPKG_ROOT to your vcpkg checkout / 指定本机 vcpkg
./scripts/windows/build-backends.ps1 -Backend ninfer -CudaArchitectures 120a `
    -CudaPath 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2' `
    -CMakeArgs @("-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake")
```

源码统一入口示例 / Source unified launcher examples:

```powershell
./scripts/windows/start-backend.ps1 -Backend ninfer `
    -Worker './build-backends/ninfer/apps/ninfer-serve.exe' `
    -Model 'D:/models/model.ninfer' -Gpu '0'

./scripts/windows/start-backend.ps1 -Backend llamacpp `
    -Worker './build-backends/llamacpp/bin/llama-kvmem-server.exe' `
    -Model 'D:/models/model.gguf' -Gpu '0'
```

统一入口默认 context=262144、B=32768、R=1024、prefill=128、输出上限1024；NInfer 默认 INT8、MTP/ngram/视觉关闭，LRFU 开启、sink=5120、recent=64。需要视觉或 MTP 时显式传 `-Vision`、`-MtpDrafts` 等。llama.cpp 对应能力使用 `-WorkerArgs` 传原生参数。

The unified launcher defaults to context=262144, B=32768, R=1024, prefill=128 and output limit=1024. NInfer uses INT8 with MTP/ngram/vision disabled; LRFU is enabled with sink=5120 and recent=64. Enable NInfer vision/MTP explicitly; pass llama.cpp-specific native options through `-WorkerArgs`.

**源码树的 `start-ninfer.ps1` 是固定配方快捷脚本，默认从仓库根目录读取 `model.ninfer`，不接受包内扩展的 `-Model`、`-NoVision` 等选项。**需要自定义模型或完整参数时使用源码 `start-backend.ps1`；前面的包内快速开始仅用于解压后的正式运行包。

**The source-tree `start-ninfer.ps1` is a fixed-recipe shortcut reading `model.ninfer` from the repository root. It does not expose the packaged shortcut's extra `-Model`/`-NoVision` options.** Use the source unified launcher for custom models and settings; the earlier package quick start applies to extracted release ZIPs.

### Linux / ROCm

```sh
# Portable core only / 仅公共核心
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release \
  -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF
cmake --build build-core --parallel 8
ctest --test-dir build-core --output-on-failure
```

GPU 构建前准备选定后端，llama.cpp 从仓库根目录配置 CMake，NInfer 从 `backends/ninfer` 配置并遵循其依赖清单。HIP/ROCm 使用已有 [ROCm 指南](docs/rocm.md)。Linux Host 测试或 CUDA 编译通过不等于 Linux GPU 实机推理验收；当前正式运行包范围仍为 Windows NInfer。详细构建、导出和打包流程见 [多后端仓库工作流](docs/multi-backend-repository.md)。

Prepare the chosen backend before GPU builds. Configure llama.cpp from the repository root and NInfer from `backends/ninfer` using its native dependency manifest. See the ROCm guide for HIP builds. Linux Host tests and CUDA compile checks do not establish Linux GPU inference qualification. Consult the repository workflow for export and packaging.

## 验证范围与限制 / Qualification and limitations

- 三个正式包完成 Release 编译、依赖与架构检查、干净 PATH 下 help、PowerShell 5.1 参数检查，以及解压校验。RTX 50 包已在 RTX 5060 Ti 上完成模型加载、健康与短 Responses 请求；RTX 30/40 包尚无对应硬件的实际推理验收。 / All packages passed build, dependency, help, launcher and extraction checks. RTX 50 has RTX 5060 Ti runtime checks; RTX 30/40 lack matching hardware inference qualification.
- 主线集成通过 17 项本地核心测试及 9 项 CI 检查，覆盖 Linux/Windows Host、CUDA 编译、完整/轻量 UI、固定后端源码和服务单元。编译通过不代表所有模型、GPU 或功能组合都已运行验证。 / Mainline integration passed 17 local core tests and nine CI checks; compile coverage is distinct from runtime qualification.
- RC3 性能结果沿用已有记录；不同配方、并发、冷 SSD、内存压力及模型可能改变性能。此次正式化没有重跑完整图像矩阵、多卡性能或 SSD 压力测试。 / RC3 measurements are reused; full image, multi-GPU performance and SSD stress matrices were not rerun for the stable promotion.
- KVMem NInfer 当前适配 Qwen3.5/3.8 Dense 边界。其他上游模型架构或 suspend/graft/router 功能不能据权重格式推断已支持。 / Supported weight formats do not imply support for additional architectures or upstream suspend/graft/router modes.
- 设备故障可能需要重启 worker；不承诺故障后的原会话可继续复用。模型权重、视觉、MTP、workspace、checkpoint 和系统缓存仍需额外内存。 / Device faults may require a worker restart; model, vision, MTP, workspace, checkpoints and system cache consume memory beyond the KV budgets.

反馈请附包版本、GPU、驱动、模型、修改后的参数与错误日志，移除日志中的私有对话。 / Include package version, GPU, driver, model, changed settings and errors when reporting issues; remove private conversation content from logs.

<a id="documentation"></a>

## 文档与仓库结构 / Documentation and layout

| Guide / 指南 | Coverage / 内容 |
|---|---|
| [0.18.0 release notes](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/v0.18.0) | 正式包、性能、来源 / Packages, performance and provenance |
| [Multi-backend repository](docs/multi-backend-repository.md) | 子模块、准备、构建、打包 / Source and build workflow |
| [Multi-backend runtime](docs/multi-backend.md) | 启动、能力与组合 / Startup and supported combinations |
| [Backend differences](docs/multi-backend-differences.md) | 默认值、预算、R、KV、视觉 / Parameter and behavior differences |
| [Active KV tiering](docs/active-kv-tiering.md) | 公共 RAM/SSD 配额与生命周期 / Shared storage and lifetime |
| [Tiering validation](docs/active-kv-tiering-validation-20261010.md) | 已测范围与限制 / Qualification evidence |
| [Memory backend contract](docs/memory-backend-contract.md) | 新后端接入 / Backend integration contracts |
| [Migration validation](docs/ninfer-all-migration-validation.md) | 固定 NInfer 基线 / Pinned engine migration |
| [Windows guide](scripts/windows/README.md) | llama.cpp 构建与快捷入口 / llama.cpp build and launcher details |
| [ROCm guide](docs/rocm.md) | HIP/ROCm 路径 / AMD builds |
| [llama.cpp multimodal report](docs/multimodal-implementation-report-2026-09-14.md) | 历史图像实现 / Earlier image implementation |

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

历史报告中的日期、参数和候选状态描述保留当时测试上下文；当前使用入口与正式包范围以本 README、最新发布说明和具体能力表为准。

Dated reports preserve their original test context and candidate terminology. Use this README, the current release notes and the capability tables to select current entry points and package scope.

## 致谢与许可证 / Credits and licenses

感谢 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)、原始 [Neroued/ninfer](https://github.com/Neroued/ninfer) 及社区分支的维护者与贡献者，包括 [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) 的整合、Ryan-gsq 的 Windows/GSQ-RCO 工作，以及 3090/4090、自适应 MTP、快速 prefill 和 ngram 的贡献者。完整 NInfer 署名见 [发布说明](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/v0.18.0#致谢)。LRFU/GQA max 参考 [melis2023/kvmem-actkv](https://github.com/melis2023/kvmem-actkv)。感谢 Bilibili 的 melis、redsnow23 等测试者。

Thanks to the upstream llama.cpp and NInfer teams, the ninfer-all integrators and community engine branches, Ryan-gsq's Windows/GSQ-RCO work, and contributors to GPU adaptations, adaptive MTP, fast prefill and ngram. Full NInfer attribution is in the release notes. LRFU/GQA max references kvmem-actkv; thanks also to community testers including melis and redsnow23.

各后端保留原许可证和第三方声明，路径见 [backend manifest](backends/versions.json)。预编译包附带相应许可证；cuBLAS DLL 遵循包内 CUDA EULA，不适用 NInfer 的 Apache-2.0 许可证。模型另行分发并遵循各自条款。

Each backend retains its upstream license and third-party notices, identified in the backend manifest. Binary packages include notices; cuBLAS DLLs follow the included CUDA EULA rather than NInfer's Apache-2.0 license. Models are distributed separately under their own terms.

## 论文与引用 / Paper and citation

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
