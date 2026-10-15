[English](README.md) | **简体中文**

# KVMem 0.18.0 · llama.cpp & NInfer

KVMem 在有限显存中管理长对话与 Agent 工具历史：完整原生 KV 保存在 Host RAM，可选下沉到 SSD；GPU 按当前查询保留所需历史和生成工作空间。`master` 已包含 **0.18.0 多后端源码**，Windows RTX 30/40/50 的 **NInfer 0.18.0 正式运行包**已发布。

**RTX 20 / TITAN RTX 与 Tesla V100：**新的 **[NInfer v0.18.1 RC1](https://github.com/kvmem/kvmem/releases/tag/v0.18.1-ninfer-rc1)** 实机试用包已经发布，并已内置 Web UI。[下载 RTX 20 / sm75](https://github.com/kvmem/kvmem/releases/download/v0.18.1-ninfer-rc1/kvmem-v0.18.1-ninfer-rc1-windows-rtx20-sm75.zip) · [下载 V100 / sm70](https://github.com/kvmem/kvmem/releases/download/v0.18.1-ninfer-rc1/kvmem-v0.18.1-ninfer-rc1-windows-v100-sm70.zip)

仓库现名为 **[kvmem/kvmem](https://github.com/kvmem/kvmem)**。旧地址 [kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp) 由 GitHub 自动跳转；旧发布与下载链接继续可用。请勿重新创建旧名称仓库，以免覆盖跳转。

**QQ 交流群：1040777853**

[下载](#downloads) · [快速开始](#quick-start) · [默认配方](#defaults) · [性能](#performance) · [源码](#source) · [文档](#documentation)

<a id="downloads"></a>

## 下载

**运行包和模型使用英文路径，完整路径不能含中文，包括用户名、上级目录和文件名。**

### RTX 20 / TITAN RTX 与 Tesla V100 · v0.18.1 RC1

| 显卡 | Windows x64 NInfer v0.18.1 RC1 | 构建 | 大小 |
|---|---|---|---:|
| RTX 20 / TITAN RTX | [下载 ZIP](https://github.com/kvmem/kvmem/releases/download/v0.18.1-ninfer-rc1/kvmem-v0.18.1-ninfer-rc1-windows-rtx20-sm75.zip) | sm75 / CUDA 13.2 | 1161.3 MiB |
| Tesla V100 | [下载 ZIP](https://github.com/kvmem/kvmem/releases/download/v0.18.1-ninfer-rc1/kvmem-v0.18.1-ninfer-rc1-windows-v100-sm70.zip) | sm70 / CUDA 12.9 | 939.4 MiB |

[RC1 完整发布说明](https://github.com/kvmem/kvmem/releases/tag/v0.18.1-ninfer-rc1) · [ZIP SHA256](https://github.com/kvmem/kvmem/releases/download/v0.18.1-ninfer-rc1/SHA256SUMS-ninfer-old-gpu.txt) · [运行包验证记录](https://github.com/kvmem/kvmem/releases/download/v0.18.1-ninfer-rc1/PACKAGE-VALIDATION.json)

两个包都内置 Web UI，当前用于老卡实机试用。构建目标、依赖和压缩包均已检查，但 RTX 20、TITAN RTX 与 V100 的实卡资格验证仍待完成。

### RTX 30 / 40 / 50 · v0.18.0 正式版

| 显卡 | Windows x64 NInfer 0.18.0 | 大小 |
|---|---|---:|
| RTX 30 / sm86 | [下载 ZIP](https://github.com/kvmem/kvmem/releases/download/v0.18.0/kvmem-v0.18.0-ninfer-windows-rtx30-sm86.zip) | 971.6 MiB |
| RTX 40 / sm89 | [下载 ZIP](https://github.com/kvmem/kvmem/releases/download/v0.18.0/kvmem-v0.18.0-ninfer-windows-rtx40-sm89.zip) | 964.5 MiB |
| RTX 50 / sm120a | [下载 ZIP](https://github.com/kvmem/kvmem/releases/download/v0.18.0/kvmem-v0.18.0-ninfer-windows-rtx50-sm120a.zip) | 859.0 MiB |

[完整发布说明](https://github.com/kvmem/kvmem/releases/tag/v0.18.0) · [ZIP SHA256](https://github.com/kvmem/kvmem/releases/download/v0.18.0/SHA256SUMS-ninfer-only.txt) · [包内使用说明](https://github.com/kvmem/kvmem/releases/download/v0.18.0/README.md)

运行包包含 NInfer 服务端、内嵌 Web UI、启动脚本与 CUDA 运行依赖，**不包含模型或 llama.cpp worker**。需要支持 CUDA 13.2 的 NVIDIA 驱动与 Microsoft Visual C++ 2015–2022 x64 运行库；运行预编译包无需安装 CUDA Toolkit 或 Visual Studio。NVFP4 **模型权重**仅适用于 RTX 50 系包。

<details>
<summary>历史运行包</summary>

| 平台 | 后端 | 模型 | 版本 |
|---|---|---|---|
| Windows / Linux / WSL2 NVIDIA | llama.cpp | GGUF | [CUDA v0.16.0-rc3](https://github.com/kvmem/kvmem/releases/tag/v0.16.0-rc3) |
| Windows / Linux / WSL2 AMD | llama.cpp | GGUF | [ROCm Beta2](https://github.com/kvmem/kvmem/releases/tag/rc3-rocm-beta2) |
| Windows Bonsai 2 | 专用 | PTQ1 GGUF | [prism.3](https://github.com/kvmem/kvmem/releases/tag/v0.16.0-rc3-prism.3) |

这些包保留各自的功能和参数范围，不会因主线更新而获得 0.18.0 功能。Linux、ROCm 和 llama.cpp 的源码构建见下文；本次 0.18.0 正式运行包仅发布 Windows NInfer。

</details>

## 0.18.0 功能

- **多后端源码**：共享 KVMem 内存、检索和存储契约；llama.cpp 与 NInfer 分别保留原生推理和服务实现。上游版本、集成补丁与完整来源由 [后端清单](backends/versions.json) 固定。
- **长历史与状态复用**：按查询选择 KV，保持逻辑位置；多轮请求、工具续接、编辑和取消按后端的精确前缀与 checkpoint 规则恢复。
- **RAM/SSD 分层**：活动和闲置历史共用配额；完整原生 K/V、量化附属数据与 MTP KV 可下沉，按实际选中的内容读取，恢复时不重新量化。
- **LRFU 与 GQA max**：Windows 启动入口默认开启历史加权，`lambda=0.5`、`top-frac=0.9`、GQA `max`；sink=12288，recent=1024。
- **工具与客户端**：NInfer 默认接入工具选择与 strict schema 约束，支持 Chat Completions、Responses 和 Anthropic Messages；仅发布完整工具调用，不支持的 schema 明确报错。Responses reasoning 返回 `summary=[]` 并保留原始思考文本；图片 detail 接受 low/high/original 并映射到 auto。原生工具约束可与思考、JSON 输出和推测解码组合。
- **NInfer 执行选项**：九种 KV 格式、固定或自适应 MTP、ngram、文本及 CPU/resident 图片并发，以及最多八张同计算能力 GPU 的按层分配。具体组合受 [能力与验收范围](docs/multi-backend.md)约束。

正式版基于 RC3 引擎，更新版本、启动默认值、说明及校验文件。默认 sink 为 **12288 tokens（12K）**、recent 为 **1024 tokens（1K）**；NInfer 的 recent 最小值仍为 64，0–63 会被拒绝。引擎二进制与 RC3 相同；版本来源见包内 `BUILD-INFO.json`。`v0.18.0` 保留首次发布源码快照，启动配方更新随 `master` 和运行包发布；`master` 也保留主线 Responses 修复和构建修复。

<a id="quick-start"></a>

## Windows 正式包快速开始

1. 下载匹配显卡的 ZIP，完整解压到英文路径。
2. 准备 **NInfer v3 `.ninfer`** 模型。可放在包根目录命名为 `model.ninfer`，也可用 `-Model` 指定。
3. 在解压目录打开 PowerShell，列出显卡并启动。可用显卡编号或完整 UUID；多个显卡建议使用 UUID。

```powershell
nvidia-smi -L

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-ninfer.ps1 `
    -Model 'D:\models\model.ninfer' -Gpu '0'
```

Web UI：`http://127.0.0.1:18200/`；OpenAI API 基础地址：`http://127.0.0.1:18200/v1`。保持终端开启，Ctrl+C 停止。添加 `-DryRun` 可预览完整参数。

默认启用 CPU 视觉和 MTP，因此模型须包含对应权重及预处理资源。缺少视觉权重时追加 `-NoVision`；缺少 MTP 权重时追加 `-MtpDrafts 0`，包内快捷入口会同时关闭自适应 MTP 与 ngram。NInfer 模型无需单独的 mmproj。只有 GGUF 时参考固定版本的 [NInfer 转换指南](https://github.com/iamwavecut/ninfer-all/blob/8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3/docs/weight-conversion.md)。llama.cpp 则使用 GGUF，图像另需匹配的 mmproj。

<a id="defaults"></a>

## 默认配方与预算

下表对应 **0.18.0 运行包内 `start-ninfer.ps1`**，1K=1024 tokens。统一 `start-backend.ps1` 和直接运行 worker 使用不同默认值。

| 参数 | 默认值 |
|---|---|
| 上下文长度 | 204800 / 200K |
| GPU 历史预算 B / 生成预留 R | 36864 / 16384 |
| 默认输出上限 | 16384 |
| Main / MTP KV | INT8 / INT8 |
| MTP / ngram | 自适应 MTP4 / ngram31，min-match=12 |
| 视觉 | CPU，每张图片 1024 tokens |
| 思考预算 | 8192 |
| Host KV 载荷 | 12288 MiB / 12 GiB |
| Prefill 分块 | 256 tokens |
| Sink / recent | **12288 / 1024 tokens（12K / 1K）** |
| 保留的闲置会话数 / 并发数 | 1 / 1 |
| LRFU / GQA | 开启，lambda=0.5，top-frac=0.9 / max |
| SSD / 设备性能分析 | 关闭 / off |
| 监听 | 127.0.0.1:18200 |

这套配方已在 RTX 5060 Ti 16GB、Qwen3.8-27B GSQ-RCO IQ3_S 图文/MTP 模型上完成加载和文本请求验证；其他模型、显卡或显存容量需自行核对。Context 是逻辑上限，不能保证任意 RAM/VRAM 条件下都可达到。

| 预算 | 含义 |
|---|---|
| B | 每条活动请求选中的 GPU 历史 tokens；sink、recent 和必选图片区域计入 B。 |
| R | GPU 追加工作空间；NInfer 可在空间不足时归档、重选并继续生成，输出不必限制在 R 内。llama.cpp 当前服务路径仍以 R 限制单次输出，包含思考。 |
| H / HostMiB | 本进程受管理原生 KV 的 RAM 配额；Main/MTP 都计入。不是进程总内存上限。 |
| D / DiskMiB | 活动 KV 和闲置状态附件共用的 SSD 配额。 |

NInfer B/R 按 64-token 页对齐，prefill 为 128 的倍数且不超过 R；B+R 不超过 context。recent 至少 64，按整页向下取整。llama.cpp 默认检索 block 为 128；默认 recent=1024 对应 8 个 block，在 NInfer 中对应 16 页。两后端参数名相同也不保证行为相同，详见 [实现差异](docs/multi-backend-differences.md)。

NInfer KVMem KV 格式：`bf16`、`int8`、`fp8`、`nvfp4`、`k8v4`、`rk8v4`、`rk4v4`、`rk4v4-e8`、`rk2v4-e8`；Main/MTP 使用相同格式。llama.cpp 的 KV 参数为 `f16`、`f32`、`q8_0`、`q5_0`、`q4_0`，Main/MTP 可分别设置。模型权重量化与 KV 格式是独立配置，NInfer `int8` 与 llama.cpp `q8_0` 也不能直接互换。

### 检索与 SSD

LRFU 用当前 query 的相关性更新历史分数，每个请求提交一次历史更新。`top-frac` 控制允许累加本轮分数的命中范围，**不是 KV 保留比例**；GQA max 对同一 KV head 对应的 query heads 取最大相关性。sink、recent、媒体保留仍占用 B。

以下示例在**正式包解压目录**运行：

```powershell
# 关闭历史加权，使用 mean 聚合
& .\scripts\windows\start-ninfer.ps1 -Gpu '0' -NoLrfu -ScoreAggregation mean

# 可选 SSD 分层，目录须使用英文路径
& .\scripts\windows\start-ninfer.ps1 -Gpu '0' `
    -DiskPath 'D:\kvmem-cache' -DiskMiB 4096
```

SSD 默认关闭。只指定 `-DiskPath` 时 D 默认 4096 MiB。H/D 不包含全部 mean-K、checkpoint、暂存和系统文件缓存；RAM 仍必须容纳活动工作集与换页余量。文件只供**当前服务进程**复用，不支持重启恢复或跨后端 KV 迁移。公共 SSD 参数不能与 llama.cpp 旧在线 NVMe/旧会话磁盘参数混用。详见 [RAM/SSD 分层](docs/active-kv-tiering.md)及其 [验收记录](docs/active-kv-tiering-validation-20261010.md)。

### 多卡与并发

```powershell
# 包内启动器；自动分配层数
& .\scripts\windows\start-ninfer.ps1 -Gpu '0,1'

# 64 层模型的手动分配示例
& .\scripts\windows\start-ninfer.ps1 -Gpu '0,1' -StageLayers 40,24
```

NInfer 使用按层 pipeline，最多八张 GPU，要求相同 CUDA 计算能力；显存容量可不同，使用匹配架构的包。每卡层数为正数，合计等于模型层数；GPU 列表顺序即 stage 顺序，第一张卡承担 embedding/head/MTP 和 resident 视觉，默认 CPU 视觉在 Host 执行。也可用逗号连接 UUID。

文本和 resident/CPU 图片最多八条活动请求，B/R 为每请求预算，H/D 为每进程预算；增加并发会增加资源需求。多路 ngram 宽度大于 15 时自动降至 15；CPU 图片编码仍串行。快速 prefill 只开放 INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8。NInfer KVMem 不支持视频、overlay 视觉、DFlash、并行 prefill 或旧 `--lookup-ngram`。参数范围、模型边界和已验收组合见 [多后端运行](docs/multi-backend.md)。

<a id="performance"></a>

## RC3 性能实测

直接使用 2026-10-13 已完成的 **RC3** 结果：Windows、**RTX 5060 Ti 16GB**、**Qwen3.8-27B GSQ-RCO IQ3_S 图文/MTP 模型**。正式版保留相同引擎二进制；下表为该测试配置的观测值，不代表默认配方、所有显卡或模型的性能。

| 项目 | 结果 | 范围 |
|---|---:|---|
| 日常文本生成 | 50.08 token/s | 3 轮聚合 |
| 日常图片生成 | 55.90 token/s | 3 轮聚合 |
| 日常 HTML/SVG 代码生成 | 64.54 token/s | 3 轮，每轮输出 512 tokens |
| 日常完整任务 | 30.10 s | 总耗时中位数 |
| 长历史工具轮次生成 | 53.24 token/s | 32 轮，包含最终代码 |
| 长历史最终代码生成 | 44.87 token/s | 末轮服务端原生生成速度 |
| 末轮首字等待 | 16.87 s | 首个思考或正文片段 |
| 长历史请求总耗时 | 855.10 s | 33 次请求合计，不含加载 |
| 最终输入 / 输出 | 261522 / 512 tokens | 接近 256K |
| 完成请求数 / 工具前缀命中数 | 33/33 / 32/32 | 一次完整长历史测试 |

测试设置：context=262144，B/R=36864/16384，Main/MTP INT8，GDN FP16；固定 MTP3，adaptive off，ngram=0；prefill=512，thinking=128，output limit=512；GPU resident vision=1024 tokens/image；LRFU(.5/.9)，GQA max；Host=12288 MiB，SSD off，sessions/concurrency=1/1，device profile=off；temperature=1，top_p=.95，top_k=20，min_p=0，seed=42；流式 Chat Completions。

日常测试预热一轮后测三轮，覆盖约 12K 文本、896×896 图片和 HTML/SVG 生成；长历史注入模拟 `read_file` 结果。聚合速度按总输出 tokens / 总服务端生成耗时计算，包含思考；不计工具执行时间，也未据此验证代码完整性或通用召回率。整卡显存峰值约 **14.12 GiB**；长历史阶段系统可用 RAM 最低 **8.78 MiB**，明显内存压力可能影响首字等待和总耗时。

[完整报告、原始配置、计时与服务日志](https://github.com/kvmem/kvmem/releases/download/v0.18.0/RC3-performance-results-20261013.zip)。历史 llama.cpp 性能结果见 [16 GiB 报告](docs/recommended-config-performance.md)和 [旧版 256K 工具测试](docs/long-context-benchmark-2026-09-14.md)，不与本表混为同一配置。

## API 与 Web UI

| 接口 | 用途 |
|---|---|
| `GET /health` | 就绪状态 |
| `GET /v1/models` | 已加载模型标识 |
| `GET /props` | 后端、预算与能力 |
| `POST /v1/chat/completions` | 兼容 OpenAI 的对话、流式输出、工具调用及支持的图片请求 |
| `POST /v1/responses` | OpenAI Responses，包含 SSE 流式输出 |

NInfer 另提供原生 Anthropic Messages 接口；具体协议、工具约束和 schema 组合以 worker 及随包说明为准。llama.cpp 支持 `/responses` 别名，NInfer 不保证相同别名。图片与视频能力按后端区分：NInfer 支持图片、不支持视频；llama.cpp 图片需 mmproj，源码视频还需 ffmpeg/ffprobe 及相应构建选项。

以下非流式示例先查询真实 model ID，避免把占位名称作为模型：

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

<a id="source"></a>

## 源码准备与构建

`master` 是当前主线；`v0.18.0` 固定发布快照。两个引擎是 Git submodule，KVMem 改动位于 `backends/patches/`，构建前须应用补丁。GitHub 自动生成的 Source ZIP 不含完整子模块，开发请使用 Git：

```sh
git clone --recurse-submodules https://github.com/kvmem/kvmem.git
cd kvmem
python scripts/prepare-backends.py
python scripts/prepare-backends.py --check
```

Python 3.9+ 和 Git 用于准备源码；`--backend llamacpp` / `--backend ninfer` 可只准备一个后端。`--check` 只检查 pin、patch 与完整树，不修改源码。准备后的子模块显示 modified 属于应用集成补丁的正常状态；脚本保留开发者修改，冲突会报错。

### Windows

需要 VS2022 C++ Build Tools、CMake/Ninja，以及选定后端适用的 CUDA Toolkit。以下 RTX 50 NInfer 路径使用 CUDA 13.2、`120a` 和 vcpkg；其他架构需选择匹配配置。默认使用 **8 个并行编译任务**，两个 worker 使用独立构建目录。

```powershell
# 公共核心，无需模型或 GPU
./scripts/windows/build-backends.ps1 -Backend core
ctest --test-dir build-backends/core -C Release --output-on-failure

# llama.cpp CUDA 工作进程
./scripts/windows/build-backends.ps1 -Backend llamacpp -CudaArchitectures 120a

# NInfer 工作进程；将 VCPKG_ROOT 指向本机 vcpkg
./scripts/windows/build-backends.ps1 -Backend ninfer -CudaArchitectures 120a `
    -CudaPath 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2' `
    -CMakeArgs @("-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake")
```

源码统一入口示例：

```powershell
./scripts/windows/start-backend.ps1 -Backend ninfer `
    -Worker './build-backends/ninfer/apps/ninfer-serve.exe' `
    -Model 'D:/models/model.ninfer' -Gpu '0'

./scripts/windows/start-backend.ps1 -Backend llamacpp `
    -Worker './build-backends/llamacpp/bin/llama-kvmem-server.exe' `
    -Model 'D:/models/model.gguf' -Gpu '0'
```

统一入口默认 context=262144、B=32768、R=1024、prefill=128、输出上限1024；NInfer 默认 INT8、MTP/ngram/视觉关闭，LRFU 开启、sink=12288、recent=1024。需要视觉或 MTP 时显式传 `-Vision`、`-MtpDrafts` 等。llama.cpp 对应能力使用 `-WorkerArgs` 传原生参数。

**源码树的 `start-ninfer.ps1` 是固定配方快捷脚本，默认从仓库根目录读取 `model.ninfer`，不接受包内扩展的 `-Model`、`-NoVision` 等选项。**需要自定义模型或完整参数时使用源码 `start-backend.ps1`；前面的包内快速开始仅用于解压后的正式运行包。

### Linux / ROCm

```sh
# 仅公共核心
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release \
  -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF
cmake --build build-core --parallel 8
ctest --test-dir build-core --output-on-failure
```

GPU 构建前准备选定后端，llama.cpp 从仓库根目录配置 CMake，NInfer 从 `backends/ninfer` 配置并遵循其依赖清单。HIP/ROCm 使用已有 [ROCm 指南](docs/rocm.md)。Linux Host 测试或 CUDA 编译通过不等于 Linux GPU 实机推理验收；当前正式运行包范围仍为 Windows NInfer。详细构建、导出和打包流程见 [多后端仓库工作流](docs/multi-backend-repository.md)。

## 验证范围与限制

- 三个正式包完成 Release 编译、依赖与架构检查、干净 PATH 下 help、PowerShell 5.1 参数检查，以及解压校验。RTX 50 包已在 RTX 5060 Ti 上完成模型加载、健康与短 Responses 请求；RTX 30/40 包尚无对应硬件的实际推理验收。
- 主线集成通过 17 项本地核心测试及 9 项 CI 检查，覆盖 Linux/Windows Host、CUDA 编译、完整/轻量 UI、固定后端源码和服务单元。编译通过不代表所有模型、GPU 或功能组合都已运行验证。
- RC3 性能结果沿用已有记录；不同配方、并发、冷 SSD、内存压力及模型可能改变性能。此次正式化没有重跑完整图像矩阵、多卡性能或 SSD 压力测试。
- KVMem NInfer 当前适配 Qwen3.5/3.8 Dense 边界。其他上游模型架构或 suspend/graft/router 功能不能据权重格式推断已支持。
- 设备故障可能需要重启 worker；不承诺故障后的原会话可继续复用。模型权重、视觉、MTP、workspace、checkpoint 和系统缓存仍需额外内存。

反馈请附包版本、GPU、驱动、模型、修改后的参数与错误日志，移除日志中的私有对话。

<a id="documentation"></a>

## 文档与仓库结构

| 指南 | 内容 |
|---|---|
| [0.18.0 发布说明](https://github.com/kvmem/kvmem/releases/tag/v0.18.0) | 正式包、性能与来源 |
| [多后端仓库工作流](docs/multi-backend-repository.md) | 子模块、准备、构建与打包 |
| [多后端运行](docs/multi-backend.md) | 启动、能力与支持的组合 |
| [后端实现差异](docs/multi-backend-differences.md) | 默认值、预算、R、KV 与视觉 |
| [活动 KV 分层](docs/active-kv-tiering.md) | 公共 RAM/SSD 配额与生命周期 |
| [分层验收记录](docs/active-kv-tiering-validation-20261010.md) | 已测范围与限制 |
| [内存后端契约](docs/memory-backend-contract.md) | 新后端接入契约 |
| [迁移验收记录](docs/ninfer-all-migration-validation.md) | 固定 NInfer 基线 |
| [纯文档 CI](docs/ci.md) | 文档修改跳过编译测试 |
| [Windows 使用说明](scripts/windows/README.md) | llama.cpp 构建与快捷入口 |
| [ROCm 指南](docs/rocm.md) | HIP/ROCm 路径 |
| [llama.cpp 多模态报告](docs/multimodal-implementation-report-2026-09-14.md) | 历史图像实现 |

```text
kvmem/                  共享内存契约、检索与 Host 存储
src/adapter/, tools/    llama.cpp 适配器、命令行与服务
backends/llamacpp/      固定版本的 llama.cpp 子模块
backends/ninfer/        固定版本的 NInfer 子模块
backends/patches/       当前后端集成补丁
backends/versions.json  固定版本、补丁哈希、准备后源码树哈希与许可证
scripts/               源码准备、构建、启动与打包
docs/                  设计、使用、发布与带日期的验收记录
patches/               历史补丁
```

历史报告中的日期、参数和候选状态描述保留当时测试上下文；当前使用入口与正式包范围以本 README、最新发布说明和具体能力表为准。

## 致谢与许可证

感谢 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)、原始 [Neroued/ninfer](https://github.com/Neroued/ninfer) 及社区分支的维护者与贡献者，包括 [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) 的整合、Ryan-gsq 的 Windows/GSQ-RCO 工作，以及 3090/4090、自适应 MTP、快速 prefill 和 ngram 的贡献者。完整 NInfer 署名见 [发布说明](https://github.com/kvmem/kvmem/releases/tag/v0.18.0#致谢)。LRFU/GQA max 参考 [melis2023/kvmem-actkv](https://github.com/melis2023/kvmem-actkv)。感谢 Bilibili 的 melis、redsnow23 等测试者。

各后端保留原许可证和第三方声明，路径见 [后端清单](backends/versions.json)。预编译包附带相应许可证；cuBLAS DLL 遵循包内 CUDA EULA，不适用 NInfer 的 Apache-2.0 许可证。模型另行分发并遵循各自条款。

## 论文与引用

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
