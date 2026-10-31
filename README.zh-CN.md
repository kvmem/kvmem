[English](README.md) | **简体中文**

# KVMem

**在有限显存中运行长上下文对话与 AI Agent。**

KVMem 将完整原生 KV 历史保存在内存中，可选下沉到 SSD；GPU 只保留当前请求需要的历史和生成工作空间。项目支持 **NInfer、llama.cpp、PolyStrata** 等多种推理后端，并持续接入更多后端，提供本地 Web UI 和兼容 OpenAI 的 API。

## 支持哪些模型？按后端选择

以下是**启用 KVMem 历史管理时**支持的主要模型范围。选择模型时，请同时确认后端和文件格式。

| 后端 | 模型 | 模型格式 |
|---|---|---|
| **[NInfer](docs/multi-backend-differences.md)** | **Qwen3.5 Dense**（0.8B / 2B / 4B / 9B / 27B）、**Qwen3.6 / Qwen3.8 27B**，以及兼容的微调模型，包括 **Ternary Bonsai 2 27B** | `.ninfer`，需使用兼容的模型文件或先转换权重 |
| **[llama.cpp](docs/multi-backend.md)** | **Qwen3.5 Dense**、**Qwen3.6 / Qwen3.8 27B**、**Ternary Bonsai 2 27B**，以及其他兼容的全注意力 GGUF 模型 | `.gguf` |
| **[PolyStrata](docs/polystrata-integration.md)** | **Qwen3.6-35B-A3B**、**Ornith 1.5**、**Tiel Coder / Cyber-Tiel 35B-A3B** 等兼容的 `qwen35moe` 模型 | `.gguf`；图片输入还需匹配的 `mmproj` |

NInfer 的 KVMem 接入目前覆盖 **Dense** 模型；PolyStrata 覆盖 **`qwen35moe` MoE** 架构。这两个上游引擎支持的其他模型尚不在上述 KVMem 范围内。llama.cpp 的 KVMem 适配器会跳过滑动窗口注意力（SWA）及纯循环架构模型。

架构支持不代表每个尺寸、量化和功能组合都已实测。具体范围见 [Qwen3.5 Dense 家族验收](docs/qwen35-dense-family.md)、[Bonsai 接入](docs/bonsai-integration.md) 和 [PolyStrata 接入](docs/polystrata-integration.md)。实际可用模型、视觉与 MTP 能力还取决于模型组件及所选[运行包](https://github.com/kvmem/kvmem/releases)。

## 支持哪些显卡？按后端选择

下表列出 **KVMem** 的主要显卡系列，并说明运行包与源码构建的区别。

| 后端 | 显卡系列 / 代表型号 | 支持范围 |
|---|---|---|
| **NInfer** | **NVIDIA RTX 20 / 30 / 40 / 50 系**、**TITAN RTX**、**Tesla V100** | 仅 CUDA。V100 需 `sm_70` / CUDA 12 构建或匹配的历史运行包。A100 类 `sm_80` 有源码兼容目标，尚无 KVMem 实卡验收。 |
| **llama.cpp** | **NVIDIA RTX 20 / 30 / 40 / 50 系**（CUDA）；**AMD Radeon RX 6800 / 6900**、**RX 7700 XT / 7800 XT / 7900 XT / XTX**、**RX 9060 XT / 9070 / 9070 XT**（HIP/ROCm） | AMD 可用范围取决于 SDK、操作系统与编译目标，见 [ROCm 指南](docs/rocm.md)。RX 9060 XT 已有 KVMem 长上下文实测记录。 |
| **PolyStrata** | **NVIDIA RTX 20 / 30 / 40 / 50 系** | KVMem 当前使用 CUDA 单卡。源码中的 AMD HIP 入口仍为**实验性**，尚待 AMD 实卡推理验收，见[接入状态](docs/polystrata-integration.md)。 |

RTX 代表型号包括 **2060 / 2080 Ti**、**3060 / 3090**、**4060 Ti / 4090**，以及 **5050 Laptop / 5060 Ti / 5090**。显卡系列符合要求后，仍需确认显存足够容纳所选模型和配置。

请在[发布页](https://github.com/kvmem/kvmem/releases)选择与显卡、系统和后端匹配的运行包。Windows 双后端统一包覆盖 RTX 20 / 30 / 40 / 50 系，包含 NInfer 和 PolyStrata；llama.cpp 与 AMD 使用独立运行包或源码构建。编译目标覆盖不等于每个型号都已实卡测试，驱动要求与实际验收范围以随包说明和验证报告为准。

## KVMem 能做什么

- **长上下文**：按查询召回历史 KV，在固定显存预算内管理长对话和工具历史。
- **Extra-long / xhigh 长推理**：循环复用 GPU 生成空间，在有限显存下持续生成长推理与回答，支持 `xhigh` 推理档位。
- **状态复用**：支持多轮对话、工具续接和会话管理，按后端规则复用前缀与检查点。
- **内存分层**：统一管理 RAM / SSD 中的 KV 历史，按需调入 GPU。

## Demo：16GB 显卡上的 xhigh 推理

**RTX 5060 Ti 16GB · 27B IQ3 模型 · 本地实测。** 下图是发布视频中使用的两份鹈鹕骑行 HTML 动画，分别由 `medium` 与 `xhigh` 推理档位生成。

| `medium` | `xhigh` |
|:---:|:---:|
| ![medium 推理档位生成的鹈鹕骑行 HTML](docs/assets/pelican-reasoning/medium.gif) | ![xhigh 推理档位生成的鹈鹕骑行 HTML](docs/assets/pelican-reasoning/xhigh.gif) |

KVMem 在生成过程中持续腾出空间，**Gen reserve 是可循环使用的工作空间，不是累计输出上限**；仍遵循上下文、请求限制与模型停止条件。这里展示的是单次生成实例，来源与范围见 [示例说明](docs/demos/pelican-reasoning.md)。

<a id="downloads"></a>

## 下载

**[稳定版](https://github.com/kvmem/kvmem/releases/latest)** · **[全部发布（含预发布与历史版本）](https://github.com/kvmem/kvmem/releases)**

在发布页选择匹配操作系统、GPU 和后端的运行包。安装包、校验文件、驱动要求、模型支持及实测范围统一放在对应发布说明中；模型需另行准备。标记为 **Pre-release** 的版本用于试用。

<!-- 维护约定：README 保持与版本无关。普通发版只更新 Releases、随包说明与相关文档；不在首页添加版本号、下载直链、包大小或校验值。中英文同步维护。 -->

<a id="quick-start"></a>

## 快速开始

1. 下载运行包并完整解压。**运行包和模型的完整路径须使用英文路径，不能含中文。**
2. Windows 控制台运行包双击 `START-HERE-CN.bat`，选择后端、模型和 GPU；英文入口为 `START-HERE-EN.bat`。
3. 启动服务，待就绪后打开控制台中的聊天入口，或将客户端接入控制台显示的 API 地址。

NInfer 通常使用 `.ninfer` 模型，llama.cpp 使用 GGUF；具体模型、量化与视觉支持以所选后端和发布说明为准。旧包没有一键入口时，按随包说明启动。

详见 [Windows 一键启动](docs/windows-one-click.md)、[Linux 启动](docs/linux-launcher.md) 和 [控制台指南](docs/unified-console.md)。

## API 与配置

服务提供兼容 OpenAI 的 `/v1/chat/completions` 和 `/v1/responses` 接口，支持流式输出；工具调用与多模态能力按后端区分。API 基础地址为 `http://127.0.0.1:<服务端口>/v1`，模型标识可从 `/v1/models` 查询。

上下文上限、GPU 历史预算、生成预留和内存配额可在控制台或启动参数中配置。长上下文仍需足够的系统内存；模型权重与推理工作空间另占显存。参数、默认值及支持组合见 [运行指南](docs/multi-backend.md) 和 [后端差异](docs/multi-backend-differences.md)。

<a id="source"></a>

## 从源码构建

```sh
git clone --recurse-submodules https://github.com/kvmem/kvmem.git
cd kvmem
python scripts/prepare-backends.py
python scripts/prepare-backends.py --check
```

引擎使用固定版本的子模块和集成补丁，GitHub 自动生成的 Source ZIP 不含完整子模块。构建要求与命令见 [源码准备与构建](docs/multi-backend-repository.md)；AMD 用户另见 [ROCm 指南](docs/rocm.md)。源码主线与已发布运行包的功能可能不同。

<a id="defaults"></a>
<a id="performance"></a>
<a id="documentation"></a>

## 文档

| 内容 | 入口 |
|---|---|
| 启动、参数与支持范围 | [运行指南](docs/multi-backend.md) · [后端差异](docs/multi-backend-differences.md) |
| KV 召回与内存管理 | [图召回](docs/graph-recall.md) · [RAM / SSD 分层](docs/active-kv-tiering.md) |
| 设计与后端接入 | [架构](docs/architecture.md) · [内存后端契约](docs/memory-backend-contract.md) |
| 构建与发布 | [开发工作流](docs/multi-backend-repository.md) · [后端版本与许可证](backends/versions.json) |
| 测试与性能 | [验证记录](docs/multi-backend-validation.md) · [性能报告](docs/recommended-config-performance.md) · [测试数据](docs/benchmarks/) |

测试报告保留各自的版本、硬件与参数条件，不代表所有运行包或设备的表现。更多专题见 [docs/](docs/)。

## 交流与反馈

**QQ 交流群：1040777853** · [GitHub Issues](https://github.com/kvmem/kvmem/issues)

反馈请附版本、GPU、驱动、模型、启动参数和错误日志，并移除日志中的私有对话。

## 致谢与许可证

感谢 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)、[Neroued/ninfer](https://github.com/Neroued/ninfer)、[iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) 及社区贡献者，包括 Ryan-gsq、3090/4090 适配、自适应 MTP、快速 prefill 和 ngram 的贡献者。LRFU/GQA max 参考 [melis2023/kvmem-actkv](https://github.com/melis2023/kvmem-actkv)。感谢 melis、redsnow23 等测试者。

各后端保留原许可证和第三方声明，详见 [后端清单](backends/versions.json) 与随包许可证。cuBLAS 遵循 CUDA EULA；模型另行分发并遵循各自条款。

## 论文与引用

[KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU](https://arxiv.org/abs/2609.04852)

<details>
<summary>BibTeX</summary>

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

</details>
