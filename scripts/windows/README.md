# KVMem Windows rc3 runtime / Windows rc3 运行包

> Device identifiers and private paths have been anonymized; reported measurements are unchanged.


Current source builds use [graph recall C](../../docs/graph-recall.md) in both backends. The 0.18 resource recipe remains. Use `-QueryScorer meanq-cosine|page-softmax`, `-ScoreAggregation mean|max`, `-GraphAlpha`, and `-Lrfu`/`-NoLrfu` for the documented scoring variants. Rebuild workers with the versioned backend patches. Published old packages keep their original behavior.

当前源码的两个后端默认采用图召回 C，保留 0.18 资源配方。可用 `-QueryScorer`、`-ScoreAggregation`、`-GraphAlpha`、`-Lrfu`/`-NoLrfu` 切换已支持的评分组合；新脚本须配合本源码重编译的 worker。旧发布包保持其原有行为。

## Multi-backend launchers / 多后端启动入口

The current launch UI calls `start-backend.ps1` internally with typed JSON settings.
NInfer exposes default/off thinking mode, thinking-token budget, reasoning effort
and preservation of prior assistant reasoning. Effort levels depend on the model.
The UI's additional native arguments remain a small allowlist; direct launchers and
worker EXEs remain available for advanced options.

当前启动 UI 通过 JSON 结构化设置调用内部 `start-backend.ps1`。NInfer 可设置默认/关闭思考、
思考预算、思考强度和保留历史思考；具体强度由模型决定。UI 的附加原生参数仍有白名单，
完整高级选项可用脚本或直接调用后端 EXE。

The source and newly built packages share `start-ninfer.ps1`, including `-Model`,
`-NoVision`, `-ThinkingBudget`, `-ReasoningEffort` and `-PreserveThinking`.
For the Qwen CPU Vision path, `-CpuVisionIdleSeconds` defaults to 300: weights load on the
first actual image encode and unload five minutes after the last encoder releases them. Set it
to 0 to retain lazily loaded weights until model unload. Both launchers forward this setting.

Its 0.18 recipe differs from the unified launcher's defaults. Older ZIPs retain their
original scripts. In a PowerShell session, pass an array; for an external `-File`
invocation put native options last after `-WorkerArgs`:

源码和新运行包使用同一份 NInfer 快捷入口；旧 ZIP 仍使用原脚本。
PowerShell 内直接调用使用数组；外部 `-File` 调用时将原生参数放在最后：

```powershell
& .\scripts\windows\start-ninfer.ps1 -Model 'D:\models\model.ninfer' `
  -ThinkingBudget 8192 -WorkerArgs @('--default-reasoning-effort', 'medium', '--preserve-thinking')

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-ninfer.ps1 `
  -Model 'D:\models\model.ninfer' -ThinkingBudget 8192 `
  -WorkerArgs --default-reasoning-effort medium --preserve-thinking
```

For exact argument boundaries across shells, save a UTF-8 JSON array in `worker-args.json`
and pass `-WorkerArgsFile .\worker-args.json`. This preserves empty strings,
embedded quotes and paths ending in a backslash. Both launchers accept this setting.
It is mutually exclusive with `-WorkerArgs`; the file contains only native arguments,
not launcher settings or executable paths. Arrays are never evaluated as code or split
on spaces. Named thinking settings must not duplicate the corresponding native options.

跨 shell 传递复杂值时，将 UTF-8 JSON 数组保存为 `worker-args.json`，使用
`-WorkerArgsFile .\worker-args.json`。空字符串、引号和路径末尾反斜杠均完整保留。
此选项与 `-WorkerArgs` 互斥；文件只写原生参数，不写启动器设置或 EXE 路径。
命名思考设置与对应原生参数不能重复指定。

```json
["--default-reasoning-effort", "medium", "--preserve-thinking"]
```

## NInfer tensor parallelism / NInfer 张量并行

Select **NInfer**, choose two compatible GPUs, and enable **Tensor parallelism · heads**
in **Multi-GPU**. For the Windows launcher, pass the GPU UUIDs in order and use
`-TensorParallel`. `-TensorSplit` is optional. Its weights follow the UUID order, and
omitting it lets NInfer allocate automatically. With TP enabled, leave `-StageLayers` unset.
The worker must advertise `heads_tensor_parallel_revision` of at least `1` in
`--kvmem-capabilities`. Starting with `-DryRun` previews the arguments without launching the worker.

在控制台选择 **NInfer**，勾选两张兼容显卡，然后在 **多卡设置** 启用 **张量并行 · heads**。
PowerShell 中使用 `-TensorParallel`，以逗号分隔的字符串传入显卡 UUID。`-TensorSplit`
可选，按 UUID 顺序填写权重，省略时由 NInfer 自动分配。启用 TP 时请留空 `-StageLayers`。
worker 须在 `--kvmem-capabilities` 中报告 `heads_tensor_parallel_revision` 至少为 `1`。

```powershell
$gpu1 = 'GPU-00000000-0000-0000-0000-000000000001'
$gpu2 = 'GPU-00000000-0000-0000-0000-000000000002'
& .\scripts\windows\start-ninfer.ps1 -Model 'D:\models\model.ninfer' `
  -Gpu "$gpu1,$gpu2" -TensorParallel -TensorSplit 3,1 -SpeculativeMode none -NoVision -DryRun
```

Replace the model path and UUIDs. Remove `-DryRun` to start. / 替换模型路径和 UUID，去掉 `-DryRun` 即可启动。

The following sections describe the older rc3 IQ3/IQ4-specific scripts.
以下章节描述旧 rc3 的 IQ3/IQ4 专用脚本。

## Quick start / 快速启动

Choose either the CUDA 13.2.86 or 12.9.86 runtime ZIP and extract the entire package.
Keep `bin`, `scripts` and `share` in their relative locations. Models are separate.
选择一个 CUDA 13.2.86 或 12.9.86 运行包并完整解压，保留目录结构；模型单独下载。

- [IQ3 main model / 主模型](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/blob/main/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf)
- [Vision projector / 视觉头](https://huggingface.co/HermiHg/Qwen3.8-27B-mmproj-Q5_K-MIX-GGUF/blob/main/mmproj-Qwen3.8-27B-Q5_K-MIX.gguf)

Open PowerShell in the extracted package directory. Check GPU indices with `nvidia-smi -L`.
Replace the example paths and GPU index in this command:
在解压目录打开 PowerShell，用 `nvidia-smi -L` 查看显卡编号，替换下面的模型路径和显卡编号：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-iq3.ps1 `
  -Model 'D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf' `
  -Mmproj 'D:\models\mmproj-Qwen3.8-27B-Q5_K-MIX.gguf' `
  -Gpu 0
```

Open **http://127.0.0.1:18200/** after loading. Keep the terminal open; **Ctrl+C** stops the service.
加载后打开上述地址，保持终端开启；按 **Ctrl+C** 停止服务。普通日志包含请求结束时的 prefill/decode 速度。

## Script settings / 脚本设置

The scripts are independent and directly call `bin/llama-kvmem-server.exe`.
They accept only `-Model` and `-Mmproj` (required), plus `-Gpu` (default `0`, index or UUID).
Executable and UI paths are relative to the script, not a fixed drive or username.
脚本独立运行，无需公共启动脚本。仅支持必填的 `-Model`、`-Mmproj`，以及默认 `0` 的 `-Gpu`。
程序和 UI 使用相对脚本位置的路径，不绑定盘符或用户名。

To change other settings, edit the script directly / 其他设置直接编辑脚本：

| Setting / 设置 | Edit / 修改位置 |
|---|---|
| Port / 端口 | `$Port = 18200` |
| Full UI (default) / 完整 UI（默认） | `$UiDir` ends in `share\kvmem\ui` |
| Lightweight UI / 轻量 UI | Change `$UiDir` to end in `share\kvmem\ui-lightweight` |
| Disable UI / 关闭 UI | Replace `--webui` with `--no-ui` |
| GPU vision / GPU 视觉头 | Replace `--no-mmproj-offload` with `--mmproj-offload` |
| Fixed API model name / 固定 API 模型名称 | Add `--alias your-model-name` to the server arguments |
| KV, context, MTP, thinking / 其他推理设置 | Edit the corresponding server argument in the script |

When adding argument lines, retain PowerShell's trailing backtick on every continued line.
添加参数行时，注意保留 PowerShell 续行反引号。不要把服务端参数当作脚本参数传入。

| Default / 默认值 | IQ3 | IQ4 |
|---|---:|---:|
| Context / 上下文 | 262144 | 262144 |
| KVMem budget / 工作集 | 36864 | 32768 |
| Generation reserve / 生成预留 | 16384 | 12288 |
| Main K/V | Q8/Q8 | Q5/Q5 |
| MTP draft / 草稿 | F16, MTP3/ReplaySSM | F16, MTP3/ReplaySSM |
| Vision / 视觉头 | CPU | CPU |
| Block tokens | 128 | 128 |
| Thinking budget | 4096 | 4096 |

IQ4 is an optional experiment requiring the separately prepared MTP-Q4_0 main model.
Use `start-iq4.ps1` with the same three parameters and your IQ4 model path.
IQ4 为备选测试，需要自行准备 MTP-Q4_0 主模型；将脚本名改为 `start-iq4.ps1` 并传入相应模型路径。

Both UIs are included. Full UI does not add backend tool execution or stream resumption.
两套 UI 均随包提供；完整 UI 不代表服务端新增工具执行或断线续传功能。

## Text-only startup / 无视觉头的纯文本启动

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
.\bin\llama-kvmem-server.exe -m 'D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf' `
  --host 127.0.0.1 --port 18200 -c 262144 -n 16384 `
  --kvmem-budget 36864 --kvmem-gen-reserve 16384 `
  -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3 `
  --enable-thinking --reasoning-budget 4096 --verbosity 3
```

## NInfer DFlash2 16 GB profile / DFlash2 16 GB 配置

`start-dflash2.ps1` selects the qualified IQ3_S + DFlash2 profile: 256K logical context,
36K history + 10K generation reserve, five drafts, FP16 GDN state, INT8 KV, CPU Vision,
no n-gram and a 64 MiB CUDA Graph planning allowance. Supply a `.ninfer` artifact containing
the DFlash2 companion. The shared NInfer startup defaults are unchanged.

`start-dflash2.ps1` 使用已验证的 IQ3_S + DFlash2 配置：256K 上下文、36K+10K、五草稿、
FP16 GDN、INT8 KV、CPU 视觉、关闭 ngram、CUDA 图规划余量64 MiB。模型必须包含 DFlash2 草稿模块。

```powershell
.\scripts\windows\start-dflash2.ps1 -Model 'D:/models/iq3-dflash2.ninfer' `
  -Gpu '0' -DryRun
```

Omit `-DryRun` to start. Source builds can also pass `-Worker` with the path to `ninfer-serve.exe`.
See [measurements and limits](../../docs/dflash2-kvmem-integration.md), including the capped Task 2
output and CPU image-encoding latency.

## Requirements and validation / 环境与验证

Windows x64, an NVIDIA driver compatible with the selected package, Microsoft Visual C++ x64 runtime,
and AVX2/FMA/F16C/BMI2 CPU support are required. CUDA Toolkit, Visual Studio and Node.js are not needed.
需要兼容的 NVIDIA 驱动、VC++ x64 运行库和支持 AVX2/FMA/F16C/BMI2 的 CPU；运行不需要 CUDA Toolkit、Visual Studio 或 Node.js。

CUDA 13.2.86 targets: 75/80/86/89/90/120a. CUDA 12.9.86 additionally includes 70.
RTX 5060 Ti was physically tested; other targets need community validation.
This script update does not rebuild or change native executables or DLLs.
本次脚本更新没有重编译或更换 EXE/DLL。两套 CUDA 包的独立 IQ3 脚本已通过短流式推理和实时日志检查；
IQ3 的 Ctrl+C 退出已验证。IQ4 无本地模型，仅完成语法检查。

See `BUILD-INFO.json`, `VALIDATION.json`, `UI-VALIDATION.json`, `TEST-REPORT.md` and `RELEASE-NOTES.md`
for binary provenance and the scope of earlier tests. `SHA256SUMS` verifies the packaged files.
