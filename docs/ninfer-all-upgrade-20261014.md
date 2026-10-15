# NInfer 上游升级记录（2026-10-14）

适配开始于 2026-10-14，构建与验收持续到 2026-10-15。

本次接入以 KVMem `master` 的 `dea1f226c74391b05a2a6ccb13cd29481874a30b` 为基础，
将 `iamwavecut/ninfer-all` 从 `8319e8f512477d0cfdef89e16c17c1ba9f9bc7b3`
升级到固定提交 [`1383a50e415348e6fca4c7ecfd85ceb79b0426d5`](https://github.com/iamwavecut/ninfer-all/commit/1383a50e415348e6fca4c7ecfd85ceb79b0426d5)。
llama.cpp 基线、公共 KVMem 策略库和已发布的 0.18.0 运行包没有改变。
实现位于 `upgrade/ninfer-all-20261014` 分支。本文主体记录接入完成时的源码与验证状态；RTX 20 / V100 的后续试用包范围见 [0.18.1 NInfer RC1 说明](releases/v0.18.1-ninfer-rc1.md)。该 RC 不代表老卡实机资格验证已经完成。

## 上游变化与接入范围

| 上游变化 | 本次接入 |
|---|---|
| Turing `sm_75`（RTX 20/TITAN RTX）、Volta `sm_70`（V100）的内核路线与设备配置 | 保留上游路线；分别用 CUDA 13.2 和 CUDA 12.9 构建。V100 要求 CUDA 12.x；构建通过不等于实卡或 KVMem 全组合资格验证。 |
| 原生 Gemma 4、Flash-Next MTP、专家卸载与查表路径 | 保留新模型的原生入口；新增模型仍明确拒绝 KVMem 和 strict/mixed 常驻组合。Gemma 原生工具格式接入约束生成与解析。 |
| 最短剩余 prefill 调度、按轮分配 decode、语法掩码重叠 | 合并新调度与约束路径，保留 KVMem 的请求预算、历史装载、取消和资源回收。 |
| 自动 proposal head、目标分布耦合验证、采样位置修正、自适应 MTP/lookup | 保留新算法；Main/MTP 使用物理页位置，采样和验证使用逻辑 token 位置。 |
| Host cache 自动预算、指标和版本信息 | 保留原生模式的新行为及服务接口；KVMem 的全局 H 预算和原生页归档仍受已有约束。 |
| Windows/MSVC、CUDA 图更新、GDN 与量化转换改进 | 采用新 CUDA 图重建处理与普通 CUDA 分配，保留 WDDM 预算及 KVMem strict/mixed 管理；不恢复上游已删除的 D3D12 分配器。 |
| 图片 detail 的原生 low/high 处理 | low/high 使用上游预处理，旧 `original` 映射为 auto；未知值默认拒绝，显式 lenient 选项允许忽略。 |

旧主线的 166 个补丁文件经过逐文件三方合并，37 个文本冲突已处理；
删除的上游源文件没有复活。固定 64-token Main/MTP 页、GDN checkpoint、
B/R/H、LRFU、RAM/SSD 分层、工具/推理输出和错误契约保留。
目前已有模型架构之外的 KVMem、suspend、prompt graft、router 等组合继续明确报错。

## 验证配置与依据

本轮正确性测试使用当时唯一可用的 RTX 5050 Laptop（8 GiB），UUID
`GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29`。
RTX 5060 Ti 当时未枚举，因此没有性能测量或与旧版速度对比。
编译默认使用 8 个并行任务，MSVC 14.44，RTX 50/Turing 使用 CUDA 13.2，
V100 使用 CUDA 12.9；转换测试使用现有 Python 3.12.13 环境。
另一批八路编译同时运行时，系统可用内存一度仅剩 0.45 GiB，V100 构建保留
已完成对象缓存后降为四路继续；与主构建重叠的原生/Turing 增量构建使用两路。

真实模型为既有 `real-bonsai-q4-mtp.ninfer`（7,441,879,808 字节）。
测试显式启用 WDDM evictable budget、FP16 GDN 和 full draft head，图执行的
allowance 为 128 MiB，目标权重保持文件中原有格式。
此配置用于 8 GiB 卡的功能检查，不代表 strict/mixed 常驻或默认显存配方通过。
该卡的默认 CUDA 空闲容量检查拒绝模型；尝试 Q4 embedding/Q6 head 的配置也被
正确拒绝，因为该文件存储的 head 是 T2。失败尝试及诊断日志保留。

正确性以输出上限、取消和恢复、并发实际执行、页所有权、H 预算及资源释放为依据；
Main/MTP KV 与 GDN 状态的搬运检查比较原生字节。
同一配置的 checkpoint 重放使用确定性输出作为补充证据，不以旧版 token 序列作为唯一裁判。

## 当前验证结果

| 范围 | 结果 |
|---|---|
| 原生 RTX 50 `sm_120a` worker、Turing `sm_75` worker | 完整 worker 构建通过，包含下文正常停止修复。 |
| V100 `sm_70` worker | CUDA 12.9 完整 worker 构建通过；实际嵌入 `sm_70` 代码，帮助入口与文件指纹核对通过。没有 V100 实卡测试。 |
| 转换与量化识别 | GGUF blocks/ternary 的 17 项定向 pytest 通过。 |
| 接口、选项与基础 CPU/CUDA 检查 | 24 项通过；包括 Chat/Responses/Responses store/Anthropic、工具约束/解析、structured output、采样与 speculative 数学、页 payload、RoPE、稀疏 INT8 和 token sums。 |
| 真实模型 ordinary 与 CUDA Graph | KVMem INT8 eager、RK8V4 Graph、native INT8 eager 均通过越窗输入和 130-token 输出上限检查。 |
| 会话与工具历史 | A→B→A、Host 压力、编辑前缀、输入 checkpoint 回退、取消后恢复通过；Main/MTP/GDN 搬运字节检查通过。 |
| 自适应 MTP4＋ngram31＋RK8V4 Graph | 冷/恢复复制、取消/恢复、1/2/3/17/65 输出边界、重复采样的逻辑 RNG 位置、编辑前缀及代码复制通过，实际接受宽 ngram 草稿。 |
| 两路并发与停止状态 | ordinary FP16、固定 MTP4 FP16、自适应 MTP4 FP32 通过并发 decode、checkpoint 重放、H 预算、页释放及异步正常停止状态检查。FP16 自适应的数值限制见下文。 |
| 最新 worker HTTP | Chat、强制严格工具调用、流式工具、Responses、流式 Responses、空 detail 的 400 拒绝、JSON opt-in 拒绝及随后恢复通过；health/stats/metrics 通过，正常停止退出码 0。工具约束无需开启全局 JSON opt-in。 |
| 新家族边界 | metadata-only Gemma 4、Qwen4Exp fixture 的 KVMem 拒绝检查通过。 |
| 源码重建 | 新补丁应用、完整 tree 校验、重复准备、再次校验在独立干净检出中全部通过；源码 diff 无空白错误。 |

完整检查清单和最终 worker 指纹见 [验证摘要](validation/ninfer-all-upgrade-20261014.json)。
原始构建与实验日志保存在本机工作目录 `ninfer-upstream-integration-20261014/results/`
及对应构建日志中；不是已发布运行包的验收结果。

## 发现并修复的接入问题

| 问题 | 触发与证据 | 修正 |
|---|---|---|
| Q2_0 GDN 控制权重未被新转换校验接受 | `test_gguf_blocks.py` 的原有 Q2 用例失败 | 根据原始 gate projection 和存储量化格式识别控制权重，包含新 Q8 路径；转换定向测试 17 项通过。 |
| 新图片 detail 占位字段把空字符串当作缺省值 | OpenAI Chat schema 的未知 detail 用例失败，而当前 master 正确拒绝该值 | 将未知 detail 标记改为 optional；空字符串与其他未知值同样遵循默认拒绝/显式 lenient 契约。 |
| 旧模板实例化少了新参数 | 首次编译发现 `StringConstraintState` 构造参数不匹配 | 使用上游新状态类，并传入最长 token 长度。 |
| 原生 NInfer 正常停止锁存故障 | 旧集成和新上游的 `fail_all_locked` 都在主动停止时设置 `failed_`，HTTP watch 因此偶发以退出码 2 关服；Engine.stop 后延迟检查可稳定复现 | 区分故障锁存与资源清理。正常停止不新增故障标志，真实故障仍锁存，既有故障不会被清除。修复后 C2 延迟停止检查及完整 HTTP 退出码 0 验收通过。 |

以上为本轮接入修正，没有据此改变原 KVMem/llama.cpp 的可疑旧行为。

## FP16 GDN 自适应窗口的数值边界

两路测试使用 FP16 GDN、自适应 MTP4 时，冷请求和重放在第29个原始 token
选择了 EOS 和逗号。冷运行对应 logprob 为 -0.654663/-0.779663，
重放两者均为 -0.715805，处于同分边界。Graph 和 eager 都能复现；
关闭 MTP、改用默认 FP32 GDN、或使用固定 MTP4 后，同配置重复输出一致。
页和 GDN 状态的恢复字节、输出预算、并发执行及资源回收检查均通过。

旧、新源码的 `run_recurrent_sequence` 完全相同：寄存器保留 FP32 递推，
在窗口结束时将状态存为 FP16。改变窗口会改变舍入边界，字节相同的 checkpoint
不保证后续不同计算形状具有相同分布。普通解码与 FP32 路径提供了独立参照，
所以本次没有以旧 token 序列为裁判，也没有为匹配旧输出改变数学或缓存算法。
默认 FP32 保留更多状态精度，用它完成核心并发资格检查更合理。

两条输出的数字正文都符合开放式提示。测试刻意关闭模型默认停止符并强制48个
原始 token，EOS 后出现的特殊标记不等于正常产品输出发生破坏。
目前记录为有源码与分布证据支持的数值边界，未确认缓存恢复逻辑缺陷；
FP16＋自适应 MTP 的逐 token 重放一致性不在本次通过范围内。
本轮观察和原生停止修复也已记入工作区 `backend-p0p1-20261002/legacy-findings.md`
的 L019/L020。

## 尚未覆盖的硬件与模型范围

没有 RTX 20、TITAN RTX 或 V100 实卡，没有双 GPU 流水线资格验证；
没有完整 Gemma 4/Flash-Next 模型推理或真实视觉模型资格验证。
Gemma 工具语法/解析及新家族拒绝边界分别通过 CPU 测试和 metadata-only fixture 检查，
不等同于对应模型的 GPU 正确性验证。
历史文档中的长上下文、所有 KV 格式、SSD 故障注入及八路图文全覆盖结果，
仍只代表各自记录的旧基线，不能转记为本次升级结果。

## 复现

固定基线、补丁 SHA-256 和 patched tree 由 `backends/versions.json` 记录。
准备源码后运行 `python scripts/prepare-backends.py --backend ninfer --check`，
验证补丁以及含新增文件的完整后端树。
本次已在独立干净检出验证初次应用和重复准备的幂等性。
使用 `NINFER_TEST_LOW_VRAM=1` 可启用上述实验测试配置；生产默认值保持不变。

Windows 正式构建入口支持显式选择目标和 Toolkit（依赖与 vcpkg toolchain 通过
既有 `-CMakeArgs` 传入；不要混用同一个 build 目录）：

```powershell
./scripts/windows/build-backends.ps1 -Backend ninfer -BuildRoot ./build-turing `
  -CudaArchitectures 75 -CudaPath 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2'
./scripts/windows/build-backends.ps1 -Backend ninfer -BuildRoot ./build-v100 `
  -CudaArchitectures 70 -CudaPath 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9'
```
