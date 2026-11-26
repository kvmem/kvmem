# ninfer-all 63428ac2 接入记录

本页记录 2026-10-26 的候选版本。当前控制台和启动器设置见 [NInfer 张量并行](multi-backend.md#ninfer-张量并行)。

日期：2026-10-26。版本：0.19.0。状态：完整构建、最终 fork 的核心功能、独立
数值检查和三种 KV 的受控性能比较通过。同步 KVMem master `47cee454` 的
checkpoint-policy 与 monitor-events；发布包仅包含 NVIDIA 50 系 ninfer 后端。

## 顺序和固定版本

1. 先在 ninfer-all `63428ac249da8a6c5d891b508d7e483d8c7def6c` 上移植现有单卡能力。
2. 单卡达到验收条件后，适配 KVMem 与上游 TP 的组合。
3. 最后移植 NVFP4 fork，单独记录来源、改动和性能结果。

第一阶段依次保留 `kvmem`、`ptq1`、`ptq1-performance`、`t5-performance`、
`query-memory` 五层补丁。非 fork 验收二进制已单独保存，随后加入 NVFP4 fork 三层、
master 的 checkpoint-policy/monitor-events 两层和本地 Blackwell scale 修复，共十一层。

用户于 2026-10-25 最终澄清：本轮 ninfer 只适配 Qwen dense。之前提到的
35B-A3B MoE 限定已撤回。单卡核心功能使用本地 Qwen3.5-4B，单卡性能及 TP
使用 Qwen3.8-27B（Qwen3.5-core dense 架构）。其他模型不做专项迁移或验收，
共享构建所需的最小兼容修复仍保留，不据此声称其他模型可用。

## 用户明确延期的上游优化

### 共享 prefill 批处理

启用 KVMem 时，仍接受普通请求组并按已有逐 lane 路径执行，不把多个请求
放进上游新增的共享 prefill pass。Query 捕获、Pressure、回放、取消和错误清理
继续经过现有路径。未启用 KVMem 时保留上游共享 pass。

上游 branched reads 依赖共享 pass，因此 KVMem 暂时报告不支持该能力：
HTTP 路径采用已有普通请求回退；直接调用该 API 时明确拒绝。
这可能影响并发短 prompt 吞吐和分支 rerank 性能。

### 首 token 提前发布

KVMem 暂时等待完整 prefill 执行完成，再发布首 token；未启用 KVMem 时保留
上游等待 token 事件、让初始 MTP draft 在后续完成的行为。后续需要共同适配
Query/统计读取、初始 draft 生命周期及取消边界，才能放开提前发布。
这可能影响启用推测生成时的 TTFT，不能把它混在 decode 吞吐中隐去。

以上两项是 2026-10-25 用户指定的延期，不是已经验收完成的优化。

## 验收要求

- 最核心的功能覆盖：普通生成、prefix 复用、Query/Pressure 和取消后的恢复。
- 同一 GPU、模型、激活 KV 量、chunk、graph 和推测设置，对比原版上游。
- 至少检查 INT8、RK8V4、NVFP4 KV；分别报告 PP、TG、TTFT，不以输出逐字相同
  作为唯一正确性依据。
- 单卡性能比较固定使用 RTX 5060 Ti，运行前检查占用。
- 用户于 2026-10-25 明确授权 TP 测试同时使用 RTX 5060 Ti（GPU-A）
  与 RTX 5050 Laptop（GPU-B）。原始本地 metadata 保存实际 UUID，公开文档使用别名。
  运行前检查两卡占用；
  优先以 5060 Ti 为 rank 0。双卡结果单独记录，不混入单卡受控比较。
- TP 需要独立记录真实多卡验证；单卡编译或测试不能当作 TP 验收。
- NVFP4 权重量化算子与 NVFP4 KV 格式分别验证，不互相代替。
- 实际合入 fork 时保留原作者署名：
  `Co-authored-by: 13thblade <333595421+13thblade@users.noreply.github.com>`。

## TP 使用边界

本轮 TP 入口是原生 `--devices 0,1 --tensor-parallel`，或 Windows 快捷脚本的
`-Gpu 'GPU-uuid0,GPU-uuid1' -NoVision -WorkerArgs '--tensor-parallel'`；原有脚本
DryRun 已确认能正确映射设备顺序和转发 TP 开关。不要同时传 `-StageLayers`。
这里的 `-NoVision` 仅用于本轮纯文本性能对照，不是 TP 的通用要求。
上游 `load.cpp` 明确拒绝的是 `vision-residency overlay` 与多设备组合；常驻 GPU
Vision 和本地 CPU Vision 没有此门禁。普通视觉 embedding 在进入 TP 层前写入
输入残差；这两种 TP 视觉组合尚未实测，不能据纯文本结果声称已验收。
原 KVMem ninfer 的 Engine/serve 参数门禁和 Windows 启动脚本本来就只接受
resident/CPU Vision，未接入 overlay；此次保留该边界，不减少原有视觉执行模式。
模型仍需满足上游的
Qwen3.5-core dense、27B 几何和 GGUF 分片权重条件；MTP 可用，DFlash 不在支持范围。
本轮没有新增控制台 TP 配置控件；控制台现有多卡描述仍针对 layer pipeline。

## 当前证据

五层补丁的文本冲突已解决并导出，公共类型迁至上游拆分后的头文件。
已修复试合并中 TP 控制数组声明 7 项、仅初始化 6 项而造成的空指针解引用。
完整构建暴露的 `DevicePlacement` 聚合初始化字段变更、Flint 的
`computed_prompt_tokens`、Flint 依赖的上游 `GrammarMaskCost` 辅助类，以及
新增 decisions HTTP 调用遗漏的 prepared 参数均已修复。不改变 Qwen 已有的
草稿策略。`ninfer-serve.exe` 和 `ninfer_bench.exe` 完整构建通过，使用
CUDA 13.2 / sm_120a / MSVC 19.44 / Release，8 个编译任务。
五层补丁已用独立 Git 索引从固定上游逐层重建并核对树值；修复归入第一层，
最终树为 `d2fcb9daf6efbb6d0289ac19b825a92fc0b177b5`。

2026-10-25：上游原始 `tests/test_tensor_parallel.cu` 通过独立入口运行，复用
当前 `ninfer_core`，CUDA 13.2 / sm_120a / MSVC 19.44。启动前两卡均无计算进程、
显存占用为零；可见顺序为 5060 Ti、5050 Laptop。CPU 参考比较、2/3/4 个逻辑
rank 在单卡及真实双卡上的 eager 归约、强制 host 传输和 CUDA Graph 重放均通过。
3/4 rank 是复用两张物理卡，不是三卡或四卡测试。外部原始记录：
`tp-634-exchange/metadata.json` 和 `run.log`。

随后完成整模型核心回归：

| 模式 | 模型 | INT8 KV | RK8V4 KV | NVFP4 KV |
| --- | --- | --- | --- | --- |
| 5060 Ti 单卡 | Qwen3.5-4B MTP Q4_K_M | 67/67 | 67/67 | 67/67 |
| 5060 Ti + 5050 Laptop TP | Qwen3.8-27B GSQ-RCO IQ3_S | 65/65 | 65/65 | 65/65 |

上述检查覆盖文档先行、独立问题请求、prefix 复用、bounded Query/replay、
工具续接 resident-only Pressure、会话切换恢复、重写、流式取消恢复，并启用
MTP 2、CUDA Graph 和原生 KV/StateImage 传输验证。原始请求、响应、检查和
启动元数据分别保存于 `single-634-core/<kv>/` 与 `tp-634-core/<kv>/`。
运行期间存在 CPU 编译任务，因此这些结果只作功能证据，不作性能结论。

原版 63428ac2 基线已独立构建通过；仅增加 bench 阶段计时观测，不改变执行引擎
或算子。同一 5060 Ti、Qwen3.8-27B GSQ-RCO IQ3_S、8192 prompt token、
128 计时 decode token、KV capacity 9216、chunk 256、Graph、MTP 4、ngram 0
下完成三种 KV 的原版与非 fork KVMem 对照；测量期间不运行编译。
各预热一次、测量两次，使用同一内置设备 profile；KVMem B8192/R1024、
H12288 MiB、单会话。TG 为包含主机开销的 wall 吞吐。

| KV | 原版 PP tok/s | KVMem PP tok/s | 原版 TG tok/s | KVMem TG tok/s | 原版 TTFT s | KVMem TTFT s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| INT8 | 782.61 | 761.09 | 88.19 | 87.56 | 10.468 | 10.765 |
| RK8V4 | 782.25 | 757.92 | 88.29 | 87.66 | 10.473 | 10.810 |
| NVFP4 | 760.06 | 754.83 | 87.13 | 86.51 | 10.779 | 10.855 |

三种 KV 的 wall TG 均下降约 0.7%；PP 分别下降 2.7%、3.1%、0.7%，
TTFT 分别增加 2.8%、3.2%、0.7%。此处只有 8K、两次重复的核心检查，
不推断其他长度或统计显著性。原始记录位于 `single-634-performance/`。

同一双卡组合另外完成 NVFP4 KV 的 HTTP `/completion` 对照，模型和 KV
容量、chunk、MTP 与上面一致，但 profile off，因此不与单卡直接比较。
每次请求禁用 prompt cache，确认实际读取 8192 token、缓存命中 0、输出 129。
预热一次、测量两次：原版/KVMem PP 为 410.97/410.23 tok/s（-0.18%），
服务端 prompt wall 为 19.933/19.969 s（+0.18%），wall TG 为 62.05/65.56 tok/s。
TP 请求为非流式，未测客户端首 token 到达时间；历史 summary 的 `ttft` 字段
取自 `prompt_ms`，只代表上述服务端阶段耗时，不包括排队和 HTTP 发布开销。
原版两次 TG 为 67.27 和 56.84，KVMem 为 64.96 和 66.16；原版波动较大，
不将平均值差异解释为 KVMem 加速。记录位于 `tp-634-performance/`。

原版首次构建因共享实验目录中的 FFmpeg/zlib 依赖消失而中断，已保留失败日志并
修复路径生命周期：依赖迁入独立 `build-deps/ninfer-x64-windows-c7684b1d96347de8`，
379 个文件固定摘要，7 个运行 DLL 与非 fork 候选完全一致；Python 3.11.15 也迁入
稳定工具目录。三个构建入口及验证脚本统一读取本任务的 `build-dependencies.json`，
构建前验证完整依赖，BuildOnly 拒绝失效 CMake 路径。缺失文件、内容变更、清单变更
及旧缓存拒绝检查通过；重新配置后复用已有 CUDA 对象完成构建，没有改变基线算子。

## NVFP4 fork 最终验收

fork 固定为 `c02c7692bd10a9b4a7effef67b60f58673985aa6`。A16 和 INT8 两层
按作者在该版本中的 TP 冲突解法移植；纳入的 130 个文件与作者快照逐字节一致。
非 fork 二进制及 manifest 保存在 `nonfork-634-binaries`。本地兼容层保留 Blackwell
原生 KV codec，并约束环境变量作用域、默认路由和 StateImage 模式绑定。master 的
checkpoint-policy 在新的 prefill 生命周期中捕获，monitor-events 沿用当前 master
的原生请求计数与耗时语义。最终十一层树值：
`2d5f335df7a9ad2bf0727ece037e06782edbba2a`。完整来源、各层 SHA-256 和树值见
`backends/versions.json`；`prepare-backends.py --backend ninfer --check` 通过。

运行 A16 独立数值参考时发现：作者软件 UE4M3 权重 scale 表把 `0x7F` 映射为零，
Blackwell 原生 E4M3 转换却产生 NaN。仅在 unified 权重 pair/quad 解码入口归一化
此值；NVFP4 KV codec 未改。新增独立 FP64 E2M1/E4M3 参考，穷举 256 个 code byte
与 128 个非负 scale 的 32768 个组合，逐位比较 BF16 pair/quad 结果，全部通过。
失败记录保留在 `fork-634-qualification/`，修复后记录在
`fork-634-scale-oracle/`、`fork-634-qualification-fixed-scale/`。

最终构建的 A16、INT8、fused norm、SwiGLU、add 五项算子参考全部通过；默认、
INT8 legacy/unified、A16 legacy/unified 五个独立进程的路由检查和 native snapshot
模式隔离通过。checkpoint pool、Qwen3.5 frontend、serve options、OpenAI schema
四项运行契约通过。单卡三种 KV 各 67/67，真实双卡 TP 的 NVFP4 KV 65/65；
原始记录分别位于 `single-634-fork-core/<kv>/`、`tp-634-fork-core/nvfp4/`。

最终 fork 在前述完全相同的单卡性能设置下，预热一次、测量两次：

| KV | 最终 PP tok/s | 相对原版 | 最终 wall TG tok/s | 相对原版 | 最终 TTFT s | 相对原版 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| INT8 | 760.58 | -2.82% | 87.59 | -0.68% | 10.772 | +2.90% |
| RK8V4 | 757.98 | -3.10% | 87.74 | -0.62% | 10.809 | +3.21% |
| NVFP4 | 754.22 | -0.77% | 86.63 | -0.58% | 10.863 | +0.78% |

三种 KV 均未达到本轮 5% 的回退调查阈值。此为 8K、两次重复的核心检查，不代表
全部长度或统计显著性。与冻结非 fork 候选的差异很小。最终原始记录位于
`single-634-performance/<kv>-fork/`；前述 TP 性能表为非 fork 阶段，最终 fork TP
另做功能验收，没有以该表声称最终 fork TP 加速。

没有本地完整 NVFP4 权重模型，因此未声称其整模型性能提升；GGUF 模型的
NVFP4 KV 功能/性能不能替代 NVFP4 权重模型验证。

NVFP4 KV 已有上述单卡/TP 功能证据；这不等价于 NVFP4 权重算子验收。
上游 TP 仍要求受支持的 GGUF 权重，fork 没有解除 NVFP4 权重的 TP 限制。

## 可选 checkpoint policy 的边界

此功能默认关闭，只有 `KVMEM_CHECKPOINT_POLICY=spread` 才启用。本轮以普通文档
请求后单独提问验证。默认 12 个 checkpoint、1024 MiB host 预算下，MTP 3、
Responses、interval 256 的核心回归通过；中段编辑复用 1408 token，前段编辑复用 26。
6 个 checkpoint、256 MiB 的专门预算压力检查也通过：计费有界，缺少空间时跳过
可选捕获，后续请求可恢复。记录在 `checkpoint-634-default-count/` 和
`checkpoint-634-budget-pressure/`。

另保留两个诊断失败：6 个 checkpoint 配 256/1024 MiB 的普通复用测试，经过多次
续接后，中段编辑分别只复用 0/26 token，未达到测试期望。字节预算或计数预算不足
时，近期端点可能挤走合法恢复前沿内的中间点；这是现有策略允许的重算回退，未将
其冒充已修复缺陷，也未降低普通测试断言。小预算仍可能增加重算耗时。
