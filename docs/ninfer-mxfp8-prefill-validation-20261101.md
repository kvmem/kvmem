# GGUF MXFP8 prefill：当前实现、补丁边界与验证

> Publication update (2026-11-03): This default-off native candidate includes the measured device routes and equivalent image stores. Physical device identities/private paths are anonymized; measurements, hashes and adverse results are retained. Statements about uncommitted local builds below describe the dated experiments, not this publication. These are validation artifacts, not an installed release. Raw evidence remains local.

更新于 2026-11-03。本轮已实现 NInfer 后端按设备、格式、N/K 和输出类型选择
FP8 tile 或 Q8 matrix，并加入原生校准器及 5060 Ti 的 25 条 T=256 内置规则。
未匹配形状仍用 256≤T<1024 的融合 FP8 tile，T≥1024 保持 64 MiB 有界 Lt。
默认关闭，仍属于有损量化选项。FP4 的进一步实验暂停。

## 2026-11-03：补齐原仓库的 128-bit 权重图像写回

本轮先回到源码与实际 SASS 比对。确认原仓库 `image_kernel` 每个32字节切片用两条
`STG.E.128`，本地移植却是八条标量 `STG.E`；两者另各有一条尺度写回。已恢复两个
对齐的128-bit写回，不改变值的解码、实际amax、FP32乘法边界、E4M3编码或累加。
IQ3_S 的 SASS 寄存器数原仓库30、本地旧版48、修复后47；旧/新均 STACK=0、LOCAL=0，
共享内存同为3072字节。没有加入新查表、模型副本或工作区分配，64 MiB图像上限不变。

原仓库还使用基于格式上界的尺度、折叠尺度乘法、直接BF16输出与BF16 gate/up中间值，
本地则保留实际amax和FP32中间值；这些不是完全相同的数值路径。普通前缀缓存仍是
本地hybrid、原仓库legacy。不能把全部差距视作同一FP8内核的调度差距，也不能为追平
直接取消这些数值或缓存边界。本次只修复不改变语义的向量写回遗漏。

同一5060 Ti UUID `GPU-A`、原27B artifact、
int8 KV、MTP4、chunk1024、并发1；各臂两次镜像进程复测，KV capacity/max-context
均25600，KVMem B24576/R1024。与此前组分开报告，本轮的新旧比较如下：

| 路径 | 旧版两轮 tok/s | 修复后两轮 tok/s | 旧均值 → 新均值 | 增幅 |
|---|---:|---:|---:|---:|
| 普通 NInfer FP8 | 1377.819 / 1377.567 | 1396.669 / 1393.013 | 1377.693 → 1394.841 | +1.245% |
| 预算内 KVMem FP8 | 1307.119 / 1307.759 | 1324.166 / 1323.032 | 1307.439 → 1323.599 | +1.236% |

本轮原仓库普通FP8为1457.172 / 1458.711，均值1457.941 tok/s；本地普通路径差距
从5.504%缩小到4.328%，尚未追平。没有把上轮KVMem的1280.335均值拼入本轮收益。
每轮均为14806个新prompt token、cache0、本地native `no_pressure`。文档先行短确认，
同会话另发真实问题，保存实际前缀复用和native计算span；十轮问题输出均相同，属于
同文档的相关行为探针，不是新增独立质量分数。数值绑定仍为既有profile-v2签名。

prefill阶段每秒采样的总卡显存最大值：普通路径旧/新均13806 MiB，KVMem旧/新均
13652 MiB。它不是allocator精确峰值；源码中的工作区容量公式和显式分配也未增加。
本轮只覆盖此卡/模型/1024，每臂两次；不宣称decode、其他设备或并发收益。

资格验证在新旧两套链接上通过：独立CPU GGUF解码/E4M3枚举，8格式×3组形状，
逐字节比较23,330,816个图像字节（含padding）及729,088个尺度；覆盖零、负零、正负
最小FP16 subnormal、最大有限值与真实K。既有公开Linear/LinearAdd/SwiGLU、混合Q8
路由、独立FP64、gather、Graph、行列尾部及canary也通过。该改动只重排同一编码字节
的写回，不新增量化误差；FP8本身仍有损，之前的工具格式反例未被本轮修复或重测。
公共Linear的三格式×三真实N/K形状另做ABBA：全部录得正收益，范围+0.593%–2.520%；
这些Op成绩不代替上面的模型收益。未因编码完全相同而重跑整套PPL。

保留的不利实验与失败：首版精确查表通过数值资格，但一次进程对照中IQ3公共Linear
下降0.555%–4.513%，IQ4仅增0.174%–0.868%，没有采用；四线程协作草案只完成编译，
在用户要求先审底层后未继续资格/计时，也未纳入补丁。聚焦链接首次缺少VS库环境
而找不到delayimp.lib，补齐后通过。附带续问检查曾把有效14799-token复用误判为
不满14800；原生证据是14799复用+44新计算=14843，现按实际native/HTTP span核对，
原失败探针保留，新实验使用独立目录且没有混入该样本。

构建每轮仅编译必要的一个CUDA对象，首次/最后一次另编主机测试对象，8 jobs；独立
链接图复用同构建既有库，基线服务和原FP8库哈希保持不变，没有重编整个引擎。
交付为第34层 `ninfer-gguf-mxfp8-image-stores.patch`，前33层字节不变，完整清单准备/
回放通过，最终tree `10f662d686ca7a92d0d2e0044d840ffd7a84162f`。真实Git索引未变。
原始证据在实验根 `fp8/image-opt-20261103`：`vector-opbench.json`、`qualify-vector-*`、
`performance-vector/analysis.json`、SASS/resource、构建计划及`export.json`。
本轮为本地验证构建，未安装、提交或推送，PR250保持原快照；所有测试服务已退出。

## 2026-11-03：测试脚本修复及 1024 原仓库对照

修复两处审计确认的实验脚本缺陷：结果仅在服务上下文正常退出（包括收尾 GPU
冲突检查）后提交；续跑严格校验不可覆盖的实验配置，绑定模型、文档、二进制、
GPU UUID、参数、环境、profile、依赖和脚本内容。结果原子写入，旧的无验证记录
不作为新实验的成功样本。8 项主机回归及 Python 3.11 语法检查通过。首次实际续跑
发现镜像计划的 tuple 经 JSON 变成 list，导致原样配置被误拒绝；现已在比较前统一
为 JSON 表示，并新增实际 round-trip 回归，保留失败日志。

另确认旧 `fp8/on-off` 队列在显存释放后因参数错误退出：普通路径显式容量 25600
小于 max-context 32768，服务报 `--kv-capacity must be at least --max-context`。
旧队列没有成功样本，并非仍在等待显存。新测试将三臂的 max-context 和 KV capacity
均设为 25600，KVMem B24576/R1024；文档 14806 token 全在预算内。

新版 FP8 开关脚本为 `fp8/on_off.py`，使用独立的 guarded 结果目录。新增
`fp8/prefill_1024_compare.py` 复用已编译、固定哈希的原仓库与当前二进制，比较原仓库
FP8、本地普通 FP8、预算内 KVMem FP8，三臂各两次镜像进程重复。固定 int8 KV、
MTP4、ngram0、并发1、真实 chunk1024、空用户 profile；短请求和独立 system 的
5018-token prime 后测原文档，要求零 prefix reuse，并核对 native 实际 chunk、
容量、计算 token 数及本地 no_pressure。本轮仅测 prefill，不作质量或 decode 结论。

此前 RTX 5060 Ti UUID `GPU-A`
被驱动报告为 `GPU is lost. Reboot the system to recover this GPU`，两次启动检查
在空闲检测阶段失败，未启动模型服务，也未保存成功速度样本。用户恢复显卡后，
同日继续并完成以下六轮；5050 未参与比较，没有更换模型或混入旧失败尝试。

| 路径，均开启 FP8 | 两轮 prefill tok/s | 均值 | 相对原仓库吞吐 |
|---|---:|---:|---:|
| 原仓库，固定提交 08c4fb70 | 1445.960 / 1458.600 | 1452.280 | 基线 |
| 当前后端，关闭 KVMem | 1376.298 / 1379.462 | 1377.880 | −5.123% |
| 当前后端，预算内 KVMem | 1308.167 / 1252.503 | 1280.335 | −11.840% |

KVMem 相对同一当前后端普通路径吞吐低 7.079%。原仓库/普通/KVMem 的平均文档
prefill 时间分别为 10.195、10.746、11.570 秒。相对原仓库的两轮配对差距：普通
4.818% / 5.426%，KVMem 9.530% / 14.130%。KVMem 两轮波动明显大于另两臂，
保留全部样本，不将较慢一轮排除或未经验证地归因于噪声。

六轮均确认真实 chunk1024、实际 KV capacity25600、零缓存命中、14806 计算 token，
文档回应均为 2 token 的 Received。本地四轮均为 native `no_pressure`，两轮 KVMem
的文档全在 B24576 内，不能用超预算冷历史检索解释这组差距；具体开销来源未在本次
计时中定位。启动前检查空闲，运行监测未记录其他计算进程冲突；结束后本任务服务
全部退出，5060 Ti 为 0 MiB、0% 利用率、无计算进程。

实验协议、每轮请求/响应、native 日志及最终 `analysis.json` 保存在
`fp8/upstream-1024-20261103/validated`；首次配置及续跑误拒绝记录保留在父目录。
仅限此卡、此 27B 模型和预热服务的两次进程重复，未新增质量、decode 或功耗结论。
本次没有原生编译、内核或补丁链变更，没有提交或推送。

### 1024 普通路径差距分析与可行方向

1377.880 → 1452.280 tok/s 需要本地吞吐提升 5.400%；正式对照的额外 prefill
时间为 550.603 ms/文档。原生日志中本地 engine boundary 和 program.post 的均值
合计约 7.431 ms，原仓库约 0.282 ms；这些前后处理无法解释大部分差距。
program.submit 包含 CUDA 提交及阻塞，不能把其整段时间当成纯 CPU 成本。

关闭 KVMem 不会恢复原仓库的整个普通路径：正式运行的 native metadata 确认
本地是 hybrid 前缀缓存，原仓库是 legacy，设备状态槽、checkpoint 发布策略及
Host tier 配置也不同。因此 5.123% 是整个普通后端的差距，不是已隔离的 FP8 Op 差距。

补抓相同模型/卡/1024 的普通路径诊断，两边 Host cache 均为 5120 MiB，沿用固定
二进制、实际最大值 FP8 与原仓库各自的数值实现；每边一次完整文档记录：

| 已记录活动累计耗时，ms | 当前普通路径 | 原仓库 |
|---|---:|---:|
| FP8 GEMM | 6066.315 | 6491.835 |
| GGUF → FP8 权重图像 | 1283.873 | 689.170 |
| 输出 store-plane / gated epilogue | 758.074 | 91.932 |
| prompt attention | 992.379 | 1010.360 |
| FP8 激活转换 | 182.066 | 169.097 |
| D2H 拷贝 | 168.983 | 39.640 |

D2H 记录量为本地 659,144,724 B、原仓库 153,954,324 B。这与不同的缓存发布实现
同时出现，未以单因素实验确认每笔拷贝的必要性，不能称为已确认的冗余或缺陷。

限制：两条跟踪均有 CUDA/NVTX 事件可能未全收集的警告，Nsight 还提示用 13.2
库追踪 13.4 驱动；两次诊断的 Host cache 上限与正式测速不同。以上仅为记录到的
活动累计，支持优化假设，不能相加解释 550.603 ms 或推算可追回的净收益。
正式测速均值保持不变，不混入 profiler 耗时。诊断原始资料在实验根
`fp8/upstream-1024-20261103/profile-host5120-60s`，结构化分析为同层 `diagnosis.json`。

建议按下列顺序做独立 A/B，均属于 NInfer 后端 patch：

1. 优化 IQ3_S、IQ4_XS、IQ3_XXS 等格式的 GGUF→FP8 转换。这三类占本地记录图像
   转换约 1.013 秒。保留解码 FP32 值、每32值实际 amax 和现有 E4M3 舍入，验证
   特定格式的编码/查表方法；不照搬原仓库的尺度上界、Q4_K FP16 中间值，或已被
   实测否定的通用整数查表草案。原始 LUT 更快的原因尚未与 slab/L2 布局分离。
2. 优先在输出本来就是 BF16 的普通 Linear Store 边界尝试融合，减少 FP32 plane
   的额外读写；另行评估保留 FP32 的 gate/up 合并计算。两边 GEMM 均 FP32 累加，
   但原仓库 gate/up 会先舍入到 BF16 再激活，本地保持 FP32，直接移植会改变数值。
   758 ms 也不代表全部可删除；需独立 Op oracle 和模型质量验证。
3. 检查普通 hybrid 缓存的 checkpoint 发布与部分/增量拷贝，保留缓存正确性与复用
   合约，不能用关掉缓存发布来宣称产品优化。确认继承缺陷时继续遵循先记录并讨论的规则。
4. 最后按真实 N/K/输出类型校准 slab 和 Lt 算法。本地图像上限64 MiB、原仓库10.5 MiB，
   本地记录到的 GEMM 调用8484次、原仓库39410次；不能简单复制较小 slab 或认为
   主乘法必然更慢。新增25条路由只覆盖T256，不优化主T1024 Lt产品；PDL两边均启用。

未改计算路径或发布配置，以上收益均未实施或经端到端 A/B 证明。诊断失败全部保留：
最初默认8 GiB本地Host tier因页锁内存不足退出；随后4 GiB本地已完成请求但脚本
以60秒等待90秒采集导致收尾超时，已改成有界短轮询；原仓库4 GiB又因15份状态
镜像占用超过预算一半而拒绝启动。最终同设5 GiB的两条记录完成，前述失败未混入。
自动审批曾因连接中断未执行一次命令，正常重提同一审批后成功，没有绕过审批。
所有诊断服务已退出，5060 Ti检查为0 MiB、0%、无计算进程。

## 上轮设备选路结果（第33层补丁）

同一张 GPU-A（RTX 5060 Ti）、原始 Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp、
int8 KV、MTP4、真实 chunk=256、无 prefix cache 命中。文档为 14,806 token；
KVMem B=24,576、R=1,024，文档与回答均在预算内，native 停止原因为 `no_pressure`。
同一候选二进制只替换新增 FP8 profile 键，保留其他设备调优；每臂两次镜像进程复测。

| 路径 | 旧 tile tok/s | 新选路 tok/s | 增幅 |
|---|---:|---:|---:|
| 普通 NInfer FP8，KV capacity=32768 | 918.899 | 934.820 | +1.733% |
| 预算内 KVMem FP8，capacity=B+R=25600 | 881.646 | 902.832 | +2.403% |

固定原仓库 FP8 在相同普通后端配置为 969.816 tok/s；普通本地差距从 5.250%
缩小到 3.609%，尚未追平。两种容量的普通/KVMem 横向差额不能全部归因于集成开销。
最后一次旧普通臂与一次 CMake configure 有重叠，未并行编译或运行其他 GPU 任务；
最终二进制的单次复核单列，不混进上述均值。
最终服务二进制 SHA256 为 `de54e2aaf4b51ba2c4b4b1cc5c451747dc23aba8d51bd6e0b5e5229447dde909`。
单次抽查：旧普通 tile 914.716、新普通内置选路 934.622、预算内 KVMem 内置选路
894.601 tok/s；KVMem 单次结果低于前述镜像均值，存在进程间波动。没有将单次结果
并入均值。此前把普通臂的 32768 容量参数误用于 KVMem 的启动被参数校验拒绝，
没有产生速度数据，保留失败记录后改回合法 B+R 配置。

两轮独立校准各测 8 格式×10 形状×4 epilogue，共 320 键。使用 11 次中位数、
3% 候选门槛、交错确认和既有 5% 输出峰值差异保护，保留两轮都胜出的 25 个 Q8 键。
其中 16 键为 IQ4_XS，其他键主要是 N=1024 的较小输出；大部分组合中 Q8 更慢，
保留 FP8。规则只覆盖测过的 T=256，不外推到 257/384/512，也不套用到其他卡。

选路改变数值路径，因此本轮专门测试 context=257，即实际 256 个预测列。
六域冻结留出语料共 127,744 scored tokens / 499 个不重叠窗口：

| 模式 | PPL | 相对 Q8 | 最差域增幅 | 最差窗口增幅 |
|---|---:|---:|---:|---:|
| Q8 | 9.810739 | — | — | — |
| 旧 FP8 tile | 9.850831 | +0.4087% | +0.7274% | +5.7071% |
| 新 FP8/Q8 选路 | 9.843478 | +0.3337% | +0.5343% | +4.5797% |

两者均通过预先确定的总体≤1%、每域≤3%门槛，新选路 PPL 比旧 FP8 低 0.0746%。
这是语料层面的通过，不能承诺任意输入无损；已有 max128 工具文档格式反例仍保留。
旧 513/2048 窗口数据在下文，属于此前实现验证，本轮未重跑。
评分完成后仅补充 host 侧旧 catalog 兼容分支及保持 Turing catalog 不变；受测 sm120
当前 catalog 路径、Op 对象和 25 条规则未改变。最终二进制另做原生、问答和速度复核。

本轮长文本回归严格使用文档先行、同会话另发问题：旧 tile/新内置选路各 3 个会话，
24/24 个事实正确，实际 prefix reuse 为 21,257、27,371、29,395 token；检查了 question
turn 的 native Query 范围。每文档 4 个事实属于相关探针。请求、响应和两阶段耗时
均保存在 `fp8/route/recall`，不能外推为 256K 或无限长度质量通过。

工程验证包含独立 CPU GGUF 解码/FP64 原始权重点积、真实 N/K 抽样、四种输出、
gather、混合 Q8/FP8 工作区、canary、Graph、非法/缺省 profile、宽度和部分行边界。
双卡测试验证设备选择互不串用、每设备 shared-memory 初始化、rank 顺序签名和
调用设备恢复；不代表 TP 整体资格通过。新增 catalog 的兼容迁移保留未变更的旧
用户校准键，旧 provenance 不能授权新 FP8 键；任意其他 catalog 不匹配仍拒绝。

成本与限制：最大工作区仍保守预留 Q8+FP8 空间；混合产品可多做一次激活格式转换，
不新增模型副本。FP8 数值签名升级到 `mxfp8-e4m3-e8m0-f32-profile-v2`，逐 rank
绑定选路规则，旧 FP8 状态/成本缓存失效；默认 Q8 数值签名不变。本轮不宣称 decode
提升；1024 新版与原仓库的 prefill 对照已补测，见上文。选路原始记录在实验根 `fp8/route` 下的
`analysis.json`、`selection.json`、`quality`、`speed`、`mixed-ops.log` 和 `two-device.log`。

本轮作为独立的第 33 层后端补丁交付，前 32 层字节不变；本地已验证完整补丁链。
GitHub PR #250 保持上一轮上传的快照，本轮未提交或推送。

## 当前后端补丁边界

当前清单以 [versions.json](../backends/versions.json) 为准，规则见
[多后端维护约定](multi-backend-repository.md)。旧报告中的 tree/hash 属于当时补丁划分。

| 补丁 | 所有权和内容 |
|---|---|
| `ninfer-5060-gguf-coop.patch` | 设备 profile；保留已验证 Q4_K/Q6_K 选择，不新增 FP8 选路 |
| `ninfer-gguf-mxfp8-prefill.patch` | 通用 FP8 计算、融合 tile、64 MiB Lt、数值签名、独立 CPU/FP64 测试 |
| `ninfer-gguf-nvfp4-prefill.patch` | NVFP4 计算、显式 A4 权限、FFN producer、测试；进一步 FP4 测试暂停 |
| `ninfer-kvmem-statistics-readback.patch` | KVMem 原生 Mean-K/Query 合并拷贝、注意力反馈批量回读和布局/字节校验 |
| `ninfer-gguf-mxfp8-device-routes.patch` | 原生校准、逐设备/格式/形状/epilogue FP8/Q8 选路、数值绑定和旧 profile 兼容迁移 |
| `ninfer-gguf-mxfp8-image-stores.patch` | 恢复权重图像128-bit写回；编码字节、数值边界、工作区上限不变 |

FP8 和 NVFP4 计算在关闭 KVMem 时也可使用。FP8 测试先引入共享 CPU GGUF oracle，
NVFP4 测试复用它；FP8 基础计算层不依赖后续 NVFP4 层或 KVMem 回读层。第 33 层
增量按完整清单应用，包含共享 dispatcher/signature 文件的上下文，不启用 FP4。性能/快照数值签名跟随
计算模式维护。前端、启动器和协议中不增加显卡→CUDA kernel 的选择规则。

## 最新 FP8 正确性与质量

原模型 Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp，权重不重转换，int8 KV。
独立 CPU packed-GGUF 解码、CPU E4M3 编码和 FP64 乘积覆盖八格式、实际 K、
gather、tails、signed/zero/subnormal/max scales、FP32/BF16 epilogue、Graph 和大 slab。
本轮已补充双 GPU 的初始化及选路隔离测试；TP 整体资格仍未完成。

固定独立六域留出语料，语料先于当前结果选定，使用相同原始模型及 Q8/FP8：

| 上下文 | scored tokens / windows | Q8 PPL | FP8 PPL | 总体增幅 | 最差域增幅 | 最差窗口增幅 |
|---:|---:|---:|---:|---:|---:|---:|
| 513 | 127488 / 249 | 7.841810 | 7.872204 | +0.3876% | +0.5744% | +6.1218% |
| 2048 | 122820 / 60 | 6.147128 | 6.175023 | +0.4538% | +0.8738% | +3.9527% |

两组通过预设总体≤1%、各域≤3%门槛，但单窗口仍可能退化。原始记录在实验根目录
`fp8/quality/{513,2048}-{q8,fp8}`，完整分域/窗口统计为 `fp8/quality/analysis.json`。
评分运行的速度不作为受控服务性能成绩。

文档先行、同会话单独提问，15 个会话的 60 个事实全部正确；保存两轮请求/响应，
核对实际 prefix reuse 和 native Query span。工具续接验证 append_pressure、空 Query、
零 replay。5050 上的 4B 双并发/MTP4 两臂通过，分别有 91 次注意力回读字节一致检查。
这属于独立功能验证，不能与 5060 Ti/27B 速度混为一组。四事实共用文档，属于相关探针。

仍保留格式反例：原始 max128 工具文档请求，Q8 只答 Received，FP8 却输出 27 token
解释。旧 FP8 路径也能复现；普通 FP8 亦有类似额外输出，全驻留受测组正常。
扩大 sink 的诊断未修复该例，未修改策略。相同 neutral-system 约束下受控工具测试通过，
不代表原始反例已解决。证据为 `fp8/format-diagnostic`、`fp8/quality-behavior.json`。

## 最新 256-token budget 内速度

同一 RTX 5060 Ti UUID `GPU-A`，
CUDA13.2/MSVC14.44/Release/sm120a，固定依赖 bundle，MTP4、并发1、int8 KV，
`NINFER_PREFILL_ALIGN=0`。不同 system 的 5018-token prime 后，新根会话输入未变的
14806-token 文档，cache_n=0，要求短中性确认。KVMem B24576/R1024，提示词完全在 B 内。
原仓库为固定提交 `08c4fb70bec11063cae42eafe1c1b0dc3a7f993e` 的实际编译版本。

原有默认内存策略三臂镜像次序，各两次独立进程：

| 版本 | 两轮 prefill tok/s | 均值 |
|---|---:|---:|
| 原仓库 FP8 | 970.30 / 970.51 | 970.40 |
| 当前后端 FP8，关闭 KVMem | 912.20 / 917.07 | 914.64 |
| 当前后端 FP8，开启 KVMem | 874.77 / 888.87 | 881.82 |

当前后端相对原仓库低5.75%；KVMem 相对同后端普通路径低3.59%，仍高于3%目标。
初始普通路径自动 KV 容量不同；另固定两边为32768后做四轮镜像对照：
本地916.43/914.99（均值915.71），原仓库969.25/969.68（均值969.47），差距仍为5.55%。
初始三臂与固定容量两臂分开报告，不拼成同条件样本。原始数据在 `fp8/budget-speed`。

Nsight 仅用于定位，不作计时成绩：本地 IQ4_XS、IQ3_XXS 矩阵累计更慢，attention
反而更快。原仓库按行数/SM wave fill 在 tile 与 Q8 matrix 间选择，本地目前不做该选择。
两边 PDL 实际均开启：本地 Native 模式独立开启 PDL，不能仅看 CMake PDL=OFF 误判。
这些差异尚未做独立优化 A/B，不承诺某一项追回全部差距。首份普通跟踪55秒截断已排除，
后用完整90秒跟踪替代；一次 host-pinning 内存不足启动失败也已保留并补跑。

## 成本、拒绝方案与未完成范围

- slab从8增至64 MiB；代表形状估算工作区约13.73→116.74 MiB，增加约103.01 MiB。
  这是形状工作区估算，不是整进程峰值显存。没有永久整模型 FP8 副本。
- 通用 decoded-int register lookup、小矩阵运行时混合选择实测不稳定或变慢，已撤回；
  shared-K attention-page-mass 四种形状均变慢，也已撤回。未打入当前补丁。
- 初次移植期间测试 oracle、MSVC includes/Ninja 元数据和受限 loopback 就绪失败均已记录并
  处理；没有伪造 Ninja 记录或为整理补丁重编整个引擎。
- 当前受控速度覆盖 256/1024；其他宽度、其他模型/卡、多 GPU/TP、256K 质量、发布安装仍未验收。
  decode 数学未因当前 FP8 prefill 路由改变，不宣称 decode 加速或输出完全等价。
- 未导出暂停的 FP4 quarter-scope 工具改动和实验 smoke/GDN probes。未提交、合并或安装。

## 历史初版记录

以下为2026-11-01的两阶段8 MiB初版及其原始不利结果。它与当前内核、预热及质量语料不同，
只保留历史证据，不作为当前实现的结论。

## GGUF MXFP8 prefill 实现与对照测试（2026-11-01）

已实现默认关闭的实验路径，并完成独立数值检查、原 q8_1 回归、真实模型 ABBA 性能和六域困惑度对照。RTX 5060 Ti 上，1024-token chunk 的 prefill 平均提升 52.2%，256-token chunk 平均下降 24.1%；decode 没有可宣称的提升。总体困惑度上升 0.619%，arXiv 域上升 4.274%。因此保留为显式有损选项，没有启用默认值，没有提交或替换现有安装包。

### 实现和范围

- 开关：进程启动前设置 `NINFER_GGUF_PREFILL_MXFP8=1`。未设置或 `0` 保持原路径，非法值报错；进程内第一次查询后固定。
- 支持 sm120 编译路径及计算能力 12.x、K 正整数且整除 256，格式 IQ3_XXS、IQ3_S、IQ2_XXS、IQ2_XS、IQ2_S、IQ4_XS、Q2_K、Q4_K。其他格式、设备和小于 256 列的调用保持既有路径。
- 激活和解码后的权重按每 32 个值实际 amax 编码成 E4M3 和 E8M0 幂次尺度；没有 FP16 权重中间值。cuBLASLt 使用原生 VEC32_UE8M0 块缩放 GEMM，FP32 累加、FP32 输出，然后执行原 GGUF residual/gate/BF16 存储边界。
- 8 MiB 权重图像 slab（最小 128 行），没有整模型 FP8 副本。激活、权重图像、尺度、补齐输出和 fallback scratch 都由 caller workspace 持有并纳入容量查询；cuBLASLt 句柄/描述符/算法按设备和几何缓存。
- 处理输入 gather、行列尾部、补零、输出裁剪和已有混合格式 fallback。没有导入 NVFP4 或前一次改变累加顺序的 IQ tile。
- 普通 decode/verify 支持上限 128 列，因此保持 q8_1。prefill 误差仍会留在后续 KV/GDN 状态里，不能由“decode 未改”推断输出等价。
- Qwen prefill 性能/状态签名加入数值模式和宽度门槛，所有 family 的 snapshot model binding 加入此模式。默认签名未变；最终 PPL 报告验证 FP8 与 q8_1 签名不同。跨模式快照实际保存/加载拒绝未单独端到端测试。
- 当前保留 256 的实验门槛以覆盖已测域；它不是收益校准门槛。此配置建议使用 chunk 1024，256 已证实变慢。

改动覆盖 Ops glue、单独 CUDA unit、静态与 Windows kernel-DLL 的 cuBLASLt 链接、CTest 注册、契约注释、缓存签名及使用文档。没有新 CLI 参数、协议字段或 ingest 状态。测试 smoke 文件仅属于实验目录，不纳入产品补丁。

上游参考是 Micky Zhang / IAMFreehandZ 的 [ninfer-5060-qwen](https://github.com/2-com-org/ninfer-5060-qwen)，最初路线来自 `7f189ba2`；本地保存的 bridge/tile/internal blob 分别为 `3932fbcfb2fb2172136794d3c5c46f007ed1f0b2`、`bf612d88a7bd0776aa006f98b54ddbbe6c07576b`、`a9d13236aca8dc95eb53639937e63e6288c9e92c`。移植了尺度布局、解码恢复和 Lt 计划方法；本实现使用实际 amax 和 FP32 输出，没有采用该分支的优化 LUT 权重图像或 BF16 两阶段乘积，不能直接等同于上游的性能。

### 对照条件

GPU 为 RTX 5060 Ti，UUID `GPU-A`。每次启动前检查无计算进程、利用率和占用，串行执行。5050 未参与比较。使用 CUDA 13.2、MSVC 14.44、sm120a、Release，同一固定依赖 bundle 的构建和运行 DLL。

模型：`models/ninfer/Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp/Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp.ninfer`。它是混合 GGUF 格式 artifact，未重转换权重。

基线为现有 exact-ops + 校准后的 5060 Q4_K/Q6_K profile，不含 IQ tile。最终比较使用同一个新 `ninfer-serve.exe`，只改变环境开关 0/1；每种 chunk 采用 A-B-B-A，各模式两次独立进程。空用户 profile 保证两臂使用相同内置路由。

上下文 32768、并发 1、int8 KV、MTP4、ngram 0、KVMem budget 24576、reserve 1024、host 4096 MiB、sink/recent 各 64。先用短请求 warm，然后普通文档请求输入 14,806 token，要求中性回应 Received；在同一会话另发实际问题，生成 193 token。没有把参考答案放进输入。

问题回合实际复用 14,807 token，仅计算 36 个新 prompt token；native log 确认为 private_endpoint。请求和响应、native 日志、启动参数均已保存。没有单独记录 native Query span，因此这里的记录复制属于相关行为探针，不作为完整长上下文召回质量结论。

### 性能：最终版完整 ABBA

以下为各两次运行的均值及范围。吞吐采用 native prompt/decode timing；wall 为客户端整请求，TTFT 为 native 指标。两臂不按不同 chunk 交叉比较。

| chunk | 指标 | q8_1 | MXFP8 | 均值变化 |
|---:|---|---:|---:|---:|
| 256 | prefill tok/s | 704.20（695.90–712.50） | 534.59（532.60–536.57） | −24.09% |
| 256 | 文档 wall 秒 | 21.227（20.969–21.485） | 27.883（27.766–28.000） | +31.36% |
| 256 | 文档 TTFT 秒 | 21.048 | 27.716 | 变慢 |
| 256 | decode tok/s | 85.136（85.115–85.158） | 85.198（85.189–85.206） | +0.072% |
| 256 | 问题 wall 秒 | 2.9290（2.921–2.937） | 2.9525（2.921–2.984） | +0.80% |
| 256 | 问题 TTFT 秒 | 0.6163 | 0.6346 | 变慢 |
| 1024 | prefill tok/s | 734.77（727.18–742.36） | 1118.42（1102.47–1134.37） | +52.21% |
| 1024 | 文档 wall 秒 | 20.461（20.375–20.547） | 13.4375（13.250–13.625） | −34.33% |
| 1024 | 文档 TTFT 秒 | 20.171 | 13.261 | 变快 |
| 1024 | decode tok/s | 85.272（85.219–85.325） | 85.266（85.242–85.290） | −0.007% |
| 1024 | 问题 wall 秒 | 2.9605（2.937–2.984） | 2.9300（2.922–2.938） | −1.03% |
| 1024 | 问题 TTFT 秒 | 0.6437 | 0.6106 | 变快 |

相邻配对 prefill 变化：256 为 −23.47%、−24.69%；1024 为 +51.61%、+52.81%。配对 decode 范围 −0.098% 到 +0.088%，没有稳定收益。所有八臂的文档回应和问题输出逐字一致；MTP 均接受 154/164，41 rounds，排除了本样本中用少生成或不同接受率换速度。

256 的问题整请求最差配对 +2.16%；1024 的两对为 +0.034%、−2.08%。冷进程第一个短请求（15 prompt token、不触发 MXFP8）wall 在 1024 组的两对为 +32.35%、−20.80%，平均 +2.33%；256 平均 −8%。不能宣称冷短请求延迟改善，也没有足够证据将这项波动归因于 FP8 算术。启动时间在 256 组为 12.2185 -> 10.883 秒，1024 为 10.961 -> 11.0935 秒，同样不作稳定收益结论。

请求结束后的 nvidia-smi 显存：256 为 13290 -> 13302 MiB（+12），1024 为 13534 -> 13548 MiB（+14）。这不是峰值显存测量，也不是理论 workspace 上界。没有测功耗积分、其他 chunk、并发吞吐或长生成性能。

窄 chunk 尚不能摊薄两阶段路径的成本；当前没有采集内核级 profile，不能把 −24% 精确归因于某个转换核或 cuBLASLt 调度。改进方向是权重图像转换/LUT、减少 host 提交和 shape 路由校准，需要单独验证，不是本次已交付的收益。

### 数值与精度

独立 CPU E4M3 编码器以数值枚举和最近偶数规则选择 FP8 code，测试零、极小 BF16、1e30、负值、gather 和 padding，GPU code/scale 全部相同。没有用新 GPU 量化器自己当参考。

公开 Linear/LinearAdd/SwiGLU 经新的生产 glue 路径，FP64 naive 点积使用独立解码的 GGUF 权重和 CPU 编码后的操作数。涵盖八种格式的 N=17、K=256、T=257、两种 gather，以及实际 K=5120/6144/17408、T=256/1024 的形状、非整齐尾部、输出哨兵、精确 workspace 容量和预热后的 graph capture/replay。

| 检查 | 结果 |
|---|---|
| 22 例 FP32 encoded-product vs FP64 | RMS 最大 1.96013e−6、相对峰值最大 3.21029e−6；阈值 3e−5 / 1e−4 |
| BF16 store / residual | RMS 最大 0.00168974 / 0.0017117，峰值 0.00355418 / 0.00312854；阈值 0.003 / 0.004 |
| 16 例 SwiGLU | RMS 最大 0.00171885、峰值 0.00275457；阈值 0.003 / 0.005 |
| 22 例 graph replay | 全通过同一 FP64 阈值 |
| 相对原权重与 BF16 输入的 FP64 点积 | 额外量化误差 RMS 3.545%–4.045%，峰值最大 4.722%；诊断阈值 8% / 20% 在测试前固定 |
| 默认关闭原 q8_1 oracle | IQ3_S / Q4_K、K256、宽度 8/256/1024 及原测试附带 embedding 检查通过，无 MXFP8 marker |

以上检查证明实现符合所选有损算术，不证明模型质量无损。误差不是对原 q8_1 输出的精确复现要求，也没有放宽原 q8_1 测试阈值。

困惑度使用现有 heldout corpus 中六域各第一个 stream 的前 16,384 Unicode 字符；在评分前统一选择，哈希见 quality-corpus/provenance.json。context 2048、disjoint、prefill_chunk/score_tile 1024、int8 KV，同一 PPL executable 切开关，不加载 MTP/vision。共 13 个窗口、26,611 个有效目标 token；尾部不足一窗口的部分按原 scorer 规则不计。

| 域 | token | q8_1 PPL | MXFP8 PPL | 变化 |
|---|---:|---:|---:|---:|
| 总体（按 token 加权 NLL） | 26611 | 5.734254 | 5.769745 | +0.619% |
| arxiv | 2047 | 7.789485 | 8.122419 | +4.274% |
| github_code | 4094 | 1.998325 | 1.995193 | −0.157% |
| own_code | 4094 | 3.265076 | 3.289813 | +0.758% |
| synthetic_chat | 4094 | 4.488762 | 4.525242 | +0.813% |
| wikipedia_en | 2047 | 6.499456 | 6.532027 | +0.501% |
| wikipedia_zh | 10235 | 11.078694 | 11.090252 | +0.104% |

总体 mean NLL 上升 0.006170256 nat/token，arXiv 上升 0.041853246。总体受中文样本权重较大影响，不能只看 +0.619% 掩盖 arXiv。两个阶段的 PPL 结果完全一致；实际复制文本相同也不能否定这里测得的分布变化。未测完整大语料、长上下文召回、推理/工具质量，不把此路径归入前面的近似无损优化线。

### 构建、修复、保留的初测和交付

复用 `ninfer-heads-20261030/build` 同一工具链、架构、编译选项的 21 个静态库和前序 109 个 implementation objects，另保留校准 profile object。本任务只新增一个 CUDA translation unit、三个相关 host implementation units 和必要测试/PPL host objects。第二轮修补的 dry-run 精确为 1 CUDA、2 host、4 link、无缺失 Ninja 命令记录；8 jobs，没有清理或全引擎重编译。最终 21 个库和前序全部 110 个 object 的哈希均未变。

固定依赖为 `build-deps/ninfer-x64-windows-c7684b1d96347de8`，每次生成/构建/运行前均执行 check.py，运行 PATH 使用同一 bundle 与 CUDA 13.2。聚焦链接是本地验证构建，不是完整发布包或已安装包。

初次 PPL 链接遗漏了 corpus/evaluation 两个 host object，已补齐并聚焦重链；不是引擎推理失败。第一次每域 4096 字符的质量样本不足 2048-token 窗口，scorer 在评分前拒绝；失败日志和原语料保存在 quality-short-corpus-failure，随后全部域统一扩到 16384 字符再评分，没有依据结果选域或窗口。

第一次实现尚未把开关写进数值签名，收尾时发现并补齐，属于本次新增路径缺项。初测全部保存在 initial-probe-before-signature，最终修补后重跑完整 ABBA 和 PPL。初测 256 prefill 705.18 -> 531.42（−24.64%），1024 为 729.29 -> 1130.18（+54.97%）；初测 1024 问题 wall 平均 +5.81%、最差配对 +10.43%，256 冷短请求最差 +26.74%。没有删除这些不利结果，也没有以它们与最终版交叉凑对照。

产品补丁为 `ninfer-exact-ops-20261101/source/backends/patches/ninfer-gguf-mxfp8-prefill.patch`，接在 gguf-coop-5060 后。versions.json 已更新，完整 30-patch chain 在隔离索引中通过逐步 tree 验证，新补丁通过 whitespace=error-all。没有更改原 checkout 的工作文件或 index，没有提交。

最终服务程序是本实验 build/ninfer-serve.exe；数值测试 qualification.exe、原回归 oracle.exe、质量程序 ninfer-perplexity.exe。使用方式见 backend/docs/gguf-mxfp8-prefill.md。运行参数、每臂输出和原生日志在 performance，质量明细在 quality-baseline/quality-mxfp8，汇总在 analysis.json，产物哈希与 GPU 释放状态在 final-checks.json。所有自建服务已退出，两张 GPU 检查时均为 0 MiB、无计算进程。

未验证 Linux / 非 sm120 编译、Windows kernel-DLL 的完整发布构建、其他模型/MoE、多 GPU、多并发、未预热的首次 graph capture、完整长上下文质量和峰值显存。现有包脚本已包含 cublasLt DLL，但未生成发布包。当前完成的是 5060 Ti dense 模型的可复现默认关闭实验路径及上述范围的比较；更广的默认启用需要后续收益门槛和质量判断。
