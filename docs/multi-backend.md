# KVMem 多后端运行

本文描述当前多后端源码，包含活跃 KV 的 RAM/SSD 分层；具体依赖以 `backends/versions.json` 为准。历史 Windows NInfer [0.18.0 正式版](https://github.com/kvmem/kvmem/releases/tag/v0.18.0)基于 RC3 引擎，默认 sink=12288 tokens（12K）、recent=1024 tokens（1K）。两后端参数差异见 [实现差异](multi-backend-differences.md)，历史验收范围见 [分层验收记录](active-kv-tiering-validation-20261010.md)。

本轮 NInfer 固定到 `iamwavecut/ninfer-all` 的 `2026-10-29.2` / `d0da39796697`，dense heads TP 的范围、测量及限制见 [本轮验收](ninfer-heads-validation-20261030.md)。0.18.0 的 `8319e8f51247` [迁移记录](ninfer-all-migration-validation.md)及下文各有日期的实验仍是历史证据，不能视为新基线已经完成同等覆盖。KVMem 暂不支持新增的 `qwen4_exp`、模型 suspend、prompt graft 和 router，开启这些组合时明确报错。

Windows 入口 `scripts/windows/start-backend.ps1` 在当前源码可选择 `llamacpp`、`ninfer` 或下文限定范围的 `polystrata`。两个 worker 分进程运行，各自拥有推理引擎和 GPU 内存；公共 KVMem 策略库直接链接在 worker 内。

当前源码统一了 MTP/DFlash2 的启动配置、能力描述及 `/props` 统计，控制台按后端、并发和显卡数量启用选项。llama.cpp 的 DFlash2 使用上游原生草稿实现和独立 GGUF，NInfer 使用模型内置组件；详见 [公共推测解码接口](speculative-interface.md) 和 [本轮功能验证](speculative-validation-20261019.md)。已有发布包需更新 worker 后才能使用新增方式。

切换后端时的参数、默认值和执行行为区别见 [两后端实现差异](multi-backend-differences.md)，包括 ninfer 固定 64-token block、recent 默认且至少保留一页（参数至少 64 tokens）、R 与输出上限，以及 Host 和磁盘预算的不同含义。

Qwen3.5 dense 的 0.8B/2B/4B/9B/27B 规格与原生 MTP 适配范围、4B 两后端验收及模型复用方法见 [家族适配记录](qwen35-dense-family.md)。

Bonsai 2 27B 的 PQ2/PTQ1 作为两个现有后端的独立补丁接入，继续使用普通 llama.cpp／ninfer worker，不新增 Prism 引擎。当前实现、5050 8GB 的 PTQ1＋MTP 预算验证、限制和未来上游接管办法见 [Bonsai 接入记录](bonsai-integration.md)。此说明针对源码，不改变历史发布包。

## 启动

ninfer 预编译包可直接运行默认脚本。将包含文本、视觉和 MTP 组件的模型放到包根目录，命名为 `model.ninfer`；包和模型的完整路径不能含中文字符。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-ninfer.ps1
```

Qwen3.5/3.6/3.8 的 CPU 视觉权重在首次实际图片编码时加载。最后一个编码器释放引用后，
默认空闲300秒自动卸载，即使没有后续请求也会回收；文本请求不触发加载，后续图片可重新加载。
统一启动器 `-CpuVisionIdleSeconds` 可调整秒数，0表示加载后保留到模型卸载；原生参数为
`--cpu-vision-idle-seconds`。首次图片请求承担读取和反量化开销，视觉计算精度不变。
`/props.cpu_vision` 提供驻留字节数和加载/卸载次数。此策略针对 ninfer 的 Qwen CPU 视觉路径，
需要本轮新编译的 worker；范围与测量见 [按需加载和回收验收](ninfer-cpu-vision-lifecycle-validation-20261020.md)。

监控读取 CPU 视觉缓存自己的实时状态，不再等待 prefill/decode 的执行锁，也不会触发权重加载。
可对已启动且空闲的 `--vision-cpu` worker 运行
`python tests/ninfer_props_responsiveness.py --url http://127.0.0.1:18200`，检查流式文字生成期间
`/props` 的响应延迟、字段和未加载状态；默认要求每次响应不超过 1 秒。已有安装包需更新 worker 才会生效。

默认端口18200，context200k、budget36k、gen reserve及输出上限16k、思考预算8192、双INT8、自适应MTP上限4、ngram31、CPU视觉（图片上限1024 tokens），Host预算12GiB。脚本调用统一启动器，可编辑模型路径与预算，通过 `-Gpu` 选择设备。小预算CPU/resident视觉的实际1024-token图片及八路图文功能已通过，范围见 [视觉并发验收](vision-concurrency-validation-20261006.md)；正式版在 RTX 5060 Ti 16GB 上完成了该默认配方的加载和短 Responses 请求检查；该检查不代表所有模型或完整长上下文质量已验收。各历史记录保留其原有测试范围。

解压包后，在 PowerShell 中运行：

```powershell
./scripts/windows/start-backend.ps1 -Backend ninfer `
  -Model 'D:/models/model.ninfer' -Gpu 'GPU-your-device-uuid' `
  -Context 262144 -Budget 2048 -Reserve 512 -Prefill 512 -HostMiB 12288 -SinkTokens 64 -RecentTokens 64
```

`-Worker` 可指定本地构建的 `ninfer-serve.exe` 或 `llama-kvmem-server.exe`；省略时使用包内 `workers/<backend>/` 的程序。`-DryRun` 输出启动参数，`-Describe` 查询静态能力。默认监听 `127.0.0.1:18200`，前台运行，Ctrl+C 结束本次启动的 worker。安装 NVIDIA 驱动及 Microsoft Visual C++ 2015–2022 x64 Runtime；包内保留各引擎的 DLL，模型另行提供。

当前源码的两个后端默认使用 [图召回 C](graph-recall.md)：Query 按 token/head 做 Mean-K page softmax 后执行一次图传播；prefill/decode pressure 按累计 accepted attention 对驻留工作集重选，不做图传播或冷页召回。原生命令行与 Windows 启动脚本支持 meanq-cosine/page-softmax、GQA mean/max 和图传播比例；LRFU 仅支持 meanq-cosine。资源配方仍保留 0.18 的 B/R/H、sink=12288、recent=1024 等参数，`-DryRun` 可查看。已发布旧 ZIP 继续采用其原有算法；新脚本需要与新 worker 配套。

当前源码的 KVMem CLI v1 由两个 worker 共用解析和默认定义。两个 EXE 都支持 `--kvmem` / `--no-kvmem`，不传 KVMem 配置默认关闭；旧命令传正的 `--kvmem-budget` 仍会启用。启用后的默认 B/R/H 为 32768/1024/16384 MiB，sessions=4；B/R/H 显式 0 会报错，需配置能容纳 B+R 的原生上下文。`--kvmem-sessions` 和脚本 `-RetainedSessions` 在两个后端都只限制非活动历史，活动请求不计入；旧 llama `--kvmem-conversations` 是总数兼容模式，不是同值别名。SSD 额度 0 表示关闭且必须省略目录。C 接受 `--no-kvmem-lrfu` 兼容写法，拒绝启用 LRFU。新脚本必须与本轮新编译的 worker 配套；已有发布包不会被自动更新。完整定义见 [参数规范](native-cli-configuration-spec.md)。

ninfer 多卡默认采用上游 layer pipeline；当前启动器也提供需配套新 worker 的 heads 张量并行入口。按层拆分时，`-Gpu 'GPU-a,GPU-b'` 保留 UUID 顺序，第一张卡承担 embedding、head、MTP 和 resident Vision；CPU Vision 保留原生 Host 编码路径。可加 `-StageLayers 40,24` 指定连续层数，省略时按各卡空闲内存规划；层数总和由模型装载器验证。统一启动器禁止重复或空设备、超过八卡，以及经 `WorkerArgs` 重复传入设备参数。新 worker 允许各卡 compute capability 不同，但每张卡都必须落在该二进制实际编译支持的架构范围内；只编译 `sm_120a` 的包不能据此用于其他架构，本轮不声明跨架构实测。B/R 仍是每请求的完整逻辑窗口，H 是共享预算；按层模式各卡只持有自己的层，不按卡数拆分 B/R 或重复建立 Host 历史。

```powershell
./scripts/windows/start-backend.ps1 -Backend ninfer `
  -Worker 'D:/build/apps/ninfer-serve.exe' -Model 'D:/models/model-vision-mtp.ninfer' `
  -Gpu 'GPU-first-uuid,GPU-second-uuid' -Vision -VisionResidency cpu `
  -MtpDrafts 3 -Budget 2048 -Reserve 512 -Prefill 512 -SinkTokens 64 -RecentTokens 64
```

启用 heads TP 时，控制台选中两张或更多显卡后，在“硬件与加速”勾选“张量并行 · heads”；命令行增加 `-TensorParallel`。`-TensorSplit 3,1` 是按 `-Gpu` UUID 顺序填写的相对权重，省略时由引擎自动规划。权重不等于最终 KV head 数，也不保证对应显存占比；实际切分以服务 `/props` 的 `execution.tensor_shares` 和监控中的“实际 KV heads 分配”为准。移动显卡顺序时，控制台一起移动权重；更换选中显卡会清空权重。TP 与 `-StageLayers` 互斥，自动 budget 测算暂仍限单卡。

```powershell
./scripts/windows/start-backend.ps1 -Backend ninfer `
  -Worker 'D:/build/apps/ninfer-serve.exe' -Model 'D:/models/dense.ninfer' `
  -Gpu 'GPU-first-uuid,GPU-second-uuid' -TensorParallel -TensorSplit 3,1 `
  -Budget 2048 -Reserve 512 -Prefill 512 -SinkTokens 64 -RecentTokens 64
```

启动器显式传入 `--tensor-attention heads`，不开放 positions，也不接受从 `WorkerArgs` 绕过专用 TP 选项。真实启动及导出配置必须通过 worker 的 `heads_tensor_parallel_revision >= 1` 整数能力检查；`-Describe` 和 `-DryRun` 只描述启动器能力及参数，不证明现有运行包支持 TP。默认仍不开启 TP，旧包不会因更新控制台配置而获得此能力。本轮在 5060 Ti＋5050 Laptop 上通过 dense heads 的普通/MTP/DFlash2、三种核心 KV、状态和 SSD 验证；[实测记录](ninfer-heads-validation-20261030.md)同时保留预填充变慢、首 token 延迟和未验证格式范围，不能承诺双卡普遍加速。

按层多卡范围与证据见 [多卡验收](ninfer-multigpu-validation-20261006.md)。当前源码也开放 NInfer DFlash2＋按层拆分：选择多张 GPU 和固定长度 DFlash2，或传 `-SpeculativeMode dflash2 -DraftTokens 5`；草稿和局部状态保留在第一张卡，其他卡回传目标层特征。新 worker 已接入 heads TP＋DFlash2，但其真实 GPU 验收不能沿用按层测试结论。[双卡验证记录](dflash2-pipeline-validation-20261029.md)限定了历史按层实测范围。需要配套新 worker；已发布的 0.19-rc1 ZIP 仍保留单卡限制。多卡用于扩展容量，尚无本轮吞吐提升结论；`--cuda-memory-policy mixed/strict` 继续要求单卡，KVMem overlay Vision、原始 DFlash（非 DFlash2）、并行 prefill 和磁盘冷快照继续沿用当前限制。

llama.cpp 示例：

```powershell
./scripts/windows/start-backend.ps1 -Backend llamacpp `
  -Model 'D:/models/model.gguf' -Gpu 'GPU-your-device-uuid' `
  -Budget 4096 -Reserve 1024 -KvType q8_0 -SinkTokens 128 -RecentTokens 0
```

已有的 llama.cpp 可执行文件、`start-server.ps1`、IQ3/IQ4 启动脚本仍可使用。原生参数通过 `-WorkerArgs @('--flag', 'value')` 传入，例如 llama 的 MTP、mmproj 和会话配置。后端参数保留各自含义；`.gguf` 与 `.ninfer` 不互换，也不迁移跨引擎 KV。当前源码的 `-HostMiB` 对应两个后端共用的原生 KV payload RAM 预算；llama 仅在显式指定或启用 SSD 时传入，保持未配置时的行为。

## ninfer 功能与组合

当前源码的召回传输保留 64 MiB 目标分批中转，并在批内合并连续物理页；可用默认关闭的
`KVMEM_TRANSFER_TRACE=1` 查看拷贝规划、字节和分段计时。使用方法、计时边界及验证范围见
[召回传输合并与诊断](ninfer-recall-transfer-validation-20261019.md)。

当前源码支持固定 MTP1–15、多个 Host 会话、最多八条文本或 resident/CPU 图像请求，可在同一服务中混合图文；BF16/INT8/FP8/NVFP4/K8V4/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8 沿用相同的请求预算规则。2026-10-10 起，磁盘入口接入公共进程内 RAM/SSD 分层，活动 KV 和非活动历史共享 D 配额；旧持久化快照已被替换。自适应 MTP 允许文本及 resident/CPU 图片 C1–8；ngram1–15 可用于九种 KV 的多路文本及 resident/CPU 图片，启动并发大于1时，配置的 ngram16–63 自动降至15并提示，单路保留原宽度。快速 prefill 允许 INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8 的文本及 resident/CPU 图片。CPU 图片编码沿用原生串行编码器，八条活动请求不表示八张图片同时编码。冻结的 P5 运行包仍只有单请求文本能力；新源码的参数不适用于旧包。最终可用组合以随包验收报告为准。原生非 KVMem 模式保持原来的参数入口。

双 NVFP4 已接入源码，5050 上的聚焦数值／页搬运测试及 Bonsai B128/R128 ordinary、MTP1 eager、MTP4 Graph 功能检查已通过。
两个 MTP 用例使用现有 D3D12/WDDM 选项、FP16 GDN 和关闭的 ngram；默认 CUDA 预算下 Q8 MTP 装载仍被容量检查拒绝。
5060 Ti＋IQ3S 已通过双 NVFP4 的 229 组数值／功能／质量／故障注入检查：ordinary/MTP1–4 × eager/Graph、Host 会话、2/4 路文本、resident 图像、冷快照和最高 128K 档检索。三次独立 B32768/R16384、MTP4 Graph 服务启动及越窗 HTTP 请求也通过，ngram/lookup 均关闭。[完整记录](nvfp4-kvmem-validation-20261004.md)保留失败尝试与测试修正；这是本地源码验收，现有运行包不因此获得该能力。
`-KvType nvfp4` 对 K/V 都使用 group-16 e2m1 packed codes 与 E4M3 scale 字节，
Main 与 MTP 页池使用相同格式。参数组合与双 INT8 相同：ordinary 或固定 MTP1–15、
1–4 条文本请求或一条 resident/CPU 图像请求、Host 会话复用；该历史验收中的冷快照目前已暂时禁用。
ngram 默认关闭。

`-KvType k8v4` 已接入 Main/MTP、Host 会话、文本并发和 resident 图像；此前已验证的冷快照入口当前暂时禁用。K 使用 FP8_E4M3FN codes 与一份 FP16 row scale，V 使用 NVFP4 codes 与 raw E4M3 group-16 scales；两者均沿用原生 D256 归一化 Hadamard。5060 Ti＋IQ3S 通过了精简的 26 次真实模型实验和 8 项基础检查，包括 MTP0 eager/Graph、MTP1–3 Graph、MTP4 eager/Graph、C4、图像、磁盘格式隔离、8K/32K 检索，以及 B32768/R16384 MTP4 Graph 服务。ngram/lookup 均关闭；本轮没有性能对比和全排列验收。[清单、配置与结果](k8v4-kvmem-validation-20261004.md)。

新增参数采用 12 个组合的定向补测，另加原生视觉边界诊断，累计 18 次模型启动，使用 5060 Ti＋IQ3S，ngram/lookup 关闭。启动器增加 `-AdaptiveMtp`、`-FastPrefill`、`-VisionResidency cpu`，并将 `-MtpDrafts` 扩至 0–15。数值检查、失败尝试与最小重跑证据见 [参数补测记录](parameter-admission-validation-20261004.md)。这次没有性能对比或全排列验收。

```powershell
# 固定 MTP3，两个活动文本请求；B/R 是每请求预算，H 是全局预算。
./scripts/windows/start-backend.ps1 -Backend ninfer `
  -Model 'D:/models/model-mtp.ninfer' -Gpu 'GPU-your-device-uuid' `
  -MtpDrafts 3 -Concurrency 2 -Budget 2048 -Reserve 512 -Prefill 512 -SinkTokens 64 -RecentTokens 64

# 图像需要包含视觉权重与预处理资源的制品；默认最多 1024 个视觉 tokens/图。
./scripts/windows/start-backend.ps1 -Backend ninfer `
  -Model 'D:/models/model-vision-mtp.ninfer' -Gpu 'GPU-your-device-uuid' `
  -Vision -VisionTokens 1024 -MtpDrafts 3 -Budget 2048 -Reserve 512 -Prefill 512 -SinkTokens 64 -RecentTokens 64

# 八条活动图文请求；B/R 是每路预算，ngram 显式关闭。
./scripts/windows/start-backend.ps1 -Backend ninfer `
  -Model 'D:/models/model-vision-mtp.ninfer' -Gpu 'GPU-your-device-uuid' `
  -Vision -VisionResidency cpu -Concurrency 8 -VisionTokens 1024 `
  -MtpDrafts 3 -NgramDrafts 0 -Budget 2048 -Reserve 512 -Prefill 512 -SinkTokens 64 -RecentTokens 64

# NVFP4 本地源码验收配置示例，使用新编译的 worker；ngram 关闭。
./scripts/windows/start-backend.ps1 -Backend ninfer `
  -Worker 'D:/build/apps/ninfer-serve.exe' `
  -Model 'D:/models/model-mtp.ninfer' -Gpu 'GPU-your-device-uuid' `
  -MtpDrafts 2 -NgramDrafts 0 -KvType nvfp4 -Budget 32768 -Reserve 16384
```

`-MtpDrafts` 默认 0，可选固定 1–15；`-AdaptiveMtp` 要求 MTP，允许文本/图片 C1–8；`-FastPrefill` 允许 INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8 的文本及 resident/CPU 图片 C1–8，可组合公共 SSD 分层；`-VisionResidency cpu` 要求 `-Vision`；`-NgramDrafts` 默认 0，可选 1–63，非零时要求 MTP：宽度 1–15 允许 C1–8；启动并发 C2–8 时，配置的 16–63 自动降至15并提示，C1 保留原宽度。文本及 resident/CPU 图片均允许 BF16/INT8/FP8/NVFP4/K8V4/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8 九种 KV。多路上限 15 来自原生 GDN 验证工作区的 16 列容量；本次移除了 KVMem 额外的单路门禁。`-NgramMinMatch` 默认 12，可选 4–64。例如八路文本用 `-Concurrency 8 -MtpDrafts 3 -NgramDrafts 15 -KvType int8`，图片再加 `-Vision`。ngram 使用完整已提交 token 历史提出复制草稿，仍由目标模型验证；重复文本或代码更容易获益，不能承诺所有输入都加速。直接运行 server 开启 MTP 时，默认 ngram15 可用于多路；统一脚本默认仍显式传入 ngram0。快捷脚本默认单路/ngram31，改成多路时会自动使用 ngram15；以启动并发上限为准，当时只有一条活动请求也不改变宽度。MTP 独立 attention window、视频、overlay Vision、并行 prefill 仍被拒绝。`-DeviceProfile auto` 可选原生设备配置；公共 SSD 分层不提供重启恢复。`-RetainedSessions` 控制非活动 Host 历史数量，默认 4，上限 16。

本次 ninfer 实测制品是 Qwen3.8-27B-GSQ-RCO-IQ3_S，设备为 RTX 5060 Ti 16GB、CUDA 13.2、`sm_120a`。8K/64K/128K/256K 档的早期口令检索和后续追问通过；这组人工构造用例不代表通用长文问答质量，也不表示其他 GPU 架构已验收。示例的 B=2048、R=512、H=12 GiB 是实际通过的配置。

- B 是每请求选中历史的 GPU token 预算，R 是追加工作空间，两者按 64-token 页对齐；C 条活动请求的 Main 页池按 C×(B+R) 配置。MTP 另有原生草稿页池，Main/MTP 的 Host payload 都计入 H。
- NVFP4 的 D256 K/V 每 token 每 KV head 共 288 字节（两份 128 字节 codes 与 16 字节 scales）；K8V4 为 402 字节（256+128+2+16），INT8 为 528 字节，RK8V4 为 408 字节。换页按原始字节传输，不重新量化；权重、GDN state、workspace 与 CUDA Graph 不随 KV 格式同比缩小。格式布局身份和冷快照执行配置区分 NVFP4 与其他 KV 格式。
- prefill 是 128 的倍数且不超过 R。追加压力触发选页，逻辑历史保留原始位置。生成上限由请求输出预算和逻辑上下文上限共同决定。
- H 只限制原生 KV payload。统计索引、每会话最多四个检查点的 GDN StateImage、模型和临时缓冲另计，H 不等于进程总内存上限。
- 长期历史保存在普通 Host 内存。KV CUDA 搬运分批复用 Program 持有的 pinned 缓冲区，以 64 MiB 为目标按完整原生页对齐；单页超过该目标时至少容纳一页，小请求按需分配。D2H、H2D 与 KV 字节校验复用同一缓冲，扩容先释放旧 pin；GDN 检查点与状态校验缓冲另计。`transfer_staging_bytes` 报告搬运暂存总量，SSD 模式还包括普通 RAM scratch，并非进程全部锁页内存。验证范围见 [分批 pin 搬运验证](ninfer-pinned-transfer-validation-20261018.md)。
- 请求在执行前检查整段合法历史的 H 需求，已知预算不足返回错误。编辑历史不匹配可信前缀时从头计算。
- 如果单请求可容纳，但并发请求的全局 H 预留暂时不足，后续请求进入原生 FIFO 队列；不会让每个请求重复独占 H。服务入口默认排队时限 600 秒，Engine API 默认 30 秒，可显式配置。
- 图像区域完整保留，连同 sink 和追加页必须能装入 B；不满足时在准入阶段拒绝，可增大 B 或降低 `VisionTokens`。换图按内容、位置和精确前缀使旧 checkpoint 失效，不依赖文件名。
- 长输入的同查询工具续写可恢复 query 之后的完整端点，并保留冻结 Q/选页；新查询恢复合法的查询前检查点，重新执行 probe 与 replay。取消保留已经提交的完整检查点，不发布尚未提交的生成端点。
- 未超过 B+R 的短输入保存输入末尾前一 token 的 Base 和正常生成结束时的执行端点。短转长时，越过新 query 探测起点且不具有同查询附件的旧检查点不被复用。
- 复用与全量计算可能经过不同的原生分块边界，不能承诺两者浮点状态或所有 greedy tokens 完全相同。相同 checkpoint 的重复请求、取消恢复，以及明确答案的检索测试分别验收。

本次同时修复了 IQ3_S 模型八路/ngram15/Graph 的原生预热启动缺陷；实际验证范围见 [ngram并发补测](ngram-concurrency-validation-20261006.md)。

旧 `--lookup-ngram` 在 KVMem ninfer 中明确禁用，统一使用 `--ngram-draft-tokens` 与 `--ngram-min-match`；启动脚本对应 `-NgramDrafts` 和 `-NgramMinMatch`。经 `WorkerArgs` 传入旧参数也会被拦截。直接运行服务或调用 Engine 时，非零 lookup 配置会明确报错并指向新参数；未启用 MTP 或同时启用新 ngram 也不能放行旧路径。

## API 与状态

`POST /v1/chat/completions` 使用原生 ninfer 的 tokenizer、chat template、采样器和协议实现。`stream: true` 返回 SSE，断开请求会触发取消；下次提交完整 messages 可恢复可信历史。`GET /health` 查询就绪状态，`GET /v1/models` 查询模型，`GET /props` 提供 `backend`、`kvmem` 预算和能力；`/stats`、`/v1/load` 提供原生运行统计及 KVMem 当前历史的 payload、mean 统计、checkpoint 图像用量。历史换入/换出计数随历史替换重置。

ninfer 直接启动参数为 `--kvmem-budget B --kvmem-gen-reserve R --kvmem-host-mib H`，其中 H 的单位为 MiB。这会关闭原生前缀缓存目录，由 KVMem 管理当前会话历史；`--no-prefix-reuse` 同时关闭 KVMem 的跨请求复用。`--kvmem-verify-transfers` 逐字节检查恢复内容，用于功能诊断，会增加传输开销。

直接指定多卡用 `--devices 0,1`，可附加 `--stage-layers 40,24`；序号对应 `CUDA_VISIBLE_DEVICES` 的 UUID 顺序。`/props` 的 `execution` 返回实际设备顺序和最终层数切分，KVMem usage 继续统计共享历史与逻辑页数量。

ninfer 保留四种用途独立的完整状态点：输入/查询前 Base、正常生成结束的已执行端点、Frontend 确认的输入回退边界、生成内容的模板重建边界。输入回退点不会被生成阶段的思考结束状态覆盖，下一轮丢弃或替换生成答案、继续追加相同工具历史时，仍可复用此前输入。未改写的 assistant/工具回放可复用生成内容；最后一个尚未执行的输出 token 留给下一轮计算。每个会话共用一份 Host KV 历史，最多四个完整状态点。长历史工具续写在查询 span 和精确输入身份一致时保留原 Q/选页，只计算新增尾部；新用户查询重新执行检索，改写后的历史仅恢复仍精确匹配的完整状态点。恢复更早状态时放弃较晚状态点，防止共享历史被覆写后仍错误命中。此补全属于 ninfer 适配，llamacpp 路径保持原有行为。

当前源码使用进程内活跃 KV 分层替代已禁用的 ninfer KVMem 冷快照实现。
两个后端的统一启动器均接受 `-HostMiB H -DiskPath 'D:/kv-cache' -DiskMiB D`；
直接启动 worker 时使用 `--kvmem-host-mib H --kvmem-disk-path PATH --kvmem-disk-mib D`。
启动器要求磁盘参数经上述顶层选项传入，不通过 WorkerArgs 重复设置。
省略 DiskPath 则不启用 SSD；仅指定 DiskMiB 或空路径会报错。

即使只有一条活跃请求，超过 H 的已完成原生 K/V 块也可放入 SSD，检索选中的块按需读取。
K、V、scale 和 MTP 数据保持原格式，活跃 mean-K 留在 RAM。H 是原生 KV 驻留预算，
不包含 recurrent checkpoint、统计、传输暂存或系统文件缓存。D 是活跃原生 KV 与闲置
会话附件共用的额度；会话转为闲置时移动原记录所有权，不再复制一份 KV。
H 必须容纳所有并发 lane 的基本工作区，H+D 还需为换页预留余量；容量不足会明确拒绝。

文件属于当前服务进程，正常退出或进程结束后释放，不导入旧 ninfer-kvmem-v5 文件，
不提供重启恢复。ninfer 的 `disk_restore=false` 保留这一含义；`active_kv_tiering=true`
和 `ssd_enabled` 分别表示分层能力及是否启用。既有 llama 会话磁盘入口保留，不能与新的
共享 SSD 配置混用。Windows 和 Linux 使用同一存储契约；当前验收状态与性能限制见
[活跃 KV 分层计划](active-kv-tiering-plan.md)，不能将 WSL 主机测试当作 Linux CUDA 验收。

已知 H 或逻辑上下文不足会在请求执行前返回错误，服务可以继续使用。实际 CUDA 驱动/设备错误仍沿用原生 ninfer 的失败处理，可能需要重启 worker；没有承诺在设备故障后继续复用同一会话。

## 源码构建和打包

引擎源码通过 `backends/llamacpp` 和 `backends/ninfer` 子模块引用既有项目，固定提交及 KVMem 补丁由主仓库管理。先运行 `python scripts/prepare-backends.py`；Windows 构建入口会自动准备所选后端。克隆、补丁维护和源码包说明见 [仓库工作流](multi-backend-repository.md)。

公共核心可用 `KVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF` 单独构建，llama 目标继续使用现有顶层 CMake。ninfer 在自己的构建目录中增加 `-DNINFER_KVMEM_SOURCE_DIR=<kvmem checkout>/kvmem`，构建 `ninfer-serve`；两边独立配置 CUDA 与编译器，公共核心维持 C++17，ninfer 使用 C++20。

在 VS 2022 Developer PowerShell 中运行 `scripts/windows/package-backends.ps1`，传入 `-LlamaWorker`、`-NinferWorker`、`-NinferSource`、`-OutputDir` 和 `-CudaPath`，可加 `-VcpkgInstalled` 与 `-ValidationReport`。输出目录必须不存在。脚本为每个 worker 收集 DLL，检查干净 PATH 下的启动，并写入许可文件、二进制校验值和验证报告；模型不会复制。打包成功只证明依赖可加载，推理结果以配套验证报告为准。

自适应组合另补 10 次 IQ3S／5060 Ti 实验，覆盖快速 prefill、2/4 路、ngram31/63、resident/CPU 图片 query 边界、冷快照跨进程恢复；复用既有数值检查，不进行全排列或性能比较。范围和证据见 [自适应组合补测](adaptive-combinations-validation-20261005.md)。

图片＋MTP＋ngram 当前放行九种原生 KV，支持 resident/CPU 及1–8路；多路宽度16–63自动降至15。新增 `fp8`、`rk4v4`、`rk4v4-e8`、`rk2v4-e8` 只补格式身份和准入，复用原生量化与换页实现。范围与实测见 [新增KV准入补测](other-kv-admission-validation-20261006.md)；此前五种格式见 [图片ngram补测](vision-ngram-all-kv-validation-20261006.md)，首次K8V4换行差异及对照见 [初次图片ngram补测](vision-ngram-validation-20261005.md)。

快速 prefill 的新增 RK 格式与图片准入仅移除额外门禁，复用原生内核，验收范围见 [快速prefill准入补测](fast-prefill-admission-validation-20261006.md)。该开关影响符合原生路由条件的 prompt Attention，不负责并行 prefill，其他 KV 仍用各自原有路径。


## PolyStrata qwen35moe 接入（2026-10-22 源码）

当前源码固定 PolyStrata `v0.1.45-poly.6` / `20d839dbbaf34`，支持 CUDA 单卡的 `qwen35moe` GGUF（F16/Q8_0 KV）及 Gemma 4 26B-A4B（F16/int8/int8k KV）。Gemma 的独立 assistant、原生状态配额和验收边界见 [Gemma 接入说明](polystrata-integration.md#gemma-4-26b-a4b)。使用同一个 GGUF 导出的 tokenizer/template；图像输入还需要匹配的 mmproj 和 `strata-vision`。Python 运行环境需具备固定上游 requirements 中的 frontend 依赖。

统一启动器和直接 Python 入口均可省略 Tokenizer：第一次启动从所选 GGUF 本地导出完整 tokenizer 与 chat template，缓存在用户本地 KVMem 目录中，不下载模型或修改模型目录。缓存按模型绝对路径、大小、修改时间、头部摘要及导出器版本隔离，并校验导出文件；显式 Tokenizer 仍要求完整的同模型导出。DryRun 只展示计划，不创建缓存。控制台的 Python、frontend、图像 worker 等可选路径留空时使用运行包内默认位置。

```powershell
./scripts/windows/start-backend.ps1 -Backend polystrata `
  -Model 'D:/models/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf' `
  -Tokenizer 'D:/models/tokenizer' -Worker 'D:/workers/strata-poly.exe' `
  -Python 'D:/python/python.exe' -Gpu 'GPU-your-device-uuid' `
  -Budget 8192 -Reserve 1024 -Prefill 128 -HostMiB 4096 `
  -SinkTokens 64 -RecentTokens 64 -KvType f16 -Concurrency 2 -MtpDrafts 2
```

并发使用一个原生 worker 的多个序列，专家矩阵计算共享；每个序列独立拥有 KV、循环状态和检索统计。RAM/SSD KV 与图统计预算按进程共享。原生 batch 还保留一个主序列用于 prompt admission，显存规划需计入这一个额外序列。C2/C4/C8 仍受显存和 Host 容量约束。batch 内使用普通解码，独占执行可恢复 1–3 个内置 MTP 草稿。此入口不开放跨 GPU 层切分、batch 内 MTP、其他架构、DFlash2 或 n-gram。

检索使用当前公共的有界 Query/replay、PageSoftmax/图传播和驻留 Pressure。工具续接不触发冷历史召回；原生统计图对没有可证明消息边界的跨度使用公共的 unknown fallback。PolyStrata 的 `RetainedSessions` 包含主序列预留位置，默认取 `max(4, Concurrency+1)`；显式减小会触发正常的 LRU 淘汰。SSD 内容仅在当前进程内有效。

`-Vision -VisionWorker ... -Mmproj ... -VisionResidency cpu` 启用 CPU 图像编码；`resident` 使用所选单卡。`scripts/windows/package-polystrata.ps1` 可单独打包文本/图像 worker、原生依赖、frontend 和控制台；可选 `PythonRuntime` 必须是可搬迁且包含 frontend 依赖的 Python 目录，并在复制后验证。未提供它的包需要已有 Python，不声称离线即用。构建/打包能力不等同于某个模型和资源组合的运行验收，查看包内 `provenance/validation.json`。

三后端统一包使用 `scripts/windows/package-backends.ps1`，保留原有 LlamaWorker/NinferWorker 参数，另传 PolystrataWorker、PolystrataPythonRuntime，可加 PolystrataVisionWorker、PolystrataGgmlSource、PolystrataValidationReport。该组合使用已安装 frontend 依赖的便携 Python，不与仅含标准库的 PythonArchive 同时使用。LlamaSource 可显式指定已有 llama.cpp 构建的源码许可来源；省略时沿用仓库子模块。统一包中的 PolyStrata 验收记录位于 `provenance/polystrata-validation.json`；其他 worker 的依赖加载检查不代表重新进行了全套模型推理验收。
