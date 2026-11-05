# Radiance 1.17.0 与 IQ3R-S v4 接入

2026-11-06 提交前审计修复了三个问题：草稿元数据预检对 `.RAD` 后缀的大小写回归、Linux/Windows 控制台包遗漏新增的 `draft_control.py`，以及已有打包清单遗漏入口直接依赖的 `console_radiance.py`。修复前均保存了失败证据，修复后 89 项 backend、16 项共享启动器、9 项打包测试及真实 Windows 离线包/HTTP 集成通过，Radiance 启动器 UI 检查通过。隔离发行包在源码目录外使用自身文件解析 MTP3/DFlash7，Windows 使用固定 Python 3.12.10/psutil 7.2.2 bundle。第二轮源码审查未发现剩余可操作问题。原生 C++、补丁链和内核未改，校验了七补丁与 1781 个文件，因此本轮不重新编译引擎。[审计记录](benchmarks/radiance-iq3v4-audit-20261106.json) 保留来源、触发条件、修复与验证范围。

后续 256K Task2 的 V4 DFlash7 和旧 MXFP4 MTP3 均完成 33 个请求，使用 B36864/R4096、prefill256 和最终 max_tokens4096；生成代码与批次确认的质量限制单独记录，不把运行合同通过等同于模型质量通过。DFlash3 按用户要求中止。下文“最终服务恢复”的 PID 与 ready 是 11 月 5 日交付快照；后续 benchmark worker 完成后已正常退出并释放 GPU，不代表当前常驻服务。

本轮已完成 Linux RX 9060 XT 16 GiB、gfx1200、TP1/C1、FP8 KV/F32 state 的文本和运行时集成验证。旧 MXFP4 关闭草稿、IQ3R-S v4 关闭草稿、旧 MXFP4 MTP3、V4 DFlash2 7 四组同构建合同和短探针均通过；最终 hostfix 构建的 12K ALL 通过 155 checks。CPU PNG 请求、缓存和 placement 检查通过，完整原始图片 suite 保留为 21/22：低分辨率 OCR `5829→5029` 在普通 native 无 observer 下同样复现，两条路径的高分辨率附加 probe 均正确。交付范围为带该明确质量限制的源码集成与 focused validation build，不宣称一般图片质量或全部 vision checks 通过。最终服务已恢复并通过 health 和启动算术检查。普通启动方法见 [V4 runbook](radiance-iq3v4-runbook-20261105.md)。

| 项目 | 固定值 |
| --- | --- |
| Radiance | v1.17.0 / `19d4e8fe0e18071b9b6300f15fc4ccf38dc4fa3d` |
| 上游 ABI / KVMem ABI | 35 / 1263927331 |
| 补丁链 | 主集成、Qwen profile、AVX levels、profile closure、bounded admission、MXFP4 cast closure、CPU domain coverage，共 7 项，SHA-256 见 `backends/radiance/upstream.json` |
| 模型仓库 | `StillDeadcode/qwen3.8-27b-iqr-v4` |
| 模型 revision | `e5f1b3ef70741d705a64a28b852acf3fcefcead7` |
| 文件 | `qwen3.8-27b-iq3r-s-v4.rad` |
| 大小 | 13,202,911,232 bytes，约 12.30 GiB |
| SHA-256 | `cac179933e3f663ba2d60f104a451be327f3657643eb087d30fd8129112cf610` |
| 插件 / 内置草稿 | `qwen35_w8a8` / DFlash2，训练 block size 8，7 draft tokens |
| 实测设备 | RX 9060 XT 16 GiB，`gfx1200` |
| ROCr UUID | `GPU-043fe07311696099` |
| 本轮构建 profile | `qwen35-dense-tp1`，AVX2，固定 FFmpeg 8.1 PNG/JPEG 静态依赖 bundle |
| 默认及本轮 prefill | 256 tokens |

模型目录、文件大小和哈希来自固定 HF revision。文件包含视觉塔和 DFlash2 载荷，GPU 使用量由加载策略及运行配置决定。[模型文件](https://huggingface.co/StillDeadcode/qwen3.8-27b-iqr-v4/blob/e5f1b3ef70741d705a64a28b852acf3fcefcead7/qwen3.8-27b-iq3r-s-v4.rad)

共享启动器使用 `SpeculativeMode=dflash2`，启动前核对容器的草稿类型与训练深度。关闭草稿使用 `SpeculativeMode=none`。独立 `backends/radiance/tools/serve.py` 根据模型目录选择架构插件和草稿，显式参数经过匹配校验。Radiance 1.17 的 prefix host cache 参数为 `--prefix-cache-mib`，普通启动入口已采用该名称。

原生缓存观察、Query 回放、已接受 token 的统计和 recurrent state 物化已迁移。DFlash 窗口在回放后通过 epoch 失效，草稿覆盖范围从恢复位置重新累计。GDN materializer 核对实际 producer 的 ABI、layout 和导出函数。物理 KV 池由 KVMem 工作集预算限制，原生 admission 使用独立的逻辑上下文上限检查请求。

本轮构建先核对既有对象、完整头依赖、编译命令与固定依赖，再审查指定目标的 dry-run。23 个旧 host 对象经过 fresh proof 重验后符合复用条件。记录中的编译阶段如下，各阶段包含对已有单元的增量编译。

| 阶段 | 审计并执行的编译范围 |
| --- | --- |
| 初始选定闭包 | 234 host、51 HIP |
| profile closure 补齐 | 1 host、4 HIP |
| logical admission 修复 | 13 host、0 HIP |
| 并行上限 | 各阶段 `-j8` |

最终 sampling-capacity helper 移入 host-only header 后，远端增量构建实际仅编译 `statistics_observer.cpp.o`、`observer_main.cpp.o` 两个 host 对象并重链接 worker，GPU 编译为 0。23 个旧兼容对象和 2 个当前对象复用经过严格内容、完整依赖、历史编译命令、clang/toolchain 锁、对象和 log 证据核对；后两者为 `attention_kernel.hip.o`、`attention_capture.cpp.o`。源文件的正常写入和恢复确实更新了 mtime，因而被 Ninja 判为 dirty。没有人为回拨源文件时间戳、touch 对象时间戳或改写/伪造 Ninja 记录。曾计划的 3 host + 1 GPU 及中间错误 header 布局均保留审计证据。最终长文/图片验收使用这个增量构建，worker SHA 为 `78fbfa36d1de082b58c66d0646de7df46aa46f0b09068fc26be935a21a6ae6a9`。

FFmpeg 8.1 复用既有源 pin `b072aed6871998cce9b36e7774033105ca29e33632be5b6347f3206898e0756a`。版本化 bundle 位于共享 `kvmem-build-deps`，通过 manifest 和任务依赖锁校验。profile 只启用 PNG/JPEG 相关解码、parser、image demuxer及原生媒体模块所需的五个库。模型图片请求实测仅为 PNG；JPEG decoder 构建支持不等于 JPEG 模型请求已验收，视频和音频沿用适配器当前的功能范围。

构建工具要求显式传入 `--target`。Linux 的 `--kernel-profile qwen35-dense-tp1` 选择本轮 TP1 内核闭包，省略该参数会保留已有 CMake cache 的 profile。新建构建目录由 CMake 采用默认 profile。原生构建前运行固定依赖检查，并审查指定目标的 Ninja dry-run。

| 验证项目 | 本轮结果 |
| --- | --- |
| GDN 实际 GPU export | F32/F16，accepted 0、1、8、16，与独立 CPU rank-one recurrence 对照通过，invalid 参数、clone 边界及 live payload 保持检查通过 |
| 插件 schema / ABI | 通过 |
| 原生 argv | 7/7，通过普通入口生成真实 argv，实际执行 observer 预解析和原生 `config_parse`，包含旧参数及未知参数拒绝 |
| 调度器 | 190 cases，63,014 checks，通过 |
| Python | backend 89、launcher 15，共 104 项通过 |
| 源码来源 | 7 个补丁从固定 archive fresh replay，1781 个 native 文件 raw bytes 一致，无差异 |
| CPU domain coverage | 7 cases、22 checks 通过；旧六补丁实现相同用例失败 5 cases，未改变 ABI/header/GPU source |
| 普通 Radiance 独立服务 | 短文本、合法工具历史、取消恢复、同一文档两阶段问答通过 |
| KVMem worker，关闭草稿 | 合法工具历史、跨轮缓存、取消恢复、两阶段问答与真实 SSD 迁移通过 |

首轮 DFlash2 在合法工具续接时触发 prefix 恢复边界拒绝。上一请求保留 363 tokens，新 prompt 为 1010 tokens，小于 drafter window 2048。adapter 返回的恢复位置与原生 refill 边界未完成协商，原生返回 invalid argument。之后的实测还定位 split-prefill 中未来 reserved +8 窗口与 Pressure 提交时机的冲突。两项本轮接入回归均已修复，最新 DFlash2 文本 phase 的合同复验通过；其余 cycle 结果完成后再汇总受控对照。

独立 HOST backend 回归通过 24 个 RAM/SSD 变体及 1100 次 checkpoint save/restore cycles，包含 reserved +8 边界。该测试不执行 GPU。WSL DrvFS 临时目录的 unlink/ftruncate 诊断出现 errno 2，同一未改变的可执行文件在 Linux ext4 `/tmp` 上通过；环境证据和验证范围见 runbook。

四种文本配置使用逻辑 context 8192、B2048/R512、Host 32 MiB、SSD 8192 MiB、prefill 256、单并发、FP8 KV/F32 recurrent state、Sink 128/Recent 512、retained sessions 2、OMP 8、RAM embedding、mm0 和关闭原生 prefix cache。实际记录的 worker SHA-256 均为 `b8710bb74777ac9d6f1d3e4f1c9b630b2b74c786ee6fdca04cd2a37a38fb0aa2`，加载的六个 worker/plugin 文件哈希集合也相同。这组文本构建早于 CPU domain coverage 和最终 12K 容量修复；后续长测构建单独记录，不将两者描述为同一二进制。

旧模型采用本轮原始实测身份：`/root/kvmem-models/qwen38-27b-radiance-task2-20261104/target-head-mxfp4.rad`，17,775,964,160 bytes，SHA-256 `271b072c89232171d679a677ae93d9aa3397b1eaccfbff418c944f531adaae87`。部分历史 cycle 的人工声明多抄了一个字符，形成 65 字符字符串；`old-model-observed-identity.json` 与 `model-identity-correction.json` 保留实测身份和纠正记录。模型文件和原 benchmark 数据没有因此更改，不能用历史声明字段或同尺寸旧报告代替实测 SHA。

短性能探针为同一 1024-token 参考文、1 次 warmup 和 2 次计量，每次实际生成 128 tokens。四组全部 3 个请求均有效，cached tokens 均为 0。参考文 SHA-256 为 `0dcdc239f5cfc6115bbff87cb7b14e837efced96cdc44f70bc5a74e4f6770755`，messages SHA-256 为 `b72a2aefa59ac385d5f06b6c74905dcdc632d7cdbf51eec80c9333338895c853`。

| 文本配置 | all 合同 | 平均 TTFT（s） | 平均端到端（tokens/s） | 原生 decode 两次平均（tokens/s） |
| --- | --- | --- | --- | --- |
| 旧 MXFP4，关闭草稿 | 113 checks 通过 | 1.244453 | 16.316827 | 19.795861 |
| IQ3R-S v4，关闭草稿 | 113 checks 通过 | 1.734821 | 17.823003 | 24.082980 |
| 旧 MXFP4，serial MTP3 | 114 checks 通过 | 1.256286 | 31.407532 | 61.574024 |
| IQ3R-S v4，block DFlash2 7 | 113 checks 通过 | 1.551383 | 40.090750 | 140.410477 |

原生 decode 列取响应中 `predicted_per_second` 的两次算术平均；端到端列取实际 completion tokens 除以完整 SSE 请求耗时，两者计时范围不同。该固定数字序列探针中，MTP3 每次计量 offered/accepted 为 99/95，DFlash2 为 112/112；不能将其当作通用文本接受率。旧/新模型均关闭草稿的两行是编码对照；MTP3 与 DFlash2 两行同时改变了草稿算法和模型编码。两次短计量不构成稳定长时吞吐结论。

all 合同按同一文档先发、问题随后在同一对话单独发送的协议执行，参考文 SHA-256 为 `090ebd7e281cec4e1f2435a1d63762671594710f8e05bf8c7c991e1422102f0c`。四组均验证原生 Query span、Query replay 和独立答案 `LANTERN-5829`；普通工具续接执行 append 加 resident-only Pressure。

| 文本配置 | 文档轮（s） | 问题轮（s） | 问题 cached tokens | SSD read / written（bytes） | history hits |
| --- | --- | --- | --- | --- | --- |
| 旧 MXFP4，关闭草稿 | 4.810270 | 1.296232 | 4160 | 1,638,400,000 / 4,069,785,600 | 9 |
| IQ3R-S v4，关闭草稿 | 6.399692 | 1.387536 | 4160 | 1,646,788,608 / 4,078,174,208 | 9 |
| 旧 MXFP4，MTP3 | 5.540385 | 1.593598 | 4160 | 1,701,445,632 / 4,196,794,368 | 9 |
| IQ3R-S v4，DFlash2 7 | 6.879402 | 4.525897 | 2048 | 825,753,600 / 3,731,750,912 | 4 |

Host payload 计账峰值为 33,030,144 bytes，MTP3 为 33,226,752 bytes，均处于 32 MiB 配额内。阶段采样包含加载、合同和 benchmark，而非单个测量请求的峰值：

| 文本配置 | device-global VRAM 峰值（bytes） | 进程 RSS 峰值（bytes） |
| --- | --- | --- |
| 旧 MXFP4，关闭草稿 | 14,336,065,536 | 5,265,489,920 |
| IQ3R-S v4，关闭草稿 | 11,155,017,728 | 3,180,285,952 |
| 旧 MXFP4，MTP3 | 14,865,100,800 | 5,245,399,040 |
| IQ3R-S v4，DFlash2 7 | 13,089,116,160 | 3,146,706,944 |

VRAM 来源为同一 GPU 的 PCI BDF `0000:03:00.0` 全局计数，RSS 为各阶段 worker 的 `/proc` 计数。GPU guard 在启动前检查空闲，在测量前要求唯一任务上下文；device-global 数值不等于模型权重或 KVMem payload。

本轮 12K 阶段暴露了现有 Radiance 适配代码中的容量参数缺陷：fixed Query replay 的合法 suffix 为 775 tokens，visible 4864；sampling 仍误用普通 B+reserve 4608 的限制，而实际物理 pool 5376 和 admission workspace 允许该范围。来源核对确认 Nov4 的 Radiance adapter `begin_attention` 已有该错误，不属于本轮版本迁移新引入的问题，也不来自 common invariant 或原 llama 路径。修复只在 replaying 时使用实际 device capacity，普通 Pressure 仍受 B+reserve 约束；冻结的 Nov4 baseline 不改。最终 mm1280 配置下的 12K 运行通过 155 checks：文档请求 12,295 prompt tokens，16.428402 s；问题请求 12,328 prompt tokens，4.495535 s，cached 10,240 tokens，独立答案 `LANTERN-5829` 正确。文档 SHA 为 `c88436410dee074bf12823ed22b6db63ca336fe58c2ec25e2c55225e32978940`。该阶段 SSD reads 2,480,668,672、writes 7,091,126,272 bytes，history hits 10，Host payload 峰值 33,030,144 bytes。

CPU 图片首轮颜色、追问、异 session 图片缓存和颜色反转共 15 checks 通过，随后 384×128 OCR fixture 因 mm256 容量返回 HTTP 400。真实模型 metadata 的 `preprocessor.size.shortest_edge` 是 `65536`，patch 16、merge 2、temporal 2，longest edge 为 16,777,216；记录见 `vision-capacity-observed-model-metadata.json`。原生保持既有 Qwen resize/ceil 规则，OCR 为 160×448、280 patches／70 image tokens，超过 mm256；方图为 256×256、256 patches／64 tokens。仅将运行容量扩大为 mm1280（共享启动器 `VisionTokens=320`）后，min pixels 65,536 和上述预处理结果均不变，宽图 280 patches 处于新上限内。320 是部署 token 容量，不是每张图的实际 token 数。该配置调整无需编译，没有修改预处理或 fixture 字节，也不作为代码缺陷记录。

mm1280 的完整重跑中，原始 PNG 图片 suite 通过 21/22：请求完成、颜色及追问、图片缓存和 CPU placement 检查通过，唯一失败是原始 OCR 将独立标准 `5829` 回答为 `5029`，其后两位追问 `29` 通过。原图、判定标准及失败结果均保留。追加的 768×256 同像素 2 倍 probe 实际使用 192 image tokens，同一 DFlash2 7 worker 回答 `5829`；这是独立诊断，不替换原始失败用例。普通 native observer-off、CPU vision、同为 DFlash2 7 的独立参考已完成：相同图片 SHA、seed 42、temperature 0、thinking 关闭，原图同样回答 `5029`／70 image tokens，2 倍图同样回答 `5829`／192 tokens，observer 统计行数为 0。这些证据未显示 KVMem observer 特有的 OCR 缺陷；当前资格保留低分辨率 OCR 质量限制，不据此宣称一般图像质量或 CPU/GPU 视觉数值等价。

最终阶段整卡 VRAM 峰值为 13,187,584,000 bytes，worker RSS 峰值为 3,090,214,912 bytes；采样覆盖加载、测试和等待期。独立参考结束后，KVMem 服务按相同 argv/env 和逐项相同的加载 artifact 映射恢复，`final-service-snapshot.json` 记录 ready、health 200 和启动算术答案 `4`，服务器地址 `http://127.0.0.1:18211`，记录时 PID 51422。原始 cycle 的 `pass=false` 及 vision suite 的失败状态保持不变，snapshot 将通过的 155 项文本、21 项 vision runtime 子集和未通过的 OCR oracle 分开记录。

| 后续验收范围 | 当前状态 |
| --- | --- |
| 四文本配置合同及同构建短探针 | 已完成，结果如上；四组共享同一 worker/plugin 哈希 |
| 旧 MXFP4 serial MTP3 | 合同 114 checks 和短探针均通过，含真实 offered/accepted 计数 |
| 最终 12K 文本合同 | 155 checks 通过，保留 Query/replay、真实 RAM/SSD 和两阶段原始请求/响应 |
| CPU PNG 图片 | 请求/缓存/placement 子集 21 checks 通过；完整 suite 21/22，OCR `5829→5029` 未通过，普通 native 同现；2 倍 probe 两路径正确 |
| JPEG 模型请求、Windows、TP2、general profile、完整模型 fp8_v4/k6_v4/f16 state | 未资格验证 |

原始构建 dry-run、依赖与对象复用审计、请求/响应、原生计数及 GPU/RSS 采样保存在本轮验证目录。`provenance-final-release` 保存最终七补丁 raw 重放与 Python 结果，早期 `provenance-final-seven-verified`、失败封装和中间构建证据也保留。`capabilities-tp1-evidence` 保存 capability 构造器，`logical-context-sched-results.json` 保存调度器结果。四文本数据来自 `evidence-download/text-evidence`，35 个 present 索引文件已核对大小及 SHA。最终资格的 262 个索引文件保存在 `evidence-download/final-qualification`，归档 SHA 为 `ede8abcd1ea38d1edd6a07b67d0b0006b52546b8e1d70c33b6f550f984e37993`；服务 snapshot 的 15 个索引文件在 `evidence-download/final-service`，归档 SHA 为 `50a166f21bec69aa1b5886a6b1604732b8e771c600ceebc533cce23ae937f58a`。`final-delivery-audit` 保存逐项提取值、公共参数对照和有限资格核验；原始失败数据没有改写。
