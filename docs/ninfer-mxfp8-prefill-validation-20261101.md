# GGUF MXFP8 prefill：当前实现、补丁边界与验证

> Publication note (2026-11-02): native backend patch candidate, not an installed release. Machine identifiers are anonymized; measurements and hashes are retained. Uncommitted-status statements below describe the original experiment; this PR publishes source only. Raw requests/logs/traces remain local. See [the public evidence summary](benchmarks/ninfer-low-precision-20261102.json). Further FP4 tests are paused.

更新于 2026-11-02。当前路径为 256≤T<1024 的融合 FP8 tile，以及 T≥1024 的
64 MiB 有界 cuBLASLt slab。默认关闭，属于有损量化选项。最新数值/模型验证已完成主要项目；
256 的原仓库速度差距仍存在，1024 的新版完整最终性能对照尚未完成，不能宣称全配置达标。
本次整理仅重分补丁并更新文档，未加入此前讨论的按设备/形状选路或精确缩放新优化。

## 当前后端补丁边界

当前清单以 [versions.json](../backends/versions.json) 为准，规则见
[多后端维护约定](multi-backend-repository.md)。旧报告中的 tree/hash 属于当时补丁划分。

| 补丁 | 所有权和内容 |
|---|---|
| `ninfer-5060-gguf-coop.patch` | 设备 profile；保留已验证 Q4_K/Q6_K 选择，不新增 FP8 选路 |
| `ninfer-gguf-mxfp8-prefill.patch` | 通用 FP8 计算、融合 tile、64 MiB Lt、数值签名、独立 CPU/FP64 测试 |
| `ninfer-gguf-nvfp4-prefill.patch` | NVFP4 计算、显式 A4 权限、FFN producer、测试；进一步 FP4 测试暂停 |
| `ninfer-kvmem-statistics-readback.patch` | KVMem 原生 Mean-K/Query 合并拷贝、注意力反馈批量回读和布局/字节校验 |

FP8 和 NVFP4 计算在关闭 KVMem 时也可使用。FP8 测试先引入共享 CPU GGUF oracle，
NVFP4 测试复用它；FP8 不依赖后续 NVFP4 层或 KVMem 回读层。性能/快照数值签名跟随
计算模式维护。前端、启动器和协议中不增加显卡→CUDA kernel 的选择规则。

## 最新 FP8 正确性与质量

原模型 Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp，权重不重转换，int8 KV。
独立 CPU packed-GGUF 解码、CPU E4M3 编码和 FP64 乘积覆盖八格式、实际 K、
gather、tails、signed/zero/subnormal/max scales、FP32/BF16 epilogue、Graph 和大 slab。
另有新增 per-device shared-memory 初始化，但未完成双 GPU/TP 整体资格验证。

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
- 256 之外新版完整受控速度、其他模型/卡、多 GPU/TP、256K 质量、发布安装仍未验收。
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
