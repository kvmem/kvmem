# RTX 5060 Ti Q4_K/Q6_K 协作加载路由（2026-11-01）

> Publication note (2026-11-02): native backend patch candidate, not an installed release. Machine identifiers are anonymized; measurements and hashes are retained. Uncommitted-status statements below describe the original experiment; this PR publishes source only. Raw requests/logs/traces remain local. See [the public evidence summary](benchmarks/ninfer-low-precision-20261102.json). Further FP4 tests are paused.

已在上一批 exact-ops 候选上完成路由校准、配置集成、受影响目标编译、数值检查和实际模型对照。
未提交、未发布、未替换安装版本，也未修改用户全局 device-profiles.json。

## 实现

使用已有的 [Q4_K/Q6_K cooperative kernels](https://github.com/2-com-org/ninfer-5060-qwen/commit/b4081ed46cc0de316284b7f21ac29ed44dc4be0d)，没有新增 CUDA 数学路径。
只给 `nvidia-geforce-rtx-5060-ti-sm120`、36 SM 的内置 profile 补齐两个 dense 路由键：

- `gguf_vector/q4_k`：1–8 列继续 scalar。普通 Linear 的少数宽度能获益，但没有宽度同时通过 Linear 和融合 SwiGLU 的选择条件。
- `gguf_vector/q6_k`：单列继续 scalar，2–8 列选择 cooperative；超过 8 列保留既有 Op 路由。
- 其他 GPU、其他路由键、MoE 专用加载选择均保持原配置；更新 profile provenance 和 calibration 文档。有效用户 profile 仍按既有规则覆盖对应内置键。

当前 Qwen3.8-27B IQ3_S 制品的主模型 Q4_K 包括输出 head 及部分投影；Q6_K 的逻辑 bindings 在 MTP 中。
因此本次没有启用 Q4_K，也没有改变 Q6_K 单列路径，普通无投机生成不应期待这项优化带来收益。

## 校准和数值验证

- GPU：RTX 5060 Ti，UUID `GPU-A`，36 SM，180 W。
- CUDA 13.2 / MSVC 14.44 / Release sm120a；driver API 13040、runtime 13020；route catalog `29b2a2a7f55b5a61`。
- 现有 ninfer-calibrate，`--only gguf_vector/ --detail`。L2 flush、11 次取中位数、3% 初筛及再次交错确认，分别测 wide/tall Linear 与融合 SwiGLU，最终取共同获益的宽度。没有降低选择门槛。
- 现有 cooperative 完整测试通过：full/partial warps、scalar/cooperative 的 decoded values 与 product bits、host dispatch、tails、Store/Accumulate、fused 和 prepared routes。
- 独立 FP64 public GGUF oracle 通过：`--formats q4_k,q6_k --k 256,5120,6144,10240,17408 --widths 1,2,3,4,5,6,7,8`。参考从 packed weights 独立解码并按 q8_1 activation 表示计算，不把旧实现当作唯一正确性标准。
- 更新前后的 profile parser/catalog/layering 单元测试均通过。

以下是每个校准点的实际耗时；这是 Op 测量，不是整机提速：

| 格式/算子 | 列数 | scalar µs | cooperative µs | 耗时减少 | 该算子选择 |
|---|---:|---:|---:|---:|---|
| q4_k/linear | 1 | 290.0 | 285.9 | +1.41% | compiled |
| q4_k/linear | 2 | 292.0 | 288.0 | +1.37% | compiled |
| q4_k/linear | 3 | 292.1 | 288.9 | +1.10% | compiled |
| q4_k/linear | 4 | 296.2 | 286.6 | +3.24% | cooperative |
| q4_k/linear | 5 | 314.4 | 317.6 | -1.02% | compiled |
| q4_k/linear | 6 | 292.1 | 308.4 | -5.58% | compiled |
| q4_k/linear | 7 | 355.5 | 373.2 | -4.98% | compiled |
| q4_k/linear | 8 | 446.9 | 428.4 | +4.14% | cooperative |
| q4_k/swiglu | 1 | 304.3 | 293.4 | +3.58% | cooperative |
| q4_k/swiglu | 2 | 304.2 | 290.0 | +4.67% | cooperative |
| q4_k/swiglu | 3 | 297.6 | 302.1 | -1.51% | compiled |
| q4_k/swiglu | 4 | 300.3 | 305.7 | -1.80% | compiled |
| q4_k/swiglu | 5 | 340.3 | 311.7 | +8.40% | cooperative |
| q4_k/swiglu | 6 | 337.5 | 321.4 | +4.77% | cooperative |
| q4_k/swiglu | 7 | 352.8 | 342.4 | +2.95% | compiled |
| q4_k/swiglu | 8 | 375.4 | 398.5 | -6.15% | compiled |
| q6_k/linear | 1 | 426.3 | 416.2 | +2.37% | compiled |
| q6_k/linear | 2 | 449.7 | 410.8 | +8.65% | cooperative |
| q6_k/linear | 3 | 439.5 | 406.0 | +7.62% | cooperative |
| q6_k/linear | 4 | 483.7 | 434.4 | +10.19% | cooperative |
| q6_k/linear | 5 | 485.9 | 429.2 | +11.67% | cooperative |
| q6_k/linear | 6 | 496.8 | 465.4 | +6.32% | cooperative |
| q6_k/linear | 7 | 573.9 | 477.7 | +16.76% | cooperative |
| q6_k/linear | 8 | 594.5 | 537.8 | +9.54% | cooperative |
| q6_k/swiglu | 1 | 437.5 | 439.5 | -0.46% | compiled |
| q6_k/swiglu | 2 | 468.2 | 423.1 | +9.63% | cooperative |
| q6_k/swiglu | 3 | 467.5 | 419.4 | +10.29% | cooperative |
| q6_k/swiglu | 4 | 508.3 | 425.2 | +16.35% | cooperative |
| q6_k/swiglu | 5 | 517.3 | 445.7 | +13.84% | cooperative |
| q6_k/swiglu | 6 | 531.7 | 468.2 | +11.94% | cooperative |
| q6_k/swiglu | 7 | 539.3 | 494.9 | +8.23% | cooperative |
| q6_k/swiglu | 8 | 576.6 | 517.3 | +10.28% | cooperative |

Q4_K 的有利宽度在两种算子间没有交集；Q6_K 单列也没有通过，所以没有为凑收益扩大启用范围。

## 真模型对照：首轮

基线是上一批 exact-ops 候选（native tree `bebe657c4b441d1e287babd36a23b6511633d1e3`），不是安装版或冻结 master。
候选只更换内置 profile。模型为同一 `Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp.ninfer`。
每种模式各 A-B-B-A，全新服务进程，各 arm 两次；每进程短预热后测 GPU 运算说明和数据库查询说明两种技术性 prose prompt。
每次均输出 256 tokens，temperature=0、seed=42、ngram=0；模式分别是无投机和 MTP4。
统一 INT8 KV、context32768、prefill chunk256、concurrency1、KVMem B24576/R1024/Host4096 MiB。
每次启动检查 GPU 空闲，并用任务内空 user profile 避免全局配置干扰；argv、GPU 状态、请求、响应及 native JSONL 都已保存。

每个 prompt 的四次输出 hash 完全一致；MTP 接受轨迹也一致。
这不是长文 recall 质量测试，未扩展成长上下文或其他模型的性能/质量结论。

| 模式/提示 | decode tok/s 基线→候选 | 变化 | 总耗时 s 基线→候选 | TTFT ms 基线→候选 |
|---|---:|---:|---:|---:|
| plain/0 | 31.004 → 31.000 | -0.014% | 8.7735 → 9.0780 | 426.76 → 572.63 |
| plain/1 | 30.993 → 30.982 | -0.037% | 8.6640 → 8.8985 | 343.47 → 335.00 |
| mtp4/0 | 44.099 → 44.287 | +0.426% | 6.3200 → 6.3280 | 413.79 → 460.08 |
| mtp4/1 | 51.645 → 51.845 | +0.387% | 5.4065 → 5.3825 | 357.38 → 356.32 |

首轮不利结果保留：plain 两种提示总耗时均增加（+3.47%/+2.71%），其中提示 0 的 TTFT 平均增加 145.9 ms，单个相邻对增加 64.70%；MTP4 提示 0 的 TTFT 平均增加 46.3 ms，总耗时 +0.13%。
这些不是输出/接受差异：native 日志显示主要差异落在 prefill/首字路径，queue wait 约 1 ms；没有直接将其归为噪声，也没有用 decode 的小收益掩盖总延迟。

## 同二进制、充分预热的隔离复验

为判断首轮不利延迟是否来自加载/二进制差异，追加一次 MTP4 的 A-B-B-A：两边使用完全相同的候选 exe，
用任务内两份同 catalog 的 profile 分别强制 dense Q6_K scalar 和本次选择；先把两个真实提示各完整跑一次 256-token 预热，再测同样两个提示。
四个进程、每 arm 两次，其他参数、模型、GPU 和提示不变。此组是后续独立诊断，不替换上面的首轮数据。

| 模式/提示 | decode tok/s 基线→候选 | 变化 | 总耗时 s 基线→候选 | TTFT ms 基线→候选 |
|---|---:|---:|---:|---:|
| mtp4/0 | 44.064 → 44.247 | +0.416% | 6.2580 → 6.2425 | 345.71 → 345.83 |
| mtp4/1 | 51.614 → 51.827 | +0.413% | 5.4530 → 5.4690 | 345.69 → 346.14 |

稳态复验的 decode 提升为 +0.416%/+0.413%，两种提示的平均 TTFT 分别仅增加约 0.11/0.45 ms，未复现首轮的大幅首字回退。
但这同时控制了二进制和充分预热，不能据此确定首轮差异究竟来自哪个因素；原始首次加载场景仍有未解释的代价。
稳态总耗时也并非一致改善：提示 0 为 -0.25%，提示 1 为 +0.29%（最差相邻对 +0.59%）。因此只认定本组中有小幅 decode 吞吐收益，不宣称端到端请求稳定加快。

两组的逐次 decode、TTFT、总耗时和相邻对变化分别在 performance/analysis.json、steady-diagnostic/analysis.json。
所有稳态测量输出及 MTP 接受轨迹也完全一致。样本有限，不能把小幅吞吐变化外推成普遍收益，也未证明所有首次加载场景的延迟不变。

## 构建与限制

- 固定同一 build-dependencies.json bundle；复用上一批 110 个明确的实现对象及 21 个 hash 固定的原静态库。没有更改 ABI。
- 启动校准/验证工具时编译 1 个已有 CUDA 测试单元和 4 个 host 对象；实际 profile 配置落地后，只重编生成的 profiles.cpp 一个 host 对象并重链接服务和配置测试。
- 所有 native build 先检查 specific-target Ninja dry run；profile 增量使用 --expect-no-cuda 通过，正常 8 jobs。没有触发全引擎或 CUDA 内核重编，也未清理/伪造原 Ninja 元数据。
- focused build 沿用原 compile_commands 的 flags、架构、工具链和库，是验证构建，不是完整多架构发布包。其他 GPU、其他模型端到端、长文 prefill 和多并发吞吐未测。
- 首次沙箱运行 profile 单元测试异常退出；校准测量本身完成，但保存文件的 atomic rename 被沙箱拒绝。保留原日志，直接恢复已写完的同一份序列化结果（记录 SHA256），没有重跑挑选校准成绩；获准本机运行的原 profile 单元测试及更新后测试均通过。
- 内核无新增持久分配。首轮请求结束后 plain 显存均为 12,464 MiB，MTP4 均为 13,222 MiB；没有测峰值 host/device 内存。
- 未提交、未安装。已有 exact-ops 补丁未重写；新增独立补丁附在其后，整条 29-patch 链已通过现有 prepare-backends.py 的独立索引验证。

## 产物与复现

- build.ps1、focused_build.py、build-driver.log、build-profile-driver.log：focused 编译、dry-run 和复用对象记录。
- qualify.py、cooperative.log、oracle.log、profile.log：数值/状态和 profile 检查；calibration.log、microanalysis.json、calibrated-profile.json：校准结果。
- perf.py / performance/：首轮八进程、16 次实际测量的完整请求、响应和 native JSONL。
- prepare_steady.py、steady.py / steady-diagnostic/：同 exe、完整提示预热后的四进程隔离对照。
- ../ninfer-exact-ops-20261101/source/backends/patches/ninfer-5060-gguf-coop.patch：产品补丁；同目录 versions.json 记录顺序与树。

最终 native tree：`35d2e889f6053428815e2d1779a6ac4b9f011add`；补丁 SHA256：`1e309e62fb784fa4604702b6ff48d39e1146c9a56756c204ce7033949e9e6ad6`。
