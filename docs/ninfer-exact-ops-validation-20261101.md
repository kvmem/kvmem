# NInfer 近似无损优化候选验证（2026-11-01）

> Publication note (2026-11-02): native backend patch candidate, not an installed release. Machine identifiers are anonymized; measurements and hashes are retained. Uncommitted-status statements below describe the original experiment; this PR publishes source only. Raw requests/logs/traces remain local. See [the public evidence summary](benchmarks/ninfer-low-precision-20261102.json). Further FP4 tests are paused.

已完成移植、受影响目标编译、数学/状态检查、Windows 导出符号检查和真模型交错对照。
结果是首个短请求延迟降低、复制 decode 小幅加快、长文 prefill 基本持平；启动更慢。
本目录和 source/ 是独立、未提交候选，未替换已安装 KVMem 或冻结基线。

## 实现与范围

- 基线：KVMem `a0645f988ab71141c66f83b080cbc8dd56281001`，已准备 native tree
  `1e7a5d621c341f2c257b81b9a5632dd19b17b07f`。对照程序从同一批原库重链接，候选重编所有受影响的实现和 ABI 消费方。
- 移植 [85a0b23b](https://github.com/2-com-org/ninfer-5060-qwen/commit/85a0b23b8f50c8233a580d2db358b6209e75973f)：
  GDN norm/gating 用 256 个物理线程保留原 512 个虚拟 lane 的归约顺序；record/fold 用 shared staging 和更多 value rows 复用输入；INT4 V 解码用等价 FP16 操作保留单次舍入。
- 移植 [88844810](https://github.com/2-com-org/ninfer-5060-qwen/commit/88844810f04dee090ad5ec7bff12bc715b19cf59)：
  Program 准备期加载所需 small-T attention CUDA modules，提前承担 lazy load 成本。保留当前 KVMem 每个 geometry/width/input 一个编译单元、一个单元包含各 KV storage 的组织；未引入 fork 后来的 per-storage 文件拆分。
- 移植 [57583f6e](https://github.com/2-com-org/ninfer-5060-qwen/commit/57583f6e91685cce7732b29b19e453a067173000)：
  CPU ngram 索引持有输入副本并异步构建，第一次 proposal 等待并接管一次；异常传播，已取消请求的 worker 完成后自释放。保留 KVMem snapshot/archive 与非 drafting 行为。
- 适配动态 heads、TP geometry、state slots 和 request retirement；Windows kernel DLL 显式导出 launch/load 两类模板，共 56 个符号。更新 native 的 ngram、build-system、replay 文档。
- 移植中修正了候选自身的 conv-history 覆盖问题：replay 从每 head 8 tiles 改成 2 后，小型 dense 的 16 value heads / 6144 conv channels 不能再假定每 CTA 只发布一个 128-channel block。改成按 CTA 总数步进覆盖，dense-family 专项测试通过。这是本次适配修正。
- 不包含上一批 IQ tile kernel，也未修改权重/KV 格式、activation 精度或投机接受规则。

## 验证

全部 GPU 检查在 RTX 5060 Ti `GPU-A` 上运行；每次启动前检查空闲状态。

| 检查 | 结果 |
|---|---|
| INT4 解码独立精确参考 | 8,126,464 个有限非负 FP16 scale / packed code 组合，0 mismatch；覆盖 subnormal 和结果溢出。未验证非有限 scale 的 NaN payload identity |
| GDN fixed-order gating | 11 个宽度 × 旋转开/关共 22 例，h/g/beta 与冻结实现逐位零差异；另通过既有独立 FP64 数值参考测试 |
| GDN record/fold | 输出、record、state、conv-history、accepted-length、masked rows、FP16/FP32、graph、TP geometry 检查通过；小型 dense conv-history 专项通过 |
| Attention | 模块加载、INT8 prompt 独立参考、rk8v4 原测试通过 |
| Ngram CPU | 原语义测试与异步/同步 proposal 一致性、take-once、abandoned worker 自释放通过，oracle proposals=13,708 |
| Windows archive exports | 使用实际 28 个 CUDA 编译单元构建 archive，经 dumpbin 和现有 ArchiveExports.cmake 检查 56 exports 通过 |
| 补丁 | 整条 28-patch 链经现有 prepare-backends.py 的独立索引 preflight 通过；改变文件应用结果与候选逐文件核验（按基线 LF/CRLF 格式）；whitespace 检查通过 |

这是保持原有表示和操作顺序的实现，测试范围内未观察到数值/输出变化；不能据此推导所有模型、输入和设备都逐位一致，也不是完整质量评测。

## 真模型性能

模型：`Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp.ninfer`，同一个 5060 Ti UUID、Release sm120a、CUDA 13.2 / MSVC 14.44。
每种 KV 为 A-B-B-A 四次全新服务进程，A=基线、B=候选；共八次。模型路径与完整 argv 在 performance/*/setup.json。
统一 context=32768、prefill chunk=256、MTP drafts=4、ngram drafts=31、concurrency=1，KVMem B=24576/R=1024/Host=4096 MiB。

先发普通短请求观察首次调用，再发 14,806 tokens 文档要求中性确认；随后在同一会话单独问复制五条记录。
保存了两次真实请求与回复、native JSONL 和 timings。问题轮复用 14,807 tokens（private_endpoint），实际重算 36 tokens，输出 193 tokens。
所有八次文档回复都是 Received，问题回复也完全一致；MTP+ngram 接受 180/291，ngram 接受 168/279、9 rounds，未出现 proposal/接受差异。

此处 decode 是有重复文本的复制场景，包含 MTP+ngram 收益，不能外推成普通自由生成 decode 的绝对速度。
没有验证 native Query span，因此不作为完整 Query/Pressure 长文 recall 质量结论。

平均值（每个 arm 两次，百分比为候选相对基线）：

| 指标 | INT8 KV | rk8v4 KV |
|---|---:|---:|
| 长文 prefill（tok/s） | 706.0 → 712.8 (+0.97%) | 712.3 → 711.0 (-0.18%) |
| 复制 decode（tok/s） | 165.3 → 170.5 (+3.11%) | 165.1 → 171.5 (+3.83%) |
| 首个短请求总耗时（ms） | 1109 → 781 (-29.61%) | 1117 → 711 (-36.35%) |
| 首个短请求 TTFT（ms） | 790 → 553 (-30.02%) | 974 → 571 (-41.35%) |
| 启动至 health（s） | 11.59 → 12.19 (+5.12%) | 10.21 → 11.12 (+8.95%) |
| 文档轮总耗时（s） | 21.164 → 20.945 (-1.03%) | 20.984 → 21.008 (+0.11%) |
| 问题轮总耗时（s） | 1.836 → 1.781 (-2.97%) | 1.790 → 1.711 (-4.39%) |
| 问题轮 TTFT（ms） | 612 → 597 (-2.33%) | 561 → 546 (-2.65%) |
| 请求完成后显存（MiB） | 13364 → 13364 (+0.00%) | 13166 → 13166 (+0.00%) |

逐次测量（不以均值隐藏分布）：

| KV | 序号/实现 | 启动 s | 首请求 ms | 文档 tok/s | 复制 decode tok/s | 问题总耗时 s |
|---|---|---:|---:|---:|---:|---:|
| int8 | 0/baseline | 13.062 | 1016 | 700.12 | 165.80 | 1.844 |
| int8 | 1/candidate | 13.578 | 765 | 711.82 | 170.60 | 1.797 |
| int8 | 2/candidate | 10.797 | 797 | 713.86 | 170.36 | 1.766 |
| int8 | 3/baseline | 10.125 | 1203 | 711.83 | 164.86 | 1.828 |
| rk8v4 | 0/baseline | 10.297 | 1109 | 717.49 | 165.14 | 1.766 |
| rk8v4 | 1/candidate | 11.281 | 828 | 704.80 | 171.52 | 1.703 |
| rk8v4 | 2/candidate | 10.968 | 594 | 717.30 | 171.41 | 1.719 |
| rk8v4 | 3/baseline | 10.125 | 1125 | 707.20 | 165.13 | 1.813 |

相邻 A/B 对的长文 prefill：INT8 +1.67%, +0.29%；rk8v4 -1.77%, +1.43%。rk8v4 最差一对下降 1.77%，不足以宣称 prefill 提速，也不将其直接归为噪声。
复制 decode 相邻对均为正收益，INT8 +2.89%/+3.34%，rk8v4 +3.86%/+3.80%；样本量仍小，未测其他 prompt/纯 decode 或其他 chunk。
启动变慢与首请求变快必须一起考虑：模块加载前移并没有消除成本，不能仅用首请求耗时宣称冷启动整体更快。

## 构建、资源与限制

- 使用固定 build-dependencies.json，并在构建/验证前检查 bundle。复用原工具链/flags/architecture 的既有可兼容静态库；根据 Ninja deps 加本地 include 递归闭包，重编 52 个 CUDA 和 58 个 host 实现对象及专项测试，明确替代全部 ABI 消费方，再重链接。21 个复用库的最终 hash 均未变。
- 每次原生构建先检查目标 Ninja 计划。初始 8 jobs 时系统 RAM 使用率约 95%、可用仅约 1.4 GiB，停止自有构建进程，保留对象与 Ninja 记录后改 4 jobs 完成；后续单个 host relink 用 8 jobs。没有清理或伪造旧 build metadata。
- 这是有据可查的 focused validation build，不是完整 CMake release/package 构建。Windows DLL 导出 archive 已实测，但未生成/验证完整多架构 kernel DLL 发布包；未验证其他 GPU/架构、Linux、TP 多卡真模型和多请求压力。
- 构建完成后 recurrent.cuh 只调整了缩进/空格，去除所有空白后的 token 字节完全相同，记录在 post-build-formatting.json；没有为纯排版重复 CUDA 编译。
- ngram 引入每次准备请求的 CPU worker 及临时 token/source/boundary 副本，忙碌请求仍可能等到首次 proposal 才完成。请求取消不打断正在执行的建表；worker 完成后自释放。未测峰值 host RAM/多请求 CPU 压力。请求结束后的显存读数相同，不代表峰值显存相同。
- 未提交、未安装；两个 GPU 最终均为空闲。既有 frozen master 与上一批 tile 候选保留独立。

## 复现与产物

- build.ps1 / focused_build.py / dependency-plan.log / build-main-complete.log：编译闭包、目标 dry run 与编译记录。
- cpu_test.ps1、qualify.py、check_exports.ps1 及 test-*.log：CPU/GPU/Windows exports 检查。
- perf.py、performance/results.json、analysis.json、各 arm 请求/响应及 native.jsonl：完整性能与输出记录。
- source/backends/patches/ninfer-exact-ops.patch 与 source/backends/versions.json：待 review 的产品补丁和版本链。

补丁 SHA256：`2d8e1513f30fce7a22599857e053c063beea30bc1e00672e546091fc42a28c68`；最终 native tree：`bebe657c4b441d1e287babd36a23b6511633d1e3`。
