# Radiance：部署配置与实测性能对照

更新日期：2026-10-31。本文对照 KVMem README 的配置及 256K 文本任务指标。上游基准固定为 [`89c66980ed88`](https://github.com/kvmem/kvmem/blob/89c66980ed885866251414e1897518bdf008d211/docs/recommended-config-performance.md)。

**下表为历史 33 请求测试；本次 PR 新版的五请求短测见 [短测记录](QUICK-VALIDATION-20261031.md)。**

## 推荐配置逐行对照

| 配置 | IQ3 | IQ4 | Radiance Windows（历史测试/当前交互分列说明） |
|---|---|---| --- |
| 启动脚本 | [start-iq3.sh](https://github.com/kvmem/kvmem/blob/89c66980ed885866251414e1897518bdf008d211/scripts/start-iq3.sh) | [start-iq4.sh](https://github.com/kvmem/kvmem/blob/89c66980ed885866251414e1897518bdf008d211/scripts/start-iq4.sh) | 见配套部署文档 |
| 主权重 | IQ3_S，GSQ-RCO | IQ4_XS，MTP 权重 Q4_0 | AMD MXFP4骨干；额外6个MTP投影和主输出头MXFP4 |
| 主模型设备 | RTX 5060 Ti | RTX 5060 Ti | RX 9060 XT 16 GiB（RDNA4/gfx1200） |
| 视觉头 / 设备 | Q8_0 / GPU | BF16 / CPU | 未加载；纯文本部署 |
| 主 KV / MTP KV | Q8_0 / F16 | Q5_0 / F16 | FP8 / FP8 |
| MTP 草稿长度 / 状态保存 | 3 / ReplaySSM | 3 / ReplaySSM | 3 / Radiance原生recurrent checkpoint + replay |
| 检索预算 | 36864（36 × 1024） | 32768（32 × 1024） | 36864 |
| 生成预留 / 默认最大输出 | 16384（16 × 1024） | 12288（12 × 1024） | 16384 / 旧测试512；新交互65536（循环生成） |
| context / batch | 262144 / 512 | 262144 / 512 | 262144 / 512 |
| query replay / policy | auto / user | auto / user | 冻结 user query / mean-key 评分 |
| 正式脚本默认思考预算 | 4096 | 4096 | 旧33轮128；新交互不设独立上限、xhigh |

## Task 2：历史 256K / 33 请求性能

| 指标 | IQ3 | IQ4 | 历史ROCm IQ3 / RX7900XTX | 历史ROCm IQ4 / RX7900XTX | Radiance WSL / RX9060XT | Radiance Windows / RX9060XT |
|---|---:|---:| --- | --- | --- | --- |
| 全任务聚合 prefill，首遍 | **437.13 token/s** | **463.18 token/s** | 384.47 token/s | 419.52 token/s | 未独立计时 | 未独立计时 |
| 全任务聚合 prefill，有效 | **242.06 token/s** | **253.41 token/s** | 216.81 token/s | 232.00 token/s | 940.83 token/s | **792.93 token/s** |
| 工具轮次聚合 decode，含思考 | **31.74 token/s** | **33.31 token/s** | 33.77 token/s | 39.31 token/s | 48.01 token/s | **45.24 token/s** |
| 最终代码 decode，512 token | 30.53 token/s | 38.15 token/s | 39.34 token/s（仅232 token） | 44.78 token/s | 44.44 token/s | 41.78 token/s |
| 工具轮次 MTP 草稿接受率 | 64.70% | 67.08% | 59.66% | 60.79% | 68.27% | 68.27% |
| 整卡采样峰值显存 | **15617.10 MiB** | **15591.69 MiB** | 16011 MiB | 16164 MiB | 15736.91 MiB（Windows适配器专用计数） | 16082.15 MiB（Windows适配器专用计数，采样覆盖加载及运行） |
| 峰值时可用显存 | **434.90 MiB** | **460.31 MiB** | 8549 MiB（24GiB卡） | 8396 MiB（24GiB卡） | 未独立采集 | 未独立采集 |
| 运行阶段进程 RAM 峰值（RSS，不含加载） | **13483.52 MiB** | **11244.75 MiB** | 13067.95 MiB | 10978.13 MiB | 11055.93 MiB（WSL VmRSS） | 10925.13 MiB（Windows working set，排除加载） |

ROCm历史两列来自 [RX7900XTX测试记录](https://github.com/kvmem/kvmem/blob/89c66980ed885866251414e1897518bdf008d211/docs/rocm-recommended-config-performance.md)。各列的硬件、权重与KV格式按配置表列出。

### 统计口径与运行记录

- 有效 prefill = 新增逻辑输入位置 / prefill总耗时；Windows为261527 /329.8222894 = **792.93 token/s**。
- 工具轮 decode = 后32请求的输出（含thinking）/生成总耗时；6469 /142.9989088 = **45.24 token/s**。
- MTP接受率 = 总接受草稿tokens / 总草稿tokens；4332 /6345 = **68.27%**。
- Windows显存采用适配器DedicatedUsage，RAM采用运行阶段working set；WSL RAM采用VmRSS。
- Windows33请求达到261528输入+512输出，完成31次精确前缀恢复；物理KV池54528tokens /2104.125MiB，spill1829、restore0。Host召回由独立短测验证，restore191/197块。

原始指标见 `evidence/windows-final33-readme-metrics.json` 和 `evidence/wsl-final33-readme-metrics.json`。表中展示各部署的实测性能；硬件、模型表示和缓存行为均随部署不同。

## 版本与验证记录

| 版本 | 实际验证 |
|---|---|
| 2026-10-28 Windows，exe `53a7d5fa…` | 上述 33 请求及独立 Host 召回 |
| 2026-10-30 DSH部署版，exe `52969463…` | prefix修复CPU回归；原生思考续接生成完整动画HTML；App短调用 |
| 2026-10-31 PR候选（KVMem `89c66980`） | Windows全量构建；5请求短测：2次Host召回+3次prefix |

新交互版配置仍B36864/R16384/batch512/context262144、FP8 K/V、MTP3。xhigh、不独立限制thinking、max_tokens65536来自新客户端/循环适配；旧33测试仍是thinking128/max512。以上性能表采用旧33轮的请求设置。
