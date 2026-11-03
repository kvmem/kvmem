# Radiance 跨会话缓存短任务验收：2026-11-03

远端 RX 9060 XT 上的本轮缓存开发和短任务验收完成。66 次请求提交包含 63 次正常响应、2 次预期的存储错误响应，以及 1 次主动断开；383 项显式检查全部通过。没有运行 256K 长任务。

修复了两处来自 `fb891db` 缓存重构的回归：

1. 异步输入检查点交接先移动对象，再通过已移空的 identity 选择目标，导致首个文档请求在 `KVMEM_INPUT_RETAIN` 后退出。C++17 起赋值右侧先求值；GCC 15.2/C++20 独立探针确认了顺序。修复改为先构造完整对象，再读取该对象的 identity 发布到 input/fallback，新增所有权和无效完成回归检查。真实请求现在通过 `KVMEM_INPUT_READY` 并正常完成。
2. 监控把每次安全物理分叉都当作跨会话命中，同一 owner 的续接也显示 `shared`。物理分叉计数继续保留，新增跨 owner 分叉来源计数供监控使用；来源只有在原生恢复成功后计入实际命中。强化的实模型断言验证了 `own=4`、`shared=6`、`miss=4`，总计 10 次实际命中。

两处均为本次重构引入的缺陷；未更改冻结基线或继承行为。来源定义区分会话归属与物理共享，同一会话的安全分叉仍可共享封存的字节。

机器为 `ms10dk-gigi31.degima.ai:10011`，GPU 架构 `gfx1200`，AMD-SMI UUID `04007590-0000-1000-803f-e07311696099`，ROCr UUID `GPU-043fe07311696099`。显存实际总量 17,095,983,104 bytes。各组启动前均检查空闲，无其他计算任务被停止。

模型来自 `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`，固定 revision `5233554c5fa56afda40150556b95573c2d7d29c0`。转换后 `.rad` 为 17,775,964,160 bytes，SHA-256 `202c350fee4fa0ccb008607745ad830e10ccca41bd2c1ba2a74de02e2f72323e`。目标输出头及 MTP 投影使用 MXFP4，MTP 使用 shared head；embedding 放 RAM。

共同配置：逻辑上下文 8192、B=2048、生成预留 R=512、prefill batch=256、block=128、sink/recent=128/512、temperature=0、关闭思考、每次最多 8–64 个输出 token。这是功能验收配置；不是 B36K/R4K 的长任务或上线配置。

| 配置 | MTP / 并发 | H / D MiB | 提交数 | 检查数 | 采样峰值显存 GiB | 进程峰值 RSS GiB | VmHWM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp8-f32-mtp0 | 0 / 1 | 3072 / 0 | 10 | 57 | 13.337 | 5.350 | 5.502 |
| fp8-f32-mtp3-c2 | 3 / 2 | 3072 / 0 | 14 | 81 | 14.214 | 5.991 | 6.092 |
| k6-v4-f16-mtp3-c2 | 3 / 2 | 3072 / 0 | 14 | 81 | 13.860 | 4.867 | 4.972 |
| fp8-f32-mtp3-c2-ssd | 3 / 2 | 384 / 4096 | 14 | 77 | 14.213 | 4.457 | 4.567 |
| fp8-f32-mtp3-c2-owner | 3 / 2 | 3072 / 0 | 14 | 87 | 14.219 | 6.094 | 6.094 |

显存为整卡已用量，每 0.2 秒采样，包含模型加载阶段；更短的瞬时峰值可能遗漏。最高采样值 15,267,553,280 bytes（14.219 GiB，整卡 89.30%），该时刻约剩 1.70 GiB。所有组 VmSwap=0。H 是 KV/checkpoint 存储配额，不是进程 RSS 上限。配置有差异、输出较短，表格不用于吞吐提升或精度对照。

主矩阵覆盖命名和匿名跨会话复用、完整长 owner 键、父会话返回、前缀编辑后原分支存活、MTP3 双并发私有尾部隔离、原生工具调用和工具结果续接。每个正常请求核对 API cached_tokens、原生 actual_cached_tokens 和 native_request_id；工具续接的 Query span 为 (-1,-1,0)，并验证原生 forks 与物理共享抵扣。

最终来源组的几个原始测点如下；它们是不同请求阶段的观测，不是受控速度比较：

| 请求 | 输入 token | 实际复用 token | 来源 | TTFT ms | wall s | decode token/s |
|---|---:|---:|---|---:|---:|---:|
| 01-seed-A | 1064 | 0 | miss | 1457.29 | 2.383 | 22.69 |
| 02-fork-B | 1094 | 1067 | shared | 421.91 | 1.237 | 46.39 |
| 04-parent-return | 1094 | 1067 | own | 411.92 | 1.349 | 46.13 |
| 12-tool-result | 1413 | 1378 | own | 275.23 | 1.257 | 45.47 |

SSD 组使用 H=384 MiB / D=4096 MiB，通过几份短文档触发真实文件层回读。首次共享恢复读取 194,215,936 bytes（185.219 MiB）；全过程累计读取 1,357,840,384 bytes、写入 2,231,762,944 bytes。末次静止点 RAM payload=401,473,536 bytes、SSD payload=1,684,209,664 bytes，均在配额内。文件层读计数不代表绕过 OS 文件缓存的物理 NVMe 带宽。

| SSD 场景 | 结果 |
|---|---|
| 一次读取 EIO | HTTP 500，明确存储错误，约 0.092 s；原文随后仍命中并回答正确 |
| 回读延迟 35 s | 约 30.080 s 明确超时，原始 I/O 后台收尾；另一请求约 0.981 s 完成 |
| 回读中客户端断开 | 原生状态 disconnected；另一请求约 1.032 s 完成，随后父缓存仍可恢复 |
| 一次可选缓存写入 EIO | 当前请求正常返回 200，日志说明丢弃可选 input parking；服务和原缓存继续可用 |
| 收尾 | parking / finishing / pending I/O / active archive / active checkpoint 全部归零，RAM/SSD 配额正常 |

最终来源组保留逻辑缓存 3,159,533,036 bytes，物理共享抵扣 973,864,960 bytes，计账后 2,185,668,076 bytes。LRU 只释放对应引用，父分支和兄弟分支的封存数据仍被保留。

基准候选 `401a71d39b3fad4717ab93907b385814d77329c1` 上，12 组原生测试（含 5 组 GPU 测试）全部通过。交接修复后重跑了检查点 CPU 回归；最终来源修复后 2 组 CPU 回归通过，并由新的 14 请求实模型组验证。旧原生测试不标作最终新二进制的重跑结果。主矩阵和 SSD 组早于来源统计修复，其 own/shared 标签不可用作准确归属统计，复用 token 数和状态验证仍有效。

初始主机无可复用原生对象，已审查仅 gfx1200 的目标闭包后构建。两次修复的 dry-run 分别为 2 个主机对象 + 3 次链接、3 个主机对象 + 3 次链接；故障夹具为 1 个主机对象 + 1 次链接。全部使用 8 并行上限。修复没有编译 HIP/CUDA 内核，没有清理构建树或改写 Ninja 成功记录。固定依赖和 SDK 在构建及验证前检查。

该阶段源码差异清单 SHA-256 `e0684f537daad342c6b0e80efa340d58312a4725ecf397105e836e36c9706229`；服务 SHA-256 `26038a7dbc0fc8744979f859d82852c4d176759497556470242c9ca732908ff9`；未改动的 libr4d SHA-256 `f424e42d5f09d698930d76e5a60de5057a91c9da0e51678ff772d34c8ce9dddd`。6 个修复/测试源码文件的本地与远端哈希一致。

原始证据保存于本任务 `remote-ms10dk-20261102` 目录：`request-metrics.csv` 是全部 66 次提交的明细，`qualification-results.json` 是汇总。三个 `qualification-evidence-*.tar.gz` 包含请求、响应、HTTP 错误、监控采样、日志和构建记录，每个成员校验 SHA-256；`evidence-phases/primary`、`ssd`、`final` 分别保留各阶段原版本。取消场景最初未另存发送 body，能够由归档执行脚本和已保存的 seed 响应还原；取消中监控和原生日志已完整保留。CSV 对主动断开不伪造 HTTP 响应码。

该阶段检查显示测试服务均已退出，GPU 无计算进程，整卡已用显存回落到 59,895,808 bytes；未部署用户常驻服务。此构建使用 RAD_WITH_FFMPEG=OFF；本报告不扩展为 CPU 图片、TP2、RDNA3 或 256K 质量验收。

## 后续审计修复：活动存储超时与多 rank 异步收尾

两项已完成修复。新增 CPU 回归和原生内存 GPU 回归通过；另有 22 次短请求提交、112 项检查通过，覆盖活动 KV 写入超时、EIO、取消和后续 14 请求共享缓存回归。CPU 存储失败仅拒绝对应请求，设备错误保持致命处理，正常完成的延迟检查点继续发布。

完整记录：本任务归档 `remote-ms10dk-20261102/storage-fix-report.txt` 和 `storage-fix-results.json`。最终源码清单 SHA256 `359cf64e36e0eaefdb70513d8717a7f52ca446657075cae95e5215ef73ebe0ac`；多 rank 已通过 CPU 协调回归，未进行双 GPU 硬件测试。活动组与最后正常收尾保护的二进制阶段分别记录，未将其合并为受控性能比较。

## 后续修复：存储队列取消与前缀读取收尾

存储队列在取消尚未开始的任务时移除该任务，在发布完成前释放其持有的资源并归还配额；已经开始的 I/O 仍需安全收尾。Radiance 在请求取消或前缀读取超时时也尝试取消排队中的读取，避免其在无关慢 I/O 后继续占用请求槽。

最终版本通过 `spill_io_test`、`radiance_storage_recovery_test`、`checkpoint_image_test`、`tiered_kv_storage_test` 和 `radiance_tiered_archive_test`。受控 CPU 测试覆盖队列取消竞争、队列析构期间的取消、慢 I/O 后的槽位释放，以及已经开始的前缀读取收尾。

RX 9060 XT 实模型短测分两个独立阶段：队列修复中间版本通过 14 个请求、85 项检查；加入前缀读取修复后的最终版本通过 3 个请求、18 项检查，跨会话和原会话回访均复用 1067 个 token。这不是性能对照。最终版本源码清单 SHA256 `c7ad49304f1cd4b6667cee27ce33de6a29b56f6b267f4748b950669dc1eeb70a`，服务 SHA256 `ee1ef31c3ff9359e7dd0690ef58d069580c472137262453f2f1c223423151021`。故障并发由 CPU 受控 I/O 验证，未进行双 GPU 测试。原始记录见本任务 `remote-ms10dk-20261102/queue-cancel-fix-report.txt` 及 `queue-fix-short-results.json`。

## 提交前对齐 master

后续对齐 master `a84443bc11406c60400031e987d6b75db4a8fae1`，保留其 NInfer 低精度 prefill、ngram 仲裁、DFlash2 conditioning 与 PolyStrata poly.7 升级。共享缓存补丁分别追加在这些最新补丁之后；NInfer 37 项、PolyStrata 17 项补丁从固定基础提交逐项重建，所有中间树及最终树哈希通过。

三方合并的两处源码冲突均保留两边功能：NInfer 同时引入 DFlash conditioning 与 cache owner 头文件，PolyStrata 同时保留 poly.7 的 `disk_sessions` 属性与共享缓存的 `cache_reuse` 属性。此步骤未重新编译原生引擎；此前 RX 9060 XT 数据仍对应上文记录的 Radiance 二进制，不扩展为新 NInfer/PolyStrata 组合的实机验收。
