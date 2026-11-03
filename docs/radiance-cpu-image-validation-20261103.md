# Radiance CPU 图片短任务验收（2026-11-03）

CPU 视觉路径短测通过：9 次正常响应、1 次预期的无效 PNG HTTP 400、1 次流式断开，共 11 次请求提交；原执行脚本的 57 项检查均通过。审计后又对原始证据进行了独立离线核验，没有重跑 GPU 或修改推理引擎。

## 环境和配置

- Intel Core i7-8700，6 核/12 线程，`OMP_NUM_THREADS=8`；RX 9060 XT / gfx1200，UUID `04007590-0000-1000-803f-e07311696099`。
- `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`，revision `5233554c5fa56afda40150556b95573c2d7d29c0`；目标输出头与 MTP 投影为 MXFP4。模型身份沿用已有转换记录和路径，本次没有重新全量哈希模型。
- 完整视觉编码器、权重及激活在 CPU，语言模型在 GPU，embedding 放 RAM。启动日志确认 `vision: CPU execution`。
- FP8 KV / F32 state、MTP3、并发 2、逻辑上下文 8192、B2048/R512、batch256、block128、sink/recent128/512、host3072MiB、disk0；temperature0、关闭思考，正常请求输出限额 16–64 token。
- `--mm-max-patches 1024`。128×128、512×512 的红蓝对半 PNG 经预处理后，分别得到 64、256 个图片 token。BF16 编码输出大小为 655360、2621440 bytes，隐藏维度均为 5120；输出有限且非零。

## 功能和时间

| 场景 | 回答/行为 | 实际命中 token | wall 秒 | TTFT 秒 |
|---|---|---:|---:|---:|
| 首个 64-token 图片 | `LEFT=red; RIGHT=blue` | 0 | 24.399 | 23.509 |
| 另一个会话续问右侧颜色 | `blue` | 192 | 2.365 | 1.587 |
| 同尺寸反转图片 | `LEFT=blue; RIGHT=red` | 0 | 2.367 | 1.627 |
| 256-token 图片 | `LEFT=red; RIGHT=blue` | 0 | 9.384 | 8.547 |
| 两张图的左侧颜色 | `red, blue` | 0 | 3.631 | 2.604 |
| 两路并发，红左蓝右 | 颜色和重复行正确 | 128 | 4.505 | 2.784 |
| 两路并发，蓝左红右 | 颜色和重复行正确 | 128 | 5.243 | 3.236 |

首次请求与后续请求分别报告；现有计时不能把首次开销精确拆成 CPU 编码、触页或运行时初始化。wall 包括视觉准备和生成；LLM prefill token/s 不是端到端图片吞吐。没有 CPU/GPU 速度对照。

无效 PNG 返回 HTTP 400，其后文本 `2+2` 回答 `4`。带图生成收到首个内容 token 后断开客户端，SSE 的请求 ID 与原生监控中的 `disconnected` 状态一致，其后 `3+4` 回答 `7`。采样确认两个原生活动请求重叠，不能将此等同于 CPU 编码器并行执行。

收尾时 parking、finishing、pending I/O、active archive 和 active checkpoint 均为零。退出前的 1633550336 bytes host payload 是保留的有效会话缓存，不是异步资源泄漏。服务正常退出，GPU 无计算进程，整卡显存回到 59895808 bytes。

每 0.2 秒采样，包含模型加载阶段：整卡显存峰值 **15274328064 bytes / 14.225 GiB**，进程采样 RSS 峰值 **6375186432 bytes / 5.937 GiB**，VmHWM **6527901696 bytes / 6.080 GiB**，VmSwap 为零。更短的瞬时峰值可能遗漏。

完整逐请求 wall、TTFT、输入/输出/命中 token、prefill 与 decode 指标及版本摘要见 [机器可读记录](benchmarks/radiance-cpu-image-20261103.json)。

## 构建与审计

远端原先无 FFmpeg 开发库。固定 FFmpeg 8.1 源码仅启用 PNG 解码/解析/解复用，其他解码器、编码器、网络和程序关闭。审计了五个必要库的 221 个公共/PNG C 对象后以 8 并行构建、封存依赖，合计约 16.12 秒。固定 bundle 为 `/root/kvmem-build-deps/radiance-ffmpeg81-png-b072aed6-gcc15-v1`。

Radiance dry-run 移除不必要的 `rad-info` 后为 7 个主机对象、7 次链接；增量构建和五组合约测试合计约 8.45 秒，没有 GPU 内核编译。libavx、libref、libr4d 哈希均未改变。

`cpu_vision_test`、`cpu_residual_alias_test`、`radiance_image_contract_test`、`radiance_media_query_test`、`radiance_observation_slice_test` 全部通过；libavx/libref 独立残差算术共 240 个用例，覆盖 F32/BF16、激活及精确/部分/跨步别名。

源码审计将 147 个 Radiance/KVMem 源码文件与合并提交 `85c3533f6d7a1e7fd1f40082c013094bddfe43df` 比对，无差异；仅为比较归一化换行，没有重写源码。原生 revision `d5b65cddc6c763861377c77692fb7ee5375222ec` 的完整源码清单与文件哈希也通过校验。

- 适配层源码差异清单 SHA256：`c7ad49304f1cd4b6667cee27ce33de6a29b56f6b267f4748b950669dc1eeb70a`。
- 启用 PNG 解码的服务 SHA256：`a224499e96b5ba2af3b67fc2fd7160adcf903952ffdbfdb2d739a070ac8ba43f`。
- FFmpeg 源码 SHA256：`b072aed6871998cce9b36e7774033105ca29e33632be5b6347f3206898e0756a`。

原执行脚本的颜色断言仅比较词语顺序；离线审计改为检查明确的 LEFT/RIGHT 赋值，并核对请求中的实际 PNG、输出字节数、缓存计数、取消 ID 和采样峰值。新增 5 项单元测试拒绝颜色误判、矛盾赋值、截断、NaN/Inf 和全零输出。上述补强不改变已保存的运行结果或历史检查数。

提交前 Radiance Python 测试集合共 83 项通过，包含新增核验器测试；严格离线核验对原始 76 成员归档通过。本轮审计未进行原生构建。

## 证据和范围

原始 `cpu-image-evidence.tar.gz` 保存在任务证据归档中；模型和二进制归档不加入 Git。归档包含 76 个逐文件大小/SHA256 已核验成员，整体 SHA256 为 `9cdacb9b51cddfc05c05af76aa70c21c3cdc479aab25b6943b1f4c455564514f`。取得原始归档后，在仓库根目录运行：

```sh
python backends/radiance/tools/check_cpu_image_evidence.py /path/to/cpu-image-evidence.tar.gz --sha256 9cdacb9b51cddfc05c05af76aa70c21c3cdc479aab25b6943b1f4c455564514f
python backends/radiance/tests/cpu_image_evidence_test.py
```

本次是两色 PNG、64/256 token、MTP3、两并发提交的功能验收，未扩展到自然图片/OCR 质量、JPEG/视频/音频、MTP0、其他 CPU ISA、RDNA3、多 GPU 或 256K。未部署用户常驻服务。
