# Radiance：Windows / Linux AMD后端

## 主线整合验证（2026-10-31）

统一控制台已接入 Radiance 的多会话前缀缓存、同前缀并行分支、最多八并发文本解码、
原生自适应 MTP，以及活动历史和闲置前缀共用配额的 RAM/SSD 分层。
MTP 默认 3、最多 7；启动器、界面和原生 worker 使用同一上限，超过上限在加载模型前拒绝。
该限制保留现有融合 GDN 实现，并保证部署窗口（草稿数 + 1）不超过其 8 个位置的容量。
Query 支持 page-softmax、meanq-cosine、Graph/LRFU 和有界回放；Prefill/Decode Pressure
只从驻留历史选块，普通工具续接不进行冷历史召回。

SSD 使用 `--kvmem-disk-path DIR --kvmem-disk-mib MiB`，与 `--kvmem-host-mib` 一起
限制共享载荷。文件仅在当前进程内有效，不支持重启恢复。磁盘读写由后台 CPU 队列执行，
GPU 状态和目录只在引擎线程发布。可选缓存落盘失败丢弃该项；必要读取失败只影响对应请求。

本次集成实测是 Linux、Radeon AI PRO R9700（gfx1201）、Qwen3.8-27B MXFP4、FP8 KV。
验证包括两阶段长文档召回、跨轮输出复用、八会话隔离、MTP0/3/5/7、并发取消及真实 SSD
延迟/EIO 注入。BF16/FP8 原生状态契约有独立测试。Windows AMD、其他模型和设备、
可配置 n-gram 草稿及 DFlash 组合尚未在这次整合中完成验证。
MTP7 通过固定及自适应八并发，以及共享启动器下的双会话长文档召回、SSD 和取消检查；
这些是功能正确性证据，不能据此声称 MTP7 比 MTP3 更快。

下文的 RX9060XT 配置和性能表属于 PR 作者原始测试记录，不能视为本次 R9700 的结果。

## 双卡接入

统一启动器的 `Gpu` 可指定一或两个不同的 ROCr UUID。双卡使用原生张量并行和
`--tp-wire exact`；Query/KV 头数必须能均分，不使用 NInfer 的层数切分设置。
各卡独立持有 KV 和递归状态，已接受 token 的统计按头拼接，Query token 数不重复累加。
工作集迁移在所有卡都准备好后统一发布；跨轮前缀、分支和 checkpoint 同时覆盖两卡。
RAM/SSD 配额为整个进程共享，不按卡重复授予。

本次双 R9700 测试采用 Linux、gfx1201、上述 Qwen3.8-27B MXFP4 和 FP8 KV，覆盖
MTP0/7、八并发、图文前缀、两阶段长上下文、两种 Query scorer、Prefill/Decode Pressure
及存储失败隔离。该机器 P2P 不可用，原生集合通信通过主机暂存；本次不据此推断多卡加速比。
Windows AMD 和其他模型/设备仍需独立资格验证。

对于 HSA 不能确认 CPU 可访问显存池的机器，AQL 内核参数环及录制参数改用已有的
一致性主机内存路径。该修复不迁移模型权重或 KV；保留可以确认 CPU 访问的原有显存路径。

## 原生图片接入

统一启动器的 `Vision=true` 开启 Radiance 原生图片处理器和视觉编码器；模型必须是
包含视觉权重及 patch/merge 元数据的 `qwen35` `.rad`。当前支持
`VisionResidency=resident|cpu`，支持原生 GPU 或完整 CPU 视觉编码器；不支持视频或音频。
CPU 图片已在 RX 9060 XT 上完成短任务验证，范围和资源指标见
[CPU 图片验收](radiance-cpu-image-validation-20261103.md)；
本轮线程配置、embedding 缓存和异步编码见[管线验收](radiance-vision-pipeline-validation-20261104.md)。
CPU 编码使用独立执行上下文在后台运行，编码结果由调度线程发布；语言模型可以继续处理其他请求。
统一启动器提供 `CpuThreads`（1–256，默认 2）和 `VisionCacheMiB`（0–65536，默认 256）。
`CpuThreads` 设置进程的 `OMP_NUM_THREADS`，也会影响其他使用 OpenMP 的 CPU 算子。
直接启动 worker 时使用 `OMP_NUM_THREADS` 和 `KVMEM_RAD_VISION_CACHE_MIB`。
缓存为进程内 LRU，0 关闭；预算计入 BF16 embedding 和用于精确核对身份的图片 patch 数据，
独立于 KVMem Host/SSD 预算。活动请求和后台编码器的工作内存另计。
重启或更换模型后不保留图片编码缓存。
`VisionTokens` 默认 1024、范围 64–16384，按模型几何换算为原生 patch 和像素上限；
这是单图压缩目标，图片数量还受上下文及 KVMem 工作集预算限制。

请求使用 OpenAI chat 的 `image_url` 内容项和 `data:image/png;base64,...` 或
`data:image/jpeg;base64,...`。原生服务不替客户端下载远程图片 URL。
图片和文字共用普通消息协议；跨轮发送完整对话历史即可复用 KV/state。
缓存身份同时检查图片内容的 128 位 key、完整展开区间和原生 MRoPE 布局，
因此相同占位 token 和尺寸不能使不同图片误命中。图片解码和预处理仍逐请求执行；视觉 embedding 可跨请求复用，
缓存同时核对内容 key、几何和完整 patch 数据，避免不同图片误命中。

图片完整页在 Query、Prefill Pressure 和 Decode Pressure 中始终保留；预算不足时
拒绝请求并提示减少图片或增加 Budget。Query 边界在图片展开后的真实 token 序列上证明，
支持后续字符串提问，以及多模态内容数组的最后一个文字项。工具续接仍不授予冷历史召回。

构建需要 FFmpeg 的 avformat、avcodec、swscale、avutil 开发库。通过
`--ffmpeg ON --ffmpeg-prefix /path/to/pinned/ffmpeg` 显式指定；静态库额外使用
`--ffmpeg-static`。`--ffmpeg OFF` 构建的 worker 会报告不支持图片，启动器会拒绝视觉配置。
本次 Linux 验证使用单独固定的 FFmpeg 8.1 最小图片解码依赖包，未改变原有 ROCm 依赖包。
分发时保留原生来源、FFmpeg 许可和对应构建信息；本次没有发布二进制包。

API 响应的 `timings.image` 和监控请求的 `image` 对象提供 `decode_ms`、
`preprocess_ms`、`queue_ms`、`encode_ms`、`cache_hits`、`cache_misses` 和 `tokens`。
`queue_ms` 从图片进入编码队列开始计时；CPU 首次执行上下文初始化计入此阶段。
CPU `encode_ms` 是视觉算子执行的 wall time，包含执行中的权重触页等等待；GPU 多图批次按 patch 数分摊批次耗时。
监控另提供正在编码/排队的图片数及当前编码耗时。跨请求缓存命中时编码耗时为零，
解码和预处理仍有开销。并行阶段存在重叠，不能将这些时间简单相加当作端到端耗时。

## 本次改动

为 KVMem 新增 **Windows / Linux 双系统 Radiance AMD 显卡后端**，实测设备为 RX 9060 XT 16 GiB（RDNA4/gfx1200）。

- 接入 KV 页管理、Host KV 换出与召回、前缀缓存和循环生成。
- 复用 KVMem 的公共压力选块逻辑，使用 mean-key 相关性统计。
- 修复 Windows GPU 地址位宽，优化显存分配与映射。
- 支持共享输出头，提供 MTP 和主输出头的 MXFP4 转换配方。
- 提供统一构建和启动入口：`--platform auto|windows|linux`。Python负责启动与资源管理，C++/HIP执行推理。

## 模型

[ModelScope：Qwen3.8-27B-Radiance-KVMem-MXFP4](https://modelscope.cn/models/Zeshion/Qwen3.8-27B-Radiance-KVMem-MXFP4)

模型文件：`target-head-mxfp4-a.rad`。仓库目前为私有，下载需访问权限。

基于 `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`，进一步量化六个 MTP 投影矩阵和主输出头为 MXFP4；K/V缓存均为FP8。

## 默认配置

| 参数 | 默认值 |
|---|---:|
| KVMem BUDGET | 36864 tokens |
| 生成预留 | 16384 tokens |
| Batch / context | 512 / 262144 |
| MTP | 3 |
| K / V cache | FP8 / FP8 |
| Block size | 128 tokens |
| Host KV 上限 | 9216 MiB |
| GPU 余量 | 256 MiB |
| Sink / recent | auto (-1) |
| Embedding / 视觉 | CPU 内存 / 纯文本 |
| 循环生成 | 启用 |
| 并发 | 1 |
| API / model ID | http://127.0.0.1:18380/v1 / radiance-local |
| 交互设置 | xhigh, 思考与答案共用输出额度, max_tokens=65536 |

生成时复用16384-token工作空间，交互请求的总输出额度为65536 tokens，包含思考和答案。

## Windows 原生部署及完整测试

依赖：Python3、Git、CMake≥3.24、Ninja、AMD HIP SDK7.2、VS x64 C++ Build Tools及Windows SDK。在 **x64 Native Tools开发者终端**、仓库根目录执行：

```powershell
# Fetch exactly the pinned upstream and apply the verified native patch.
python backends/radiance/tools/prepare_native.py --destination ../radiance-abi15

# Native Windows build and service; adjust model/SDK paths.
python backends/radiance/tools/build.py --platform windows --source ../radiance-abi15 --build ../radiance-build-windows --rocm "C:/Program Files/AMD/ROCm/7.2" --jobs 2
python backends/radiance/tools/serve.py --platform windows --exe ../radiance-build-windows/bin/kvmem-radiance-observer-probe.exe --home ../radiance-build-windows/radiance_home --model D:/models/target-head-mxfp4-a.rad --rocm "C:/Program Files/AMD/ROCm/7.2"
```

**2026-10-28 Windows原生版，256K / 33请求完整运行记录：**

| 指标 | 结果 |
|---|---:|
| 有效 Prefill | 792.93 token/s |
| 工具轮聚合 Decode（含思考） | 45.24 token/s |
| 最后代码请求 Decode | 41.78 token/s |
| MTP接受率 | 68.27% |
| 适配器专用显存峰值 | 16082.15 MiB |
| 运行阶段进程内存峰值 | 10925.13 MiB |
| 精确前缀恢复 | 31次 |

## Linux / WSL 部署及完整测试

依赖：Python3、Git、CMake、Ninja、C/C++构建工具及ROCm。使用对应系统编译产物，在仓库根目录执行：

```bash
python3 backends/radiance/tools/prepare_native.py --destination ../radiance-abi15
python3 backends/radiance/tools/build.py --platform linux --source ../radiance-abi15 --build ../radiance-build-linux --rocm /opt/rocm-7.2.3 --arch gfx1200 --jobs 2
python3 backends/radiance/tools/serve.py --platform linux --exe ../radiance-build-linux/bin/kvmem-radiance-observer-probe --home ../radiance-build-linux/radiance_home --model /data/models/target-head-mxfp4-a.rad --rocm /opt/rocm-7.2.3
```

WSL在服务命令后追加 `--wsl-runtime-dir /path/to/verified-rocdxg-reserve-v`。配套运行库构建步骤见 `backends/radiance/wsl-runtime/README.md`。

**Linux测试环境：WSL Ubuntu24.04、ROCm7.2.3，256K / 33请求历史完整运行记录：**

| 指标 | 结果 |
|---|---:|
| 有效 Prefill | 940.83 token/s |
| 工具轮聚合 Decode（含思考） | 48.01 token/s |
| 最后代码请求 Decode | 44.44 token/s |
| MTP接受率 | 68.27% |
| 适配器专用显存峰值 | 15736.91 MiB |
| 运行阶段进程内存峰值 | 11055.93 MiB |

两组33请求的性能设置为BUDGET36864、生成预留16384、batch512、context262144、MTP3、FP8 K/V；每请求输出上限512、思考上限128。表中数据对应各自历史测试版本。

## 与 README 性能指标对照

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

有效Prefill按新增输入总量除以Prefill总耗时计算；Decode按工具轮输出总量（含思考）除以生成总耗时计算。Windows RAM采用working set，WSL采用VmRSS，显存采用Windows适配器专用计数。各列使用表中对应硬件及模型配置。


## 接入客户端

- API：`http://127.0.0.1:18380/v1`
- 模型ID：`radiance-local`
- 交互参数：`reasoning_effort=xhigh`、`max_tokens=65536`、`temperature=1`、`top_p=0.95`、`top_k=20`。
- `--print-command`预览启动参数；`Ctrl+C`结束本次服务。

实现及部署说明：`docs/radiance-integration.md`；客户端示例：`docs/radiance/CLIENT-QUICKSTART.md`；页管理代码：`backends/radiance/adapter/`。引擎固定使用Radiance ABI15，保留上游许可和作者归属。
