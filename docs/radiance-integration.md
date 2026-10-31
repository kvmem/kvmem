# Radiance：Windows / Linux AMD后端

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
