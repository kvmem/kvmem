# Windows RX 9070 XT 上启动 KVMem Radiance

这套配置使用普通 `qwen3.8-27b-iq3r-v4.rad`、DFlash2 固定深度 7、单卡单会话、纯文本。实测平台为 Windows 原生、USB4 外接 RX 9070 XT 16 GiB、驱动 `32.0.31041.5005` 和 ROCm Core SDK `10.1.0`（HIP 运行时版本 `7.16.26385`）。[256K 测量与输出对照](evidence/windows-gfx1201-dflash7-task2-256k.json)

## 准备 worker、模型和前端

1. 在 x64 Visual Studio 开发者终端中将 `ROCM_PATH` 指向 SDK 根目录，目录下应有 `bin\amdhip64_7.dll`。已有配套 worker 时可直接使用。源码构建使用 [gfx1201 构建命令](../../backends/radiance/README.md)。保留同次构建的 `bin\radiance-kvmem-server.exe`、`rad_core.dll`、`radiance_home\kernels.rkb`、`kernels\` 和 `architectures\`。将 SDK 的配套运行库放到 worker 的 `bin\` 或加入 DLL 搜索路径。使用 LLVM OpenMP 的 CPU 插件还需要构建时使用的 `libomp140.x86_64.dll`。
2. 下载固定 revision `e5f1b3ef70741d705a64a28b852acf3fcefcead7` 的普通 IQ3R。文件大小为 14,261,252,096 字节，SHA-256 为 `99f306975bf5f48d6b023a177893fa85c6469ee080f2d7955caa3eaa16b84542`。[模型文件](https://huggingface.co/StillDeadcode/qwen3.8-27b-iqr-v4/blob/e5f1b3ef70741d705a64a28b852acf3fcefcead7/qwen3.8-27b-iq3r-v4.rad) · [国内镜像](https://hf-mirror.com/StillDeadcode/qwen3.8-27b-iqr-v4/blob/e5f1b3ef70741d705a64a28b852acf3fcefcead7/qwen3.8-27b-iq3r-v4.rad)
3. 完整控制台包内置聊天资产。用源码启动时，安装 64 位 Python 3.10+ 及控制台依赖，在仓库根目录运行下面的命令。Windows 源码目录宜用短路径，如 `D:\kvmem`，以便检出并复制前端依赖中的深层文件。轻量 UI 构建需要 Git、Node.js 和 npm，也可以在另一台构建机执行，再把整个输出目录复制到部署目录的 `ui/dist`。验收时该目录应包含 `index.html` 和 `console_chat_protocol` 为 1 的 `ui-build.json`。

```powershell
python -m pip install -r scripts/requirements-console.txt
python scripts/build-webui.py --output ui/dist
```

## 启动控制台并选择模型

在同一 PowerShell 窗口替换 SDK 路径并启动控制台。以下加载参数是该 16 GiB 实机使用的配置。控制台在启动时从 Windows HIP 运行库枚举 AMD 显卡，更换 SDK 或插拔外接显卡后重新打开控制台。

```powershell
$env:ROCM_PATH = 'D:\AMD\ROCm'
$env:HIP_PATH = $env:ROCM_PATH
$env:RADIANCE_LOAD_STAGING_MIB = '128'
$env:RADIANCE_LOAD_WORKERS = '1'
$env:RADIANCE_LOAD_WINDOW_MIB = '16'
$env:RADIANCE_LOAD_RELEASE_PAGES = '1'
python scripts/local-console.py --host 127.0.0.1 --port 18199 --open-browser
```

完整控制台包可在同一环境中运行 `.\START-HERE-CN.bat`。在「启动配置」选择 Radiance，指定 worker、普通 IQ3R 模型、`radiance_home`、SDK 根目录和实际枚举出的 RX 9070 XT。按下表设置后检查配置并启动服务。

| 设置 | 9070 XT 文本配置 |
|---|---|
| 上下文 / KV 预算 / 生成预留 / prefill 分块 | 262144 / 36864 / 4096 / 256 tokens |
| 并发 / 保留会话 | 1 / 1 |
| 草稿 | DFlash2，固定 7 |
| KV / recurrent state | FP8 / F32 |
| 文本 embedding / 图像 | RAM / 关闭 |
| Host / SSD | 4096 MiB / 8192 MiB，填写可写的 SSD 缓存目录 |
| GPU headroom / CPU 线程 | 256 MiB / 8 |
| 默认输出 / 思考预算 | 0 / 0，服务不额外设置上限 |
| 模型别名 / 服务地址 | `radiance-local` / `127.0.0.1:18200` |

生成预留是 GPU 工作空间。4K 是历史任务 2 末轮请求自己的生成上限。保留启动窗口，以便控制台管理 worker 和同一会话的缓存。

## 两轮验收

- 打开 [监控](http://127.0.0.1:18199/monitor/)，确认服务已就绪。worker 日志应出现 `both compute lanes use one stream (gfx1201 DFlash7)`。若要回退到双流，在启动窗口设置 `$env:KVMEM_RAD_HIP_SINGLE_STREAM='0'` 后重启控制台和 worker。
- 打开 [聊天](http://127.0.0.1:18199/chat/)，发送一段超过一个缓存页（128 tokens）的资料，附上自选口令。收到答复后在同一对话追问口令。在第二轮请求详情或原生 timings 中核对 `cache_n > 0`。页面提示 `Unified chat assets missing` 时，补齐前述完整轻量 UI 目录并刷新。
- 其他客户端可连接 `http://127.0.0.1:18200/v1`，模型别名与启动配置相同。结束测试时从控制台停止服务。

2026-11-10 在独立 Windows 部署目录中，使用前次 256K 实测的同一套 worker 和模型，以及从固定源码重新构建的轻量 UI，通过 Edge 聊天页面实际发送两轮。首轮处理 580 个输入 tokens；第二轮显示口令 `LANTERN`，共享缓存命中 632 tokens，新处理 24 tokens。两轮均以 `stop` 结束，监控返回了同一请求的计数，日志确认启用了单流。

[Linux IQ3R-S 实机部署](../radiance-iq3v4-runbook-20261105.md) · [WSL 运行库准备](../../backends/radiance/wsl-runtime/README.md) · [早期 MXFP4 部署记录](../radiance-integration.md)
