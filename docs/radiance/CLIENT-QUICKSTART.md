# 给接手者的启动和客户端接入说明

## 先分清三个东西

1. **模型文件**：ModelScope 私有仓库 `Zeshion/Qwen3.8-27B-Radiance-KVMem-MXFP4` 中的 `target-head-mxfp4-a.rad`。使用Radiance服务加载。
2. **推理服务**：Radiance 的 C++ / AMD GPU 程序，加上 KVMem adapter。Python启动脚本负责参数、环境和内存保护，C++/HIP执行模型算子。
3. **Harness / DSH**：调用本机 HTTP API 的客户端。部署流程为准备模型、编译并启动服务、连接客户端；当前DSH通过生命周期插件管理服务。

## 当前机器：现成可用版本

直接使用 Windows DeepSeek Harness。已部署的本地 provider 为 `radiance-local`，API为 `http://127.0.0.1:18380/v1`。新建会话默认xhigh；旧会话可能保留以前的Off设置。源码包提供通用启动例程，现用客户端配置备份保留在本机部署目录。

同一时间运行一个模型进程。

## 新机器：统一入口及系统选择

解压源码包后进入其中的 `kvmem` 目录。旁边的 `../radiance-native` 已包含完整配套源码和校验清单，可离线用作 `--source`。源码与插件统一使用配套ABI15版本。模型另外从ModelScope仓库获取；当前是私有仓库，需账号有权访问。

统一构建入口为 `backends/radiance/tools/build.py`，统一服务入口为 `backends/radiance/tools/serve.py`。
两者均支持 `--platform auto|windows|linux`；`auto` 检测当前系统。在Windows用 `windows`，在原生Linux或WSL终端内用 `linux`。该参数选择当前系统的实现。各系统分别编译自己的程序和插件。

### Windows原生

准备Python3、CMake、Ninja、VS x64 C++ Build Tools/Windows SDK、AMD HIP SDK7.2及兼容驱动。以下在 **x64 Native Tools 开发者终端**运行，模型/SDK路径按本机替换：

```powershell
python backends/radiance/tools/build.py --platform windows --source ../radiance-native --build ../radiance-build-windows --rocm "C:/Program Files/AMD/ROCm/7.2" --jobs 2

python backends/radiance/tools/serve.py --platform windows --exe ../radiance-build-windows/bin/kvmem-radiance-observer-probe.exe --home ../radiance-build-windows/radiance_home --model D:/models/target-head-mxfp4-a.rad --rocm "C:/Program Files/AMD/ROCm/7.2"
```

### 原生Linux

准备Python3、CMake、Ninja、C/C++构建工具、ROCm开发SDK及正常GPU驱动。以下在Linux shell中运行；SDK版本和模型路径按真实安装调整：

```bash
python3 backends/radiance/tools/build.py --platform linux --source ../radiance-native --build ../radiance-build-linux --rocm /opt/rocm-7.2.3 --arch gfx1200 --jobs 2

python3 backends/radiance/tools/serve.py --platform linux --exe ../radiance-build-linux/bin/kvmem-radiance-observer-probe --home ../radiance-build-linux/radiance_home --model /data/models/target-head-mxfp4-a.rad --rocm /opt/rocm-7.2.3
```

### WSL

在WSL发行版终端内使用Linux构建命令。模型路径可为 `/mnt/d/models/target-head-mxfp4-a.rad`。
WSL配置使用独立的scratch运行库修复，按下述步骤准备。
必须先备好文档指定的同版本运行库，再传 `--wsl-runtime-dir`：

```bash
python3 backends/radiance/tools/serve.py --platform linux --exe ../radiance-build-linux/bin/kvmem-radiance-observer-probe --home ../radiance-build-linux/radiance_home --model /mnt/d/models/target-head-mxfp4-a.rad --rocm /opt/rocm-7.2.3 --wsl-runtime-dir /path/to/verified-rocdxg-reserve-v
```


### 启动前检查与停止

在对应服务命令末尾加 `--print-command` 可核查路径/参数而不加载模型；去掉后才真正启动服务。`Ctrl+C` 停止这个启动器创建的模型子进程。同一时间运行一个模型进程。

实测设备为RX9060XT/gfx1200；新机器部署后先执行短请求。

## 默认部署配置

| 名称 | 当前显式部署profile |
|---|---:|
| KVMem选择预算 | 36864 tokens |
| 生成/验证KV工作空间预留 | 16384 tokens |
| batch / context | 512 / 262144 |
| MTP | 3 |
| K和V类型 | FP8 / FP8 |
| Host archive上限 | 9216 MiB |
| GPU headroom | 256 MiB |
| 块大小 | 128 tokens |
| 日常单请求总输出上限 | 客户端传65536，含思考和答案 |
| 思考 | xhigh，不传独立thinking_budget |

max_tokens65536为单请求总输出上限，长生成循环复用GPU工作空间。并发为1，服务监听本机loopback。

## 客户端最小调用

服务ready之后，任何支持OpenAI兼容chat/completions的客户端可用：

- base URL：`http://127.0.0.1:18380/v1`
- model：`radiance-local`
- API key：本机服务不验证云端密钥；客户端若要求非空可使用无敏感含义的本地占位值。
- temperature=1、top_p=.95、top_k=20；xhigh；不另设thinking_token_budget。

Python标准库示例（无需安装OpenAI SDK）：

```python
import json
import urllib.request

payload = {
    "model": "radiance-local",
    "messages": [{"role": "user", "content": "写个鹈鹕骑自行车的html网页，要求使用html 和 svg 动画实现，有速度调节功能"}],
    "reasoning_effort": "xhigh",
    "max_tokens": 65536,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "seed": 42,
    "stream": False,
}
request = urllib.request.Request(
    "http://127.0.0.1:18380/v1/chat/completions",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=3600) as response:
    result = json.load(response)
print(result["choices"][0]["message"].get("content", ""))
print("finish_reason:", result["choices"][0].get("finish_reason"))
print("usage:", result.get("usage"))
```

此示例演示客户端调用。模型效果对照时保存包含system prompt、历史及采样参数的完整请求。

## 发给维护者的话（可直接改称呼使用）

> 为 KVMem 接入 Radiance AMD RDNA4 后端，支持 Windows 原生与 Linux/WSL。提供 packed KV 页布局、Host KV 换出与召回、前缀缓存、循环生成，以及 Windows 显存分配与映射优化。复用公共 pressure_selection，采用 mean-key 评分。默认 BUDGET 36864、生成预留 16384、batch 512、context 262144、MTP 3，K/V 均为 FP8。模型通过 ModelScope 提供，配套文档包含双系统完整测试数据、量化清单及构建和启动命令。

部署说明及完整测试表见 `docs/radiance-integration.md`。
