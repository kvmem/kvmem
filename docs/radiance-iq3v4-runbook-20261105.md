# IQ3R-S v4 启动与来源核对

本页只说明启动入口和证据边界。四阶段验收不会由普通服务启动入口触发。四种文本模式的合同和短探针已完成，最终 12K ALL 通过 155 checks。CPU 原始 PNG suite 为 21/22，OCR 将 `5829` 读为 `5029`；普通 native 无 observer 也得到相同结果，两条路径的 2 倍分辨率 probe 均回答正确。该质量失败保持可见，详见集成报告。最终服务已按原测试 argv/env 恢复并通过 health 与启动算术检查，地址为服务器上的 `http://127.0.0.1:18211`，记录时 PID 51422。它仍在运行时直接使用该服务，不要同时启动第二个实例。

本轮文件是 focused validation build，尚未制作安装包。实际服务入口为 `/root/kvmem-radiance-iq3v4-20261105/build-gfx1200/bin/radiance-kvmem-server`，插件 home 为同一构建目录的 `radiance_home`。已固定的设备是 RX 9060 XT 16 GiB、gfx1200、ROCr UUID `GPU-043fe07311696099`；启动前核对该卡和端口没有其他服务占用。

## 从最终服务 snapshot 单独重启

需要与验收逐项一致的服务设置时，读取 `/root/kvmem-radiance-iq3v4-20261105/final-service-snapshot.json` 中的 `argv` 和环境白名单。该 snapshot 的 `ready=true` 表示真实服务已恢复；`text_12k.pass=true` 与 21 项 `vision_runtime.pass=true` 分别记录通过范围，`vision_suite.pass=false` 和 OCR 质量限制仍明确保留。原始 cycle 的失败状态没有改写。下面片段只检查来源并启动一次 worker，不执行编译或四阶段测试，不扫描或停止其他进程；只有当前服务已经正常停止后才使用它。

```bash
python3 -B - /root/kvmem-radiance-iq3v4-20261105/final-service-snapshot.json <<'PY'
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

task = Path('/root/kvmem-radiance-iq3v4-20261105')
snapshot = json.loads(Path(sys.argv[1]).read_text())
assert snapshot['ready'] and snapshot['health']['status'] == 200
scope = snapshot['qualification']
assert scope['text_12k']['pass'] and scope['vision_runtime']['pass']
if not scope['canonical_ocr_quality']['pass']:
    print('Known low-resolution OCR limitation:', scope['canonical_ocr_quality'])
argv = snapshot['argv']
assert isinstance(argv, list) and all(isinstance(a, str) for a in argv)
assert Path(argv[0]).resolve() == task/'build-gfx1200/bin/radiance-kvmem-server'
assert '--api-key' not in argv, 'credentials must not be copied into this evidence example'
allowed = {'ROCR_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'LD_LIBRARY_PATH',
           'KVMEM_RAD_VISION_CACHE_MIB'}
recorded = snapshot['environment']
assert set(recorded) <= allowed and recorded['ROCR_VISIBLE_DEVICES'] == 'GPU-043fe07311696099'
env = dict(os.environ, **recorded)
for key in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
    env.pop(key, None)
for name in ('build-dependencies.json', 'native-source-dependencies.json',
             'media-build-dependencies.json'):
    subprocess.run([sys.executable, '/root/kvmem-build-deps/check.py', str(task/name)],
                   env=env, check=True)
for name, expected in snapshot['artifacts']['mapped_task_artifacts'].items():
    with Path(name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected['sha256'], name

# Reuse the reviewed guards without invoking their four-phase main function.
sys.path.insert(0, str(task))
from qualify_delivery_cycle import gpu_guard, port_available, process_record
run = task/('delivery-service-restart-' + str(time.time_ns()))
run.mkdir()
gpu_guard(run/'gpu-before-start.json', 0)
port_available(run/'port-before-start.json')
with (run/'worker.log').open('wb') as log:
    child = subprocess.Popen(argv, cwd=task, env=env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
(run/'command.json').write_text(json.dumps({'argv': argv, 'environment': recorded,
    'pid': child.pid, 'process': process_record(child.pid),
    'source_snapshot': str(Path(sys.argv[1]).resolve()),
    'qualification': scope}, indent=2) + '\n')
print(json.dumps({'pid': child.pid, 'log': str(run/'worker.log')}))
PY
```

该片段保留记录中的 SSD 路径。它不承诺跨进程恢复上一服务的历史缓存；需要独立空缓存时，在明确记录该差异后，为 `--kvmem-disk-path` 指定新目录。端口 guard 对无监听的残留占用最多等待 120 秒，发现真实监听立即拒绝。停止服务应使用保存的 PID、启动时间、exe 和 argv 核对身份后发送 SIGTERM，并等待正常退出。

## 日常使用共享启动器

共享 UI 和 `scripts/launcher-cli.py` 使用相同 structured configuration。V4 选择 `SpeculativeMode=dflash2`、`DraftTokens=7`、`DraftPolicy=fixed`；启动前读取容器并验证训练深度。关闭草稿使用 `SpeculativeMode=none` 和 `DraftTokens=0`。V4 不需要外置草稿文件。

下面是一份明确选择 CPU vision、OMP 8、64 MiB 图片缓存及 prefill 256 的日常配置。`VisionTokens=320` 会结合容器的 patch/merge geometry 生成 `--mm-max-patches 1280`；这是部署容量，不代表每张图片实际使用 320 tokens。本例和实机 cycle 都明确使用 Sink 128、Recent 512；cycle 的完整原生 argv 以其记录为准。共享 schema 没有 `Deterministic` 字段，因此本例不作为验收 argv 的逐 flag 重放。

```json
{
  "schema_version": 1,
  "parameters": {
    "Backend": "radiance",
    "Worker": "/root/kvmem-radiance-iq3v4-20261105/build-gfx1200/bin/radiance-kvmem-server",
    "RadianceHome": "/root/kvmem-radiance-iq3v4-20261105/build-gfx1200/radiance_home",
    "RocmPath": "/opt/rocm",
    "Model": "/root/kvmem-models/qwen38-27b-iqr-v4-e5f1b3ef/qwen3.8-27b-iq3r-s-v4.rad",
    "Gpu": "GPU-043fe07311696099",
    "ListenHost": "127.0.0.1",
    "Port": 18211,
    "ModelAlias": "radiance-local",
    "Context": 32768,
    "Budget": 4096,
    "Reserve": 512,
    "HostMiB": 32,
    "DiskPath": "/root/kvmem-radiance-iq3v4-20261105/service-v4-ssd",
    "DiskMiB": 8192,
    "Prefill": 256,
    "Concurrency": 1,
    "RetainedSessions": 2,
    "SinkTokens": 128,
    "RecentTokens": 512,
    "QueryScorer": "page-softmax",
    "ScoreAggregation": "mean",
    "GraphAlpha": 0.2,
    "Lrfu": false,
    "QueryMaxTokens": 512,
    "QueryReplayMaxTokens": 2048,
    "KvType": "fp8",
    "StateCacheType": "f32",
    "EmbeddingPlacement": "ram",
    "VramReserveMiB": 512,
    "SpeculativeMode": "dflash2",
    "DraftTokens": 7,
    "DraftPolicy": "fixed",
    "Vision": true,
    "VisionResidency": "cpu",
    "VisionTokens": 320,
    "VisionCacheMiB": 64,
    "CpuThreads": 8,
    "MaxTokens": 0,
    "ThinkingMode": "default"
  }
}
```

先前的 `VisionTokens=64`/mm256 小配置适合方形低容量请求，宽图可能由原生容量检查拒绝。真实模型 metadata 的 shortest edge 为 65,536、patch 16、merge 2、temporal 2，longest edge 为 16,777,216。本轮未改变的 384×128 OCR fixture 经既有 Qwen resize/ceil 后成为 160×448、280 patches／70 image tokens，超过 mm256，得到 HTTP 400；方图是 256×256、256 patches／64 tokens。扩大到 mm1280 只解除硬 cap，min pixels 65,536 和实际预处理几何不变，OCR 仍为 280 patches。该运行容量调整没有修改图片预处理或 fixture。容量通过后，原始 OCR 的 `5829→5029` 质量失败仍然保留；同一 worker 的 768×256 probe 使用 192 image tokens，回答正确，但不替换原始 suite。

保存为 `v4-cpuvision.json` 后，在目标 Linux 上预览和导出。`plan` 不加载模型到 GPU；`export` 会读取 worker 的 host-only capability，再生成脚本。运行导出结果给出的 `script` 路径即可启动一次服务，不需要注册 systemd 服务。

```bash
python3 -B /root/kvmem-radiance-iq3v4-20261105/source/scripts/launcher-cli.py plan --config v4-cpuvision.json
python3 -B /root/kvmem-radiance-iq3v4-20261105/source/scripts/launcher-cli.py export --config v4-cpuvision.json --profile radiance-v4-cpuvision
```

本轮 `qwen35-dense-tp1` worker 的二进制 capability 收窄为最多 1 device、TP1；不能用它运行 TP2。通用源码 manifest 的能力未收窄，通用 profile 仍独立保留，但 Windows、TP2 和 general profile 尚未获得本轮资格验证。预览 plan 后，还应核对实际 worker 的 `--kvmem-capabilities` 输出。

## 独立 serve.py 的范围

`backends/radiance/tools/serve.py` 已能根据容器选择 V4 的 `qwen35_w8a8` 插件和 DFlash2。`--drafter auto` 自动采用内置训练深度 7，也可显式传 `--drafter dflash2 --speculative-tokens 7`；错误类型或深度会在加载前拒绝。使用 `--print-command` 可先查看命令：

```bash
python3 -B /root/kvmem-radiance-iq3v4-20261105/source/backends/radiance/tools/serve.py \
  --platform linux \
  --exe /root/kvmem-radiance-iq3v4-20261105/build-gfx1200/bin/radiance-kvmem-server \
  --home /root/kvmem-radiance-iq3v4-20261105/build-gfx1200/radiance_home \
  --rocm /opt/rocm \
  --model /root/kvmem-models/qwen38-27b-iqr-v4-e5f1b3ef/qwen3.8-27b-iq3r-s-v4.rad \
  --drafter dflash2 --speculative-tokens 7 --print-command
```

该入口保留历史容量 profile：context 262144、B36864/R16384、Host 9216 MiB、prefill 256、`mm-max-patches=0`。Linux child 环境重设 OMP 为 2，清理继承的 KVMEM 设置；CLI 没有 CPU vision、图片缓存或预算覆盖选项。它用于保留既有启动行为，不能据此宣称复现本轮有界 CPU vision 配置。需要可调服务配置时使用共享启动器；需要精确复现验收参数时使用上面的 cycle 记录。

## 固定来源与验证边界

原生源码固定于 Radiance v1.17.0 / `19d4e8fe0e18071b9b6300f15fc4ccf38dc4fa3d`，archive SHA-256 为 `40062b0345225cb27cfeb64776e179dd6b414122bfc73c82853530093f8abbd1`。`backends/radiance/upstream.json` 固定七个补丁及各自 SHA：主集成、Qwen profile、AVX levels、profile closure、bounded admission、MXFP4 cast closure、CPU domain coverage。`provenance-final-seven-verified` 的 fresh replay 核对全部 1781 个 native 文件原始字节完全一致，没有使用换行规范化。CPU domain coverage 的独立 HOST 回归为 7 cases/22 checks。运行前的三份锁分别是 `build-dependencies.json`、`native-source-dependencies.json`、`media-build-dependencies.json`。

本轮窄内核 profile 为 46 HIP units / 159 rows；默认 all profile 保留 111 HIP units。固定 FFmpeg 8.1 bundle 仅提供 PNG/JPEG CPU 图片所需的媒体依赖，AVX 部署 profile 为 level 2。更改 profile、依赖或源码后，应先审查准确目标的 Ninja dry-run 与脏原因，再决定最小增量编译；普通服务启动不执行编译。

DFlash 工具续接的 prefix/refill 协商及 split-prefill Pressure reserve 边界属于本轮接入回归，已修复。四种文本模式的合法工具、取消、缓存、4K 两阶段文档召回及 SSD/Query 合同均通过。该组共享 worker/plugin 的历史对照构建早于 CPU domain coverage 与最终 12K sampling-capacity 修复；最终长测 worker SHA 为 `78fbfa36d1de082b58c66d0646de7df46aa46f0b09068fc26be935a21a6ae6a9`。本轮 12K 暴露的 sampling 容量参数缺陷已存在于 Nov4 的 Radiance 适配代码，不是本轮版本迁移新引入的问题，也不来自 common invariant 或原 llama 路径。只修候选源码，冻结的 Nov4 baseline 保持原样。最终 12K 155 checks 和 PNG 请求/缓存/placement 检查通过，完整图片 suite 21/22 的低分辨率 OCR 限制在普通 native 同样复现；JPEG 模型请求、一般图片质量、Windows、TP2 和 general profile 未资格验证。

独立本地 `--host-test` 使用 Radiance HOST backend，通过 24 个 RAM/SSD 变体和同一 engine stream 上 1100 次 checkpoint save/restore cycles，并验证未来 reserved +8 窗口的 trim/Pressure/anchor 边界。它不执行 GPU，不能替代实机模型验收。WSL 的 DrvFS `/mnt/c/...` 临时目录上，unlink 后继续 ftruncate spill descriptor 的诊断出现 errno 2；同一未改变的可执行文件在 Linux ext4 `/tmp` 上通过。证据是 `dflash-pressure-unlinked-spill-diagnostic.json` 和 `dflash-pressure-native-memory-linux-tmp-result.json`。该结果描述文件系统环境差异，不据此修改缓存实现或推断 GPU 故障。

cycle 的 `environment` 只记录 ROCr UUID、OMP、运行库路径和图片缓存预算四个必要字段；不会写出完整 `os.environ`。独立 MTP runner 另外明确记录 `KVMEM_RAD_MTP_HEAD=shared` 与 `PYTHONDONTWRITEBYTECODE=1`。发布 compact evidence 时仅保留这些非敏感配置字段。来源核对、实际 argv、worker/plugin hashes、请求与响应、原生计数和采样结果应各自保留明确的阶段索引。
