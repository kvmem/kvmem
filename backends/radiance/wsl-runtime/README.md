# WSL 独立 ROCDXG scratch 修复材料

这是 WSL Ubuntu 24.04、ROCm 7.2.3、RX9060XT/gfx1200 验收所用的独立运行库补丁，通过模型子进程加载。

## 来源与修改

- 上游：[ROCm/librocdxg](https://github.com/ROCm/librocdxg)，v1.2.0。
- 固定 commit：`fbd6f28765809e485d63a6a3edeb3fdbf76dc63c`。
- 补丁：`rocdxg-v1.2.0-scratch-reserve.patch`，仅修改 `src/wddm/queue.cpp`，10 行新增、1 行删除。
- 对非零 scratch 请求，初始每 wave 至少预留 8192 bytes；历史设备为 1024 waves，因而初始分配 8 MiB。分配量随 GPU 的 wave 数量变化。
- `UpdateScratch` 分配失败后立即返回 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`，避免原调用方忽略失败继续提交。保留 `SCRATCH_RESERVE_V` 日志。

原作者、版权与许可归属见 `UPSTREAM-LICENSE.md`。大部分源代码为 MIT；`src/thunk_proxy/libthunk_proxy.a` 采用单独的 AMD 二进制许可。本目录分发补丁、许可与说明。

## 在 WSL 构建到独立目录

需要 Git、CMake >=3.15、Ninja、GCC >=11.4，以及 Windows SDK 头文件。以下命令依据历史构建缓存整理，使用 GCC13、Release/Ninja 和 Windows SDK 10.0.26100.0。

在本目录执行，目标目录必须尚不存在，Windows SDK 版本按本机调整：

```bash
git clone https://github.com/ROCm/librocdxg.git ./librocdxg-source
git -C ./librocdxg-source checkout --detach fbd6f28765809e485d63a6a3edeb3fdbf76dc63c
git -C ./librocdxg-source apply --check ../rocdxg-v1.2.0-scratch-reserve.patch
git -C ./librocdxg-source apply ../rocdxg-v1.2.0-scratch-reserve.patch

cmake -S ./librocdxg-source -B ./librocdxg-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DWIN_SDK='/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/shared'
cmake --build ./librocdxg-build --parallel 2
```

运行库保留在独立构建目录，通过子进程环境加载。 编译目录内保留 `librocdxg.so`、`librocdxg.so.1`、`librocdxg.so.1.2.0` 的正常符号链接关系。

## 仅为模型子进程选择该库

历史验收采用以下环境逻辑（`runtime_dir` 和 `sdk_dir` 改为各自绝对路径）：

```bash
runtime_dir='/absolute/path/to/librocdxg-build'
sdk_dir='/opt/rocm-7.2.3'
env -u LD_PRELOAD \
  HSA_ENABLE_DXG_DETECTION=1 \
  LD_LIBRARY_PATH="$runtime_dir:$sdk_dir/lib:$sdk_dir/llvm/lib" \
  /absolute/path/to/kvmem-radiance-observer-probe [其余完整模型参数]
```

完整服务启动使用统一 launcher 的 `--wsl-runtime-dir` 参数，它为模型子进程设置上述加载环境。

## 验证记录

历史完整 33 请求使用 `librocdxg.so.1.2.0`，进程 maps 确认加载路径，日志确认初始 `requested_bytes=8388608 result=0`。来源和构建记录见 `provenance.json`、`historical-build-evidence.json`。

新构建依次执行 GPU scratch/复制/回读测试和固定模型请求，记录实际加载路径、scratch 分配、MTP 一致性及资源占用。模型测试串行运行。历史退出日志中的 `destroy_allocation -22` 留在原始证据中。
