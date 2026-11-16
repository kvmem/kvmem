KVMem + Radiance 2.0 RDNA3 源码预览包
======================================

此包与同日期的 Windows RDNA3 预编译测试包对应，包含：
- kvmem-source/：KVMem 工作区源码快照，包含 Radiance 适配、补丁和测试。
- radiance-native/：已应用 KVMem/RDNA3 补丁的 Radiance 2.0 上游源码；
  .radiance-source-manifest.json 记录每个文件的 SHA-256 与补丁身份。
- third_party/include/nlohmann/：构建使用的 JSON 单头文件及许可证。
- source-info.json 和 SHA256SUMS.txt：版本、包含范围和逐文件校验值。

快照基于 KVMem Git HEAD，但包含尚未提交的本地修改。Radiance 上游版本与补丁哈希
见 source-info.json。压缩包不包含模型、GPU 驱动、ROCm SDK、编译器或构建产物。
为避免带入测试机器的路径与会话记录，省略了 docs 下的测试证据及含个人路径的
报告，个别原有文档链接可能因此失效。llama.cpp、ninfer 和 PolyStrata 是独立
Git 子模块，当前工作区未检出，未包含在本包内；重建本包的 Radiance 后端不需要它们。
KVMem 图形控制台源码位于 ui/monitor、ui/lightweight 和 scripts/ 目录。若要重新
构建聊天页面，需按 .gitmodules 和 backends/versions.json 准备固定版本的 llama.cpp
子模块，然后使用 Node.js 22、npm 运行 python scripts/build-webui.py；本源码包不含
预编译 UI、Node.js 或离线 npm 依赖。预编译测试包已包含 UI 和便携 Python 运行时。

Windows 构建示例
----------------
需要 64 位 Visual Studio C++ 构建环境（MSVC 及 Windows SDK）、CMake、Ninja、
Python 3.10+、可编译 HIP 的 ROCm/HIP SDK。预编译测试包使用 ROCm Core 10.1.0
构建，其他 SDK 版本尚未验证。以下命令在 x64 Visual Studio 开发者 PowerShell
中运行，将 $env:ROCM_PATH 设为实际 SDK 根目录后，在本文件所在目录执行：

  $env:ROCM_PATH = 'C:\path\to\rocm'
  $helper = '.\kvmem-source\backends\radiance\tools\build_windows.py'
  $argsCommon = @('--source','.\radiance-native','--build','.\build-rdna3',
      '--rocm',$env:ROCM_PATH,'--gpu-target','rdna3','--ffmpeg','OFF',
      '--json-include','.\third_party\include')
  python $helper @argsCommon --configure-only
  ninja -C .\build-rdna3 -n -d explain radiance-kvmem-server qwen35_w8a8 libr3d libr3diqr r3di_iqr_gpu_test
  python $helper @argsCommon --target radiance-kvmem-server --target qwen35_w8a8 --target libr3d --target libr3diqr --target r3di_iqr_gpu_test

--gpu-target rdna3 为 gfx1100、gfx1101、gfx1102、gfx1103 生成代码；如仅为单个
目标构建，可指定对应的 gfx 名称。构建前的源码校验会拒绝修改过的 native 源文件；
若需改源码，先理解并更新源码清单和适配器补丁身份，不要绕过版本检查。

Linux 构建示例（RX 7900 XTX / gfx1100）
------------------------------------------
需要 Linux 64 位、可编译 HIP 的 ROCm SDK、CMake、Ninja、Python 3.10+ 和 C++ 编译器。
在本文件所在目录执行，按实际 SDK 位置修改 ROCM_PATH：

  export ROCM_PATH=/opt/rocm
  helper=kvmem-source/backends/radiance/tools/build_linux.py
  python3 "$helper" --source radiance-native --build build-rdna3 --rocm "$ROCM_PATH" \
    --arch gfx1100 --ffmpeg OFF --json-include third_party/include --configure-only
  ninja -C build-rdna3 -n radiance-kvmem-server qwen35_w8a8 libr3d libr3diqr libref libavx r3di_iqr_gpu_test
  python3 "$helper" --source radiance-native --build build-rdna3 --rocm "$ROCM_PATH" \
    --arch gfx1100 --ffmpeg OFF --json-include third_party/include \
    --target radiance-kvmem-server --target qwen35_w8a8 --target libr3d --target libr3diqr \
    --target libref --target libavx --target r3di_iqr_gpu_test

构建助手会为 gfx1100–gfx1103 启用 Qwen IQR 算子集合。空闲显卡上可先运行
build-rdna3/bin/r3di_iqr_gpu_test build-rdna3/radiance_home/kernels/libr3diqr.so；
该测试包含 u5/bf16 分组标度的解码与预填充数值检查，默认选择第一个 HIP 设备，
要求显卡显存至少有 90% 空闲。模型需自行准备。

RDNA3 文本路径此前只在 Windows RX 7800M 上实机验证。Linux、RX 7900 XTX、
多卡、图像及其他模型仍需要分别实机验证。使用的模型及 GPU 工具链受各自许可证约束。
