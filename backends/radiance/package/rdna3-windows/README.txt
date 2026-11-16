KVMem + Radiance 2.0 / RDNA3 Windows 预览测试包
================================================

这是供兼容性测试使用的独立便携包，不会覆盖现有 KVMem 安装。
内核目标为 gfx1100、gfx1101、gfx1102、gfx1103；目前实际验证的显卡只有
Windows RX 7800M（gfx1101）。请勿把此包视为已通过所有 RX 7000 型号验证的正式版。
已在 RX 7800M 上短测单卡、单会话的 Qwen3.8 27B IQ2R、IQ3R 和 IQ4R-S 文本，
包括 IQ3R/IQ4R 的 DFlash2。图像、多卡和其他模型尚未验证。

准备
----
1. 使用 Windows 11 64 位和支持该显卡的 AMD 驱动。包内含 HIP 运行库，但不含显卡驱动。
2. 将压缩包完整解压到可写目录，建议路径为 C:\KVMem-RDNA3，避免直接在压缩包内运行。
3. 自行准备 .rad 模型。此前验证的模型：
   https://hf-mirror.com/StillDeadcode/qwen3.8-27b-iqr-v4/blob/main/qwen3.8-27b-iq2r-l-v4.rad
   文件大小 11,990,364,160 字节，SHA-256：
   89fb6bfbd602594ed8a79b655bee7e15e7732b7c2c78e34db3d145b2e58706bd
   可将模型放到本目录 models 文件夹（只有一个 .rad 时自动选取），也可启动时输入
   完整路径。模型未包含在本包内。8 GB 显卡不能按这次 27B 的已测配置装下模型；
   更小的模型或卸载方案需要另行验证，单纯减少 KV budget 不保证足够。

先检查显卡
----------
关闭模型服务并让 AMD 显卡空闲，双击 check-gpu.cmd。它运行 IQR 和 DFlash 算子
数值测试，结果写入 logs 目录。检查会要求 GPU 至少约 90% 的显存空闲；如果失败，
请保留日志，不要把其他 GPU 上的结果当作 AMD 卡结果。

启动图形界面并聊天
------------------
1. 双击 START-HERE-CN.bat；提示监听地址及端口时直接回车可使用本机默认值。
   浏览器将打开 KVMem 控制台，包含启动配置、聊天、监控和日志。请保持启动窗口打开。
2. 在启动配置中选择 .rad 模型和 AMD 显卡，检查配置并启动服务。默认是 RDNA3 文本
   模板：BF16 KV、budget 1024、无草稿。选择“RDNA3 · DFlash2 测试”模板可启用
   DFlash2（budget 3072、recent 2176），模型必须包含兼容的 DFlash2 草稿权重。
   如果模型无法放入显存，可选择“RDNA3 · 主存卸载”或“RDNA3 · 主存卸载 + DFlash2”。
   这两个模板使用 8192 MiB 模型权重主存预算和 RAM 嵌入表；确保有足够的可用 RAM，
   按实际机器调整。主存卸载受 GPU 连接带宽影响，可能明显变慢。该预算与 KV 主存预算
   分开，可在“缓存与历史”中分别修改。
3. 服务就绪后切换到“聊天”，也可在“监控”查看性能。默认控制台地址：
   http://127.0.0.1:18199/monitor/；OpenAI 兼容 API 默认地址：
   http://127.0.0.1:18200/v1。通过界面停止服务；关闭启动窗口会停止其启动的服务。

命令行备用入口
--------------
也可双击 start.cmd 或 start-dflash2.cmd 启动命令行服务，再双击 chat.cmd 对话。
该备用服务默认使用 127.0.0.1:18380，不要同时启动多个会占用显存的服务。

高级选项（PowerShell）
----------------------
可按需调整模型路径、端口、KV 类型和预算。例如：
  powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 -Model 'D:\models\model.rad' -Drafter off -Budget 1024 -Recent 256 -KV bf16
  powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 -Model 'D:\models\model.rad' -Drafter dflash2 -Budget 3072 -Recent 2176 -KV bf16
FP8 KV 是可选项，但未对本模型做完整实机验证。DFlash2 的 recent 不能低于 2176。
多 AMD 显卡系统可在启动前设置 HIP_VISIBLE_DEVICES；设备编号由 HIP 运行时决定，
请以服务日志中实际显示的 GPU 为准。图形界面已内置 Python，无需另行安装。

反馈
----
请提供显卡具体型号及显存、AMD 驱动版本、Windows 版本、模型文件和启动模式；
说明能否启动、输出是否正常、decode token/s，并附 logs 中对应的 gpu-info、
gpu-iqr、gpu-dflash 和 server 日志。日志可能包含本机路径和输入内容，发送前
   请自行检查并删除不希望分享的内容。不同显卡的速度和功耗不保证相同。

如果程序提示缺少 MSVCP140.dll 或 VCRUNTIME140.dll，请安装微软官方的
Microsoft Visual C++ 2015-2022 Redistributable（x64）后重试。
许可证及各组件说明见 licenses 目录；使用的模型受其自身许可证约束。
