# KVMem GGUF NVFP4 prefill：接入与资格验证

> Publication note (2026-11-02): native backend patch candidate, not an installed release. Machine identifiers are anonymized; measurements and hashes are retained. Uncommitted-status statements below describe the original experiment; this PR publishes source only. Raw requests/logs/traces remain local. See [the public evidence summary](benchmarks/ninfer-low-precision-20261102.json). Further FP4 tests are paused.

> 当前维护状态（2026-11-02）：进一步 FP4 测试暂停。下文保留此前完成的历史测试、
> 当时的31段补丁树和不利结果，不代表新增验收。当前计算补丁与 KVMem 统计回读补丁
> 已拆分，准确版本以 `source/backends/versions.json`（产品源码中为 `backends/versions.json`）
> 为准；最新 FP8 状态见 `source/docs/ninfer-mxfp8-prefill-validation-20261101.md`。

日期：2026-11-02。结论：仍有宽度未通过 3% 集成开销线，不能宣称全部达标。精度模式仍默认关闭；完整 NVFP4 配置未通过质量门槛。该结论只覆盖本文指定的模型、GPU、权限范围及工作负载。

实验根目录：`$ARTIFACTS\ninfer-nvfp4-prefill-20261102`。

## 实现与来源

候选源码在 `source`，原生源码在 `backend`。从既有 FP8 / 精确算子 / 5060 Ti Q4_K、Q6_K 协作加载补丁链继续，基树 `43eb8590c3d1ee149b12494ae9bf1b2c13c81b17`，最终树 `7e6395b7c5db966f713be4e4a374eb577c764666`。完整 31 段补丁链已校验，新增补丁 SHA-256 `0dda4b16236af902dbd1ec734b4af1685bc84141408a39c8da14442c1c603a64`。没有提交、合并或替换安装包。

移植来源为 https://github.com/2-com-org/ninfer-5060-qwen 的固定提交 `08c4fb70bec11063cae42eafe1c1b0dc3a7f993e`。原仓库实际独立编译；两处 Windows 主机模板兼容适配及来源核验沿用 `../ninfer-upstream-fp8-20261101/report.md`。不是按 README 成绩估算。

`NINFER_GGUF_PREFILL_NVFP4=1` 在明确 AllowA4 的 GGUF Use 上启用 E2M1 权重/激活、每 16 值一个 E4M3 尺度、FP32 累积和 BF16 输出。FP8 开关在本地 NVFP4 验证中保持 0。129 列起、sm120/cuBLASLt、K 为 256 倍数、对齐 BF16 目的地才可使用；窄 decode/verify、FP32 目的地和不支持形状维持原路径。权重最大值和两位尺度选择表由 Parameters 持有，瞬时图像和工作区由 Program 持有。已有 q8 内核保持原算术。精度加入前缀签名和会话绑定，切换模式需要重新 prefill。

artifact 工具只提升已有 GGUF AllowA8 权限，新 UUID、1183 个物理对象逐一读回校验，权重字节不变。完整范围 521 Uses、全层 MLP gate/up 128 Uses、中间半数层 64 Uses；最终保守配置为 64 层中的 16–47 层 MLP gate/up，其他权限保留。原始模型未改。

## GPU 与固定设置

RTX 5060 Ti 16GB，UUID `GPU-A`。Dense 优先卡始终一致，5050 未参与。模型为 Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp，混合 GGUF 格式。Windows、CUDA 13.2、MSVC 14.44、Release、sm120a，固定依赖 bundle `build-deps/ninfer-x64-windows-c7684b1d96347de8`，构建及每次验证前校验，运行 DLL 使用同一 bundle。

本地 Native=ON/PDL=OFF；原仓库按其推荐 Native=OFF/PDL=ON。跨项目绝对差距包含整套后端及构建差异。原版 NVFP4 必须同时保留其 FP8 开关，会对合适的非 A4 Use 使用 FP8，因此不能把跨项目差值当成纯粹 KVMem 集成开销。

统一 max-context 32768、并发 1、int8-g64 KV、MTP4、ngram0、temperature0/seed42、thinking关闭、内置 auto profile；`NINFER_PREFILL_ALIGN=0` 保证真实 256/1024 宽度。KVMem 预算24576、reserve1024、host4096MiB、sink/recent64、sessions4。普通路径关闭 KVMem，使用自身 context cache 和自动设备 KV 容量；这些内存策略差别属于所测功能开关，不把显存样本视为纯 workspace 增量。

## 速度：最终同二进制、同 artifact、镜像顺序

最终组共12次独立进程，每宽度 Q8-KV / NVFP4-普通 / NVFP4-KV，再反向执行，每项两次；都使用同一个最终二进制 SHA `3005e19fe758770a3f1a91b6c65de9cae25adfdfc96e2257df72a6fb374c950c` 和同一份 middle-half artifact。以下为均值，单位 tok/s。开销为 1−KV/普通吞吐；decode 使用独立31-token短提示生成256 token，按255个间隔归一。

| chunk | Q8 KVMem | 保守FP4普通 | 保守FP4 KVMem | FP4相对Q8收益 | prefill集成开销 | decode KV相对普通 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 723.42 | 751.96 | 736.59 | +1.82% | 2.04% | +1.63% |
| 1024 | 744.87 | 892.82 | 859.40 | +15.38% | 3.74% | +1.29% |

完整数值、两次范围、冷启动及首个长提示成本见 `acceptance-results.json`。默认启动均值 Q8→FP4：256 为 10.219→10.274s；1024 为 10.726→10.352s。首个5018-token长提示的 prompt_ms：256 为 7231.6→7240.7；1024 为 6916.9→6414.2。不宣称所有冷延迟、短问题 TTFT 或 decode 都改善。

每臂先做短 warm，再以不同 system 前缀做约5K首用预热，测量14,806-token完整文档，单独问题回合读记录，最后独立短生成。所有计时文档均断言 cache_n=0、prompt_n=14806；短生成也断言cache_n=0和256输出。后续问题的普通/KV缓存边界不同（通常14799/14807复用），其 TTFT 不是严格等工作量的矩阵比较。原始请求、响应、启动参数和所有不利结果均保留。

先前28项对照和12项批拷贝复测在 `performance`、`performance-coalesced`；最终通过与否只看 `performance-final` 的同时期同二进制组，不筛选其中的好样本。初始同前缀预热误使普通路径少计算4994 token，已整体保留并排除在 `performance-shared-prefix-invalid`。另有多次与其他任务的启动竞争，停止的仅为本任务服务器；没有把这类显存不足描述为模式自身 OOM。后续十秒连续空闲检查加全程外来进程监测，发现竞争即排除该轮并只停止自己的子进程。

## 原仓库 NVFP4 参考

同一 GPU、物理 GGUF 字节、完整A4 artifact、工作负载和预热协议，两次均值：

| chunk | 原版Q8 prefill | 原版完整NVFP4 prefill | 原版内部收益 | 原版Q8/FP4短decode |
|---:|---:|---:|---:|---:|
| 256 | 806.20 | 744.24 | -7.69% | 105.41 / 105.44 |
| 1024 | 912.55 | 1762.88 | +93.18% | 105.43 / 105.49 |

此组位于 `performance-reference`，不参与保守配置的3%集成门槛。原版完整模式的质量代价和本地完整模式都大于保守范围；原版本来就有 decode 优势，不能把整套后端差距归功于 FP4。此前 FP8 真宽度对照仍见前一报告：原版有256融合tile，本地初版只移植两阶段 Lt 大批次路径，因而256显著变慢；本次没有宣称补齐了 FP8 小批次路径。

本地完整模式的诊断速度如下。普通控制来自批拷贝前的同算术版本，KV来自批拷贝后；保留这些不同时期的数值作完整模式诊断，不用它们判定保守模式的3%门槛。

| chunk | 本地完整NVFP4普通 | 本地完整NVFP4 KVMem |
|---:|---:|---:|
| 256 | 745.06 | 731.90 |
| 1024 | 1705.94 | 1592.69 |

## 质量：先选范围，再看独立留出

第一份固定六域语料每域65536字符，122820目标token/60个2048窗口，prefill/score tile1024、disjoint、int8-g64。同一文本，Q8总体PPL 5.6631368515：

| 模式 | 总体PPL | 相对Q8 | 最差域 |
|---|---:|---:|---|
| 本地 FP8 | 5.683685 | +0.3628% | arxiv +0.8601% |
| 完整 NVFP4 | 5.761726 | +1.7409% | wikipedia_en +3.3325% |
| 全层 MLP gate/up | 5.721168 | +1.0247% | wikipedia_zh +1.6026% |
| 中间半数层 MLP gate/up | 5.692220 | +0.5136% | wikipedia_zh +1.0065% |

完整模式 +1.7409% 和全层 MLP +1.0247% 均严格拒绝，没有把后者四舍五入算通过。保守范围在查看第二份结果前固定。第二份互不重叠语料同样122820目标token/60窗口：Q8 5.828769601→FP4 5.848386049，总体 +0.336545%，最差域 arxiv +1.191258%，通过总体≤1%、单域≤3%。

局部不利结果仍存在：独立验证最差窗口 arxiv #0 为 +6.092939%；校准保守模式最差中文窗口约+6.38%。PPL总体通过不等于逐窗口无损。原版完整NVFP4同一第一语料总体+1.91246%、最差单窗口+16.2623%；原版最初关FP8导致A4降级的误标结果已保留为disabled-control并排除。

默认关闭的整模型算术回归：上一版与批拷贝前候选Q8的全部 60 个窗口 total_nll 逐项相同，PPL 5.663136851498。末尾主机拷贝改动仅在KVMem统计量开启时执行，不进入这些causal_scoring；最终候选的KVMem恢复、召回和工具路径另外实测。不以旧输出作为唯一数学oracle，另有下面的独立解码和FP64检查。

## 文档先行的长上下文与工具续接

3份未改动的英文/中文/代码文档约21K/27K/29K token。每次先普通文档请求，仅中性 Received，再在同一会话独立提问4个精确事实。标准答案只用于外部判分，未发给模型。普通、全驻留KVMem、8K压力预算以及256宽度压力分别比较；开启原生KV/GDN字节校验，核对问题回合真实缓存复用和native Query span。

| 模式 | 路径 | 预算 | chunk | 精确事实 |
|---|---|---:|---:|---:|
| q8 | 普通 | 31744 | 1024 | 12/12 |
| q8 | KVMem | 31744 | 1024 | 12/12 |
| q8 | KVMem | 8192 | 1024 | 12/12 |
| q8 | KVMem | 8192 | 256 | 12/12 |
| nvfp4 | 普通 | 31744 | 1024 | 12/12 |
| nvfp4 | KVMem | 31744 | 1024 | 12/12 |
| nvfp4 | KVMem | 8192 | 1024 | 12/12 |
| nvfp4 | KVMem | 8192 | 256 | 12/12 |

共 24 个文档后问题回合、96 个事实，正确 96，新增NVFP4错误 0。这是3份文档跨配置复用，每次4个相关探针，不是96个独立任务。`recall/analysis.json` 保存逐臂文档/问题独立耗时、缓存token、Query边界、replay和原生恢复证据。工具续接两臂均检查echo参数和返回码；native日志确认append_pressure、空Query、零replay，未重新选择旧Q冷历史。诊断传输检查有额外同步，这些质量耗时不用于正常性能排名。

仍有一个明确的格式遵循反例：最初含工具定义的8K压力文档回合，Q8仅回复Received，NVFP4却复述四个事实并耗满128-token预算。该轮返回码虽正确，但未满足短中性确认协议，已将两臂整体保留并排除为 `tool-continuation-nonneutral-excluded`。这是实际观察到的行为退化，不能把它当成无损。新工具受控组对两臂均加相同的明确system确认指令、文档max_tokens=8，并断言中性确认后才续接；它通过不代表原反例被修复。另以原始不变请求做普通/全驻留NVFP4的文档单轮诊断，两组均正常返回3-token中性确认，未重现反例；输出不喂给任何后续问题。详见 `tool-format-diagnostic` 和 `tool-acknowledgement-counterexample.json`，不把未经隔离的变化武断归因于某一个机制。

## 数学、资源、修复及边界

独立CPU标量GGUF解码、独立E2M1/E4M3和FP16候选计算、FP64乘积，覆盖11格式、14形状、真实K5120/6144/17408、tails/gather/residual、分离/联合父矩阵、融合FFN、host尺度缓存、保护区与CUDA Graph。编码后乘积的BF16舍入误差不能误称为相对原始模型的FP4总误差。窄batch/受保护目的地开关前后checksum `9756319403813693913` 一致，旧q8格式回归通过。Nsight实际观察到E2M1 GEMM与FP4图像内核，详见 `profile-conservative/kernels.json`；跟踪只作路由证据，不作正常计时。

移植中发现并修复了IQ1_M Slice适配错误：原版使用无符号grid加浮点offset，本地精确解码已折入有符号整数及1/8尺度，新适配器曾读未写字段。独立CPU解码在任何本地模型质量/速度运行前抓到它，修复后完整复验。原q8解码无改动。本机中文MSVC包含前缀与Ninja记录不匹配还曾留下过期wrapper，已用构建局部ASCII归一器修复依赖追踪；保留成功CUDA对象，未伪造记录。初始ABI变化全量构建必要；后续拷贝优化只编译两个主机对象并链接，无CUDA重编。

KVMem正确的逐层D2H统计拷贝作为原设计优化为连续层批量2D拷贝；独立全buffer字节oracle覆盖不均匀head、分层、MeanQ、部分Query和canary，20组布局通过，接受/选择策略和CUDA算术不变。原生时序支持后处理开销下降，但不能把全部post暴露时间归因于CPU或拷贝。

保守模式固定页锁定尺度选择缓存按实际shape和分配规则为85MiB，另有临时GPU图像/工作区和启动准备；没有永久FP4权重副本。权限副本需要一整份约12.8GB artifact磁盘空间。显存采样不是精确峰值，也未积分功耗。保留原始数据、构建目录和固定依赖。

测试结束后，仅删除完整/全层MLP两个未通过质量线的派生权限副本，约释放25.6GB；保留保守artifact、源模型和全部原始请求、报告、物理对象哈希。`derived-artifact-cleanup.json` 记录校验过的artifact身份及精确重建argv/cwd。若重跑完整实验，先按该记录或artifact工具重新生成权限副本。

未覆盖：其它模型/卡、MoE、多GPU/TP、并发、Linux/WSL速度、256K语义质量、完整推理/代码执行基准、跨版本磁盘快照兼容和发布打包。此次通过只限定所测配置；NVFP4是有损计算，不能作所有输入无损保证。精度范围由artifact显式授权、运行时默认关闭。

复现：`build.ps1`、`prepare_corpus.py`、`quality.py`、`run_native_tests.py`、`final_perf.py`、`reference_perf.py`、`parent_quality.py`、`recall.py`、`tool_continuation.py`、`profile_nvfp4.py`、`analyze.py`。所有服务由各自Popen拥有并关闭，不操作其他任务进程。最终校验见 `final-checks.json`。
