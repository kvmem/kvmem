# Radiance 1.17 合并审计

本轮审计将 Radiance 1.17 / IQ3R-S v4 和原生 DFlash 窗口保留合入 master `a2d8d9bf194d5ebbea34d3b923aec9e3d552be0e`，保留 master 的 canonical input、内容缓存修订 2、跨 owner 精确前缀查找及工具续接证明。审计发现的三项问题均已修复并复测，最终复查没有未解决的实现缺陷。机器记录见 [审计证据](benchmarks/radiance-v117-merge-audit-20261106.json)。

## 发现与修复

1. **DFlash recent 页数不能直接等同于实际窗口 token 数。** 最新页可能只有一个有效位置，原先按 W 向上取整的 recent 页数不能始终覆盖 W 个接受位置。独立位置断言在旧实现上失败；修复按 `W + block - 1` 预留尾页空隙，再由统一分配器向上取整。W2048、block128 的最小 effective recent 为 2176，sink128 时最小 selected budget 为 2304。36K 测试的 auto recent4096 不受此下限变化影响。测试遍历四种窗口及 4096–4351 每一个接受位置，按页面实际起点验证覆盖范围。
2. **可选 terminal 快照准入漏算 WINDOW 载荷。** 最终停车计费已包含窗口字节，但提前准入估算遗漏了它，可能为放不下的快照驱逐其他缓存。新增与保存共用几何逻辑的 `required_bytes`，在任何驱逐之前计入每个 rank 的原生窗口。原生测试将估算与独立读回的接受窗口字节数比较；原有 RAM/SSD、分支及配额测试通过。
3. **版本号补丁的 CI 路由不完整。** `radiance-1.17-kvmem.patch` 未匹配原文件名规则，且已有路由后的 Radiance 专属 job 修改仍被误判为全引擎修改。规则接受安全的点分版本段，拒绝空段；仅允许 Radiance job 自身变化时走该路径，其他 job 和全局配置必须相同。新增两项测试在修复前失败，修复后整套 15 项通过。原生主机合同明确加入 mem/kv，共 10 个主机编译单元，无 HIP/CUDA 编译。

## 合并及构建验证

冲突解决同时保留 Query restore 上界、实际容量检查和 master 的 canonical 内容校验。普通工具续接仍为 append 加 resident-only Pressure；没有引入冷历史重新召回或新请求协议。

实机验证使用独立的合并源码快照，旧源码和二进制保持原样。构建前复核既有 1509 项传递输入、源码、头文件、ABI、工具链、架构、标志和固定依赖锁；仅 9 个已知替换输入不同，其余不变。涉及的 adapter 头文件使用者全部重新编译。公共 checkpoint_policy 仅增加内联 capacity 方法，类布局与既有方法不变；当前 worker 不链接已变化的 KvMemRuntime/KvMemSession 实现。

构建 dry run 为 4 个主机对象、1 个静态库更新、3 个链接，共 8 个作业；使用 8 并行，没有 HIP/CUDA 编译。14.11 秒完成。最终 worker SHA256：

`fadba3a7881cd2821893ae898660c71d52c79fd1fa14afe4ecc8ad556f9c4047`

这是复用已证明兼容对象的定向验证构建。原生 prefix、HOST 和 GPU 测试通过，包含 24 种 host/device 配置、窗口字节与配额、分支隔离、RAM/SSD、回滚/恢复，以及同一 stream 上 1100 次 state 保存/恢复。

## 实际模型复测

设备为 RX 9060 XT 16GiB gfx1200，UUID `GPU-043fe07311696099`，TP1；每次运行前检查空闲状态。prefill256，selected budget4096，reserve512，sink128/requested recent512。DFlash effective recent2176，MTP recent512。下列两个模型权重不同，这不是隔离草稿算法的性能比较。

| 模式 | 合同检查 | 文档单独发送耗时 | 问题耗时 | 问题输入/实际复用 |
|---|---:|---:|---:|---:|
| IQ3R-S v4 DFlash7 | 153/153 | 11.784 s | 1.868 s | 8286 / 8254 |
| MXFP4 MTP3 | 156/156 | 10.319 s | 2.704 s | 8286 / 8254 |

两次均按相同普通会话先发送未改动文档，再单独发送真实问题，不提供答案。两者均返回预设访问码 LANTERN-5829，native Query replay 为 32；合法工具、取消恢复、SSD、缓存额度及 native drafter 接受检查全部通过。检查数量随 monitor 采样次数变化，所有记录的断言均通过。另有 Python 后端 89、Radiance 启动器 16、启动计划 21、CI 路由 15、打包 9 项通过，以及 Radiance UI 检查通过。Windows one-click 3 项通过，1 项需要额外离线运行时夹具而跳过；使用已有带 psutil 的固定 Python，默认系统 Python 缺少该依赖的环境失败也保留在审计日志中。

## 资格边界

本轮没有重跑 256K 性能或 Task2 内容质量。此前 [窗口保留报告](radiance-dflash-conditioning-20261106.md) 的 560.76 token/s prefill 等数据属于提交 `3268d34`、worker `ca9…`，不能作为本轮合并 worker 的新测量。历史原始请求、响应与结果保持不变。

此前 V4 DFlash7 的最终输出没有交付 Python 工具；历史 MTP3 虽生成可解析 Python，仍有字符串诱饵统计错误。MTP3 的 31 个中间响应中有 22 个字面工具标记、23 个批次确认失败；这些类别重叠，不能相加。不能将这些质量问题宣称为本轮已解决，也不能仅凭单次生成认定是窗口保留造成。实机合同通过仅证明所列运行行为。

原有低分辨率 CPU PNG OCR 21/22 限制仍在；Windows 原生、TP2、JPEG 模型请求和通用模型质量不在本轮资格范围内。详见 [V4 集成报告](radiance-iqr-v4-integration-20261105.md)。
