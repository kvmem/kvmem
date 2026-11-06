# Radiance DFlash 窗口保留

按用户指定的 melis 补丁方向，recent 至少覆盖模型草稿窗口。Pressure 保留这些最近页面，prefix 和 Query checkpoint 保存接受位置对应的原生 WINDOW KV 及 recurrent state；恢复时同时恢复它们，再发布有效窗口范围。checkpoint 不再固定提前 2048 tokens。原生 quantum、自然 prefill 边界、Query 上界和最后一个未缓存 logits 的 token 仍约束复用。

窗口快照只复制接受位置需要的最近页面，排除未来 reserved lookahead 及上一 prefill chunk 已不再需要的旧页。快照不长期持有滚动窗口的 GPU 引用；源请求继续生成或释放不会改写保存的字节。RAM/SSD 迁移、分支和准备好的 CPU payload 都包含窗口载荷并计入原有缓存额度。sink 加完整窗口超出既有 selected budget 时拒绝配置，不扩大预算。

新原生补丁为 `backends/patches/radiance-dflash-conditioning.patch`，接在既有七个补丁后。`KVManager` 仅新增内部非虚方法及表描述类型，没有更改类布局或 C ABI。原生独立 prefix cache 缺失窗口时的 refill 保护仍保留；新行为依赖完整 KVMem 原生窗口恢复，不能单独删除恢复边界保护。

定向验证构建沿用同一 GCC 15、ROCm 和 gfx1200 设置，复核既有 1509 项传递依赖证据。首次只编译 6 个主机对象、更新 2 个复制的静态库并链接 3 个程序，未编译 HIP/CUDA。后续窗口头文件修复只重编译实际依赖的 3 个主机对象。原构建目录、库、旧测试二进制及 Ninja 记录保持不变。该产物是有复用证据的验证构建。

原生主机和 RX 9060 XT 测试通过，覆盖完整窗口字节、分支隔离、RAM/SSD 迁移、截断和外来载荷拒绝、重复页面发布拒绝、Query 式 rewind 后恢复，以及同一 stream 上 1100 次 state 保存/恢复。Python 后端和启动器 105 项测试通过，另有 Linux 打包 9 项通过。GPU UUID 为 `GPU-043fe07311696099`；prefill 256、TP1。

实际 V4 DFlash7 的 155 项文本合同检查全部通过，包含合法工具、取消恢复、SSD 配额及两阶段文档召回。文档先单独发送，随后问题为 8286 tokens、复用 8254 tokens，Query 回放跨度 32；草稿提出 21、接受 5，返回独立预设的正确访问码。

256K Task2 的新旧 33 份请求逐项完全相同；同模型、同 GPU、同内核、B36864/R4096、prefill256、DFlash7 和相同采样参数。数据与哈希见 [验收记录](benchmarks/radiance-dflash-conditioning-20261106.json)。

| 指标 | 固定回退的历史版本 | 保留原生窗口 |
|---|---:|---:|
| 实际 native prompt rows 总和 | 322704 | 259088 |
| Prefill 总耗时 | 542.29 s | 460.01 s |
| 有效 Prefill（新增位置 / 总耗时） | 475.68 token/s | 560.76 token/s |
| 工具轮聚合 decode（原生计数） | 50.36 token/s | 53.47 token/s |
| 最终响应 decode（原生计数） | 49.82 token/s | 50.57 token/s |
| 原生设备总显存采样峰值 | 13759.54 MiB | 13760.73 MiB |
| 运行期进程 RSS 峰值 | 8320.95 MiB | 8291.04 MiB |
| 进程 swap 峰值 | 0 MiB | 0 MiB |
| 正缓存命中续接 | 31/32 | 32/32 |

减少了 63616 个额外 native prompt rows，prefill 总耗时降低约 15.2%，有效 prefill 提高约 17.9%。最后输入 257958 tokens，输出上限仍为 4096，实际输出 255 tokens。计时不包含权重加载。decode 以各自实际生成的 tokens 聚合；生成长度和内容不同。

运行合同通过不等于 Task2 内容质量通过。本次最终原始响应没有交付 Python 工具，而是解释文字及字面 read_file 标记；它自然 stop，没有触及 4096 上限。批次确认命中 15/31，历史版本为 7/31；历史版本虽生成可解析代码，仍未通过字符串诱饵和行尾注释的独立功能检查。此次保留原始输出，未修写或执行不存在的工具代码。单次采样输出不足以确定后端缺陷；窗口字节、接受边界及独立召回/工具合同单独验证。新策略保留原接受状态，避免为恢复窗口重算旧 trunk 状态；不同重算边界可改变量化/浮点路径和采样结果，不能以历史输出相同作为正确性标准。

本改动仍限 Linux RX9060XT gfx1200 TP1 文本验收。历史 CPU PNG OCR、native Windows 和 TP2 的资格边界见 [V4 集成报告](radiance-iqr-v4-integration-20261105.md)。
