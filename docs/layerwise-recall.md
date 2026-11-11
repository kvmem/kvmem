# 分层召回的启动与配置

当前完成接入与模型验证的后端为 **NInfer dense Qwen 27B**。其他后端尚未交付分层实现；
下述公共配置和能力协议为通用接口，启动器有字段或模板不代表所选后端已经支持。
未声明分层能力的 worker 会在启动前拒绝该策略。
NInfer Layerwise 暂不支持同时加载 LoRA 或 control-vector，启动时会明确拒绝该组合；Legacy 保留原有支持。

`legacy` 和 `layerwise` 是独立的召回执行策略。未填写策略的旧配置继续使用 Legacy。
Layerwise 在目标模型每层 attention 前使用本轮刚产生的当前层 Q，恢复该层选择的历史 KV；
目标模型不做 Query replay。普通工具续接仍是 append 加 accepted 驻留集合内的 Pressure，
不会因保存过旧 Q 而开启冷历史召回。请求沿用现有普通对话和 OpenAI 兼容接口，没有专用 ingest 状态。

在控制台的「召回策略」页选择 **Layerwise · Current-layer Q**。每个后端也有单独的
Layerwise 文本、无草稿起点模板；套用模板不会启动服务。现有模板、保存的配置及显式用户值不自动迁移。
Legacy 的评分器、聚合、graph alpha、LRFU 和 Query/replay 上限保存在草稿中，在新策略下不发送。

## 资源与命令行

结构化配置继续使用 `schema_version: 1`。原生参数统一使用 `--kvmem-*`，PowerShell 使用表中的配置名。

| 配置字段 | 原生命令行 | 未填写时 |
|---|---|---|
| `RecallStrategy` | `--kvmem-recall-strategy legacy\|layerwise` | `legacy`；兼容旧 worker 时不发送该 flag |
| `LayerwiseMtpHistory` | `--kvmem-layerwise-mtp-history upstream-global\|first-layer` | NInfer 缺省 first-layer；其他 worker 以能力声明为准 |
| `LayerwiseActivationRamMiB` | `--kvmem-layerwise-activation-ram-mib` | 所选后端的原生默认值 |
| `LayerwiseActivationDiskMiB` | `--kvmem-layerwise-activation-disk-mib` | 所选后端的原生默认值 |
| `LayerwiseStagingMiB` | `--kvmem-layerwise-staging-mib` | 所选后端的原生默认值 |
| `LayerwiseCheckpointTarget` | `--kvmem-layerwise-checkpoint-target N` | 支持该选项的 NInfer worker 默认保留 10 个热检查点，可设为 1–256 |
| `LayerwiseMetadataMiB` | `--kvmem-layerwise-metadata-mib auto\|N` | 新版 NInfer 根据模型自动计算，显式 MiB 数值是硬上限 |
| `LayerwiseActivationMetadataMiB` | `--kvmem-layerwise-activation-metadata-mib` | 所选后端的原生默认值 |

未填写的资源字段不生成对应原生 flag，Windows、配置预览与 Linux 导出均保留此语义。
界面可读取所选 worker profile 的可选 `default_resources`（上述五项的标准 `*_bytes` 名称）；
只有新建或缺省字段才显示该后端默认，展示值不写成配置覆盖。用户编辑后才成为显式值。
切换 worker 或刷新能力会重新读取默认，过时响应不能覆盖当前选择；未知默认保持未填写。
已保存的显式 `256` 或允许的 `0` 原样导入、保存和传递，不因新版本默认增大而自动提升。
导出的脚本校验固定 worker 的 SHA-256；更换 worker 后须重新生成，不能把另一版本的缺省值当作原配置。

激活 RAM 与 SSD 至少一项为正；staging、召回 metadata 和激活 metadata 必须为正。
这些是独立有限限额，不隐式借用 `HostMiB` 的 canonical KV 历史配额。界面显示的 configured bytes
是限额，不是实时用量。后端仍检查真实峰值、磁盘位置及可用空间。

**NInfer 检查点保留。** 只想保留最近约 10 个恢复点时，将元数据预算留空即可。数量按整个模型实例计算。多个分支共享同一次捕获只计一个，只有原生状态图像仍在主机内存中的恢复点计入热目标。`RetainedSessions` 单独限制内容组数，同一组可以有多个恢复点。

自动模式按加载模型的原生状态图像、上下文上限下的索引及反馈大小估算单点成本，并为每个并发执行槽留下恢复和捕获重叠空间。额度按需使用。填写 `LayerwiseMetadataMiB` 后，实际分配必须服从这个字节上限，空间紧张时实际可保留数可能低于目标。总额度中的 3/4 用于历史状态和索引，两个 1/8 分别供运行中反馈与冻结索引使用。

到达数量目标后，优先回收最近最少使用的空闲恢复点。有 SSD 空间时可异步转冷，写入完成前热数可能暂时超过目标。正在恢复或写入的状态受保护。旧分支的所有合格恢复点都被回收后，再次请求该前缀会重新预填充。运行中可从 `/props` 的 `kvmem.usage.checkpoint_retention` 查看数量目标、实际热数、仅 SSD 数量、稳态容量估算、有效字节上限和淘汰计数。

在 dense 27B、上下文 16,384、并发 1 的 5060 Ti 验证中，10 个热检查点对应的自动额度为 3,382 MiB。第 6 轮达到 10 个，后续连续请求与分支均保持 10 个。显式限制为 1,024 MiB 时，同一配置实测保留 4 个。[逐轮验证记录](benchmarks/ninfer-checkpoint-retention-27b.json)

Layerwise 起点使用历史预算 K=32768、追加预留 G=1024、canonical RAM 16384 MiB；
显式参数优先。Radiance 新启动的 prefill 缺省为 256；显式 512 和历史比较协议不变。
K/G 和显式 sink/recent 必须按该后端块大小对齐，不能靠启动器取整改变含义。
sink/recent 留空使用原生 auto；显式 `0` 原样传递。完整 Query evaluation 的准入仍由原生请求计划验证。
普通 rolling 的 G 最低块数由该 worker 的能力契约声明，不假设所有后端都为一块或三块。

例如，以下结构化配置选择文本、无草稿的 NInfer Layerwise；路径与 GPU UUID 需换为本机值：

```json
{
  "schema_version": 1,
  "parameters": {
    "Backend": "ninfer",
    "Model": "/models/target.ninfer",
    "Worker": "/opt/kvmem/bin/ninfer-serve",
    "Gpu": "GPU-your-device-uuid",
    "RecallStrategy": "layerwise",
    "KvType": "bf16",
    "Budget": 32768,
    "Reserve": 1024,
    "HostMiB": 16384,
    "Prefill": 256,
    "SpeculativeMode": "none"
  }
}
```

Windows 公共入口接受同名参数，例如 `start-backend.ps1 -RecallStrategy layerwise`。
PolyStrata 的 Python 前端使用 `--recall-strategy`、`--layerwise-*`，再翻译为 worker 的统一参数。
配置导入、浏览器个人模板、检查预览、启动和 Linux 导出脚本使用同一组字段；省略 MTP 历史选项时使用固定 worker 声明的默认策略，不把展示默认写成显式配置。

## 三种草稿路径

| 产品选项 | 历史来源与要求 |
|---|---|
| `SpeculativeMode: mtp` | OWN MTP。NInfer 未填写 `LayerwiseMtpHistory` 时使用 first-layer；显式 upstream-global 保留。其他 worker 未声明默认时仍须选择。`first-layer` 使用第一目标 attention 层的 fresh-Q 历史 ID，恢复 draft 自己的 KV；`upstream-global` 通过隔离的私有 probe/global 评分选择 draft 自己的历史。两种的主模型均使用逐层 fresh-Q。 |
| `SpeculativeMode: dflash2` | 独立完整原生滑动窗口，保留模型指定的 target 特征层与原生 ring/carry/CP。不按目标层选集裁剪 draft 历史，也不接受 OWN MTP history 参数。 |
| `SpeculativeMode: guess` | 仅 PolyStrata Gemma 原生 SharedTarget/SWA Guess；要求可信 GGUF 架构和匹配 assistant，以及 worker 的 Guess 能力。映射现有原生 MTP 开关，不使用 OWN MTP 两策略。旧 Legacy `mtp` 配置保留原行为。 |

`query_replay: false` 与 llama 的 MTP `state_modes: ["replay"]` 描述不同机制；后者是原生 recurrent state 回滚。
草稿物理宽度、adaptive、并发、设备数、KV 格式和每种历史策略均须由当前二进制声明支持。
界面不会把请求的 draft 5 静默改成已验证的 3，也不会替用户切换 BF16 或设备。

## 能力与运行确认

检查配置和实际启动读取选中程序的 model-free `--kvmem-capabilities`，禁用 GPU 可见性。
Layerwise 要求 `recall_strategy_revision: 1` 和 `recall_strategies.layerwise` 的显式声明；
缺少字段表示该新组合不可用。旧 Legacy 的契约兼容行为不变。
新 profile 分别校验 current-layer Q、无目标 Query replay、position policy、KV/视觉、owner 数、
最低 G 块数，以及 `parallel_modes: ["single", "pipeline", "tensor"]` 中实际声明的子集。
`max_devices: 2` 不等于支持 TP。不同发行二进制可有不同资格范围，后端整体支持表不是新策略的组合保证。

可选数值约束也由 worker 给出：`minimum_window_tokens` 限制每 owner 的 `Budget + Reserve`，
与最低 G 块数、总 Context 分开校验；`speculative.<mode>.max_ngram_draft_tokens` 限制 NGram 宽度，
`max_ngram_draft_tokens_concurrent` 在并发大于 1 时追加更窄的上限。MTP history policy 的能力项可再收窄这些值。
启动检查和界面保留超限的显式输入并报错，不用自动截短来满足条件；完整 Query/物理 proposal 的准入仍由原生执行器验证。
旧 profile 缺少这些可选字段时保留原有 boolean 能力行为，不推测某个后端的数值。

运行时只用 native 启动报告确认策略、Q/replay、OWN policy 和五个 configured 限额。
即使资源 flag 全部省略，也须等五项原生限额完整上报，才确认运行配置；
缺少完整报告时显示「等待完整服务报告」，编辑草稿不能改变该报告。正在运行的配置保持只读。
Legacy 自动预算建议尚未覆盖 Layerwise 的独立资源，当前新策略手动设置预算，不返回旧格式的自动 KV 切换建议。

长文质量比较应先发送完整文档，让模型给出简短中性确认，再在同一对话独立发送真实问题。
保留文档、两次请求/响应、实际 prefix/CP 复用与 native Query span，分别报告文档与问题时间。
只通过参数/界面检查不等于模型质量、吞吐或某个 GPU/组合已经验证。

English: choose the independent `layerwise` strategy; old profiles remain Legacy. NInfer OWN MTP defaults to `first-layer`; explicit `upstream-global` remains available. Other workers require a choice unless they declare a default. DFlash2 keeps its native full window; PolyStrata Gemma Guess
uses SharedTarget. All combinations are gated by the selected worker's explicit per-strategy capabilities.

## NInfer 的 DFlash2 反馈回读

使用 `--kvmem-recall-strategy layerwise` 时，单 GPU 的 DFlash2 CUDA Graph 路径自动按每个请求实际采样的行计算输出反馈，支持多个并发槽位。每个请求保留自己的批次行号、原生位置、接受前缀与 graph epoch 校验；其他执行路径沿用各自的反馈实现。

回读使用已有的有限 pinned 暂存区。当前批次内的请求借用互不重叠的切片，按容量分组提交；只有对应 GPU 工作完成、结果校验并消费后才复用切片。部分草稿接受和没有采样行的情况仍校验原生控制信息。取消或异常会先排空已提交工作，再释放相关所有权。

分层 KV 恢复也复用已有暂存区，每批最多装入 64 个层片段，并按实际容量缩小批次。K/V 与量化 scale 原字节保留，未发布的目标检查、共享快照保护和传输完成同步不变。这不引入异步 refresh 调度。

排障时可分别设置 `NINFER_SELECT_LAYER_FEEDBACK=0`（关闭采样行选择）、`NINFER_BATCH_LAYER_FEEDBACK=0`（关闭 mass 批量回读）或 `NINFER_BATCH_COHORT_FEEDBACK=0`（按请求分组回读），然后重新启动服务。清除这些变量即可恢复默认自动启用。它们不关闭独立的 KV 层片段批量搬运优化。

性能取决于负载：PRO 6000 / NVFP4 27B + DFlash2 Q3 的单路 256k 任务2中，prefill 提升约 21.3%，decode 基本持平；每路约 64k 输入、固定 8k 输出的 4/8 并发测试中，完整并发 decode 分别下降约 0.7%/2.8%，提问批次输出吞吐分别提升约 3.0%/1.2%。两种负载与校验设置不同，不能作为只改变并发数的扩展性对照，也不构成通用 decode 提速或语义质量保证。[配置、计时口径与验证范围](benchmarks/ninfer-layerwise-performance-20261111.json)
