# 分层召回的启动与配置

当前完成接入与模型验证的后端为 **NInfer dense Qwen 27B**。其他后端尚未交付分层实现；
下述公共配置和能力协议为通用接口，启动器有字段或模板不代表所选后端已经支持。
未声明分层能力的 worker 会在启动前拒绝该策略。
NInfer Layerwise 暂不支持同时加载 LoRA 或 control-vector，启动时会明确拒绝该组合；Legacy 保留原有支持。

`legacy` 和 `layerwise` 是独立的召回执行策略。未填写策略的旧配置继续使用 Legacy。
Layerwise 在目标模型每层 attention 前使用本轮刚产生的当前层 Q，恢复该层选择的历史 KV；
默认路径不做 Query replay；下述实验性全局 Mean-K 会额外执行一次私有探测。普通工具续接仍是 append 加 accepted 驻留集合内的 Pressure，
不会因保存过旧 Q 而开启冷历史召回。请求沿用现有普通对话和 OpenAI 兼容接口，没有专用 ingest 状态。

在控制台的「召回策略」页选择 **Layerwise · Current-layer Q**。每个后端也有单独的
Layerwise 文本、无草稿起点模板；套用模板不会启动服务。现有模板、保存的配置及显式用户值不自动迁移。
Legacy 的评分器、聚合、graph alpha、LRFU 和 Query/replay 上限保存在草稿中，在新策略下不发送。

界面按「召回方式 → 当前策略设置 → 历史保留」组织；经典评分细节与分层传输/元数据预算可展开编辑。
页面加载及切换后端或程序时自动检测能力。不可用的策略在下拉选项中置灰，并标注「此推理后端暂不支持」。
检测中或检测失败时也暂不可选；检测失败后重新聚焦策略选择框会自动重试。
可选只代表程序声明了分层能力，模型、KV 格式或草稿组合仍需检查配置。
旧配置中的不可用策略（包括旧版 NInfer）保持原值并阻止启动；可切回 Legacy，已填写的参数仍会保留。

## 实验性混合 Query 评分

NInfer dense Qwen 27B 可设置 `NINFER_LAYERWISE_HYBRID_QUERY=1`：最终 Query 在模型层 **3、7、11、15**（从 0 编号）按各层已接受的 prefill attention `mass / exposure` 排序；沿用 Pressure 的衰减、曝光归一化和同分时偏向较新页的规则。候选仅限该层此时驻留的历史页，已淘汰的冷页不能借用残留统计重新进入。必选 sink／recent／显式保留页仍计入相同历史预算；统计缺失、必选页已不驻留或候选不足时明确报错。

模型层 **19、23、…、63** 保留当前问题 Q 对该层 Mean-K 的评分。所有层仍正常计算最终 Query 的投影和 attention，前四层仅跳过问题 Mean-K 评分；它们改变 KV 后，后续层实际产生的 Q 和选页也可能改变。MTP `first-layer` 历史跟随第一目标层的结果。页大小和预算不变，未引入新 CUDA 内核或设备缓冲；不可变的 prefill 统计在整个 Query 事务期间保留原有元数据租约。

该开关默认关闭，设置 `0` 或移除变量恢复全层 Mean-K；空历史沿用原路径。它只改变新的最终 Query 选页，普通 prefill 的 Pressure 规则不变。4090 上四道固定 NIAH 题的均分为全层 Mean-K 68.75、混合版 43.75；0 题提高、3 题同分、1 题降低，其中 UUID 检索题从 100 降至 0。它仍保持实验性、默认关闭；这不是整套 130 题得分。验证范围、逐题结果与选页对比见 [测量记录](benchmarks/ninfer-hybrid-query-20261114.json)。

后续 UUID 单题诊断定位到第 15 层策略：仅此层改为 prefill 排序就会失分；混合版仅将此层改回 Mean-K 即恢复 100。给失败版本的第 15 层回补 UUID 所在页也能恢复，但基线只删除该页仍答对，因此不能把失败完全归结于单页缺失。全部干预保持 576 页预算，细节见 [逐层与单页对照](benchmarks/ninfer-uuid-diagnosis-20261114.json)。

## 实验性前层全局 Mean-K

设置 `NINFER_LAYERWISE_GLOBAL_QUERY=1`，并关闭 `NINFER_LAYERWISE_HYBRID_QUERY`，可让本模型的第 **3、7、11、15** 层共用一组全局 Mean-K 页（实现范围为编号不大于 15 的目标 attention 层）。第 **19、23、…、63** 层继续使用正式前向中各自当前问题 Q 选择历史。

全局分数复用现有 `LayerwiseProbeSelection`：对当前问题进行一次独立的目标模型探测，每层按 Query token／head 计算 Mean-K softmax，平均 head、累加真正的问题 token，再平均所有目标 attention 层，最后做一次 TopK。没有先把 Q 向量跨 token／层平均。前四层共用这个 TopK，仍执行各自完整的真实投影和 attention；深层会受到它们输出变化的影响。

探测读取原有驻留窗口，随后还原并逐字节验证 GDN 状态、恢复 KV frontier 和位置坐标，不发布探测产生的状态或统计。全局分数和选页保留独立的有限元数据配额。MTP 继续沿用所配置的历史策略；`first-layer` 跟随第一目标层选页。若同时使用 `upstream-global` MTP，两者复用同一次探测。历史全量能装入预算时不需要探测。

该实验开关默认关闭，需要现有物理窗口能容纳完整探测的 Query 尾部；空间不足时明确拒绝，不扩大预算或静默改变 prefill 窗口。探测增加一次短前向和全局打分，不能视作零开销。四题验证结果与局限见 [实验记录](benchmarks/ninfer-global-query-20261114.json)。

4090 同四题均分：全层逐层 Mean-K 68.75、前层累积 attention 43.75、前层全局 Mean-K 43.75。新方案未提高这四题的得分。UUID 页全局排名第 2，前四层均选中 key 和 UUID 页，但模型生成重复文本并在最终答案前结束，官方得分仍为 0。额外单题对照保留相同探测、丢弃全局选页，恢复全部逐层 Mean-K；16 层选页及最终答案与基线相同，得分 100，支持将本题退化归于共享选页路径而非探测本身。这只是四道固定题的小样本实验，暂不替换默认策略；没有重跑完整 130 题。

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

分层 KV 恢复复用已有暂存区，每批最多装入 64 个层片段，并按实际容量缩小批次。非诊断恢复在容量允许时把同一暂存区分为两半，打包下一批时可以上传上一批；复用某一半前等待它自己的完成事件。K/V 与量化 scale 原字节保留，固定暂存预算不增加。

排障时可分别设置 `NINFER_SELECT_LAYER_FEEDBACK=0`（关闭采样行选择）、`NINFER_BATCH_LAYER_FEEDBACK=0`（关闭 mass 批量回读）或 `NINFER_BATCH_COHORT_FEEDBACK=0`（按请求分组回读），然后重新启动服务。清除这些变量即可恢复默认自动启用。它们不关闭独立的 KV 层片段批量搬运优化。

性能取决于负载：PRO 6000 / NVFP4 27B + DFlash2 Q3 的单路 256k 任务2中，prefill 提升约 21.3%，decode 基本持平；每路约 64k 输入、固定 8k 输出的 4/8 并发测试中，完整并发 decode 分别下降约 0.7%/2.8%，提问批次输出吞吐分别提升约 3.0%/1.2%。两种负载与校验设置不同，不能作为只改变并发数的扩展性对照，也不构成通用 decode 提速或语义质量保证。[配置、计时口径与验证范围](benchmarks/ninfer-layerwise-performance-20261111.json)

## NInfer 的驻留页 Pressure 整理

Layerwise 的单 GPU 普通目标或 DFlash2 路径，默认在 GPU 内整理已经驻留的目标 KV。它仅接受原驻留页的有序子集；先保护共享页，再按不会覆盖未读源的顺序搬移各层原生字节。选页策略、完整 canonical 历史和 DFlash2 自身的滑动窗口保持原有语义。独立 MTP KV、多 rank、诊断 readback 和 import/probe 准备沿用原恢复路径。

未完成的搬运持有自己的复制流、完成事件和已计费的元数据；该请求暂时让出执行，其他就绪请求仍可调度。原生地址在此期间不发布有效覆盖，完成后才校验请求版本和 lane epoch 并发布全部目标层。取消或失败先排空复制流，再使未完成的目标失效。部分 prefill 进度按实际切分累计服务预算，纯等待不消耗计算配额。

归档新接受的 KV、选页和建立共享页写入保护仍发生在提交前；本改动没有把所有准备工作移到后台。在 Qwen3.5 系列的 NInfer 后端，执行新 Query 的请求可以用满配置的历史预算 K。生成时按实际接受的 token 占用剩余窗口，到达物理块边界后按需执行 Pressure。没有新 Query 的请求仍按声明输出上限预留空间。分析刷新性能应核对实际刷新事件，并单独观察混合 prefill/decode。

排障对照可设置 `NINFER_RESIDENT_LAYER_PRESSURE=0` 恢复 Host 搬运，或设置 `NINFER_ASYNC_LAYER_PRESSURE=0` 保留 GPU 搬运但等待完成。`NINFER_LAYER_RESTORE_PIPELINE=0` 关闭 Host 恢复的双缓冲。清除变量恢复默认；`NINFER_TIME_LAYER_REFRESH=1` 仅用于阶段计时，性能对照应关闭计时。

PRO 6000 的频繁整理诊断中，平均刷新占用时间约从 75 ms 降至 39 ms；常用 16k reserve 的完整任务2总时间降低约 0.9%，decode 持平，4/8 路总体变化较小。[实现、计时口径及验证范围](ninfer-resident-pressure-20261111.md)

`--kvmem-layerwise-score-tokens 32` 让 NInfer 每 32 个 token 保存一个 Mean-K。每个查询位置和注意力头在所有候选半页上做 softmax，然后把同一物理页的两个概率相加，按总概率选页。预算、固定区、Pressure 和 KV 搬运仍以 64-token 物理页计量。默认值 64 每页保存一个 Mean-K。选 32 时统计记录数加倍。

`--kvmem-layerwise-score-tokens 32 --kvmem-layerwise-half-reduce max` 改用两个半页分数中的较大值。每个查询位置、每个头仍在该层全部可评分半页上做 softmax；先对头求平均、累加所有查询行，再取 `max(left, right)` 给原来的 64-token 页排序。查询分块不影响取 max 的位置。固定区和 TopK 仍以整页计费；`--kvmem-layerwise-half-reduce sum` 为默认。

`--kvmem-layerwise-unique-query-anchor` 为文本查询增加可选的精确匹配。对于 32–64 个 token 的查询，当已接受历史中只有一个最长连续匹配、且至少匹配 32 个 token，选择器从原预算中保留实际相交的至多两个 64-token 页。重复、过短、图像或预算不足的情况沿用原打分。其余候选页继续根据当前层的 Mean-K 分数选择。

## NInfer 的 FP8 prefill 页分数复用

单序列、FP8 E4M3 Row256 KV、24 个 Q 头 / 4 个 KV 头 / head dimension 256 的 eager prefill，
可在启动服务前设置 `NINFER_PREFILL_PAGE_LSE_REUSE=1`，复用 attention 已计算的页 log-sum-exp，
省去独立的页分数 QK 扫描。再设 `NINFER_PREFILL_FP8_WIDE_KV=1`，可将每层可见 KV 展开一次并跨查询块共享。
两个开关默认关闭；不适用的形状和调用沿用普通 attention 与独立反馈计算。

Blackwell 的 page-LSE 支持沿用 [PR #305](https://github.com/kvmem/kvmem/pull/305)；本次增量扩展 Turing 的内核形状选择，
并按共享内存需求放开 wide-KV，不重复叠加 Blackwell 的收益。架构判定如下：

| 后端目标 | page-LSE 复用 | wide-KV 复用 |
|---|---|---|
| Turing，sm_75 / RTX 20 系 | 使用 Turing 小共享内存形状 | 不启用：内核需 96 KiB，硬件上限 64 KiB |
| Ampere / Ada，sm_80/86/89 | 支持 | 支持 |
| Blackwell，sm_120a / RTX 50 系、RTX PRO 6000 | 支持 | 支持 |
| Volta，sm_70 / V100 | 保留独立反馈路径，其专用 prompt 实现未接入捕获 | 不启用 |

本次通过 sm_75/80/86/89/120a 编译检查；实际 GPU 验证使用 RTX PRO 6000，不能视为 Turing 等其他实卡已验证。
后端的原生编译目标范围未扩大到 Hopper/B200。
wide-KV 仍要求 prefill 宽度至少 128、可见 KV 不超过 65536 tokens，额外暂存空间在 40960 tokens 时为 160 MiB、
65536 tokens 时为 256 MiB。窄 prefill 可能不足以摊薄展开和额外读写成本。

以下独立算子及 NVFP4 整模型数据来自合入 #305 前的原生树 `9fb10dda69e2`，各报告保留原始构建标识。
PRO6000 上，前两组固定 36864 历史、6 轮轮换顺序、每组每轮预热 5 次并计时 10 次，
attention 加页分数计算的单次耗时中位数如下（单位 ms）：

| Prefill / 历史 tokens | 全关闭 | 仅 page-LSE | 两项全开 |
|---|---:|---:|---:|
| 1024 / 36864 | 16.771 | 7.908 | 6.792 |
| 4096 / 36864 | 65.743 | 29.770 | 21.133 |
| 128 / 65408 | 5.078 | 3.319 | 4.138 |

前两组两项全开的算子吞吐分别为基线的约 2.47 / 3.11 倍；第三组 wide-KV 比只开 page-LSE 慢约 24.7%，
因此不能把两项全开作为所有负载的最快设置。这些是独立算子结果，不是整模型 tokens/s，未测 decode 或 RULER 质量。

45 组边界/开关组合完成 90 项独立 FP64 参考检查，覆盖非连续物理页、页/分块边界及 65536-token 阈值，
并检查归一化、因果不可见页为零和输出/暂存区哨兵。attention 输出在三组之间逐位相同；
复用页分数与独立 FP32 反馈存在舍入差异，最大相对 L2 约 0.508%，不可宣称页分数逐位一致。
新测试初稿的 0.9% 单点相对上限也被未优化基线触发；独立舍入诊断确认来自已有 BF16 Q/K 暂存及 softmax 放大。
最终对所有样本统一采用 1.2% 相对 L2 和 `0.004 + 1.2% × max(|reference|)` 单点上限后全量重跑，
未改变生产计算或原有测试标准。[完整计时、数值标准与验证范围](benchmarks/ninfer-prefill-architectures-20261114.json)

同日补测 PRO6000 整模型：Qwen3.8-27B NVFP4、FP8 KV、单并发、262148-token 固定输入，
历史预算 36864、生成预留 4096、prefill 分块 4096；关闭最终 Query 重选和投机解码。
每次独立重启服务并预热，三组轮换顺序各测三次，全部长请求缓存命中为零。
中位数如下；首 token 时间从客户端发出请求到收到首个非空文本片段，排除服务启动和预热：

| 设置 | Prefill tokens/s | 相对全关闭 | 首 token 秒 |
|---|---:|---:|---:|
| 全关闭 | 1977.4 | — | 132.86 |
| 仅 page-LSE | 2286.7 | +15.6% | 114.94 |
| 两项全开 | 2329.8 | +17.8% | 112.84 |

两项全开让首 token 时间减少约 20.0 秒（15.1%），其中 wide-KV 相比仅开 page-LSE 额外增加约 1.9% 吞吐。
三次吞吐范围分别为 1974–2003、2271–2295、2328–2331 tokens/s。
这验证了该配置下的整模型 prefill 收益；分块不同于 PR 的 1024-token 配置，不能视为 PR #302 的 RTX4090 数据的精确复现。
输出仅为短确认 `OK`，本轮未评估召回质量或 decode 吞吐。两个开关仍默认关闭。
[整模型配置、逐次计时和证据校验](benchmarks/ninfer-prefill-pro6000-model-20261114.json)

PR #305 的作者另在 Linux / CUDA 13.1、`NINFER_SM120_NATIVE=OFF` 的兼容构建上报告：

本轮实测设备为 RTX PRO 6000 Blackwell，未单独测试 RTX 5090。36K 历史、1024-token chunk、32 个采样位置的 attention 加统计约从 31.81 ms 降至 9.75 ms；四组内核用例的主 attention 输出逐 bit 一致，页概率通过独立 FP64 参考检验，但与独立 FP32 Q·K 统计存在浮点累计差异。

完整服务对照使用同一 Qwen3.8-27B NVIDIA NVFP4 模型、FP8 KV、36K 预算、64-token 物理页及评分、1024-token chunk、单路请求，只切换统计复用开关；另一个 MTP 驻留整理优化在两组中固定开启，不包含在此 SM120 支持改动内。每种输入各测三次：16,384-token 输入从平均 5,005 提升到 7,899 token/s（+57.8%）；98,304-token 输入从 1,988 提升到 3,910 token/s（+96.7%），耗时从 49.44 降到 25.14 秒。口径包含主机处理、KV 搬运和单 token 完成，排除模型加载及预热；未重跑 130 题准确率或测量 decode／并发吞吐。[测量条件、误差和验证范围](benchmarks/ninfer-sm120-prefill-page-lse-20261114.json)

本地独立复测 #305 使用 Windows / CUDA 13.2、`NINFER_SM120_NATIVE=ON` 的 NInfer，
模型为 Qwen3.8-27B GSQ-RCO IQ3_S，FP8 KV、36864-token 历史预算、4096-token reserve、1024-token prefill chunk，
关闭 Query 重选、投机解码和 wide-KV，仅切换 page-LSE。冻结原始 token 输入，每组每种长度三次，
关闭组分布在开启组的前后；每次重启服务后均预热。全部计时请求的缓存命中为零。
吞吐口径为输入 token 数除以平均请求到单 token 完成耗时，不包含启动和预热：

| 输入 tokens | 关闭 tokens/s | 开启 tokens/s | 吞吐提升 | 关闭 / 开启秒 |
|---|---:|---:|---:|---:|
| 16384 | 3383.8 | 4079.3 | +20.6% | 4.842 / 4.016 |
| 98304 | 1737.2 | 2700.7 | +55.5% | 56.588 / 36.399 |

这验证了 #305 在本机 IQ3 配置下的收益；模型、操作系统、原生构建及 MTP 配置不同，不能与作者 NVFP4 数据直接比大小，
也不能把它与前述 262148-token、4096 chunk 的结果当作只改变 chunk 的对照。
[本地复测配置、12 次结果及证据校验](benchmarks/ninfer-pr305-pro6000-20261114.json)

增量合入后的原生树 `dc3be91cb6e42` 再次通过五种架构编译、45 组用例 / 90 项 FP64 参考检查；
PRO6000 的两项全开版本另完成 16K、98K IQ3 输入的集成验证。后两次仅为功能验证，不作为增量性能的统计结论。
原有 #305 的实验测试保留不变，两个环境变量仍默认关闭。
[增量构建、数值与模型验证记录](benchmarks/ninfer-prefill-architecture-integration-20261114.json)

### RTX PRO 6000 上的 1,024-token 分块

在 RTX PRO 6000 Blackwell Server Edition 上运行 Qwen3.8-27B NVIDIA NVFP4，使用 FP8 KV、单并发、
36,864-token 历史预算和 4,096-token 生成预留。启动时设置 `NINFER_PREFILL_PAGE_LSE_REUSE=1`，
保留 `--prefill-chunk 1024`，再用 `NINFER_PREFILL_FP8_WIDE_KV=1` 开启 KV 展开复用。
每层每个分块展开一次可见的 FP8 KV，供该层的多个查询块复用。两组仅切换 wide-KV，均关闭前缀复用和最终 Query 重选。

| 输入 tokens | 仅 page-LSE | page-LSE + wide-KV | Prefill 吞吐提升 |
|---:|---:|---:|---:|
| 16,384 | 8,606 tokens/s | 9,378 tokens/s | +9.0% |
| 36,864 | 5,942 tokens/s | 6,874 tokens/s | +15.7% |
| 98,304 | 3,840 tokens/s | 4,516 tokens/s | +17.6% |

吞吐使用实际计算的输入 token 数除以平均服务端 prefill 时间。每档各测两次，98,304-token 的关闭组取自同配置的三次先前运行。
36,864-token 的 Nsight 对照中，attention GPU 耗时由 3.07 秒降至 2.05 秒，后者包含 KV 展开成本。
三个固定输入均各生成 256 tokens，两组返回文本一致。本轮未测召回准确率。
[完整启动参数、输入标识与逐次计时](benchmarks/ninfer-prefill-pro6000-wide-kv-1k-20261118.json)

## NInfer 的 age-adjusted prefill Pressure（实验开关）

设置 `NINFER_LAYERWISE_PRESSURE_SCORER=lift` 后，普通 prefill Pressure 按
`decayed attention mass / decayed expected attention` 给每层的驻留历史页打分。
默认仍为 `attention`，使用原来的 `mass / exposure`。开关只作用于 NInfer 的
Layerwise prefill Pressure；最终 Query 的 Mean-K 和 rolling decode Pressure 保留各自策略。

每个请求、每个目标 attention 层独立累计 14 个距离桶的 attention 和可见样本对数。
距离是采样行在原始文档中的位置减去逻辑页末尾，页末尾按实际 `block_tokens`
计算（NInfer 当前为 64）；窗口压缩后的物理坐标只判断可见性。
桶边界为 32、64、128、…、131072 tokens，最后一桶容纳更远的历史。
整个已接受 packet 先更新距离分布，再把每行的桶均值归一化为总和 1，累计 expected。
重复物理片段分别贡献 mass/expected，exposure 仍按同一逻辑页、同一行去重。

mass、exposure 和 expected 使用同一衰减；距离桶的累计分布不衰减。
lift 默认半衰期 32768 tokens，attention 默认 8192。
`NINFER_LAYERWISE_PREFILL_HALF_LIFE=<正有限数>` 可独立覆盖半衰期，供消融实验使用。
从未获得 expected 的页按最高 lift 加 1 排序；没有正 lift 时保留原评分回退。
sink、必选页、预算、只从当前驻留候选中整理的约束都保留。

expected、完整距离分布及评分模式一起进入 accepted snapshot、原生检查点、fork、
来源标识和元数据计费。恢复时校验模式；退回早于统计边界的位置会同时清空这些统计，
避免混入未来信息。未接受或校验失败的 packet 不发布任何一层的新统计。
rolling decode 的输出分数和 prefill fallback 继续使用原来的 ratio。

可与 `NINFER_LAYERWISE_GLOBAL_QUERY=1` 组合：prefill 使用 lift，最终 Query 的
3/7/11/15 层共享全局 Mean-K，19–63 层继续逐层 Mean-K。
这些环境变量在启动服务前设置；同一服务的多个请求仍各自维护独立统计。
该移植不新增 CUDA kernel，沿用已有 Ada/SM120 attention mass 采集路径。

数值校验包含独立 long-double 参考实现、32/64-token 页、距离桶边界及大坐标、
压缩前后坐标等价、重复片段、检查点恢复/回退，以及关闭 lift 后原统计逐位一致。
原生测试另外用独立 FP64 attention 检查 expected，并通过真实 Program 检查点恢复
验证统计完整性。GPU 正确性及小规模 RULER 结果以实际运行报告为准，不能由编译通过推断。

与 kvmem-vllm 冻结实验 `runtime-step8-combined` 的 CPU 对照在相同 32-token
输入和 32768 半衰期下，384 个 packet、27,904 个键的 mass、exposure、lift score
逐位相同；由 vLLM 原生分数导出的 384 组 Top-K 与 NInfer 的实际 picker 一致。
该对照使用合成 attention 数据，不等于两个推理后端端到端输出一致。实际 NInfer
服务仍按 64-token 整页累计、计算 Mean-K 和选页，vLLM 实验副本按 32-token 页；
分页本身会改变距离桶、softmax 域和可保留内容。vLLM 还需要显式把半衰期设置为 32768。

vLLM 实验副本的 `from_values`、普通序列化恢复和 values-based GC 重建会丢弃
lift 字段；快速 `StateRows` 对象共享路径可以保留，但旧的状态相等/行序列标识
没有包含 expected/profile。NInfer 的检查点和身份校验已经纳入这些字段。

本次移植的构建、数值与检查点验证，以及 4090 上的准确率测试结果见[实验记录](benchmarks/ninfer-lift-pressure-20261115.json)。

最初的 67 补丁集成完成了主机编译与 CPU 校验；上述 4090 测量来自更早的实验树，记录保留于[原始集成验证](benchmarks/ninfer-combined-pr-integration-20261115.json)。

2026-11-16 合并验证在 RTX PRO 6000 上完成，最终补丁链为 68 项。修正了 MSVC 大坐标测试参考中的浮点精度问题，以及原生测试中 restore 事务、MTP scratch 页和实验 Query 计数的夹具；未改变这些运行时约束。独立数值、检查点、五种策略的真实 Engine 跨轮流程，以及组合模式的 MTP／DFlash2、部分接受与取消校验通过。

长文档采用普通对话的“文档→OK→单独问题”协议，固定 512-token prefill、4096-token 选择预算与 1024-token reserve。基线、开关全关、lift、同半衰期 lift、hybrid、global 和组合模式各完成两份约 15.6K-token 文档，均命中 6/6 个目标事实；原生日志确认问题区间、前缀复用，以及工具续写不启动 Query 召回。默认关闭时答案与 token 用量同基线一致。两份同类合成文档不足以证明普遍质量提升，单次计时也不代表吞吐改进；原 4090 实验的失分结论仍有效，实验开关继续默认关闭。[PRO6000 合并验证与审计](benchmarks/ninfer-pr310-pro6000-20261116.json)
