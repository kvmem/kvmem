# KVMem 原生命令行与公共配置规范

## 独立分层策略

新增 `RecallStrategy=legacy|layerwise` 和五个有限激活/metadata 资源字段；
完整字段、原生 flag、显式 OWN MTP 双策略、Gemma Guess 与 DFlash2 窗口语义见
[分层召回配置](layerwise-recall.md)。下文旧参数对齐、评分器和 Query replay 说明属于 Legacy。
Layerwise 的显式块值不自动取整；sink/recent 空缺与显式 0 分开处理，
每个 worker 用独立策略能力声明实际最低 G、KV、并行和草稿组合。Radiance 新 prefill 缺省为 256。

## 启动参数对齐（2026-10-27）

llama.cpp 服务端与独立 CLI、PolyStrata 的默认 KV 管理块大小为 32；ninfer 仍为 64。
显式选择的块大小不因默认值改变而覆盖。PolyStrata 继续支持 32/64。

Budget、Reserve、SinkTokens、RecentTokens 在容量计算前按后端实际块大小向下取整。
ninfer 的 Prefill 按 128 向下取整；ninfer 和 PolyStrata 的 Prefill 同时限制在生效 Reserve 内。
每个改变的参数输出一次 `KVMEM_CONFIG 参数: 请求值 -> 生效值 tokens (原因)`。
Windows 启动预览返回 `token_settings` 的 requested、effective、block_tokens 和 adjustments；
控制台检查页面展示这些值，启动日志记录同一调整。配置草稿保留用户输入。

最低容量与保护语义不变：ninfer/PolyStrata 的 Budget 至少 128；Reserve 至少一块；
ninfer 的 Prefill/Reserve 必须容纳至少 128 个预填充 tokens。llama.cpp 保留已有的不足一块
Budget/Reserve 补至一块规则。llama.cpp/ninfer 的 Sink=0 仍表示至少一块，PolyStrata 的
Sink=0 仍为不额外保护前缀。llama.cpp 的 Recent 可向下取成 0；其他两个后端至少一块。
取整后的 Sink+Recent 必须不超过 Budget，Budget+Reserve 必须不超过 Context。
这些组合冲突继续返回具体错误，不擅自削减历史或保护区来容纳它们。

Context、输出/思考上限、Query/replay 上限和 RAM/SSD MiB 配额不套用 KV 块取整。
llama.cpp 的内部 Context 补齐和 batch/ubatch 限制仍由原生运行时负责；启动预览中的
token_settings 表示规范化后的启动参数，不代表模型加载后的显存可用容量。

## 当前源码修订：图召回 C

当前实现以 [图召回 C 规范及验证](graph-recall.md) 覆盖本文件早期设计/验收记录中的以下部分：

- Query 默认使用逐 token/head 的 Mean-K page softmax 与一次图传播。原生命令行和 Windows 启动脚本可切换 `--kvmem-query-scorer meanq-cosine|page-softmax`、GQA mean/max、LRFU 和 graph alpha；组合限制与默认值见 [评分参数](graph-recall.md#startup-scoring-options)。UI 继续使用默认 C，旧草稿/模板的历史设置不自动导出。
- PrefillPressure 和 DecodePressure 仅在 accepted 驻留集合中按累计注意力重选；R 是可重复使用的追加空间。两个后端都允许输出超过 R，仍受 context、请求输出预算和结束条件限制。
- UI 内置配置保留 0.18 资源参数并标为“0.18 资源配置 + C”。发布的 0.18 二进制和下文带日期的历史验证不因此改变。
- 2026-10-22 更新到 ninfer-all `db8c114802a0` 后，未启用 KVMem 的原生默认缓存为 hybrid；显式 catalog 专用选项会选择原有 checkpoint catalog。启用 KVMem 时仍由 KVMem 管理历史，不启用原生 hybrid 缓存。

下文保留原有命名收敛、原生参数映射与历史验收依据；其 LRFU/max 控件和 llama R 上限描述属于本次修订前的行为。


> Device identifiers and private paths have been anonymized; reported measurements are unchanged.

本规范以当前 `llama-kvmem-server.exe` 和 `ninfer-serve.exe` 的命令行接口为基础，覆盖原生参数、KVMem 自有接口、公共配置映射和组合校验。llama.cpp/NInfer 原生参数名称和含义保持不变；KVMem 自己增加的接口统一名称、定义和配置来源，必要时修改两个适配层。启动脚本的现有参数不限制本规范的范围。

基线为 2026 年 10 月 17 日的 KVMem CLI v1；本轮 `feat/ninfer-all-27b-refresh` 工作分支将 NInfer 更新到 `08fd7019925f`，精确来源以 [版本清单](../backends/versions.json) 为准。“公共入口建议”和“UI 建议”仍包含后续配置系统设计。当前范围是 llama.cpp 与 NInfer 的服务 worker；离线实验工具 `llama-kvmem-cli` 的 RAM/SSD 参数已迁到公共 H/D；其余实验参数和底层 Engine API 不等同于服务 CLI。

## 一 接口归属与命名边界

| 归属 | 命名和定义规则 | 例子 |
|---|---|---|
| 后端原生接口 | 原名、已有别名、原生含义保持不变；公共入口可做显式映射 | ctx-size/max-context、spec-type/spec、chat-template、kv-dtype、原生缓存及内核选项 |
| KVMem 公共扩展 | 一套规范名称、一套定义、一份配置来源；由两个适配层实现，不保留按后端分叉的公共定义 | kvmem-budget、kvmem-gen-reserve、kvmem-sessions、kvmem-host-mib、LRFU |
| KVMem 适配高级扩展 | 名称和含义仍统一，但按后端能力开放；暂不支持时明确不可用 | kvmem-query-policy、kvmem-mtp-state、kvmem-block-tokens |
| KVMem 旧接口 | 归入迁移说明；只有语义等价才能作为兼容别名，不生成新的模板或重复的 UI 字段 | kvmem-conversations、旧会话 GiB 软上限；旧磁盘参数已退役并明确报错 |

参数出现在后端 EXE 中，不代表它就是上游原生接口。尤其 `--kvmem-*` 是本项目扩展，不能借“原生兼容”保留两套公共定义。下文第三、四节是当前 EXE 的接口盘点，包含这类扩展；其旧行为用于迁移核对，不是新规范必须保留的契约。

### KVMem 自有接口的统一契约

两个 EXE 已共用 [CliOptions](../kvmem/include/kvmem/cli_options.h) 解析、默认值和基础校验；适配层继续校验各自能力。R 的持续追加能力仍有明确差异，见下表。

| 规范接口 | 唯一定义 | 实现与能力限制 |
|---|---|---|
| `--kvmem` / `--no-kvmem` | 同一个 enabled 布尔配置的正反开关；启停不编码到 B=0 中 | 两边均已实现；无 KVMem 参数时默认关闭，正 B 保留隐式启用兼容；显式关闭与有效预算冲突时报错 |
| `--kvmem-budget B` | 每条活动请求最多选入的历史 tokens；规范化后为正整数；不含 R；0 不表示自动或关闭 | 共用解析，启用后省略值为 32768；显式 0 报错 |
| `--kvmem-gen-reserve R` | 每条活动请求的追加工作空间，tokens；规范化后为正整数；与用户输出 token 上限分别定义 | llama 现有 R 输出上限列为待处理的实现限制，不把它写成另一种 R 定义；解除限制需要独立的正确性设计和回归 |
| `--kvmem-sessions N` | 允许保留的非活动历史数量上限；活动请求不计入；至少 1，资源不足可提前淘汰 | llama 单路和多路已按运行时非活动历史计数并淘汰；默认 4，不采用固定 N+并发换算 |
| `--kvmem-host-mib H` | 全部活动与非活动历史共享的原生 KV payload RAM 硬预算，MiB；规范化后为正整数 | 两边共用；明确不包含整个进程 RAM，不把旧软上限并入此定义 |
| `--kvmem-disk-mib D` | 全部活动与非活动历史共享的进程内 SSD 预算，MiB；0 统一表示关闭 | 两边均接受 0；D=0 时不得携带目录；启动器关闭 SSD 时不生成目录实参 |
| `--kvmem-disk-path PATH` | 上述 SSD 层的目录；D>0 时必需，不代表跨重启持久缓存 | 配对校验使用同一规则；不与后端原生 disk-kv-path 混同 |
| `--kvmem-sink-tokens` | B 内必须保留的前缀；0 表示后端最小一个检索块；正值按块对齐 | 同一计算规则，块粒度作为能力信息；展示有效 tokens |
| `--kvmem-recent-tokens` | B 内必须保留的最近后缀；0 的概念是无额外后缀要求 | NInfer 当前至少一页属于能力约束，输入 0 时明确拒绝；不能把 0 静默解释成一页 |
| `--kvmem-lrfu` / `--no-kvmem-lrfu` | 同一个 lrfu 布尔配置；提供明确开启/关闭以覆盖模板值 | 两个扩展解析器共用定义；UI 始终使用正向标签 |
| `--kvmem-lrfu-lambda`、`--kvmem-lrfu-top-frac`、`--kvmem-score-agg` | 分别为 [0,1] 衰减、(0,1] 命中比例、mean/max 聚合 | 名称和含义已经一致，继续沿用 |
| `--kvmem-block-tokens`、`--kvmem-query-*`、`--kvmem-mtp-state` | 一个名称表示一种高级能力或策略 | 支持范围作为 capability，不为另一后端另起同义参数，也不强行模拟不具备的策略 |
| `--kvmem-trace` 与 `--kvmem-verify-transfers` | 分别是诊断日志与搬运正确性校验，两种不同操作 | 保留独立名称，按能力开放；不为“统一”合并为一个开关 |

KVMem 配置对象只保存一份 `sessions`、`budget`、`gen_reserve`、`host_mib` 等规范值。状态接口分别报告 `active_sessions`、`retained_sessions`、`retained_sessions_limit`，不再让同一个“会话数”字段在两个后端计不同对象。旧 conversations 的值只能按旧版本和计数方式迁移，不能注册成无条件的同值别名。

KVMem 新公共配置的默认策略由一份版本化定义提供，模板可显式覆盖。后端原生参数仍可选择“后端默认”。两个 EXE 的默认值来自同一 C++ 定义；启动模板显式传入的值是预设覆盖，不是第二套 EXE 默认。具体粒度、最小值、最大值和未实现功能属于能力限制，不改变公共名称和含义。

### 本次迁移与兼容边界

- 后端原生参数名、别名和含义不变；没有给 `ctx-size/max-context`、`spec-type/spec` 等 EXE 参数强行改名。
- KVMem 整数采用统一十进制输入，范围为 0 或 1 至 2147483647（是否允许 0 见各项定义）；MiB 在校验后转换为 64 位 bytes。更小的后端能力上限另行校验。
- 新启动器对两个后端都输出 `--kvmem`、B/R/H 和 `--kvmem-sessions`；关闭 LRFU 输出 `--no-kvmem-lrfu`。同一公共项不得再从 WorkerArgs 重复覆盖。
- 旧 `--kvmem-conversations` 只保留显式兼容模式，并提示弃用；它和 `--kvmem-sessions` 同时出现时报错。旧 GiB 软预算要求显式选择旧模式，不转换为 H 硬预算。
- llama 旧的“省略参数自动开启 / B=0 自动推导 / R=256 / H 无上限”不再是新默认。旧命令应显式传入正 B/R/H 和合适的原生上下文；不是机械替换参数名。
- llama R 的持续追加改造不在本次命名收敛中完成。当前检索工作集被固定，R 耗尽时没有安全的自动重选；因此保留输出能力校验，并在帮助和状态中暴露 `generation_limit`。后续需要单独验证检索保留、MTP 接受边界与原生 KV 状态，不能直接删掉校验。
- 状态中 `active_sessions`、`retained_sessions`、`retained_sessions_limit` 分别表示活动数、非活动历史数、非活动上限。llama 的旧兼容模式上限为 null，并带 `session_limit_kind=legacy_total`；旧 `count/max` 仅供旧客户端迁移。

### 参数记录格式

| 字段 | 定义 |
|---|---|
| 接口归属 | 后端原生、KVMem 公共扩展、KVMem 适配高级扩展或旧接口；决定名称是否应统一 |
| 原生名称和别名 | 按后端分别记录原名称、短参数、兼容别名和位置参数；不覆盖原生含义 |
| 含义和作用范围 | 控制模型加载、进程服务、默认请求设置、当前请求或缓存中的哪一部分 |
| 输入定义 | 开关、整数、小数、枚举、路径、文本、JSON 或有序列表，以及单位 |
| 省略行为 | EXE 未收到该参数时的默认行为；与模板预设分开 |
| 特殊值 | 明确区分未设置、0、负数、自动、关闭；只接受原生支持的形式 |
| 约束 | 范围、对齐、依赖、互斥、模型要求、设备要求和 KVMem 模式要求 |
| 公共映射 | 公共语义、对应原生参数、转换规则；不可等价转换时保留独立设置 |
| UI 展示 | 所属分组、常用或高级、当前可用状态及原因 |
| 验证状态 | 分别记录参数已实现、组合允许、组合实测通过；不把三者合并为“支持” |
| 来源 | EXE 版本、解析代码、原生帮助和对应验收记录 |

## 二 公共配置映射表

“直接沿用”表示复用现有名称并执行约定的校验。“入口别名”和“公共 UI”用于适配后端原生参数，不能代替 KVMem 自有接口的统一。KVMem 扩展按上一节的统一契约实现；两个 EXE 的当前差异是适配与迁移工作。

### 模型与服务

| 公共设置 | 类型 | llama.cpp 原生参数 | NInfer 原生参数 | 公共入口建议 |
|---|---|---|---|---|
| 后端 | 枚举 | 由 EXE 选择 | 由 EXE 选择 | 统一入口提供 `--backend llamacpp\|ninfer`；不传给 EXE |
| 模型 | 文件 | `--model` / `-m` | 首个位置参数 | 入口沿用 `--model`；NInfer 导出为位置参数 |
| API 模型名 | 文本 | `--alias` / `-a` | `--model-id` | 入口推荐 `--alias`，兼容 `--model-id`；按后端导出 |
| 监听地址 | 文本 | `--host` | `--host` | 直接沿用；UI 提供 `127.0.0.1`、`0.0.0.0` 和自定义 |
| 服务端口 | 整数 | `--port` | `--port` | 直接沿用；1–65535；校验端口占用 |
| API key | 文本或列表 | `--api-key` 可多 key，另有文件入口 | `--api-key` 单 key | 单 key 为公共设置；多 key 和文件作为 llama 扩展 |
| 内置聊天页面 | 继承/开/关 | `--ui` / `--webui` / `--no-ui` / `--no-webui` | 默认提供已编入的页面；`--no-webui` 关闭 | 关闭沿用 `--no-webui`；开启是否可用取决于构建包 |
| 日志级别 | 枚举 | `--verbosity`，0–5 | `--log-level`，文字枚举 | 公共 UI；保留两套原生名；公共级别可映射 off/error/warning/info/trace/debug，NInfer 的 critical 单独保留 |

### 上下文与输出

| 公共设置 | 类型与单位 | llama.cpp 原生参数 | NInfer 原生参数 | 公共入口建议 |
|---|---|---|---|---|
| 逻辑上下文 | 正整数，tokens | `--ctx-size` / `-c` | `--max-context` | 入口推荐 `--ctx-size`，兼容 `--max-context` |
| 默认输出预算 | 后端默认或正整数，tokens | `--n-predict` / `-n` / `--predict`；另接受 -1 | `--default-max-tokens`；显式值必须为正 | 公共 UI；保留原名和特殊值，不直接互设别名 |
| 活动请求数 | 正整数 | `--parallel` / `-np`，1–8 | `--max-concurrency`，1–8 | 入口推荐 `--parallel`，兼容 `--max-concurrency`；按组合校验 |
| 预填充设置 | tokens | `--batch-size` 逻辑批次与 `--ubatch-size` 物理批次 | `--prefill-chunk` | 放同一公共分组，显示各自原名和概念；不把三个参数绑定成一个值 |
| 长上下文位置缩放 | 后端选项 | 当前 worker 无相应启动覆盖项 | `--rope-yarn` 等 | 公共 UI 可显示该能力；只有 NInfer 开放具体设置 |

### 设备与并行

| 公共设置 | 类型 | llama.cpp 原生参数 | NInfer 原生参数 | 公共入口建议 |
|---|---|---|---|---|
| GPU 选择及顺序 | 有序设备列表 | `--device CUDA0,CUDA1,...` | 单卡 `--device N`；多卡 `--devices A,B,...` | 公共 UI 保存 UUID 顺序；启动时解析成各后端原生标识，不修改原生参数的输入格式 |
| 切分方式 | 枚举 | `--split-mode none\|layer\|tensor` | 多卡为 layer pipeline | 公共 UI 按能力展示；不把 tensor 与 layer pipeline 互相替代 |
| 主 GPU | 设备 | `--main-gpu` | 第一阶段承担 embedding、head 等工作 | llama 独立设置；NInfer 用设备顺序表达；两者不作数值别名 |
| 多卡分配 | 比例或层数列表 | `--tensor-split` 比例 | `--stage-layers` 每阶段层数 | 公共 UI 按后端切换输入形式，不转换比例与层数 |
| GPU 层数 | 整数或 all | `--n-gpu-layers` / `--gpu-layers` / `-ngl` | 无同等启动项 | 公共高级项，仅 llama 可配置 |
| CPU 生成线程 | 整数/自动 | `--threads` / `-t` | 无对应项 | 公共高级项，仅 llama 可配置 |
| CPU 批处理线程 | 整数/自动 | `--threads-batch` / `-tb` | 无对应项 | 公共高级项，仅 llama 可配置；不映射媒体预处理线程 |

### KV 与 KVMem

| 公共设置 | 类型与单位 | llama.cpp 原生参数 | NInfer 原生参数 | 规则 |
|---|---|---|---|---|
| 启用 KVMem | 继承/开/关 | `--kvmem` / `--no-kvmem` | 同名 | 扩展接口统一为显式正反开关；继承在公共配置中解析为明确值 |
| 历史选择预算 B | tokens | `--kvmem-budget` | 同名 | 新契约为正的历史选择预算；统一消除 0 的两种定义 |
| 追加预留 R | tokens | `--kvmem-gen-reserve` | 同名 | 统一为追加工作空间；llama 当前输出限制作为待处理的适配限制 |
| 固定前缀 | tokens | `--kvmem-sink-tokens` | 同名 | 直接沿用；0 表示一块/一页，正值按本后端粒度向下取整且至少一块/页 |
| 最近历史 | tokens | `--kvmem-recent-tokens` | 同名 | 直接沿用；NInfer 至少 64，llama 允许 0；计入 B |
| LRFU | 继承/开/关 | `--kvmem-lrfu` | 同名 | 公共正反开关统一；旧 EXE 关闭时移除正向开关；当前 EXE 默认关，0.18 模板开启 |
| LRFU 衰减 | 小数 [0,1] | `--kvmem-lrfu-lambda` | 同名 | 直接沿用，默认 0.5 |
| LRFU 命中比例 | 小数 (0,1] | `--kvmem-lrfu-top-frac` | 同名 | 直接沿用，默认 0.9 |
| GQA 分数聚合 | mean/max | `--kvmem-score-agg` | 同名 | 直接沿用，EXE 默认 mean；模板可选 max |
| RAM 原生 KV 预算 | MiB | `--kvmem-host-mib` | 同名 | 统一为全局原生 KV payload 硬预算，规范化后为正；不等于进程 RAM 或旧会话软上限 |
| SSD 原生 KV 预算 | MiB | `--kvmem-disk-mib` | 同名 | 新契约中 0 统一关闭；D>0 需要目录；当前旧 EXE 可用省略实参表达关闭 |
| SSD 目录 | 目录 | `--kvmem-disk-path` | 同名 | 直接沿用；公共层是进程内缓存，不表示跨重启恢复 |
| KV 格式 | 后端枚举 | `--kv-dtype` | 同名 | 参数名沿用，格式名保持原样；后端切换时重新确认格式 |
| 独立 K/V 格式 | 后端枚举 | `--cache-type-k` / `--cache-type-v` | 无对应项 | 公共高级项，仅 llama 开放；不把 NInfer 组合格式拆成任意 K/V |
| 历史保留数量 | 整数 | `--kvmem-sessions` | 同名 | 只计非活动历史；旧 conversations 不是同值别名，不进入 UI 或新模板 |
| 检索块大小 | tokens | `--kvmem-block-tokens`，默认 128 | 固定 64，无对应参数 | llama 可编辑，NInfer 显示固定值 |
| 检索与 query 策略 | 能力相关选项 | `--kvmem-method`、`--kvmem-query-*` | 两个长度参数同名：`--kvmem-query-max-tokens` / `--kvmem-query-replay-max-tokens` | 默认 512/2048；评分窗口和完整回放后缀独立限制 |

### 采样与思考

| 公共设置 | 类型或范围 | llama.cpp 原生参数 | NInfer 原生参数 | 规则 |
|---|---|---|---|---|
| 温度 | 0–2 | `--temperature` / `--temp` | `--temperature` | 推荐沿用 `--temperature`；统一入口可兼容 `--temp` |
| Top P | 0–1 | `--top-p` | 同名 | 直接沿用 |
| Top K | 整数 | `--top-k`，非负 | 同名，0–20 | 0 关闭；UI 使用当前后端范围，不把 llama 的范围缩为共同交集 |
| Min P | 0–1 | `--min-p` | 同名 | 直接沿用 |
| 存在惩罚 | -2–2 | `--presence-penalty` | 同名 | 直接沿用 |
| 频率惩罚 | -2–2 | `--frequency-penalty` | 同名 | 直接沿用 |
| 重复惩罚 | 正数 | `--repeat-penalty` / `--repetition-penalty` | 无对应项 | 保留 llama 原名；不映射到存在/频率惩罚 |
| 随机种子 | 无符号整数或继承 | `--seed` / `-s`，uint32 | `--seed`，uint64 | 名称沿用，范围按后端；相同 seed 不承诺跨后端相同输出 |
| 默认思考 | 继承/开/关 | `--enable-thinking` / `--no-think` | `--no-thinking`；普通聊天默认思考 | 公共 UI 正向显示；生成原生开关；不把所有模型默认推断为同一值 |
| 思考预算 | 继承或 tokens | `--reasoning-budget`：-1 不限、0 立即结束、正整数 | `--default-thinking-budget`：仅正整数，省略未设置 | 公共 UI 的可移植输入为继承/正整数；llama 0 独立保留，不转换成 NInfer 关闭思考 |
| 思考预算提示 | 文本 | `--reasoning-budget-message` | `--thinking-budget-message` | 公共 UI；保留原生名及结束控制语义 |
| 思考强度 | 模型相关枚举 | `--reasoning-effort` | `--default-reasoning-effort` | 公共 UI；按模型和后端校验，不仅按名称翻译 |
| 思考后采样 | 开关和采样参数 | 无对应项 | `--post-thinking`、`--post-thinking-*` | 公共高级项，NInfer 开放 |
| 模板文件 | 文件 | `--chat-template-file` | `--chat-template` | 公共文件入口推荐 `--chat-template-file`，NInfer 导出为其原生名 |
| 模板正文 | 文本 | `--chat-template` | 无正文入口 | llama 高级项；原生 `--chat-template` 永远按所选后端解释 |
| 模板变量 | JSON 对象 | `--chat-template-kwargs` | 无同等入口 | llama 高级项 |

### 推测解码与媒体

| 公共设置 | llama.cpp 原生参数 | NInfer 原生参数 | 规则 |
|---|---|---|---|
| 推测解码方式 | `--spec-type none\|draft-mtp\|draft-dflash` | `--spec mtp\|dflash\|dflash2`；省略关闭 | 公共 UI 通过 `SpeculativeMode` 选择关闭/MTP/DFlash2，按后端映射；普通 DFlash 不在公共入口开放 |
| 草稿数量 | `--spec-draft-n-max` | `--draft-tokens` | 公共 `DraftTokens`；固定数量与自适应上限明确区分，范围按组合校验 |
| 草稿模型 | `--spec-draft-model` | 使用模型内置组件 | 公共 `DraftModel`；llama DFlash2 必填匹配的 GGUF，MTP 可选；NInfer 禁止旁挂文件 |
| 自适应 MTP | 无对应项 | `--adaptive-mtp` | NInfer 可选；依赖 MTP；关闭时保留固定草稿数量 |
| Ngram 草稿 | 无对应项 | `--ngram-draft-tokens`、`--ngram-min-match` | NInfer 可选；开启 spec 后省略宽度会默认 15，明确关闭须传 0 |
| MTP KV 格式 | `--spec-kv-dtype` | 跟随主 KV 格式 | llama 可单独配置，NInfer 显示跟随主 KV |
| 视觉启用 | 提供 `--mmproj` | `--vision` | 公共 UI；llama 需要独立 projector，NInfer 需要模型包含视觉资源 |
| 视觉位置 | `--mmproj-offload` / `--no-mmproj-offload`，可选 `--mmproj-device` | `--vision-residency resident\|overlay\|cpu` | 公共 CPU/GPU 选择按后端映射，overlay 不是普通 GPU 开关，KVMem 下不可用 |
| 图像 token 控制 | `--image-min-tokens`、`--image-max-tokens` | `--vision-max-merged` | 同组展示，保留原生命名和计量含义，不承诺相同数值得到相同图像质量/大小 |
| 视频采样 | `--video-fps` | 无对应启动项；当前 KVMem 不开放视频 | llama 高级项并按组合校验 |

### 服务高级项

以下项目也可出现在公共“服务与日志”分组，但只有支持它的后端可配置。它们不因位置相同而建立跨后端别名。

| 配置 | 原生参数 | 当前后端 |
|---|---|---|
| HTTP 读写超时，秒 | `--timeout` / `-to` | llama |
| HTTP 线程数 | `--threads-http` | llama |
| 独立统计端口 | `--stats-port` | NInfer |
| 请求等待数 | `--max-pending-requests` | NInfer |
| 准备与准入等待，毫秒 | `--pending-timeout-ms` | NInfer；不等同于 HTTP 超时 |
| 请求体大小，MiB | `--max-request-mib` | NInfer |
| 请求记录与轮转 | `--request-log-jsonl`、`--request-log-max-mib`、`--request-log-keep` | NInfer |
| 统计输出 | `--log-stats-interval-ms`、`--log-stats-panel` | NInfer |
| 原始 KVMem 诊断 | `--kvmem-trace` / `--no-kvmem-trace` | llama |
| KV 搬运校验 | `--kvmem-verify-transfers` | NInfer；与日志开关含义不同 |

## 三 llama.cpp 当前 EXE 参数完整清单

本节盘点当前 worker 的公开参数及别名，包含上游风格原生参数和 KVMem 自有扩展；不是把全部参数归为不可修改的上游接口。未写默认值的项没有在本表中另定默认；省略时保留当前后端或模型行为。KVMem 扩展迁移后的目标定义见第一节。

### 基础服务与执行

| 原生参数及别名 | 输入和原生行为 |
|---|---|
| `--help` / `-h` / `--usage` | 显示帮助并退出 |
| `--model` / `-m` | GGUF 文件；必需，可由支持的环境变量提供 |
| `--alias` / `-a` | API 模型名；省略使用模型文件名 |
| `--host` | 默认 127.0.0.1 |
| `--port` | 1–65535，默认 8080 |
| `--api-key` | 一个或逗号分隔多个 key；可累计 |
| `--api-key-file` | 每行一个 key；忽略空行和 # 注释行 |
| `--timeout` / `-to` | 非负整数秒；默认 1800；HTTP 读写超时 |
| `--threads-http` | 固定 HTTP 工作线程数；省略或非正数自动使用 `2 × --parallel`，不随宿主 CPU 核数或请求队列扩容 |
| `--ui-dir` / `--path` | 静态页面目录，启用页面时需有可读 index.html |
| `--no-ui` / `--no-webui` | 禁用内置页面 |
| `--ui` / `--webui` | 启用内置页面；默认启用；可覆盖环境变量关闭 |
| `--verbosity` / `--log-verbosity` / `-lv` | 0 silent、1 error、2 warn、3 info、4 trace、5 debug；默认 3 |
| `--kvmem-trace` / `--no-kvmem-trace` | 开启/关闭原始 KVMem 诊断；显式值覆盖 KVMEM_TRACE |
| `--ctx-size` / `-c` | 正整数 tokens，默认 2048 |
| `--n-predict` / `--predict` / `-n` | 正整数或 -1；默认 -1 不额外设固定上限；不接受 0；仍受服务 generation limit 限制 |
| `--batch-size` / `-b` | 正整数 tokens；逻辑批次，默认 512 |
| `--ubatch-size` / `-ub` | 正整数 tokens；物理批次，省略继承 batch-size |
| `--threads` / `-t` | CPU 生成线程；非正数使用硬件并发数 |
| `--threads-batch` / `-tb` | CPU 批处理线程；省略继承 threads；非正数自动 |
| `--flash-attn` / `-fa` | on/off/auto；解析还接受 1/true、0/false；默认 auto |
| `--parallel` / `-np` | 1–8 个推理 lane，默认 1；上下文和设备 KV 预算按 lane |
| `--load-mode` / `-lm` | auto/none/mmap/mlock/mmap+mlock/dio；默认 auto |
| `--mmap` / `--no-mmap` / `--mlock` | 原有加载方式别名；保留解析与生效顺序 |
| `--n-gpu-layers` / `--gpu-layers` / `-ngl` | GPU 层数，默认 99；支持 all；当前拒绝 auto |
| `--device` / `-dev` | 原生有序设备名列表，如 CUDA0,CUDA1；none 为 CPU |
| `--list-devices` | 列出可用设备并退出 |
| `--main-gpu` / `-mg` | 主设备索引，默认 0；具体可用值由设备配置校验 |
| `--split-mode` / `-sm` | 当前服务可用 none/layer/tensor；解析器也识别 row，但服务设备配置另行限制，不作为 UI 可用模式 |
| `--tensor-split` / `-ts` | 按设备顺序的非负比例，逗号或 / 分隔；不能全为 0 |

### KV 检索和存储

| 原生参数 | 输入和原生行为 |
|---|---|
| `--kvmem` / `--no-kvmem` | 默认关闭；显式开启或正 B 启用；显式关闭与预算冲突时报错 |
| `--kvmem-budget` | 正整数 tokens；启用后默认 32768，显式 0 拒绝 |
| `--kvmem-block-tokens` | 开启 KVMem 时必须为正；默认 128 |
| `--kvmem-sink-tokens` | 默认 0 表示一块；正数向下取整且至少一块 |
| `--kvmem-gen-reserve` | 正整数 tokens，启用后默认 1024；R 是可复用追加空间；持续输出按 DecodePressure 腾空间 |
| `--kvmem-recent-tokens` | 非负 tokens，默认 64；按 block 向下取整，占用 B；默认 block=32 时保留两个块 |
| `--kvmem-method` | 文档化选项 recency/retrieval；默认 retrieval；解析还接受 retrieve |
| `--kvmem-lrfu` / `--no-kvmem-lrfu` | 开启/关闭历史衰减排序；默认关，后出现的开关覆盖前者 |
| `--kvmem-lrfu-lambda` | [0,1]，默认 0.5 |
| `--kvmem-lrfu-top-frac` | (0,1]，默认 0.9 |
| `--kvmem-score-agg` | mean/max，默认 mean |
| `--kvmem-query-last` | 缺少最后用户 span 时的 query 回退长度；默认 64 tokens |
| `--kvmem-query-max-tokens` | 正整数；从最后用户 span 尾部取 query 的最大长度，默认 512 |
| `--kvmem-query-replay-max-tokens` | 两个 worker 均接受正整数；默认 2048，限制完整依赖后缀；还须满足实际页容量与 MTP 工作区检查 |
| `--kvmem-query-replay` | llama 专用 legacy/auto，默认 auto；legacy 仅强制尝试有界回放，仍须通过准入 |
| `--kvmem-query-policy` | 仅接受 user，作为兼容参数；省略时使用同一流程。legacy 已删除，显式传入会报错 |
| `--kvmem-mtp-state` | snapshots/auto/replay；MTP 默认 replay，受切分组合限制 |
| `--kvmem-gpu-ratio` | [0,1]，默认 0.50；限制槽池显存比例，不是整个进程的显存百分比 |
| `--kvmem-host-mib` | 正整数 MiB；启用后默认 16384；原生 KV payload 全局硬上限 |
| `--kvmem-disk-mib` | 非负整数 MiB；默认 0 关闭公共 SSD 层；关闭时目录必须省略 |
| `--kvmem-disk-path` | 公共 SSD 层目录，与 disk-mib 配对 |
| `--kvmem-cpu-gb` | 非负 GiB，0 关闭旧 CPU spill arena |
| `--kvmem-nvme-gb` | 已退役；报错提示改用公共 host-mib/disk-mib/disk-path，不自动换算旧预算 |
| `--kvmem-nvme-dir` | 已退役；报错提示改用公共 host-mib/disk-mib/disk-path，不自动换算旧预算 |
| `--kvmem-harvest-v` | 预填充时将 V 连同 raw-K 搬到 Host，默认关 |
| `--kvmem-raw-k-nvme` | 已退役；报错提示改用公共 host-mib/disk-mib/disk-path，不自动换算旧预算 |
| `--kvmem-sessions` | 正整数，默认 4；非活动历史上限，不包含活动请求或空 lane |
| `--kvmem-conversations` | 已弃用的显式兼容模式：活动及非活动总数，并发时至少为 parallel；与 sessions 互斥，不自动换算 |
| `--kvmem-conversations-gb` / `--kvmem-session-ram-gb` | 非负 GiB；旧会话 RAM 软上限，默认 0 不限；活动会话可超过软上限 |
| `--kvmem-session-nvme-gb` | 已退役；报错提示改用公共 host-mib/disk-mib/disk-path，不自动换算旧预算 |
| `--kvmem-session-cache-dir` | 已退役；报错提示改用公共 host-mib/disk-mib/disk-path，不自动换算旧预算 |
| `--kv-dtype` | 同时设置 K/V；f16/f32/q8_0/q5_0/q4_0，默认 q8_0 |
| `--cache-type-k` / `-ctk` | 单独设置 K 格式；组合能力由原生适配器校验 |
| `--cache-type-v` / `-ctv` | 单独设置 V 格式；组合能力由原生适配器校验 |

### 采样 推测解码与思考

| 原生参数及别名 | 输入和原生行为 |
|---|---|
| `--temperature` / `--temp` | [0,2]；0 贪心；思考/非思考预设 1.0/0.7 |
| `--top-p` | [0,1]；思考/非思考预设 0.95/0.80 |
| `--top-k` | 非负整数，0 关闭；默认 20 |
| `--min-p` | [0,1]，默认 0 |
| `--presence-penalty` | [-2,2]；思考/非思考预设 0/1.5 |
| `--frequency-penalty` | [-2,2]，默认 0 |
| `--repeat-penalty` / `--repetition-penalty` | 正数，默认 1 |
| `--seed` / `-s` | uint32；省略使用随机行为；请求字段可覆盖进程默认 |
| `--spec-type` | none/draft-mtp/draft-dflash，默认 none；mtp/dflash2 为对应别名 |
| `--spec-draft-model` | DFlash2 必填独立草稿 GGUF；MTP 可选旁挂 GGUF；DFlash2 当前最多八路、单 GPU |
| `--spec-kv-dtype` | 草稿 KV 格式，默认 f16；接受 f16/f32/q8_0/q5_0/q4_0，仍需组合校验 |
| `--spec-draft-n-max` | 默认 3；当前 KVMem replay 模式要求 1–5 |
| `--spec-draft-p-min` | [0,1]，默认 0；最小草稿概率 |
| `--jinja` | 兼容开关；本服务始终启用 Jinja；`--no-jinja` 被明确拒绝 |
| `--chat-template` | Jinja 模板正文 |
| `--chat-template-file` | 模板文件；与正文入口二选一 |
| `--chat-template-kwargs` | JSON 对象；默认模板变量 |
| `--reasoning-effort` | 传递给模板的强度；default 使用模板默认，none 关闭思考 |
| `--enable-thinking` / `--no-think` | 默认关闭思考；请求可覆盖进程默认 |
| `--reasoning-budget` | 整数 >=-1；默认 -1 不限，0 立即结束，正数为思考 token 上限 |
| `--reasoning-budget-message` | 强制结束思考前插入的文字；默认不额外插入 |

### 视觉

| 原生参数及别名 | 输入和原生行为 |
|---|---|
| `--mmproj` / `-mm` | 视觉 projector GGUF |
| `--mmproj-offload` / `--no-mmproj-offload` | GPU/CPU 视觉编码，默认 GPU |
| `--mmproj-device` / `-mmdev` | 原生视觉设备名；none 使用 CPU |
| `--image-min-tokens` | 显式正整数；图片最小 token 配置 |
| `--image-max-tokens` | 显式正整数；不得小于显式设置的最小值 |
| `--video-fps` | 有限小数；默认 2；<=0 使用原视频帧率 |

## 四 NInfer 当前 EXE 参数完整清单

本节同时盘点 NInfer 原生参数和 KVMem 自有扩展的当前行为。原生缓存、路由、休眠等设置不能因为出现在帮助中就默认用于 KVMem；组合限制见第五节。KVMem 扩展的目标定义见第一节。

### 基础 模型与设备

| 原生参数 | 输入和原生行为 |
|---|---|
| `--help` / `-h` | 作为首参数显示帮助并退出 |
| `--version` | 显示原生版本和构建标识并退出 |
| `<model.ninfer>` | 普通模式首个位置参数；路由模式不提供该参数 |
| `--model-id` | API 模型名，省略使用 artifact metadata.name |
| `--chat-template` | 模板文件路径，不是模板正文 |
| `--max-context` | 正整数 tokens，默认 8192 |
| `--max-concurrency` | 1–8，默认 1 |
| `--prefill-chunk` | 向下对齐到 128 的倍数，KVMem 下限制到 R；0/省略由引擎自动选择（KVMem 使用 1024 再应用上述限制） |
| `--decode-rounds-per-prefill` | 每个 prefill chunk 之间的 decode 轮次，0–4096；默认 0 按 chunk/64 自动计算；1 为交替执行 |
| `--default-max-tokens` | 显式正整数；省略按上下文和并发准入预算计算；0 被当前解析器拒绝 |
| `--device` | CUDA 可见设备索引，默认 0；与 devices 互斥 |
| `--devices` | 解析器接受 1–8 个有序设备索引；2–8 项启用 layer pipeline；原生还允许重复设备索引用于单卡阶段测试，普通 UI 的显卡选择不产生重复项 |
| `--stage-layers` | 按 devices 顺序给每阶段分配层数；省略按可用显存规划 |
| `--device-profile` | auto/off/calibrate，默认 auto |
| `--device-profile-path` | 自定义设备 profile 文件 |
| `--context-cost-presets` | 自定义上下文成本配置文件 |
| `--expert-residency` | device/host/disk，默认 device；上游 Qwen MoE、Flash-Next、Gemma MoE 专家放置；当前 KVMem 不接 host/disk 专家路径 |
| `--expert-cache-mib` | 非负 MiB 或 auto；host/disk 常驻专家缓存，默认 auto；Qwen MoE 的 0 只保留层 bank，Flash-Next disk 模式仍需要缓存 |
| `--expert-misses` | Flash-Next 原生 staged/mapped 专家缺失处理；默认 staged；不作为公共 KV 设置 |
| `--expert-dma-share` | Flash-Next 保留的 CPU/GPU 专家实验比例，0–1，默认 1；本轮不扩展 CPU 专家执行，也不接 KVMem |
| `--expert-cpu-threads` | 上述原生专家实验的 CPU 线程，1–256，默认自动 |
| `--expert-cache-adaptive` | 原生专家实验的动态缓存替换开关 |
| `--expert-profile` / `--expert-profile-out` | 原生专家计数配置的读取/记录路径 |
| `--ngram-table` | Flash Next 的模型 n-gram 表文件；不同于推测解码 ngram |
| `--ngram-residency` | Flash-Next 模型表的 disk/ram/ram-hot，默认 disk；上游已用 `--ngram-residency ram` 替代旧 `--ngram-ram` |
| `--ngram-io` | 模型表文件的 buffered/direct/mmap，默认 buffered |
| `--ngram-io-depth` | 模型表并行行读取数，1–1024，默认 64 |
| `--ngram-hot-profile` | ram-hot 热行配置，覆盖表内配置 |
| `--ngram-ram-mib` | 模型表的 RAM 行与索引预算；disk 默认 0，ram-hot 默认 4096 MiB；不是 KVMem H |
| `--ngram-lock` | ram/ram-hot 下锁定表行的物理内存；受系统权限约束 |
| `--no-ngram-table` | 禁用模型 n-gram 表的原生实验选项；保留专属入口 |

### KVMem

| 原生参数 | 输入和原生行为 |
|---|---|
| `--kvmem` / `--no-kvmem` | 默认关闭；显式开启或正 B 启用；显式关闭与预算冲突时报错 |
| `--kvmem-budget` | B，启用后默认 32768；必须 >=128 且为 64 的整数倍；显式 0 拒绝 |
| `--kvmem-gen-reserve` | R，启用后默认 1024；必须 >=64 且为 64 的整数倍 |
| `--kvmem-host-mib` | H，MiB，启用后默认 16384；必须为正；为原生 KV payload 硬预算 |
| `--kvmem-sink-tokens` | 默认 0 表示一页 64 tokens；正值向下取整且至少一页 |
| `--kvmem-recent-tokens` | 默认 64；开启 KVMem 时至少 64；按整页向下取整 |
| `--kvmem-sessions` | 非活动 Host 历史数，1–16，默认 4 |
| `--kvmem-disk-path` | 进程内公共 SSD 层目录 |
| `--kvmem-disk-mib` | 非负整数 MiB；0 关闭且目录须省略，正值必须配目录 |
| `--kvmem-lrfu` / `--no-kvmem-lrfu` | 开启/关闭 LRFU，默认关，后出现的开关覆盖前者 |
| `--kvmem-lrfu-lambda` | [0,1]，默认 0.5 |
| `--kvmem-lrfu-top-frac` | (0,1]，默认 0.9 |
| `--kvmem-score-agg` | mean/max，默认 mean |
| `--kvmem-verify-transfers` | 校验恢复后的 KV/GDN 字节，默认关 |

### KV 精度与执行内核

| 原生参数 | 输入和原生行为 |
|---|---|
| `--kv-capacity` | tokens 或 auto；普通模式默认 max-context，hybrid/strict 默认 auto；KVMem 派生为 (B+R)×并发 |
| `--kv-headroom-mib` / `--vram-headroom-mib` | auto KV 的空闲显存预留，默认 1024 MiB；仅 default 显存策略 |
| `--kv-dtype` | bf16/int8/fp8/nvfp4/k8v4/rk8v4/rk4v4/rk4v4-e8/rk2v4-e8；默认 bf16 |
| `--no-cuda-graph` | 关闭默认启用的 CUDA Graph |
| `--cuda-graph-allowance-mib` | Graph 驱动状态预算；默认 0 按 profile 计算 |
| `--fast-prefill-kernel` | 快速预填充；默认关；有效格式和 chunk 调整随执行组合变化 |
| `--prefill-cublas` | 宽预填充 GEMM 使用 cuBLAS，默认关 |
| `--no-prefill-cublas-projections` | 启用 cuBLAS 时让 attention/GDN 输入投影保持其他路由 |
| `--no-prefill-a8` | 关闭默认的整数激活预填充路由，使用 A16 |
| `--mlp-a8-decode` | decode/verify 的 MLP 使用整数激活路由，默认关 |
| `--lm-head-q4` / `--lm-head-q6` | 加载时将输出头按所选格式存储；原生格式以模型为准，选项互斥 |
| `--embedding-q4` / `--embedding-q6` | 加载时按所选格式存储 embedding；选项互斥 |
| `--mtp-experts-q4` | 对适用模型的 MTP 专家使用文本层格式；模型专属 |
| `--gdn-state-fp16` | GDN recurrent state 使用 FP16，默认关 |
| `--rope-yarn` | 超出模型原生窗口时按上下文比例使用 YaRN |
| `--rope-yarn-factor` | 固定 YaRN 因子 1–4；默认 1 由 rope-yarn 决定 |
| `--rope-scaling-factor` | 线性插值因子 1–32；默认 1 关闭；不是 YaRN 因子的别名 |
| `--rope-scaling-original-context` | 线性插值阈值，默认模型原生窗口 |
| `--cuda-memory-policy` | default/mixed/strict/strict-RESERVE-STEP；默认 default；当前 KVMem 只允许 default |
| `--wddm-evictable-budget` | 新基线的 Windows 原生容量预算：普通 CUDA 分配并在启动时提交；不再依赖旧 D3D12 arena，不可与 mixed/strict 混用；KVMem 仍使用显式 B+G 容量 |
| `--gpu-clock-floor` | 原生请求运行期间的 GPU 最低 graphics clock（MHz）；默认关闭，驱动权限不足时给出诊断；不是 KV 预算参数 |

### 原生上下文缓存

下面两套原生缓存目录/池由 NInfer 管理。KVMem 开启时管理自己的历史，不把这些预算映射到 kvmem-host-mib 或 kvmem-disk-mib。`--no-prefix-reuse` 和 `--derive-session-keys` 对 KVMem 历史仍有作用，单独保留。

| 原生参数 | 输入和原生行为 |
|---|---|
| `--no-prefix-reuse` | 关闭兼容前缀复用，默认允许；KVMem 下也控制请求历史复用 |
| `--derive-session-keys` | 为没有 session key 的请求推导身份，默认关；KVMem 可使用该身份 |
| `--use-original-prefix-caching` | 显式选择 checkpoint catalog；KVMem 下拒绝显式选择 |
| `--use-alt-prefix-caching` | 选择 hybrid 原生前缀缓存；KVMem 下拒绝 |
| `--device-state-slots` | checkpoint 的额外设备槽位，默认并发数；catalog 专用 |
| `--host-state-slots` | Host checkpoint 槽位，默认 8；catalog 专用 |
| `--host-kv-mib` | Host 原生缓存 KV，默认 8192 MiB；catalog 专用 |
| `--host-cache-mib` | 原生缓存 Host 总预算，默认 8192 MiB；auto 按加载权重后的可用内存计算；0 仅设备缓存；不等于 KVMem H |
| `--host-cache-reserve-mib` | auto 时为后续增长保留的 Host 内存，默认 3072 MiB |
| `--host-cache-max-mib` | auto Host 预算上限；不是 KVMem H |
| `--host-cache-percent` | auto Host 预算上限占整机内存的百分比，1–100 |
| `--max-private-continuations` | 私有 continuation 目录规模，默认 2×并发；catalog 专用 |
| `--max-shared-prefixes` | 共享前缀规模，默认 max(并发,7)；catalog 专用 |
| `--max-long-anchors-per-continuation` | 每 continuation 的长锚点上限，默认 2，自动锚点模式为 4 |
| `--auto-long-anchors` | 自动保留消息边界锚点，默认关 |
| `--branch-anchors` | 在历史分叉处保留锚点，默认关 |
| `--no-endpoint-anchors` | 关闭默认保留的 endpoint 锚点 |
| `--long-anchor-spacing` | tokens，默认 1024；0 表示每个边界；依赖自动长锚点 |
| `--max-cache-markers-per-request` | 每请求缓存标记上限，默认 4 |
| `--auto-prefix-grid` | 自动提出共享前缀网格候选 |
| `--context-cache-policy` | default/rolling；catalog 保留策略 |
| `--release-diverged-checkpoints` | 优先释放失去复用价值的分叉 checkpoint |
| `--thorough-admission-search` | 更充分搜索请求准入复用方案；与 HTTP 等待超时不同 |
| `--recency-eviction` | 压力下按最近使用情况淘汰 |
| `--value-aware-demote` | 按重建成本选择淘汰/下沉 |
| `--kv-lease-growth` | 逐段扩展输出 KV 租约；原生缓存设置，不作为 KVMem 追加空间选项 |
| `--concurrent-prefill` | 预填充时向空闲 lane 准入其他请求；当前 KVMem 拒绝 |
| `--disk-kv-path` | 原生持久磁盘缓存目录；与 KVMem 进程内 SSD 层不同 |
| `--disk-kv-gib` | 原生磁盘缓存预算，默认 64 GiB；依赖 disk-kv-path |
| `--disk-kv-restore` | 从原生磁盘缓存恢复匹配前缀 |
| `--disk-kv-directstorage` | 使用 DirectStorage；依赖相应 Windows 构建 |
| `--slot-save-path` | slot 手动保存/恢复/删除的目录 |
| `--auto-save-evicted` | 淘汰前写回 slot 文件；依赖 slot-save-path |
| `--device-snapshot-slots` | hybrid 设备快照槽；默认并发+1，无 Host 层时并发+2 |
| `--cache-taps-per-request` | hybrid 每请求新增快照数，默认 8，无 Host 层时 2 |
| `--cache-tap-ladder` | hybrid 历史快照间隔基数，默认 max(4096,2×prefill chunk) tokens |
| `--cache-tap-min-gap` | hybrid 快照最小间距，默认 max(1024,prefill chunk) tokens |
| `--prefix-cache-file` | hybrid Host 层持久文件，默认关；需相同 artifact/KV 格式及对应二进制兼容性 |

### 推测解码

| 原生参数 | 输入和原生行为 |
|---|---|
| `--spec` | mtp/dflash/dflash2；省略关闭；KVMem 允许 mtp/dflash2，DFlash2 当前最多八路单卡 |
| `--draft-tokens` | 启用推测解码时 1–15；省略为 0，不能只写 spec 而漏掉有效草稿数量 |
| `--lm-head-draft` | 使用优化的 proposal head；有该组件时默认自动采用；依赖推测解码 |
| `--full-draft-head` | 对草稿使用完整输出头；与优化 proposal head 选择相对，保留原生名称 |
| `--draft-min-p` | Flash-Next MTP 置信度阈值，0–1，默认 0；不作为 Qwen Dense KVMem 参数 |
| `--adaptive-mtp` | 自适应 MTP 宽度，draft-tokens 为上限；默认固定策略 |
| `--mtp-attention-window` | 默认 0 看完整历史；正值至少为 drafts+1；当前 KVMem 不允许非零 |
| `--ngram-draft-tokens` | 0 关闭，开启 1–63；启用 spec 且省略时默认 15；并发>1 时大于 15 自动降为 15 |
| `--ngram-min-match` | 4–64，默认 12 |
| `--ngram-archive-mib` | 跨请求草稿来源归档预算，默认 0 关闭 |
| `--ngram-session-mib` | 每 session 归档预算，默认 128 MiB；启用归档时须在 [1,total archive] MiB |
| `--ngram-native-sessions` | 识别受支持客户端原生 session 身份；依赖启用 archive |
| `--lookup-ngram` | 旧 context lookup 草稿，默认 0；KVMem 禁止非零，应使用 ngram-draft-tokens |

### 视觉与媒体

| 原生参数 | 输入和原生行为 |
|---|---|
| `--vision` | 启用模型的视觉资源，默认关；原生视频能力不代表 KVMem 视频可用 |
| `--vision-residency` | resident/overlay/cpu，默认 resident；KVMem 仅 resident/cpu |
| `--vision-cpu` | 同时启用 vision 并选择 CPU |
| `--cpu-vision-idle-seconds` | Qwen CPU视觉权重按需加载；最后一个编码器释放后空闲超时卸载，默认300秒，0禁用自动卸载 |
| `--vision-offload` | 必须带 on/off；on 选择 overlay，off 选择 resident，本身不启用 vision；KVMem 下不允许 on |
| `--vision-max-merged` | 每媒体项 merged tokens，64–16384；默认 16384，CPU 且未显式指定时为 256 |
| `--media-cache-mib` | 保留的预处理媒体，默认 1024 MiB；0 关闭 |
| `--media-live-mib` | 正整数 MiB；活跃预处理媒体总预算，默认 2048 |
| `--media-preprocess-threads` | 0–64；默认 0 自动，自动值最多 16；不是 CPU 生成线程 |

### 采样 思考与协议

| 原生参数 | 输入和原生行为 |
|---|---|
| `--temperature` | [0,2] |
| `--top-p` | [0,1] |
| `--top-k` | 整数 0–20，0 关闭 |
| `--min-p` | [0,1] |
| `--presence-penalty` / `--frequency-penalty` | 各为 [-2,2]；未显式指定的采样值来自模型和思考模式 |
| `--seed` | uint64；省略时每请求新 seed |
| `--greedy` | 设置温度 0 |
| `--post-thinking` | 思考结束后启用答案采样预设，默认关；该预设温度 0.2 |
| `--post-thinking-temperature` | [0,2]，隐含开启 post-thinking |
| `--post-thinking-top-p` | [0,1]，隐含开启 post-thinking |
| `--post-thinking-top-k` | 0–20，隐含开启 post-thinking |
| `--post-thinking-sampler` | temp=F,top_p=F,top_k=N[,min_p=F,presence=F,frequency=F] |
| `--no-thinking` | 默认关闭思考；省略交由模型/请求规则 |
| `--preserve-thinking` | 在后续提示中保留已闭合轮次的 assistant reasoning |
| `--default-thinking-budget` | 正 uint32；省略不显式设置；不接受 0/-1 |
| `--thinking-budget-message` | 非空文本；达到预算时使用，必要时附加标准思考结束标记 |
| `--default-reasoning-effort` | none/minimal/low/medium/high/xhigh/max；模型还可进一步约束 |
| `--structured-output` | 开启 JSON/JSON Schema 响应约束；工具默认采用 `tool_constraints:auto`，显式 basic 或强制/strict 等工具约束独立于这个开关；none 默认不计算工具语法、不发布结构化调用；词表里恰好有一个文本为 `<tool_call>` 的 token 时，采样只禁掉这一个 token，其它拼法仍可能输出工具标签文本 |
| `--unconstrained-response-format` | 未开结构化输出时允许无约束生成对应格式请求；与 structured-output 互斥 |
| `--assistant-prefill` | 继续末尾 assistant 内容；要求关闭思考 |
| `--lenient-assistant-history` | 兼容 function_call 后的 assistant text/reasoning 历史布局 |
| `--lenient-image-detail` | 将未知图像 detail 按 auto 处理；默认拒绝未知值；既有 original→auto 兼容值仍直接支持，low/high 按上游新语义传递；不改变是否支持图像 |
| `--usage-chunk-choice` | 为流式 usage chunk 加空增量 choice，兼容特定客户端 |

### 网络 日志与响应存储

| 原生参数 | 输入和原生行为 |
|---|---|
| `--host` | 默认 127.0.0.1 |
| `--port` | 1–65535，默认 8080 |
| `--stats-port` | 默认 0 关闭；非零须为独立的合法端口 |
| `--api-key` | 单个 bearer/x-api-key 值；默认无认证 |
| `--cors` | 发送宽松 CORS 响应头，默认关 |
| `--no-webui` | 禁用构建时编入的页面 |
| `--webui-mcp-proxy` | 启用页面 MCP HTTP 代理；原生专属选项 |
| `--max-request-mib` | 正整数 MiB，默认 384；解析前的请求体限制 |
| `--max-pending-requests` | 正整数，默认 16 |
| `--pending-timeout-ms` | 正整数毫秒，默认 600000；准备及准入等待 |
| `--recover-invariant-failures` | 内部不变量失败时终止活动请求并继续服务队列的原生选项 |
| `--response-store-max-records` | Responses 保留数量，默认 1024 |
| `--response-store-max-mib` | Responses 状态预算，默认 256 MiB；不等于模型 KV 历史预算 |
| `--request-log-jsonl` | 请求和服务 JSONL 日志文件，省略关闭 |
| `--request-log-max-mib` | 日志轮转阈值，默认 0 不轮转；非零依赖日志路径 |
| `--request-log-keep` | 轮转文件保留数，默认 4 |
| `--log-stats-interval-ms` | 默认 5000 毫秒，0 关闭周期统计 |
| `--log-level` | trace/debug/info/warning/error/critical/off，默认 info |
| `--log-colours` | on/off；默认按终端情况决定 |
| `--log-stats-panel` | on/off，默认 off；交互终端底部统计面板 |

### 模型路由 休眠与 graft

这些原生能力保留在 NInfer 专属设置中；当前 KVMem 不能与模型路由、模型休眠、prompt grafts 组合启用。

| 原生参数 | 输入和原生行为 |
|---|---|
| `--models-dir` | 路由模型目录；路由模式不提供模型位置参数 |
| `--models-preset` | 路由 INI 配置文件 |
| `--models-max` | 同时加载的模型数，默认 1，0 不限 |
| `--models-autoload` / `--no-models-autoload` | 开/关按请求自动加载，默认开；正向开关解析器支持但帮助未单列 |
| `--sleep-idle-seconds` | 空闲休眠秒数，默认 0；普通模式需要 model-suspend |
| `--unload-idle-seconds` | 空闲卸载秒数，默认 0 |
| `--model-suspend` | 启用模型暂停/恢复；依赖 CUDA 虚拟内存能力 |
| `--suspend-snapshot` | pageable/pinned，默认 pageable |
| `--suspend-weights` | artifact/host，默认 artifact |
| `--no-auto-resume` | 休眠期间请求不自动恢复模型，返回失败 |
| `--graft` | NAME=PATH，可重复；加载 prompt graft |
| `--default-graft` | 指定已加载 graft 为默认 |

## 五 能力与冲突规则表

### 当前实现约束与待统一项

Windows 启动器先按原生参数顺序解析 `WorkerArgs` 中的 Context、Prefill（llama.cpp 的逻辑 batch）及 llama.cpp 块大小覆盖，再做对齐和容量检查。支持短/长别名及 `--flag=value`，重复项以最后一次指定为准；这些参数只输出一份最终值，`token_settings` 与启动命令一致。llama.cpp 的物理 batch 仍独立设置；ninfer 原生 `--prefill-chunk 0` 先按 KVMem 默认值 1024 解析，再受 R 和 128 对齐限制。

解析时保留原生选项和值的边界：模型名称、模板或其他选项的值即使形如 `--max-context=524288`，也作为原值传递，不参与 token 参数覆盖。无值开关按锁定后端的 CLI 规则识别，后端新增此类开关时需同步更新启动器。

| 条件 | 当前规则 | UI 与命令生成要求 |
|---|---|---|
| NInfer KVMem 开启 | 归一化后 B>=128、R>=64，均为 64 的倍数；B+R>=256 且不超过 max-context；H>0；prefill 为 128 的倍数且 <=R | 先自动取整并显示生效值，再校验最低容量及组合约束 |
| NInfer KV 总容量 | KVMem 按 (B+R)×并发生成；显式 kv-capacity 必须与之相等，不能为 auto | 展示推导值；不再提供一个相互独立的自动容量开关 |
| sink/recent | 占用 B；NInfer 按 64 向下取整，sink 至少一页，recent 至少一页，二者不能超过 B | 同时展示输入值和有效值；媒体等其他必选区域还受运行时准入校验 |
| 默认输出 | llama 接受 -1 或正数；NInfer 只接受正数，省略为动态预算；两边显式 0 均不可作为公共关闭值 | 公共 UI 使用“后端默认”与“固定上限” |
| R 与输出 | llama 当前开启 KVMem 时，R>0 会限制单次输出；NInfer 可在输出中复用 B+R 追加空间 | R 的公共定义只表示追加工作空间；当前限制如实展示，行为对齐须单独完成正确性设计和回归 |
| 会话数 | 两边 sessions 均只计非活动历史；llama 的旧 conversations 仅显式兼容 | llama 在请求退出时执行配额，活动与切换中的状态不得淘汰；排队中的缓存预览可失效并重算 |
| 公共 SSD 层 | 目录与预算配对；llama 还要求 H>0 且 KVMem 开启；两边不代表跨重启持久恢复 | SSD 开关关闭时省略目录和预算；不与原生持久缓存合并 |
| llama 公共 SSD | 需 Flash Attention；不可与独立旧 CPU arena 混用；旧磁盘实现及其构建开关已移除 | UI、模板和 CLI 统一使用公共 H/D；旧磁盘参数明确报错，不把原每会话预算当成全局预算 |
| llama 并发 | 1–8；多 lane 受 GPU、切分、旧磁盘路径等限制；显式 HTTP 线程至少为 2×并发 | 按实际组合校验，不只检查数字范围 |
| llama MTP | replay 草稿 1–5；tensor+KVMem MTP 只允许 snapshots | 根据所选状态策略和切分限制草稿及状态选项 |
| llama DFlash2 | 独立匹配 GGUF、最多八路、单 GPU；公共入口固定 1–5，无 adaptive/ngram | 使用上游原生 draft-dflash；各路共享只读草稿权重，上下文/KV/回滚独立。5060 Ti 的 27B IQ3＋Q4 四路验证使用 CPU 卸载与 snapshots，全 GPU 显存不足，详见并发验收记录 |
| NInfer MTP | 需要 spec=mtp 和 drafts 1–15；adaptive 依赖 MTP；KVMem 拒绝 DFlash 和非零 mtp-attention-window | 分别显示固定数量与自适应上限 |
| NInfer DFlash2 | 内置草稿组件、固定 1–15、最多八路单卡；可使用原生 ngram，不支持 adaptive | 按模式启用控件，不把 MTP 自适应策略应用到 DFlash2 |
| NInfer ngram | spec 开启且省略宽度时默认 15；显式 0 关闭；并发>1 时 >15 自动降至15；KVMem 还要求草稿验证空间不超过 R | 预览请求值与有效值；不能把省略等同关闭 |
| NInfer 快速预填充 | 当前 KVMem 仅允许 int8/rk8v4/rk4v4/rk4v4-e8/rk2v4-e8；具体有效 chunk 可能被调整 | 格式切换时重新校验，展示调整结果 |
| llama 多卡 | 显式选择至少两张同一 CUDA 后端的设备；不允许重复设备；需要 gpu-layers=all 和 layer/tensor；比例列表逐卡对应；row 被服务拒绝 | 切分方式、GPU 层数和设备列表一起校验；不因解析器识别 row 就展示为可用 |
| GPU 顺序 | NInfer devices 与 device 互斥；stage-layers 依赖多卡；设备要求匹配 compute capability | UUID 持久保存，启动时解析可见索引；不假定索引长期稳定 |
| NInfer KVMem 模式 | 不支持模型路由、休眠、graft、Flash Next 家族、overlay 视觉、并行 prefill、非 default CUDA memory policy | 参数仍保留在原生清单；当前配置禁用并说明原因 |
| NInfer 原生缓存 | catalog/hybrid 与 KVMem 历史管理不是同一套；KVMem 明确拒绝多项模式和专属参数，部分其他缓存字段被解析后原生缓存路径关闭 | UI 禁用不适用的原生缓存配置；不要把“参数被接受”标为“对 KVMem 生效”；no-prefix-reuse、derive-session-keys 单独保留 |
| NInfer no-prefix-reuse | 与 derive-session-keys 和相应原生缓存容量/目录设置冲突 | 关闭复用时检查依赖项，不悄悄保留矛盾设置 |
| NInfer hybrid | 与 catalog 专属参数、显式 original 模式和多卡 pipeline 互斥；prefix-cache-file 需要 Host 层 | 作为原生模式独立配置，不混入 KVMem SSD |
| 原生 Host 缓存预算 | host-cache-mib 与 host-state-slots/host-kv-mib 互斥 | 不把多个预算同时传入 |
| KV 格式 | llama 与 NInfer 枚举不同；q8_0 与 int8 不等价；后端间 KV 不互通 | 保存后端限定的原值；切换后端要求选择可用格式 |
| 思考预算 | llama -1/0/正值；NInfer 仅正值；默认思考模式也不同 | 保留原义，禁止把 0 自动翻译为 no-thinking |
| 聊天模板 | llama chat-template 是正文，NInfer 是文件；llama 正文和文件入口二选一 | 导入命令时先确定后端，再解释参数 |
| 采样 | top-k、seed 的范围不同；请求可覆盖进程默认 | 范围按后端，UI 明确这些是默认请求设置 |
| 结构化输出 | NInfer structured-output 与 unconstrained-response-format 互斥；assistant-prefill 需要关闭思考 | 各开关独立描述并校验依赖 |
| 视觉 | NInfer KVMem 仅 resident/CPU 图片；CPU 未指定 token 上限时默认 256；llama 需要 projector | 不把 0.18 模板的视觉 token 值写为 EXE 默认 |
| 服务与日志 | stats-port 不能等于主端口；request-log-max-mib 非零需要日志路径 | 生成命令前校验；HTTP 超时和排队超时分别保存 |

### 统一入口新增的配置规则建议

| 事项 | 规则 |
|---|---|
| 原生命令兼容 | llama.cpp/NInfer 原生参数原名与含义不变；KVMem 扩展可统一，旧名进入有版本的迁移表；原生映射别名不宣称已由 EXE 支持 |
| 设置优先级 | 用户显式值覆盖模板，模板覆盖 KVMem 公共默认策略；后端原生参数未设置时交给后端；新 KVMem 接口不再读取两套不同默认来源 |
| 三态开关 | 继承/开启/关闭分别保存；KVMem 自有布尔项在解析后成为单一明确值并支持必要的正反开关；原生开关按各自语义生成 |
| 参数来源 | 记录当前值来自模板、用户、导入命令或后端默认，预览同时显示来源与有效行为 |
| 环境变量 | llama 支持的 LLAMA_ARG_* 先于 CLI 生效，API keys 累加；统一入口识别相关环境影响，不承诺“省略=纯编译默认” |
| 同一语义重复设置 | 导入时按当前 EXE 的原生顺序解析，保留原命令；标准化编辑时提示覆盖情况；公共项与高级项冲突必须解决后生成规范命令 |
| 组合设置 | cache-type-k/v 与 kv-dtype、模板变量与思考开关等可能共同决定结果，要按有效设置归并，不能仅按参数字符串去重 |
| 可重复参数 | API keys、graft 等遵守原生累加规则；不同于普通标量覆盖 |
| 原生高级参数 | 用参数列表保留原名、值和顺序；已识别的公共项回填公共控件；未知参数保留为待核对项，不静默丢弃 |
| 参数执行 | 生成参数数组交给进程，单独生成适合 cmd/PowerShell 的复制文本；路径、JSON 和模板文本分别正确转义 |
| 后端切换 | KVMem 公共值保持一份并按能力重新校验；保留各后端原生高级设置；格式、特殊值、GPU 能力等重新校验 |
| 配置保存 | 保存规范版本、目标后端、已知后端版本、单份 KVMem 配置、用户显式值、模板来源和各后端原生高级项；自动派生值不伪装为用户输入 |
| 校验结果 | 分清错误、将被原生调整的值、以及当前环境未验证的能力；不可用时给出具体条件 |
| 预览 | 同时显示最终 EXE、实际参数、B/R/H/D、设备顺序、输出有效上限和必要的原生默认；API key 脱敏 |

## 六 默认值与页面组织

### 当前 EXE 默认与模板预设分开

下表记录本地 CLI v1 的省略行为。KVMem 预算仅在启用时填入；B+R 不适合原生默认上下文时明确报错，不能偷偷修改原生 ctx-size/max-context 默认。

| 设置 | llama.cpp EXE 省略行为 | NInfer EXE 省略行为 |
|---|---|---|
| 主端口 | 8080 | 8080 |
| 上下文 | 2048 | 8192 |
| 并发 | 1 | 1 |
| KVMem | 默认关；正 B 兼容启用 | 相同 |
| B/R/H | 启用后默认 32768 / 1024 / 16384 MiB | 相同 |
| sessions | 4，非活动历史 | 相同；能力范围 1–16 |
| KV 格式 | q8_0 | bf16 |
| sink/recent 输入默认 | 0 / 64；按本后端块粒度规范化 | 相同；64-token 页 |
| LRFU/聚合 | 关 / mean | 关 / mean |
| 输出预算 | n-predict=-1，仍受 generation limit | 动态上下文和并发准入预算 |
| 推测解码 | 关闭 | 关闭；启用 spec 后默认 ngram=15，仍须指定有效 drafts |
| 思考 | 普通聊天默认关闭 | 模型/请求决定，当前普通 Qwen 聊天默认思考 |
| 视觉 | 无 projector 时关闭 | 默认关闭 |

0.18 启动配方是一个明确的模板，继续独立保存。后端原生设置可选择“后端默认”；KVMem 自有设置选择“公共默认”或模板/用户显式值。两者都不把 0.18 配方的数值伪装为 EXE 编译默认。

### UI 分组建议

| 位置 | 参数范围 |
|---|---|
| 顶部常用区 | 后端、模型、GPU 选择和顺序、主端口；模板放在旁边 |
| 上下文与输出 | 逻辑上下文、默认输出预算、并发、预填充、位置缩放 |
| KV 与存储 | KVMem、B/R、sink/recent、LRFU、格式、Host/SSD、历史数量 |
| 性能与并行 | 切分、分配、CPU 线程、模型加载、MTP、ngram、当前后端的性能开关 |
| 采样与思考 | 采样参数、思考开关与预算、思考后采样、模板 |
| 图像与视频 | 视觉资源、运行位置、图像 token、适用的视频和媒体配置 |
| 服务与日志 | 地址、认证、页面、队列、超时、日志、统计、Responses 存储和协议兼容 |
| 引擎高级设置 | 其他原生专属参数、原生缓存、路由/休眠；可搜索原生命令行名称；旧 KVMem 存储另列迁移说明，不与新公共设置并列 |

“公共/专属”是语义属性，“常用/高级”是展示属性。可以把后端专属的常用参数放在对应分组，也可以把两边都支持的少用参数折叠起来。

### 已接入控制台的常用原生设置

UI 只提供常用设置，不复制完整原生命令行。以下名称属于 UI 配置和 PowerShell 启动器，EXE 的原生名称保持不变：

| 配置字段 | UI 位置 | llama.cpp 输出 | NInfer 输出 |
|---|---|---|---|
| `ApiKey` | 服务与并发 | `--api-key KEY` | 同名，公共入口限单个 key |
| `ModelAlias` | 服务与并发 → 高级 | `--alias NAME` | `--model-id NAME` |
| `LogLevel` | 服务与并发 → 高级 | `--verbosity`：off=0、error=1、warning=2、info=3、trace=4、debug=5 | `--log-level` 对应文字值 |
| `ChatTemplateFile` | 模板与图像 → 高级 | `--chat-template-file PATH` | `--chat-template PATH`（此处原生含义就是文件） |
| `ThinkingMode` | 生成与思考 | default 不传覆盖，on=`--enable-thinking`，off=`--no-think` | default 不传覆盖，off=`--no-thinking`；缺少原生启动 on 选项，UI 禁用 on，不模拟成其他参数 |
| `Concurrency` | 服务与并发 | `--parallel N`，1–8；当前控制台单卡，允许公共 SSD，H 需容纳所有并发工作集 | `--max-concurrency N`，1–8 |

模型名、日志和模板文件留空时不产生覆盖参数；思考模式继承各自原生默认，不能把 llama 的默认关闭和 NInfer 的模型默认描述为相同的运行状态。思考预算仍仅对 NInfer 开放，UI 关闭思考时不发送预算。温度、top-p 等继续交给聊天请求。

密钥支持随机生成与显示/隐藏，只在页面内存和正在运行的控制器/worker 中保留。浏览器草稿、个人模板和控制器保存的草稿都排除密钥；刷新后重启服务需要重新输入。预览/复制命令使用 `<redacted>`，手动执行前要替换占位符。控制器不把密钥写入启动计划临时 JSON；状态只报告是否配置，日志接口屏蔽已用密钥。当前控制器对自己管理的服务自动携带密钥读取健康状态和固定监控路由；外部聊天客户端仍须提供密钥。

启动器的同一设置不能同时从命名字段和 `WorkerArgs` 覆盖。未使用相应命名字段时，直接脚本仍可保留原有原生高级参数；UI 的原生参数数组继续使用小范围白名单，不开放任意参数传递。

NInfer 启动器和 UI 另提供 `ReasoningEffort` 与 `PreserveThinking`，分别映射到
`--default-reasoning-effort` 与 `--preserve-thinking`；当前两个控件只对 NInfer 开放。
思考模式为 off 时不得设置思考强度；保留历史思考独立控制。`ThinkingBudget` 保持原有映射。
UI 继续通过 JSON 参数数组调用内部脚本。直接脚本增加 `WorkerArgsFile`，读取 UTF-8 JSON
字符串数组，和 `WorkerArgs` 互斥，读取后走同一套冲突校验及原生参数编码。
外部 `-File` 调用可将多个原生参数放在最后的 `-WorkerArgs` 后；复杂值使用参数文件。

### 已实现的布局、帮助与条件联动

顶部资源区展示逻辑上下文、GPU 历史预算 B 和生成预留 R；RAM 历史预算 H 放在“缓存与历史”。参数说明通过信息按钮悬浮、键盘聚焦或点击显示，Escape 或点击外部关闭，不占用表单行高。禁用控件的具体原因也通过该提示展示。

| 条件 | 受影响的控件与导出规则 |
|---|---|
| SSD 关闭 | 目录与预算变灰，预览和启动均省略两个实参；`DiskEnabled` 只属于浏览器草稿/模板，不传给启动器或 EXE |
| LRFU 关闭 | 衰减系数与候选比例变灰；分数组合方式仍可设置；导出明确的关闭开关 |
| 图像输入关闭 | 图像 token 上限、编码器存放位置变灰并省略 |
| 默认思考模式关闭 | 默认思考预算变灰并省略；不改写后端原生开关 |
| 草稿方式关闭 | 长度、策略、草稿模型、N-gram 及匹配设置变灰并省略；保留浏览器草稿以便恢复 |
| 草稿方式不支持 adaptive/ngram | 对应选项变灰并省略；DFlash2 只使用固定长度 |
| N-gram 推测为 0 | 最短匹配设置变灰并省略 |
| KV 格式不支持快速预填充 | 快速预填充变灰并省略，预填充分块仍独立设置 |
| 少于两张显卡 | NInfer 层数分配不可编辑；显卡数量变化时清除已失效的层数分配 |
| llama.cpp 并发大于 1 | 公共 SSD 开关及目录/预算继续可用；配置 1–8 路时保留 SSD 实参，并检查实际 H/D 容量 |

由于依赖条件暂时禁用的值仍保存在草稿和个人模板中，刷新、套用模板和后端来回切换后可恢复。后端本身不支持的控件不会因其他开关变化而被启用。旧草稿缺少 `DiskEnabled` 时，按已有非空目录与非零预算推断开关状态。

## 七 来源与帮助差异

| 内容 | 当前来源 |
|---|---|
| llama 主参数与组合 | [llama-kvmem-server.cpp](../tools/llama-kvmem-server.cpp) |
| llama 服务参数与别名 | [kvmem-server-options.h](../tools/kvmem-server-options.h) |
| llama 采样与思考预算 | [kvmem-chat-sampling.h](../tools/kvmem-chat-sampling.h) |
| llama 环境变量 | [kvmem-server-env.h](../tools/kvmem-server-env.h) |
| NInfer 帮助与参数校验 | [serve_options.cpp](../backends/ninfer/src/serve/serve_options.cpp) |
| NInfer 服务默认值 | [serve_options.h](../backends/ninfer/src/serve/serve_options.h) |
| NInfer KVMem 配置 | [types.h](../backends/ninfer/include/ninfer/types.h) |
| NInfer 推测默认与依赖 | [speculative_options.h](../backends/ninfer/src/product/speculative_options.h) |
| 现有后端行为说明 | [multi-backend-differences.md](multi-backend-differences.md)；涉及旧默认值时以当前解析器为准 |

当前帮助/旧说明中需要注意的差异：NInfer 正向 models-autoload 被解析器接受但帮助未单列；devices 帮助强调 2–8 阶段，解析器也接受单项及用于阶段测试的重复设备；vision-offload 需要 on/off 值。此前“默认输出固定 8192、显式 0 不限”的说明已在比较文档中纠正：当前 NInfer 省略输出上限时动态计算，显式值必须为正整数。以上差异是本规范对当前接口的说明，不改变原生执行行为。


## 八 本次实现验证（2026-10-17）

本地 Release 编译已完成，两个 worker 均可执行；保持 8 路编译。以下记录对应本轮源码验证，不改变已发布的 0.18.0 ZIP。

| 验证 | 结果与范围 |
|---|---|
| llama 参数、输出预算、旧会话策略 C++ 测试 | 3 项通过 |
| llama 服务 CLI 兼容检查 | 98 项通过；包含新 KVMem 规则、原生参数及别名 |
| Windows 统一启动器 | 参数数组、字面路径、能力选择、无效组合及新公共参数映射通过 |
| 本地控制台 | 19 项单元测试通过，覆盖密钥不落盘/不回显、两后端原生映射、能力拒绝和模板文件选择 |
| 新增 UI 设置实际启动 | RTX 5050 Laptop + Qwen3.5-0.8B-Q8_0，23 项检查通过：鉴权、模型别名、自定义模板、双并发、健康检测、监控自动鉴权及脱敏；临时服务已停止 |
| UI 条件联动专项 | 30 项浏览器与原生启动预览检查通过；覆盖禁用值保留、有效参数导出、模板恢复和后端能力边界，不启动推理 |
| 仓库内 UI 回归 | `node ui/test-launcher.cjs` 通过，包含新增的 `test-launcher-settings.cjs`：悬浮/键盘帮助、资源位置、依赖条件、有效预览请求、密钥和草稿/模板恢复；该部分使用设备与预览响应 fixture，不启动推理。`node ui/test-monitor.cjs` 也通过 |
| llama 实际会话配额 | Qwen3.5-0.8B-Q8_0；单路 N=2、双路 N=1，共 54 项检查通过；覆盖复用、隔离、淘汰、两个活动请求、排队预览失效、错误及断线收尾 |
| NInfer 参数及缓存策略 | cli_options、context_cache_defaults、kv_capacity、serve_options、unique_strings 共 5 组通过 |
| NInfer 实际启动及请求 | Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp；通过公共 CLI 启用 KVMem，B=384、R=128、H=512 MiB、sessions=1、SSD=0；19 项检查通过；普通聊天与流式返回正常，流式完成 192 个输出 tokens，活动/非活动计数分开报告 |
| 源码补丁 | NInfer 改动已记录到版本化补丁及版本清单；精确源码准备检查通过 |

llama 功能测试使用 RTX 5050 Laptop（`GPU-A`）；NInfer 27B 功能测试使用空闲 RTX 5060 Ti（`GPU-B`）。本轮为功能验证，不形成性能比较结论。测试仅启动和关闭自身的临时服务。

可复现入口：[llama 会话配额测试](../scripts/test_server_session_quota.py)、[NInfer CLI 实际请求测试](../scripts/test_ninfer_kvmem_cli.py)、[启动器测试](../scripts/windows/test-backend-launcher.ps1)、[启动配置浏览器测试](../ui/test-launcher.cjs)及[设置专项测试](../ui/test-launcher-settings.cjs)。本次实际模型验证使用纯文本和单卡，没有重新验收视觉、MTP 或多卡推理；llama 的 R 输出能力限制继续保留。

## 九 推理统计与后端兼容

各 worker 对齐仓库锁定版本的原生后端。HTTP 适配器读取对应后端的原生字段；
KVMem 的扩展放在 `kvmem.speculative` 中。已有 NInfer 的 `llamacpp:` 和 `ninfer:`
监控指标保持原样，不为了跨后端统一展示而改名。

- llama worker 增加原生 `--metrics` 开关，默认关闭，`/props.endpoint_metrics` 如实报告。
  `/metrics` 使用 Prometheus 格式，提供 prompt、cache、generation、排队/活动请求、
  普通 decode 调用、草稿数、接受数、验证轮数及按草稿位置的接受计数。
  累计数据在请求收尾时结算，保留已观察到的中断请求工作；速率桶每次抓取后重置。
- llama `timings` 复用锁定上游的序列化和计算：首个输出 token 归入 prompt 阶段，
  `predicted_n` 仍为全部输出数，生成速率和每 token 耗时使用 `predicted_n - 1`。
  计时截至最后一个采样，后续传输、清理和末尾状态提交不延长原生生成区间。
- llama 流式默认只在末尾携带 `timings`；`timings_per_token: true` 才逐块附带统计。
  `stream_options.include_usage` 控制最终 usage 块；关闭时，timings 在 finish 块。
  Responses 的最终完成/截断事件同样携带 timings。
- NInfer 保持原生 first-to-last committed token 的计时与首 token 约定。
  KVMem 扩展直接使用 Engine 的 decode round、verification 和 fallback 计数。

扩展中的接受率只统计草稿，不含额外的 bonus token。llama 的 `generation_rounds`
覆盖整个生成循环中的验证尝试、恢复重放和零草稿回退；`wall_ms` 包含该循环的工作，
`timing_scope` 为 `generation_loop`。NInfer 的该字段来自 Engine decode round，
`output_tokens` 和 `wall_ms` 不含首 token，范围为 `first_to_last_token`。
两个范围应分别解释，不能据此直接做跨后端的严格性能比较。

`verification_steps` 是原生接受流程完成的推测轮数；`restore_rounds` 单列 llama
恢复尝试，`uncommitted_rounds` 单列在提交前中止的尝试。每轮耗时与产出必须使用
同一范围的总时间、输出数和实际轮数。缺少轮数时，派生指标为空并注明原因。
`tok/s` 继续表示输出速率，接受率用于解释推测解码效率，不能当作生成质量分数。

核心验证入口：`kvmem-server-metrics-test`、`kvmem-server-options-test`、NInfer 原生
Chat/Responses schema tests，以及 `scripts/test_native_metrics_http.py`。GPU 验证必须
传入明确 UUID 和适合该卡容量的模型路径；四层执行夹具只验证接口与执行，不形成生成质量或性能结论。

### 2026-10-19 首轮核心验证范围

两项 C++ 统计/参数测试和 NInfer 原生 Chat/Responses schema tests 通过。
GPU 验证全部使用 RTX 5050 Laptop（`GPU-A`）：
llama 普通解码使用 Qwen3.5-0.8B-Q8_0；llama MTP 与 NInfer 使用已有四层执行夹具。
覆盖单 token、普通/流式末尾统计、usage 开关、取消后恢复、Responses 终态及监控计数。
NInfer 的 KVMem 监控直接复用其原生计时序列化，以保持首 token 和零耗时约定一致。

NInfer 尝试 0.8B 模型时，其现有引擎拒绝该投影尺寸，因此改用四层夹具。
小型 llama MTP 夹具使用支持的 snapshots 路径；27B 专用 GDN replay 路径未做 GPU 验证。
夹具不用于生成质量判定；没有运行大模型、跨后端性能对比或完整回归套件。
Windows 构建的声明/类型适配问题已修正；补丁打包中的行尾差异经逐文件比对后清理，
两个后端的版本化补丁和精确源码树检查通过。

### 2026-10-19 RTX 5060 Ti 补充验证

经用户授权，使用 RTX 5060 Ti（`GPU-B`）
和实际 27B 模型补测，所有临时 worker 已退出。

- llama 使用 `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`：MTP replay 30 项检查通过，
  snapshots 33 项检查通过，均包含 HTTP canary 报告读取。
- NInfer 使用 `Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp.ninfer`：MTP 33 项检查通过。
  覆盖原生 token/首 token 计时、接受位置求和、流式 usage 开关、断线收尾、后续
  Responses、原生 Prometheus 字段及 HTTP canary。模型包含视觉权重，本轮请求均为文本。
- llama CLI replay 使用固定 seed 42、temperature 0.8，实际输出 32 tokens；报告的
  11 个生成轮、32 个提出草稿和 21 个接受草稿均与逐轮 trace 对齐，接受位置合计同样为 21。
  报告使用实际输出数和生成循环耗时，没有使用该请求中 libllama 的 `n_eval=1`。
- 修正验证脚本中的跨后端字段断言：NInfer 保持原生 `ninfer:draft_tokens_total`
  和 `ninfer:draft_accepted_tokens_total`；llama 继续检查其原生 speculative 系列。
  同时保存抓取的 `metrics.prom`，增加接受位置合计检查。产品代码无需修改。

本轮没有发现补丁阻塞项。replay 路径已实际执行；snapshots 使用原生 GPU GDN 回滚平面，
已观察请求的 host checkpoint `restore_rounds` 均为 0，不能据此声称覆盖了非零 host restore。
这次仍是功能和统计一致性检查，没有运行性能 A/B、生成质量评测或完整回归套件。

### 速度 canary 报告

`scripts/speed_canary.py` 保留 llama CLI 的 KVMem on/off 检查，并增加 HTTP 读取模式。
HTTP 模式只读取已启动 worker 的原生输出，不启动或修改 worker；可选的 baseline URL
由调用者分别配置。`KVMEM_API_KEY` 环境变量用于鉴权，报告不保存密钥。

```sh
python scripts/speed_canary.py --backend llamacpp --url http://127.0.0.1:8080 --runs 3 --seed 42 --output canary.json
python scripts/speed_canary.py --backend ninfer --url http://127.0.0.1:8081 --baseline-url http://127.0.0.1:8082 --runs 3 --seed 42 --output canary-ninfer.json
python scripts/speed_canary.py --cli build/bin/llama-kvmem-cli.exe -m model.gguf --spec-type draft-mtp --spec-draft-n-max 3 --seed 42 --output canary-cli.json
```

HTTP 报告保存原生 usage/timings、worker 的 props、请求设置和逐次结果。CLI 模式支持
`--spec-kv-dtype` 与 `--mtp-state`，记录种子、GPU UUID、源码锁定的后端版本和测试配置。CLI 普通
解码仍读取 `KVMEM_PERF`；MTP 使用新增的单条 `KVMEM_SPEC_METRICS` JSON 汇总，以已输出
的 token 数和完整生成循环耗时计算实际吞吐量，避免把验证调用数当作输出数。
该汇总不依赖逐轮 trace，也不要求增加 trace 级别；显式 `--seed` 同时传入普通与 MTP 采样。
CLI 未显式指定种子时保留原有默认行为；canary 显式使用固定种子，默认 0。

报告同时给出 tok/s、ms/round、tokens/round 和草稿接受率；缺少可靠轮数时显示 n/a。
CLI 原有的可选 tok/s 回归门槛保留；HTTP 对独立配置的 worker 只出报告。
没有把 ms/round 视作与文本或验证宽度无关的唯一门槛。比较前应固定 GPU、模型、上下文、
草稿设置、采样参数及 trace 级别；相同 seed 不保证不同后端或不同 KV 配置生成完全相同的文本。

工具核心测试：`python -m unittest discover -s scripts -p test_native_stats.py -v`。
HTTP 小模型验证可加 `--canary`，在同一临时 worker 上检查报告读取链路。
