# 生成历史的服务端续接

## 目标与客户端协议

客户端照常发送完整消息。服务端在生成结束后保存实际执行的 token IDs，并在续聊时用结构化历史和模板渲染前缀查找可复用的记录。有匹配记录时，即使 KV 已被逐出，也能按本轮选中的 IDs 重建输入。目录位于持有聊天状态的服务进程中，找不到合适记录时使用普通分词。客户端无需增加字段或保存 IDs。

续聊保证延续相同的结构化历史，允许同一文本用不同的 token IDs 表示。例如，模型可以生成 `[64, 65]`（`a`、`b`），也可以用 ID `365` 表示 `ab`。如果旧记录已被逐出，另一条生成记录的结构化历史、渲染前缀与模型实例等约束均匹配，服务端可以沿用那条记录的 IDs。分词差异可能影响下一轮生成。目录仍保有歧义标记时，服务端会跳过该回合的直接匹配。社区在多轮生成和缓存中也记录过分词差异。[verl 多轮 token 连续性讨论](https://github.com/verl-project/verl/issues/6719) · [ds4 多轮 KV 未命中报告](https://github.com/antirez/ds4/issues/1214)

## 两种独立的命中

历史记录存放在模型实例所属服务进程内，与 KV 目录分别管理。记录关联鉴权域、连续的结构化历史、实际送入引擎的前缀 IDs、已接受的输出 IDs，以及它们对应的完整渲染前缀。后端以本轮选中的有效 IDs 校验 KV 前缀、位置和循环状态。原生恢复到第 r 行时，引擎从有效输入的第 r 行继续 prefill。r 可以为零。

## 请求与发布

1. 沿用各协议的标准解析和聊天模板。记录绑定服务端的模型和词表，在已通过鉴权的访问域内，根据完整父消息及回传的 assistant 消息查找候选。后端已有的 owner、会话或适配器约束继续生效。
2. 本轮追加 user 或 tool，并能识别之前完整的 assistant 回合。逐字节比对上轮渲染的 prompt 加原始生成输出。接缝需要一个可识别、由模板产生的独立结束标记。后端如已将该标记作为生成 ID 接受，需要验证其位置并避免重复添加。单独编码新后缀，再检查其 IDs 能逐字节还原后缀。模板证明失败时走普通分词路径。
3. 在选择 lane、匹配 KV 和采样之前构造有效 IDs。Query、消息边界、usage、位置和输出预算使用有效行号。媒体、无法映射的外部 token 坐标以及超过本轮容量限制的候选沿用原路径。
4. 生成结束后，用后端实际接受的输出 IDs 和对外返回的 assistant 构造记录。停止字符串裁剪、解析丢弃了部分输出等不完整情形需阻止误命中。若模板移除了思考、改写了工具调用或改动了前缀空白，下一轮的字节证明会失败。
5. 对相同身份遇到不同的实际 IDs，记录歧义并退回普通路径。限制总保留字节数和记录数，并发查找获取不可变引用。该窗口内的记录独立于原生 KV 的淘汰；不承诺进程重启后的历史恢复。

## 各后端接入契约

llama.cpp、NInfer 和 Radiance 使用同一份 C++ `kvmem::GeneratedHistory` 目录实现。PolyStrata 的目录位于持有模板、tokenizer 和 HTTP 输出的 Python 前端，记录打包后的 32 位 IDs。四个适配器共用同一种续接语义，各服务实例管理自己的 KV 张量、模型状态和恢复能力。切换引擎、模型或词表时需重新建立历史。

| 后端 | 输入处 | 发布处与恢复验证 |
| --- | --- | --- |
| llama.cpp | 模板渲染后、构造 kvmem_prompt 前 | 两种 HTTP 输出路径的最终 assistant、生成 IDs 和成功 commit。保持 lane 与 recurrent checkpoint 验证。 |
| NInfer | 在原生 frontend 编码渲染结果时拼接历史，沿有效 IDs 映射 Query、消息和缓存边界 | 从 outcome 的真实生成 IDs 建立记录，由 Engine 验证设备或主机 checkpoint。Chat、Responses、Anthropic 以及相应的计数入口共用准备逻辑。 |
| PolyStrata | Python Service 在 serve.kvmem.encode 拼接已保存的 IDs，重算 Query 和 canonical_end | Python 目录使用与 C++ 相同的有界保留和歧义规则，从原生 worker 的接受事件记录 IDs。GPU worker 卸载重启时，仍在运行的前端可保留目录。 |
| Radiance | 聊天模板完成后，核对结束标记的 Jinja 来源，并在交给 scheduler 前重建有效文本 IDs 和行号 | 从原生 sink 收到的终态发布，实际 KV 恢复由原有 snapshot、recurrent 与 draft 状态验证决定。 |

## 验证与判定

构造两组不同 ID 序列解码出相同字节的固定用例。命中时，最终输入应逐项等于本轮选中的历史 IDs 加新后缀。分别验证原记录命中、旧记录淘汰后命中同文记录，以及歧义标记保留时的回退。历史被修改时最多沿用改动前可验证的祖先；鉴权域不符或接缝无法验证时跳过该候选。覆盖多个连续回合、取消、并发和实际 token 坐标。HTTP 用例发送普通 messages，同时记录完整有效 IDs、逻辑历史复用数与后端 actual_cached_tokens。用相同有效 IDs 做热续接与冷回放对照。所有模型测试显式指定 prefill 4096。

## llama.cpp 的具体行为

开启 llama.cpp 的 KVMem 后，每个服务进程最多保留 32 条记录，按字符串与 ID 向量的容量合计设 32 MiB 上限。容器开销及正在处理的请求持有的引用另计。进程内的多个 lane 共享目录。查找会检查较早的完整回合，因此最新回复有编辑时，仍有机会从前一个完整回合分支。保留窗口之外的回复使用普通分词。

Chat Completions 与 Responses 共用发布和查找逻辑。结构化 key 将单一 text 片段与相同的字符串统一表示，渲染字节仍需匹配。客户端的 cache_reset 和显式 Query token 坐标跳过本次查找。成功的重新生成也参与冲突识别。只要停止字符串裁剪使输出 IDs 无法还原返回文本，同一结构化回复就标记为有歧义。

在响应的 kvmem.cache_reuse 中，generated_history_tokens 是沿用的历史 ID 数，actual_cached_tokens 是实际恢复的原生 token 数。二者可独立为零。目录按服务端鉴权域隔离，新字段不改变现有原生缓存的跨请求选择策略。

Qwen3.5-0.8B-Q8_0自带模板在普通 user 追问时会移除已完成回复的空思考块，实测此处回退。相同模型在保留历史字节的工具返回路径上可以命中。验证时同时覆盖保留字节的 ChatML 模板和这份原生模板。其他模板由运行时逐次验证接缝。

## 读取续聊结果

llama.cpp 和 NInfer 的 Chat Completions 可以读取 `kvmem.cache_reuse.generated_history_tokens`。PolyStrata 和 Radiance 的同步聊天结果在 `timings.cache_reuse.generated_history_tokens` 提供逻辑复用数。物理恢复数应查看该后端的原生缓存指标，例如 `actual_cached_tokens` 或 `timings.cache_n`。多进程部署时，续聊需要路由到仍保有该记录的服务实例。
