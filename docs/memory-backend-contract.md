# 公共内存契约

本契约在独立 C++17 核心中实现。最先接入的是 llama 的单 GPU、Flash Attention、RAM packed 路径，以及 ninfer 的 KVMem 工作集切换。当前产品组合以 [多后端运行](multi-backend.md) 为准，包括 ninfer 的九种 KV、文本和 resident/CPU 视觉，以及多卡 layer pipeline。公共类型的存在不表示所有后端都支持对应功能。llama 的其他物理配置保留原生兼容桥。

入口是 `kvmem/include/kvmem/memory_contract.hpp` 与 `native_kv_payload.hpp`，实现分别位于 `kvmem/src/host/`。新接口没有引擎、GPU runtime 或 Tensor 类型。

## 公共 RAM/SSD payload 所有权（活跃分层改造中）

`host_kv_storage.hpp` 提供不依赖后端的原生字节记录、容量预留和读租约。
llama 的 packed K/V 与 ninfer 的原生页使用同一实现；mean-K、逻辑位置、
有效 token 数和内容版本仍由原来的统计/内存契约管理，不在物理字节容器中复制一套。

`HostKvStorage` 的容量只计算 payload 与尚未兑现的 payload 预留，不表示整个
进程的内存上限。统计、GPU 传输缓冲区、容器元数据和引擎状态分别计量。
预留先占用容量；分配失败保留预留，预留析构归还未使用额度。记录释放之后，
仍被读租约引用的字节继续占用容量，直到最后一个读者完成。字节实际释放后才归还额度。

适配器在引擎安全边界串行修改同一个记录。同步借用要求所有者在操作期间保持有效；
跨异步传输必须持有读租约直到 DMA 完成，存在读租约时禁止原地写入。
记录只能移动，避免无意复制完整历史。不同引擎所有者可共用一个容量账户；
纯 RAM 路径的账户锁只用于预留、释放和查询，不进入每次记录读写。

显式 `fork()` 只建立已封口记录的只读引用，不复制物理 payload，也不读取冷 SSD。
修改共享尾部先复制到私有记录。`RawKvStore::fork_prefix()` 复制可变统计和目录，
共享原生 packed 字节；部分尾部的精确 MeanK 由所选原生检查点恢复。
`KvContent::fork()` 只接受已评估且已提交的 frontier，为子分支创建独立会话与版本。
分叉时取得的前缀证明不会因父会话编辑而失效；子分支自身截断、重置或失效会撤销它
持有的相关证明，之后重新增长也不会恢复旧证明。普通追加和相同内容的执行重放保留
逻辑前缀证明。物理记录与原生 recurrent/draft 状态仍需要各自的恢复校验。
不同会话可以并发验证和撤销证明：继承证明只查询子会话本地的继承集合，不能读取
父会话可变的撤销标志；自身证明的验证与撤销仍由同一会话的执行锁串行保护。

传入 `SpillFile` 后，同一账户在 RAM 压力下将完整、已封存且没有读租约的记录
写入 SSD。K、V、量化 scale 和 MTP 的原生字节均可分层；mean-K 保留在 RAM。
适配器完成捕获后调用 `seal()`；可写借用会取消封存，尾块修改完成前不能驱逐。
分层记录禁止无租约的 `data()` 借用，读取使用 `read()` 或 `lease()`。
冷记录的读取直接进入调用方缓冲区，不强制整条记录回驻 RAM。

延迟任务持有 `HostKvSource`，锁定内容版本而不锁定 RAM 驻留。真正读取时再取得短期
租约；内容有读者时禁止原地修改，适配器需要修改尾页时准备新副本。驱逐选择在公共
策略锁内完成，实际文件写入以及冷记录可写回驻时的读取均释放策略锁，只保护所选记录；其他记录的驻留读写、记账
和有容量的预留可继续。系统读写错误统一为 `MemoryError::TransferFailed`，与容量
不足分开。等待另一个驱逐释放 H 的操作只允许在 CPU 存储任务或独立 llama lane 上执行。

写盘先预留 D，完整写入成功后才发布 SSD 副本并释放 H；任何失败保留原有效副本。
H 计算实际驻留字节及 RAM 预留，D 计算仍有所有者或读租约的文件区间。
过渡期间的两份物理副本分别计费。操作系统文件缓存、统计和传输工作区另计。
活跃与闲置历史移动同一组记录所有权，共用 H/D；不得另行序列化同一份 KV 历史。
逻辑历史准入容量与物理 H 分开：启用 SSD 的适配器需保留一次工作集换入/换出余量，
不能将 H+D 全部许诺给历史，再在换页时额外分配不计费的 payload。

`SpillFile` 使用 Windows 重叠偏移 I/O 或 Linux `pread/pwrite`，支持 64 位偏移。
文件属于当前进程，最后一个区间所有者释放后删除；不承诺重启持久化或跨进程共享。
`StorageIoQueue` 同时限制任务数和持有的缓冲区字节；取消的排队任务只有在回收后才
报告完成，运行中的任务先排空。worker 仅执行 CPU/文件操作，不能访问 GPU 或发布
引擎状态；发布仍须回到持有原会话版本的引擎安全边界。

ninfer 把工作集交换、追加页准备、query 重放和冷 checkpoint 读取接入已有调度边界。
等待磁盘的 lane 不领取计算份额；其他就绪 lane 继续运行。强制控制文本按不超过
prefill chunk/R 的块提交，未提交部分仍属于输出控制状态。取消只撤销排队工作，
已经开始的 CPU I/O 排空前保留 lane 的源和临时容量，期间不阻塞引擎线程。单请求
存储错误在其他 context transaction 稳定后结算，不清空其他活动会话。

idle checkpoint 写出借用由引擎持有的不可变 pinned 源；worker 不拥有 CUDA 对象。
完成时核对历史 generation 和原 checkpoint，再发布 SSD 附件并归还原 RAM；历史已
恢复或被淘汰时仅丢弃未发布目的。活跃原生 KV 不再被 idle serializer 重复写出。
llama 的并发 lane 共用一个 H/D 账户；会话附件 I/O 在目录锁和 GPU 执行锁之外运行，
目录 switching 状态保护原生可变执行绑定；请求引用保护不可变源。源可被多个请求
分别恢复，但同一可变 GPU 视图不能同时交给两个请求。淘汰目录引用不撤销其他读者。

公共存储已通过 Windows 和 WSL/GCC 主机测试及地址/未定义行为检查；WSL 的
ThreadSanitizer 在进入测试前因地址映射失败退出，并未提供竞态检测结论。
Windows 已有双后端活跃/idle 正确性及并发、取消证据；最终性能验收尚未完成。
Linux 目前只有公共层 WSL/GCC 证据，没有原生 Linux CUDA 推理验收。
阶段状态见 [活跃 KV 分层计划](active-kv-tiering-plan.md)。

## 三种坐标

| 坐标 | 持有方与字段 | 用途 |
|---|---|---|
| 原始 token 位置 | `BlockDescriptor::original_start`、`MemoryStamp::frontier` | 历史身份、模型 RoPE 位置、提交进度 |
| 紧凑 attention 地址 | `ViewBlock::compact_start` | 按原始时间排序后的工作集读取和 causal 边界 |
| 物理页/槽 | 引擎私有 | 显存分配、地址表、native generation、stream 和 graph 生命周期 |

例如取原始位置 0、128、512 开始的三个完整 128-token 块，新视图的紧凑地址为 0、128、256；K 仍对应原始位置 0、128、512。公共契约不提供“把 K 重新旋转到紧凑位置”的操作。

这个示例是普通一维文本位置。`original_start` 始终是逻辑 token 序号；多模态的多轴 RoPE、rope delta 等真实位置元数据仍由引擎保存，公共层不能用这个序号伪造或覆盖它们。

`KvMemRemap` 的 `from_base/to_base/raw_refresh` 是旧 re-RoPE 设计留下的元数据。`KvMemStore::set_selection` 仍维护这些字段，旧测试也仍覆盖其行为；当前 llama 的 `apply_plan_to_kv` 使用 stage-in/out，`write_block_to_gpu` 恢复 packed K/V，`occupy_in` 使用保存的原始 row positions。`remaps` 在适配器中主要用于块遍历、trace 和统计，不能据其字段名推断产品又旋转了一次 K。

`MemorySession::prepare_reselect` 只调用既有 `pick_topk_blocks`，核对 policy/native catalog 的粒度、块身份和有效长度后，生成 `WorkingSetPlan`，不调用会修改旧 remap 元数据的 `set_selection`。选块算法没有复制。原生兼容桥尚在使用的旧执行元数据暂时保留。

## LCP 与前缀 checkpoint

尾部用户消息改写导致反复全量 prefill 的问题定位、LCP 边界留点方案和最初的三轮回归
由 [Iscream132](https://github.com/Iscream132) 在 [#101](https://github.com/kvmem/kvmem/issues/101)
与 [#102](https://github.com/kvmem/kvmem/pull/102) 中贡献。当前共享实现通过
[#137](https://github.com/kvmem/kvmem/pull/137) 合入主线；原始贡献及证据更正见
[贡献记录](prefix-reuse-checkpoint-validation-2026-10-03.md)。

主线的公共前缀恢复核心来自 `9d88c5c`，接入 llama.cpp 和 NInfer。
`prefix_checkpoint.hpp` 统一恢复边界、留点规划与 `CheckpointCatalog`，不解释原生状态字节。
适配器仍须证明 token、媒体、位置、KV/统计及执行状态在同一已提交边界有效。

当旧 checkpoint 晚于本轮最长公共前缀（LCP）时，不能直接恢复它；公共层仍保留已观察到的
LCP，作为重新计算时捕获复用锚点的提示。`prefix_anchor` 将提示限制在恢复点与 query 之间，
避开不可拆分的媒体组。捕获完成后才能发布，捕获失败允许后续请求重新计算。

`ReuseAnchor` 服务下一轮尾部改写，`QueryReplay` 服务本轮 query 重放；相同边界可共享状态。
新锚点替代旧锚点的保留优先级，后续 endpoint 不应将它挤出目录。回退撤销更晚的目录项，
活跃请求持有的状态引用继续存活。目录默认保留 llama 四项、NInfer 五个用途槽；共享状态按
所有者地址只计一次。checkpoint 及统计附件仍独立于 KV payload 的 H 预算计量。

候选选择以内容和原生恢复证明为准。`prompt_cache_key`、`promptCacheKey`、
`kvmem.conversation_id` 等兼容字段不限制候选、不绑定单一可变历史、不提供发布优先权，
也不能触发按同 key 批量清除快照。真正的状态式会话接口继续使用其显式状态引用。
内部 `SessionIdentity`、incarnation 和 generation 仍防止过期引用或完成回调错配，
它们不等于客户端身份。admission 在安全边界重新核对 token/media/position 与源版本。

新用户 Query 可查找已验证的内容前缀；工具续接只允许原生可证明的 continuation，
随后 append 加 resident-only Pressure。读取恢复所需的冷字节不授权用旧 Q 重选冷历史。
LCP 提示仍只规划新的捕获点，不能恢复任意 recurrent 中间状态。回退或编辑只撤销本次
私有分支的后缀，不得改写已发布源快照。

## 公共内容快照目录

`content_cache.hpp` 的公共策略不接收客户端 ID。`rank_content_candidates` 对适配器
提供的候选执行 Query/continuation 门及 frontier、冷读取量、最近使用排序；原生精确
证明仍由适配器完成。`select_content_publication` 只认可同一封存对象的身份去重，
否则按空槽/容量/LRU 选择；相同 token 或 frontier 不表示可以合并不同执行状态。

`ContentSnapshotDirectory<Record>` 为可发布不可变原生对象的适配器提供有界索引。
它按兼容域与前缀摘要筛选，先取得源引用，再在锁外调用原生证明；摘要碰撞必须被精确
证明拒绝。目录用完整准备好的元数据副本原子发布，不按请求开始顺序压制兄弟分支。
原生目录也可复用这些公共策略并保留自己的 checkpoint、调度和存储封装；这不等于所有
适配器已经切换到同一种 hash 索引。

源快照包含恢复所需的原生状态和策略基线。`shared_ptr<const Record>` 要求调用方封存
原生内容并停止可变别名写入；它不能把一个仍会被 GPU/CPU 改写的对象变成快照。
请求分支独立保存选择器、统计更新、Query、LRFU、图反馈、回滚及活动 recurrent 状态。
独占的驻留执行绑定可走热续接，同时保留可供其他请求恢复的不可变源。

容量 group 仅服务公平保留。公共 `plan_content_group_publication` 分别限制组数与物理
目录项，整组 LRU 用组内最近访问，多版本组优先让出重复占用；同组不意味着覆盖许可。
NInfer 用原生 continuation 关系继承或创建内部组，深分支可建立新组，候选匹配始终对
全部内容执行原生证明。未知组不合并为一个匿名 owner。
组分类须精确证明完整且已接受的 canonical input（token、媒体和位置），不使用短模板尾
容忍来合并不同问题。发布时重新检查组内所有已完成 input 与本次 input 均在同一精确
前缀路径，再选择最深匹配的组；冲突或未知边界建立独立内部组。这样从旧祖先同时准备的
两个分支也不会因发布先后而混成一组。此分类只影响容量公平性，不提供原生恢复权限。

`content_snapshot_covers` 仅检查非空、非零 immutable-origin 集合覆盖。调用方还必须证明
这些 origin 保留完整 KV/位置、原生状态和策略基线；NInfer 的覆盖合并还要求原 archive
分配身份一致，COW 尾页保守地阻止合并。一般容量淘汰与这种无损覆盖压缩分别记账。

目录元数据限额独立于 payload、checkpoint 池、临时缓冲区和在途引用。队列中的快照
使用 `HostKvSource` 类内容引用保留字节，不长期钉住 RAM 驻留；实际复制才取得短读租约。
普通目录淘汰不使已取得引用失效，最后一个分支/读者结束后才释放实际字节及配额。
公共目录锁内不调用原生证明、不等待 I/O/GPU，也不销毁最后一个原生资源引用。
`publish`/`clear` 返回 retired 引用；适配器必须在设备安全边界或通过延迟 deleter 回收。
包含 GPU/rank 对象的目录本体和最后一份外部引用同样遵守该销毁约束。

llama 旧 CPU arena 不能作为封存共享页：分支复制独立 arena 及 tier table，按独立分配
计费。lane 目录在 create/fork 前预留一个配置容量，承诺上限为
`(snapshot_slots + lanes) * cpu_bytes`，涵盖 active、parked 和 in-flight store；失败释放
预留并保留源。该保守账户与 raw H/D、RSS 分开报告，复制直接写入已分配目标，避免第三份
完整临时 arena。原生 recurrent checkpoint 池也有独立上限；保留源再执行私有分支会增加
其存活状态，目录未满不代表这个池仍有空间，容量不足仍可淘汰空闲源或重新 prefill。

只有已接受、实际执行完成、KV/统计/原生状态附件齐全的边界可发布。恢复或发布失败不能
提前取走源目录项；取消后的运行中 I/O 必须安全排空。worker 不得安装 GPU 状态或发布
目录。多 rank 原生对象只有在必需 rank 的边界与完成条件一致时才能作为完整恢复源。
这些规则不增加跨后端 KV 格式、跨进程持久化、GPU 页共享或新的 ingest 协议。

`content_cache_test` 的主机测试覆盖并发反序发布、原生状态变体、hash 碰撞、工具门、
源引用、容量组公平性、元数据限额及失败保持源。四后端的 MTP/视觉/SSD/多卡实机状态
必须看对应新版本的验证记录，公共主机测试和旧二进制报告都不能替代这些验收。

验证入口：公共层 `prefix_checkpoint_test`；llama 的 `kvmem-prefix-reuse-regression` 在没有
`KVMEM_TEST_MODEL`/`KVMEM_TEST_GPU` 时以 77 跳过，有模型时运行六轮尾部改写 HTTP 回归；
`kvmem-prefix-reuse-mtp-regression` 使用 `KVMEM_TEST_MTP_MODEL`（含 MTP 的模型）与同一 GPU UUID，
运行单会话、MTP3 的十二轮回归，要求第三轮起持续复用并确实产生 draft 验证；缺少模型或 GPU 时也以 77 跳过。
NInfer 的 `ninfer_p4_history` 提供 `prefix-anchor` 与 `prefix-anchor-ssd` 模式，检查原生状态、
复用进度和取消恢复。SSD 模式还要求实际发生冷 KV 读写，不能用仅开启参数代替验收。

## 职责

| 对象 | 负责 | 引擎继续负责 |
|---|---|---|
| `KvMemStore` | 既有排序、sink/recent 和强制保留策略 | 提供已提交的统计，选择前冻结 query |
| `WorkingSetPlan` | 块身份、版本、紧凑映射、换出/换入、预算 | 具体物理页号与临时 buffer |
| `MemorySession` | 版本检查、换页顺序、发布门槛、取消与失效 | 串行执行安全边界和线程同步 |
| `PayloadLayout` / `NativeKvPayload` | 原生 bytes、所有 planes、stride、alignment、格式版本 | 解释量化编码，执行实际复制 |
| `MemoryBackend` / `MemoryTransfer` | 必须实现的适配器契约 | 唯一的设备分配器、reservation、stream、event 和 graph |
| `CheckpointIdentity` | 精确检查后端/模型/布局/会话/执行历史/frontier | checkpoint 字节、GDN 还原与 query 重放 |
| `CheckpointCatalog` | 分支内最大合法恢复点、用途、覆盖与数量/字节约束 | 原生状态、完整前缀证明、状态捕获/恢复和安全执行边界 |
| `content_cache.hpp` | 无客户端身份的候选排序、发布与组容量策略、可选不可变目录 | 原生完整性/续接证明、资源账户、设备安装和安全销毁 |

新后端不能继承旧 `KvMemBackend` 的默认空复制来表示已支持。`MemoryBackend`、`MemoryTransfer` 全部必需操作均为纯虚函数；缺失 backend、返回空 transaction 或缺少所需能力会明确失败。

## 身份、版本与位置边界

- `BackendIdentity` 固定后端、源码/实现版本、实际模型身份和 state schema。一个 backend 对象在绑定期间不能热换模型、布局或能力；更换必须创建新的会话/后端对象。
- `SessionIdentity` 包含会话 ID 和 incarnation。清空、编辑历史、重建会话时使用新 incarnation，不能只靠可重复的 block ID 区分历史。
- 每个块分别记录 `content_version` 与 `statistics_version`。Native payload 的复用只看 KV 身份/版本/有效长度，统计更新不会冒充 KV 写入；统计的来源与当前 revision 仍需适配器保证。
- `MemoryStamp` 覆盖 valid、incarnation、history revision、view version、逻辑 frontier 和 execution history。逻辑内容或统计变化推进 history revision；工作集发布推进 view version。任何相关状态变化必须反映在 stamp 中。
- prepare 使用完整逻辑 catalog；按时间连续，最后一个块可以不满。resident 必须表示当前完整版本，Host version=0 表示缺失，旧版本不能作为换入来源。输入中的未知 ID、重复 ID、漏掉 mandatory、粒度不匹配会报错，不静默修正。
- 即使没有选中某个冷块，它也必须保有当前 Host 副本；不能在新计划中默默接受已经丢失的冷历史。请求 Retrieval 策略时，后端必须声明 pre-RoPE 统计能力，否则明确拒绝；不把 sink+tail 能力当成检索已经接通。
- 选择结果按原始位置排序；compact 地址不会修改 `frontier`。多 query causal mask、图的 buffer 绑定和最大原始位置仍由实际引擎验证。
- `CheckpointIdentity::compatible_with` 采用严格相等。相同 token 文本不足以让 full/sparse/replayed GDN 状态互换。ninfer 的 query checkpoint 保存完整原生 StateImage、精确前缀和当时的驻留视图；恢复与重放仍由 Program 执行，公共核心不解释其字节。

## prepare → transfer → publish / abort

调用方必须暂停旧视图的新写入，并在引擎安全边界单线程驱动 `MemorySession`。异步设备传输可以进行，协调器本身不提供锁、调度器或后台线程。

```text
Idle / Published
   prepare(expected_stamp, selected, mandatory, budget)
Prepared
   advance()：start_spill
Spilling
   advance()：poll；Pending 保留所有源页
   Complete 后：start_restore，可回收 evict 页
Restoring
   advance()：poll；Pending 不允许发布
Ready
   publish()：检查版本 → 引擎发布 → 核对新 stamp
Published
```

1. 纯规划先检查布局、身份、所有选中来源及预算。`prepare` 必须预留所有目标 Host buffer、native 页 entitlement、复制暂存和新视图 metadata；失败必须释放预留，保持旧视图，不能提前启动复制。
2. `start_spill` 只复制尚无当前 Host 副本的 outgoing 块。源页在 poll 返回 Complete 之前不得释放；Complete 包含 stream/event 完成和内容版本确认。
3. `start_restore` 可以释放 evict 页并用同一原生页池恢复。适配器也负责按新紧凑顺序排列 retained 块，不引入第二套 GPU 分配器。
4. 所有恢复完成后进入 Ready，尚未自动发布。`publish` 必须在安全边界同时暴露页表、有效长度、Host/resident metadata 与新 view version。graph 不能引用已释放或仍 pending 的 buffer。
5. `abort()` 必须等待或取消在途工作，然后才释放 buffer/leases。能保证旧字节和旧视图都完整时返回 `PreviousViewPreserved`；否则返回 `SessionInvalidated`，协调器调用 backend.invalidate，拒绝继续推理直到建立新会话并重新 prefill。

每次推进和发布前检查 stamp；传输中出现过期计划会触发 abort。发布阶段抛异常或后端未公布预期版本时，强制失效，不能报告成功。析构和 disconnect 同样先 abort；`shared_ptr` 使后端生命周期长于 pending transaction。后端的 prepare/reservation 还需防止多个协调器同时取得同一 session 的所有权。

## 预算与原生 payload

`selected_tokens` 限制有效历史数量；`device_tokens_required` 按每个块的 native 分配粒度向上对齐，再加独立对齐的 `reserve_tokens`。配置必须尊重粒度，不把 32-token ninfer 配置偷偷当成 64 tokens。

`host_payload_bytes_required` 包括所有已占用的 Host 记录和此次换出的新记录，以 `record_bytes` 计量。旧 Host 副本占用的空间也必须计入。设备权重、GDN、workspace、graph、Host metadata、checkpoint 和传输临时空间不包含在这个字段中，由引擎自己的内存计划和 prepare 预留负责。

`PayloadLayout` 明确格式版本、block tokens、record bytes、alignment，以及每个 attention layer 的 K/V/可选 K-scale/V-scale 平面。平面不能重叠或越界；packed 行内 scales 保持在对应 opaque K/V bytes 内。`token_stride=0` 表示原生整页布局，公共层不猜测其内部张量顺序。

`NativeKvArchive` 是有容量约束的普通 Host byte archive；它保存的是字节序列，不能将其 vector 地址当作满足 native alignment 的设备或 pinned buffer。引擎可继续使用自己的 pinned Host store，并兑现相同版本/完成规则。ninfer 的页池无需替换为这个 portable archive。

`raw_kv_payload_layout/export_raw_kv_payload/import_raw_kv_payload` 为现有 `RawKvStore` 的 packed token-row K/V 提供独立边界：

- 只包含显式列出的 attention layers，缺失 KV 层不能伪造成全零记录；GDN 层不产生 KV planes。
- 不读写 mean-K sums，不调用重新量化或 RoPE；部分尾页导出将未使用的序列化字节清零。
- 导入前核对 backend/model/layout、原始位置和行 stride。旧 raw packed frontier 更长时，拒绝短恢复，调用方必须先显式 invalidate packed tail，避免暴露旧尾部。
- 该桥接适用于已有 packed-row 路径；其他 native layouts 和 checkpoint 恢复由实际适配器负责。
- 来源冻结与内容版本来自引擎；`RawKvStore` 没有模型或会话标识，桥接不能自行推断它们。导入中若发生分配或 I/O 错误，调用方必须将本次恢复视为失败并按事务规则处理。

## 两后端能力表

P3 首步已让 llama 的检索、预填充压力及会话 attach/detach 共用公共 `plan_residency`。
这个函数只计算原生块的保留、换出和换入集合，供 `WorkingSetPlan` 和旧槽位执行器共同使用；不解释位置、不分配内存。
适配器通过过渡类型 `NativeResidencyPlan` 调用它，准备阶段不修改 `in_working_set`，成功分配槽位后才提交驻留成员；`baked_pos`、`remap_count`、`remap_abs_delta` 不再参与产品执行。
关闭 stage-in 复用的配置仍会强制换出并恢复选中块；重复或未知 ID 明确报错。

llama 的预填充计划可包含已登记、尚未计算 KV 的行，而 `MemorySnapshot.frontier` 只允许已提交 KV；旧 Recency 路径也不承诺所有冷历史都有 Host 副本。`NativeResidencyPlan` 保留这些区别；已经接入 `MemorySession` 的路径使用下文的 evaluated / accepted frontier。

| 契约能力 | llama.cpp 当前产品 | ninfer KVMem 首版 |
|---|---|---|
| 原始位置与物理存储分离 | 原始 row metadata 和 packed 恢复 | 原始 RoPE position 与紧凑 cache position 分离 |
| native K/V payload | 既有 packed rows/raw mirror | BF16 / INT8，包含独立 scale planes，普通 Host 归档和有界 pinned 暂存 |
| 有限设备工作集 | 既有预算语义 | 显式 B+R，H payload 上限，执行前 admission 检查 |
| mean-K / query 统计 | 既有捕获与统一接受提交 | RMSNorm 后、RoPE 前捕获；同一公共 mean-K 选择算法 |
| GDN / checkpoint | 保留引擎专用实现 | 精确 query 前 StateImage、前缀与驻留视图，探测后恢复重放 |
| 搬运 / 图 | 保留专用实现 | 同步安全边界搬运；图开关、原生字节与状态一致性验收 |
| 连续多轮、编辑、取消 | 保留已有能力 | 可信 checkpoint 复用；失配或无 checkpoint 时从头计算 |
| MTP、并发、多模态、磁盘恢复 | 保留已有能力 | 首版不开放 |
| 接入 `MemorySession` | 单 GPU FA + RAM packed；其他配置保留兼容桥 | 所有已支持的 KVMem 工作集切换 |

上表的「ninfer KVMem 首版」只描述 P3 第一步。后来的 MTP、并发、视觉和多卡在 ninfer 适配器里实现，不以该行的「首版不开放」为当前产品边界。磁盘入口现接入本文的公共进程内分层。当前组合见 [多后端运行](multi-backend.md)。

## 验证与复现

```sh
cmake -S . -B build-host -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF
cmake --build build-host
ctest --test-dir build-host --output-on-failure
```

P2 独立构建目标：Windows / MSVC 19.44、C++17，无 llama 子模块构建、无 ninfer 链接、无 GPU runtime。P2 的两个新增可移植测试与 P3 追加测试覆盖：

- `memory_contract_test`：packed-row、BF16 native page、INT8 多平面三种形状；按原始位置排列、尾页和生成预留；两次选页的数据恢复；mandatory、重复/未知块、过期来源与容量错误；pending 阶段禁止回收/发布；复制/恢复/发布失败；取消、disconnect、析构生命周期；复用原选择算法而不改变 legacy remap；checkpoint 隔离。
- `native_kv_payload_test`：全部 planes 和 padding 的精确复制、跨模型/会话/layout 拒绝、旧版本/短记录/坏布局、Host 预算；32/64/128 tokens 三种块大小 × 1/部分尾页/完整块共九种 RawKvStore 往返，验证源和目标 mean-K 保持不变。
- P3 `native_residency_test`：400 组独立集合检查、32-token 块及 5-token 尾块、四个物理槽位上的精确字节往返、强制重载、未计算的预填充行、换出完成前不释放槽位、会话换绑、分配/复制失败时保留 Host 源与回收未发布目标槽。验证原始位置和全部 legacy 重旋转元数据不变。

测试后端只在测试文件里模拟完成事件和故障，不能用作生产后端。这些测试证明公共库的行为与生命周期，不替代 P3/P4 的实际 GPU、GDN、取消、吞吐与质量验证。Windows 不运行既有 POSIX/NVMe 专用测试，本次没有修改其实现。


## P3.1b：生产原生事务与三种 frontier

`KvContent` 随 `ConvStore` 移动。它区分预留地址 `reserved`、已过图完成 fence 的 `evaluated`、已接受的 `MemoryStamp.frontier`。预填充登记位置不会生成有效 `BlockDescriptor`；分多次 ubatch 完成时，只能导出已经完成的前缀。MTP target 验证前通过显式 hook 标记推测执行，宽度为一也不会自动推进接受边界；`decode_mean_commit(n_keep)` 推进接受前缀，随后 truncate 丢弃拒绝尾部。未完成计算或接受尚未结算时，不能发布或迁移会话。

会话 ID 在进程内唯一，reset/replacement 更换 incarnation；append、重放和截断更新内容/历史版本。尾块的有效长度属于版本描述，不能拿完整旧块覆盖一个截断后的尾块。mean-K/Q 仍由现有捕获与接受逻辑管理；这里的 `statistics_version=0`，适配器不宣称公共 `pre_rope_statistics` 或 `state_checkpoint` 能力。GDN checkpoint 仍归引擎。

实际 `NativeBackend` / `NativeTransfer` 使用公共 `MemorySession`，目前启用范围是单 GPU、Flash Attention、packed K/V 保存在 RAM、旧 CPU/NVMe arena 关闭且启用驻留复用。其他组合保留原桥接，尚未宣称完成公共接口迁移。每个 lane 有自己的物理槽池；事务对象仅活在持有引擎安全边界的同步调用内，不会比引擎所有者活得更久。

准备阶段校验槽池容量，预留 packed Host 目的容量与 GPU 搬运/重排 scratch，不释放旧 GPU KV。spill 等待真实复制和 Host 写入完成并检查所有 attention 层覆盖后，协调器才允许回收、恢复与布局。沿用原有 packed 复制和 MTP follower，不改变 RoPE。publish 验证原始位置、多模态位置元数据和完整行数，再公布 view version；pending 期间 attention proof 无效。准备/换出失败保留旧视图；进入可能破坏旧字节的 admission 后发生异常，则排空、清空并使会话失效，要求重新 prefill。

历史保留规则是 **RetainAll**：即使选择策略是 recency，RawKvStore 仍保留已有历史的 packed K/V。普通选择中的预算是有效历史；压力 admission 额外保留未来块槽位，未来行不进入已提交 catalog、Host 有效长度或公开执行视图。部分尾块新增行使用现有槽位，只有新块计入独立 reserve。

真实 GPU 故障测试已覆盖 prepare、spill、restore、publish 四个阶段以及 fresh prefill 恢复，并检查 pending 视图不能逃逸。0.8B 多 ubatch、三组 KV 格式重放和 1/2/4 lane 服务回归已通过。5060 Ti 上的 27B KV/MTP/GDN 矩阵为 31 项通过；WDDM 下无法可靠制造物理显存 OOM 的 10 个检查明确跳过。普通解码和 MTP 服务回归通过，连同小模型共 54 个请求与 P3.1a 的回答及 usage 一致。性能对照详见工作区 P3.1b 报告。旧全序列删除后残留 packed KV 的路径已修正，hybrid clear 也先排空捕获再清除源 KV。

## P3.2：统计附件身份与统一接受提交

`ContentPrefix` 是 `KvContent` 发出的进程内逻辑前缀凭据，包含会话身份、接受位置和撤销状态。引擎在同一序列化边界调用其创建/校验/失效操作；它不是跨进程快照，也不证明全量与稀疏 attention 的 recurrent 状态等价。追加和同历史重放保持旧前缀，截断穿过前缀、reset 和 invalidate 永久撤销。弱引用目录清理已释放的凭据，不记录每一次生成事件。

llama query 附件带捕获范围和前缀凭据，尾块 mean-K 附件带 checkpoint 行号和前缀凭据。导入还核对维度、有限值和有效行数。默认 user-query / 多模态 checkpoint 路径在恢复任何 recurrent/draft 字节前检查该凭据；选择时跳过已失效的候选，无法找到可信状态则从完整输入重新 prefill。输入/media 的逐前缀匹配继续由 driver 负责，统计附件的凭据不能替代该匹配。

MTP 产品路径使用 `begin_speculative_evaluation` / `commit_speculative`。开始时绑定当前会话和执行历史，完成图计算后仅记录 evaluated 范围；接受时检查同一 capture、起点和宽度，为 GDN fold（若 recording）建立完成依赖，再推进 accepted KV 与 mean-K。未接受的草稿不生成 query 统计，不经 prefill mean-K 路径提交；零接受、部分接受、宽度一拒绝、超量/重复提交有真实 GPU 回归。原专用 hooks 保留，不要求 ninfer 使用 llama 的 tensor 或 recurrent 布局。

GPU 状态提交可以与不读取该状态的 Host 准备重叠。后续设备消费者必须排在提交之后，检查点、状态读取、迁移和释放必须排空相关依赖；逻辑接受不等于允许提前复用缓冲区。该规则由适配器兑现，公共核心不持有 CUDA stream/event，也不放宽 `MemoryTransfer::poll` 的 Complete 或 publish 门槛。

llama ReplaySSM 在 Fold 流记录事件，并将依赖加入目标 context 的对应设备后端；原生 context 同步和检查点读取会排空该依赖。无事件/消费者支持时保持同步完成。`KVMEM_GDN_FOLD_SYNC=1` 是性能诊断用同步对照。NInfer 在普通 Fold 前取回验证阶段产生的候选统计，使后面的 Host 统计提交不再额外等待 Fold；重建锚点仍使用原来的完整捕获与恢复边界。

这一步明确了适配器的统计/状态边界，没有把 Host worker 的完成事件伪装成公共 statistics version。`statistics_version=0`，公共 `pre_rope_statistics` / `state_checkpoint` 能力仍不声明。通用统计搬运接口、旧 server 编排分支和其余 legacy bridge 配置继续在 P3/P4 整理。

P3.2 主机测试 20/20，0.8B GPU 7 项、27B GPU 32 项通过（WDDM 物理显存 OOM 10 项跳过）。54 个既有请求与 P3.1b 的回答及 usage 一致，另有六次客户端断流后恢复通过；查询重放不改变首次 Q 的 sum/count。完整数据与范围见隔离工作区 `p32-report.md`。

## LRFU retrieval history and GQA max

Retrieval TopK may rank blocks by `H_t(b) = lambda * H_parent(b) + hit_t(b) * score_t(b)`. The hit threshold is the ascending score at `floor((1 - top_fraction) * N)`; ties qualify. Scores remain raw cosine values, including negatives. Sink, recent and media requirements consume their existing budget before the remaining blocks are ranked by H. There is no quota mixture or adaptive budget.

A logical request opens one round. Every preview derives from the same immutable parent scores. The final accepted input view commits once, after complete query capture and retrieval replay; later pressure selections and speculative verification cannot reward the round again. New blocks start with zero history. GQA max compares each query head's query-token sum with the matching KV head's mean K, takes the maximum cosine inside that group, then averages over KV heads and active layers. It changes no Q capture, RoPE, payload format or KV acceptance rule.

Physical reconstruction frontiers do not define the logical history parent. Each retained conversation has an ordered timeline of accepted input boundaries and immutable committed scores, plus one canonical input identity. When the frontend proves a message boundary before the transient assistant generation header, that boundary defines the input identity; otherwise the full input is required. Header-only blocks are not inherited. Only boundaries strictly before the incoming input end and proven to match its tokens/media/positions may be inherited. An exact repeat therefore uses its parent history. An edit drops later timeline entries only in the private execution branch when the new round is accepted; published source snapshots retain their own timelines. A physical rewind inside a verified continuation retains the selected logical history while rebuilding missing KV blocks; a same-request query rewind retains the open round. Reset or eviction drops the timeline. llama conversation swaps move the timeline and its input identity with the corresponding host store; failed multimodal requests restore their chosen parent history. ninfer's retained histories carry the same timeline alongside query and recurrent checkpoints.

With process-local SSD tiering, ninfer retains the LRFU timeline, input identity and checkpoint scores in the owning history metadata while native KV and recurrent-state bytes may move to SSD. Restoring cold bytes therefore preserves the same logical parent selection as a RAM-only history. The former whole-history snapshot codec has been removed by the tiering implementation. llama's session cold files also remain private to one server run; history and input identities stay in that run's conversation metadata. Neither backend gains cross-process recovery from these retrieval changes.
