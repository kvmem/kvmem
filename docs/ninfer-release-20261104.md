# NInfer 2026-11-04.2 升级

本次将固定上游从 `2026-10-29.2 / d0da39796697` 升级到
[`2026-11-04.2 / 6cc1f9fa438b`](https://github.com/iamwavecut/ninfer-all/releases/tag/2026-11-04.2)。
基于 KVMem `203530674f68` 迁移后，整合执行期间新合入的 `d469a933f49c`（PR #259
内容缓存）；上游已包含旧 `nvfp4-a16` 和 `nvfp4-int8` 两层的实现，因此删除重复补丁并重连依赖。
保留主线内容缓存并新增一层版本兼容适配后为 43 层，
每层的 SHA256 和 Git tree 均由 `backends/versions.json` 锁定。

这是源码升级。已发布包与本地运行进程的版本需要分别核对，不能从源码 gitlink 推断。
此次没有改变 heads TP 默认值，也没有启用 positions attention 或低精度 TP partials。

## 适配要点

- 保留上游 TP prefill 的计算/传输重叠和自动分片重拟合，继续按 KVMem 的 B+R 实际
  设备窗口计费，不将完整逻辑 max-context 当作常驻 KV 需求。KVMem 统计缓冲区继续参与各 rank 预算。
- 新版将较大的 TP prompt chunk 再切成多个 piece。同一逻辑 chunk 的 Key 页面统计、
  紧凑 Query 捕获、Mean-Q 累加及 post-RoPE Q 写入均携带 piece 偏移；whole-chunk origin
  不变，按同一 rank 的计算流顺序初始化、保留或累加。避免后一个 piece 覆盖前面的统计。
- 辅助 KV 的描述枚举实际 cache 的 layer、rank、head 范围和全部物理 planes，
  不再假定它是 rank 0 上的一组 planes。辅助 cache 的 BF16/量化 plane 数按其自身几何核对。
  对应 schema 身份更新，防止旧镜像与新布局误配。
- 保留上游 untied BF16 GGUF output head 的精确字值读取、steering/LoRA 接口和新版 NVFP4
  数据布局。KVMem 的 Query/Pressure、checkpoint 清理、InputReuse、共享缓存、量化及 dense
  规格扩展继续保留。启用 steering/LoRA 时不做未绑定 steering 的临时自动 graph 探测，仍使用静态额度。

新增 release 的 MTP K/V 分别投影会先检查 pair 算子的真实几何准入，避免将不支持的
宽度交给固定几何算子。INT8 TP 传输保留为上游显式选项，默认仍为 FP32；本次真实模型验证
不为低精度 partials 的质量背书。

## 验证

本地验证完成，版本与可公开核对的结果见 [验收摘要](benchmarks/ninfer-release-20261104.json)。

- 标准准备脚本逐层重放及再次 `--check` 通过，43 层输出与原生工作树一致。
- 转换器测试 60 项通过；补丁链准备契约测试 10 项、CI 分类测试 13 项通过；公开证据检查通过。
- 12 项主机契约通过，覆盖启动选项、容量、frontend、checkpoint 元数据/清理、统计布局/预算和 MTP graph profile。
- 两张物理 GPU 的通信测试通过；加载、8 种不等比 TP 精度策略、完整 StateImage 往返、NVFP4 A4/SwiGLU 和 rowsplit 数值回归通过。
- RTX 5060 Ti 上的统计 CUDA 回归通过：24 组分片、28 组候选、636 组既有用例。
  分片测试覆盖 512+482、384+384+221 的不整齐切分、未对齐原始位置、跨页/Query 边界、
  空 Query、Graph 多次重放与输出 canary；使用独立 FP64、逐词 FP32 和 BF16 位精确参考。

真实模型均使用 5060 Ti / 5050 Laptop 的 3:1 heads、INT8 KV、FP32 TP partials 和 Graph，
开启原生传输验证。文档与问题分开请求，检查真实前缀复用、Query 范围、工具续接的 resident-only
Pressure、冷启动工具调用和流式取消后的恢复。

| 模型/草稿 | prefill | B / R | H | 结果 |
| --- | ---: | ---: | ---: | --- |
| 27B IQ3_S / MTP2 | 128 | 768 / 1024 | 512 MiB | 11 个完成请求及一次取消恢复通过 |
| 27B IQ3_S / MTP2 | 1024 | 768 / 2048 | 512 MiB | 11 个完成请求及一次取消恢复通过；实际观察到 1024-token prefill |
| 27B IQ3_S / Q3 DFlash2 K5 | 128 | 2112 / 1024 | 1024 MiB | 11 个完成请求及一次取消恢复通过 |
| 4B / MTP2 InputReuse | 128 | 384 / 128 | 2048 MiB | T0 默认策略的 7 个请求通过，含模板边界、取消和新用户输入 |
| 4B / MTP2 内容缓存 | 256 | 8192 / 256 | 1024 MiB | 17 个请求通过，保留 3 个独立资源组；旧父分支实际复用 931 tokens |

两档 MTP 日志各包含 20 次 `payload_exact=1 state_exact=1` 恢复验证。测试脚本现在允许调整
resident 夹具长度，并独立检查其满足 `B < prompt <= B+R`；此前两档 MTP 保存的实际长度
1045 tokens 也满足这个条件。

迁移中修正了 converter 常量位置和类型夹具、测试目标的 KVMem 依赖、adapter 身份恢复参数、
新 DFlash2 调用的可选 floor 参数，以及旧跨 rank gather 夹具的格式。后者改用 GGUF 权重，
原生格式必须拒绝 gather 的检查保留。一次独立重链漏设 build ID 时，复用校验正确阻止了
不匹配生成头，随后使用固定环境重链通过。失败日志保留，没有放宽数值或恢复断言。

现有 Q3 DFlash2 artifact 的草稿环为 2048，B768、B1088 两次启动均被预算校验拒绝；
最终按 sink64 + recent2048 使用 B2112，并保留该约束。不能将 MTP 的 B768 配置直接套到这份草稿。

构建使用固定依赖、MSVC 14.44、CUDA 13.2、sm_120a。先检查完整传递依赖与编译参数，
用独立 Ninja 图显式复用 371 个对象；必要重编范围为 290 个主机对象、257 个 CUDA 对象。
旧 Ninja 日志和依赖记录没有修改。正式版 .2 发布后的增量阶段保存已完成对象及完整传递依赖，
经审计复用 745 个对象，补编 167 个主机对象、11 个 CUDA 对象；只有未引用该类型的编译单元
可以复用仅新增末尾 INT8 枚举值的头文件，原有 FP32/BF16 值和 enum 底层类型均未改变。统计算子的先行验证只另编译一个主机测试，链接本次已构建的
生产 CUDA 对象和 core 库。

本次不把上游性能声明视作本地收益，历史速度数据也不作为本次升级的性能结论。
本轮没有新增 SSD、双卡视觉或真实 LoRA 模型验收；SM70 专用路线在 SM120 卡上跳过。
没有发布新包或替换已安装程序，也没有做完整语义质量评估或三张以上物理卡测试。
