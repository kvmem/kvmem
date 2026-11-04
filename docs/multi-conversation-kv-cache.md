# Multi-conversation KV cache

The current rules are in the [content-snapshot contract](checkpoint-policy.md#content-snapshots-across-backends). The native recurrent checkpoint remains the restore authority in both `legacy` and `spread`; token-prefix equality alone is insufficient. See [multi-lane conversations](multi-lane-conversations.md) for scheduling and [active KV tiering](active-kv-tiering.md) for public RAM/SSD quotas.

The server retains content-selected versions and creates an independent execution branch before modifying a reusable source. `--parallel P` controls physical execution lanes; model weights are shared, while GPU working sets and recurrent/MTP execution state remain per lane. P=1 serves requests serially; several lanes can execute independent requests concurrently.

`--kvmem-sessions N` is the modern retention configuration. The explicit `--kvmem-conversations N` compatibility path is deprecated and cannot be combined with it; startup still normalizes its requested value to at least P. The current lane catalog uses N internal content groups and a physical version limit `C = max(2*N, P)`. This is a deliberate change from the old N-total-store directory. It must not be described as an unchanged-N physical-capacity comparison. Active/switching branches may temporarily hold additional protected resources; publication and physical budgets remain bounded separately. `/slots` reports the actual group/version limits.

`kvmem.conversation_id`, `prompt_cache_key` and `promptCacheKey` are accepted compatibility metadata. The parser preserves valid UTF-8 values up to 256 bytes and checks types, explicit empty conversation IDs and conflicting aliases. None selects, owns or replaces a mutable cache lineage. Missing, changed or shared keys use the same content/native proof. An explicit stateful protocol handle has its own lifecycle; a cache alias does not create one. Applications must preserve the order of dependent turns rather than infer ordering from a cache hint.

## Decision table

[`tools/kvmem-conversation-store.h`](../tools/kvmem-conversation-store.h) returns the following reason strings. The legacy driver logs its selection as `KVMEM_TRACE store_select`; the lane catalog calls the shared policy and reports lane/fork and actual-reuse observations separately. A planned reason is not evidence that native restore succeeded.

| Reason | Actual policy condition | Result |
|---|---|---|
| `extend` | A continuation passes the native checkpoint boundary and the absolute continuation gate | Use the continuation path; source-preserving lane execution still makes the needed independent host branch |
| `shared_prefix` | Shared selection finds an eligible completed prefix beyond the continuation gate on a new-user Query | Fork sealed KV/checkpoint references and private mutable state; preserve the source |
| `new_conversation` | The policy received no stores | Fresh execution |
| `no_recurrent_checkpoint` | Stores exist but none has a checkpoint at or before its legal keep cap | Fresh execution |
| `fork_zero_keep` | A checkpoint was found, but the largest usable frontier is zero | Fresh execution |
| `fork_shared_prefix` | The compatibility selector found a positive checkpoint but no continuation, and the shared selector selected no eligible candidate | Fresh execution; this is not a special pinned-CPU rejection reason |

There is no current `client_id` selection reason. Tool continuation cannot discover an unrelated deep historical prefix: it needs content continuation plus a legal native checkpoint, then uses append and resident-only Pressure without cold-history reselection.

## Native boundaries

| Formula | Implementation |
|---|---|
| `keep_cap = min({lcp, live_row, eval_end - (spec_ok ? 0 : 1)})` | `kvmem_store_keep_cap` in `tools/kvmem-conversation-store.h` |
| `keep` is the largest checkpoint row `<= keep_cap` | `kvmem_store_checkpoint`, followed by adapter restore validation |
| `continuation = (rows - lcp) <= max(0, last_n_gen) + 64` | `kvmem_store_continuation`; a content classification, not a restore proof |

The media-aware `kvmem_prompt::common_prefix` supplies LCP. Candidates rank by the checkpoint frontier that can actually be restored, then recency; LCP alone cannot authorize restoring KV or recurrent state. The absolute tail allowance covers dropped assistant output and template-tail changes. A deep edit does not gain continuation permission merely because most of a long prompt matches.

The adapter still verifies model/layout, media, original positions, accepted frontier, packed K/V, recurrent/draft state and policy compatibility. Native restore can fail or fall back after planning. Revision-2 observations distinguish `planned_cached_tokens` from `actual_cached_tokens` and expose `restore_path=continuation|snapshot|miss`; continuation does not promise physical GPU residency.

## Ownership and physical retention

Published source recovery points are preserved while private branches truncate, replay, generate, fail or are released. Sealed packed KV and immutable native images can share physical references. MeanK, selectors, Query, feedback, rollback and checkpoint wrappers stay private where required. Directory eviction releases one reference; physical bytes remain charged until the last branch/reader or transfer releases them.

Internal groups allocate capacity fairly and never authorize reuse or overwrite. The common policy first evicts an old redundant version before another group's sole version, both for directory pressure and physical Host/checkpoint pressure. Pinned execution bindings and admitted source leases stay protected. Genuine quota pressure can still remove a unique group or reject optional reuse; no finite budget guarantees every cache hit.

A source-preserving restore can keep more checkpoints live than the former consume/move path. Their native image bytes and read/write staging remain charged to the checkpoint account. A larger checkpoint-pool functional arm is a separate configuration, not proof of an improvement at unchanged memory cost.

## Legacy CPU arena ledger

`--kvmem-cpu-gb` remains a per-store legacy CPU-tier setting. Native forks copy its mutable tier table and packed arena independently; they receive no shared-payload credit. `clone_cpu_tier_from` prepares potentially throwing metadata first, then copies into the already configured equal-size destination arena. It does not allocate a third full arena merely to perform the copy.

Let P be lanes, C the reported `retained_snapshot_limit`, and A configured CPU bytes per store. The lane catalog's separate arena commitment limit is `(C + P) * A`. Before a source-preserving fork or a fresh runtime constructor, it reserves a commitment under the catalog mutex against active lanes, parked bundles and in-flight work. Completion or failure accounts installed bundles before returning the reservation. Reusing an existing parked runtime needs no additional arena. Exhaustion can refuse the fork before native source mutation.

`cpu_arena_live_charge_bytes`, `cpu_arena_reserved_bytes`, `cpu_arena_charged_bytes` and `cpu_arena_budget_bytes` report this independent ledger. It conservatively charges configured per-store commitments; actual arena payload is rounded to complete slots. It is separate from raw H/D, checkpoint-image quotas, allocator overhead and RSS. Default A=0 commits no legacy CPU arena. Public SSD cannot be combined with this legacy arena.

For example, N=3 and P=2 currently give C=6. With A=0.5GiB, the arena commitment bound is 4GiB, plus raw KV, checkpoint and other state costs. Arenas are allocated lazily, so this is a bound, not an assertion that startup commits 4GiB. The former three-store 1.5GiB example does not describe the current version directory.

## Soft host bytes and switching

`--kvmem-conversations-gb` is a legacy soft target for accounted retained host bytes, accepted only with the explicit legacy configuration and N>1. It includes cached raw/runtime and server-state measurements; shared logical references can make it differ from physical storage use. Busy, resident and switching stores cannot be discarded merely to meet it, so it can remain above target with a warning. Zero disables that soft byte target. Hard raw H/D, checkpoint and arena reservations remain independent.

Switching drains pending stage-in, mean-K and harvest writers before changing bindings. The outgoing GPU target/MTP mappings are invalidated; row positions, raw KV, runtime state and Query metadata move together as a bundle. Incoming native restore must finish before successful reuse is reported. A warm resident continuation preserves an independent host source before later request writes. A failed allocation or transfer is handled through the request's failure path; native bundle APIs can throw and must not be treated as infallible handle swaps.

Host switching requires the supported Flash Attention/KV geometry. Configuration checks and per-restore proof remain authoritative; a token match or key cannot enable an unsupported V layout, model, MTP geometry or backend. No public GPU-page pool, cross-backend KV format or restart-persistent cache is introduced.

## Validation and historical records

Current host tests cover content selection and native boundaries (`conversation-store-test.cpp`), controlled lane admission/failure and arena reservations (`lane-conversations-test.cpp`), immutable source leases and fair retention (`kvmem/tests/content_cache_test.cpp`), and CPU-tier copy independence (`kvmem/tests/kvmem_runtime_test.cpp`). Their scope is host logic; they do not qualify every model, GPU, pinned configuration or storage fault.

The older `scripts/test_server_conversations.py` and `scripts/test_server_compat.py` retain evidence for their recorded server versions, including legacy total-store/LRU behavior and input validation. Those expectations are not a correctness oracle for the new group/version policy, and historical successful counts must not be presented as a new content-cache pass. Runtime qualification must record its binary, model, GPU UUID, budgets, actual native reuse, failures and physical resource scope.

The original native-bridge notes also reported stale host blocks after full `seq_rm` and incomplete repair after drain failure. They are inherited findings with their own evidence and review requirement; this refactor does not claim to have reproduced or resolved them. Native accepted boundaries and byte/state proofs, rather than exact legacy output, determine correctness.

## 中文摘要

外部 cache key 仅作兼容元数据，不决定缓存归属、优先选择或覆盖。完整原生证明通过后，父快照保留，子分支独立修改；工具续接仍走 append + resident-only Pressure。

当前 llama 目录按 N 个内部内容组、`max(2N,P)` 个版本槽保留，不能称为旧 N 物理槽不变。旧 CPU arena 独立复制，并在分配前以 `(版本槽上限+lane数)*每store配置字节` 预留；它与 raw H/D、checkpoint 和 RSS 分别计费。预算不足允许正常淘汰、拒绝复用或请求失败，不承诺所有预算都命中。历史测试数字只证明其原记录配置。
