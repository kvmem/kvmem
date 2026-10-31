# Windows latest-core quick validation — 2026-10-31

The fresh Windows build on KVMem 89c66980 passed an independently reviewed five-request compatibility screen: two ~65K trailing-query Host recall cases followed by three prefix-history requests. BUDGET36864, generation reserve16384, batch512, context262144, MTP3, FP8 K/V and rolling enabled were unchanged.

| Check | Result |
|---|---|
| Exact archive answers | AMBER-7391 / COBALT-2846, both correct |
| Actual Host blocks restored | 191 / 197 |
| Recall native decode | 49.20 / 49.62 token/s |
| Recall processed prefill | 985.37 / 984.00 token/s |
| Three prefix-request native decode | 43.22 / 38.90 / 54.85 token/s |
| Exact prefix resume | 8192 tokens, same_query=1 |
| Runtime memory guards / performance pauses | None |

The raw driver reports false because the combined runner accidentally supplied the recall response list to agent_history: five accumulated rows were compared with a three-request limit. All three actual prefix requests completed and passed their request screens. The original result and error are retained; quick-validation.json records the independent scoped review and input evidence hashes. The runner now uses a separate list for prefix requests. No GPU retry was needed to correct this accounting defect.

One intermediate acknowledgement was not correct and one response hit its 512-token request cap. Both recall answers were exact. Restore counters record actual Host transfers.

An earlier attempt was interrupted by VALORANT starting after test launch; it is retained as a contaminated performance run, excluded from the reported performance results. The successful retry began after the game exited. Existing DSH was preserved; the test process was stopped after requests completed. Historical 33-request tables remain attached to their old binaries.
