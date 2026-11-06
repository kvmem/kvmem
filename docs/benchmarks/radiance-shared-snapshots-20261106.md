# Radiance immutable checkpoint sharing, 2026-11-06

All measurements in this report use **ordinary IQ3R**, not IQ3R-S, on the replacement test machine. Comparisons against earlier machines or S-model runs are excluded.

## Fixed environment

- AMD Radeon RX 9060 XT, gfx1200, UUID `GPU-81455e5c06401be0`; Ryzen 9 3950X; approximately 15 GiB system RAM and 4 GiB swap.
- Ubuntu 26.04.1, ROCm HIP 7.16, clang 24, GCC 15.2. Radiance 1.17 at `19d4e8fe0e18071b9b6300f15fc4ccf38dc4fa3d`, repository-pinned patches, `qwen35-dense-tp1` build profile.
- Model: `StillDeadcode/qwen3.8-27b-iqr-v4` revision `e5f1b3ef70741d705a64a28b852acf3fcefcead7`, `qwen3.8-27b-iq3r-v4.rad`, downloaded through HF Mirror. Size 14,261,252,096 bytes; SHA-256 `99f306975bf5f48d6b023a177893fa85c6469ee080f2d7955caa3eaa16b84542`.
- Kernel SHA-256: `a73a3d77c1bd1f6f6b2403b21c600a95853a088434b557d7993b54a2a3dab6d3`.
- Context 262,144; budget 36,864; generation reserve 4,096; prefill batch 256; concurrency 1; content groups 1; RAM payload allowance 4 GiB; SSD allowance 8 GiB; DFlash7 fixed; deterministic mode; automatic sink/recent allocation; rolling generation.
- Canonical Task2 fixed tool history: intermediate output cap 512, final output cap 4,096, thinking budget 128, temperature 1.0. These are benchmark controls; the user chat service has no startup output/thinking cap.

## Controlled 14-round attribution

Each arm starts a fresh worker on the same GPU/model. These are single runs, not a statistical confidence interval. All arms processed exactly **106,734 native prompt rows** and passed the request, cache and memory checks.

| Arm | Record limit | Prefill seconds | Actual checkpoint KV captured | Logical checkpoint KV views | Observed disk writes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline, master `40749e33` | legacy hidden floor 16 | 183.155 | not instrumented | not instrumented | 16,272,293,888 B |
| Retention change only | 1 | 182.279 | not instrumented | not instrumented | 16,272,293,888 B |
| Immutable sharing only | 16 | 162.862 | 3,626,500,096 B | 28,138,012,672 B | 0 B |
| Sharing plus retention change | 1 | 162.845 | 3,626,500,096 B | 28,138,012,672 B | 0 B |

The combined arm reduces measured prefill time by 11.1%; the sharing-only arm has essentially the same result. Actual checkpoint KV capture is 87.1% below the logical full-view byte count. Record-limit changes alone do not explain the improvement. Observed device memory peaks stayed within 14,603–14,610 MiB.

`prompt_ms` includes native scheduling and storage waits; it is not pure GPU kernel time. Capture counters cover attention KV and exclude recurrent/draft-window state. Disk counters are sampled immediately after the final response: parking may still be pending, so zero is **not** a claim of zero writes over the full worker lifetime. RAM/SSD payload limits exclude model/runtime workspace and transient restore staging.

Worker SHA-256 values:

- Baseline: `ad9930d0c5b95d4345892c3fa1517e4774a7a7e4ec228a95bd715f195939b5a1`.
- Retention only: `bc2bc78c0d7157a19f14384898a5fe636bee08fc8a0b8b5328687be4cd4f71f0`.
- First shared candidate: `74b0b49d7c2027e5f08348e6ec5c860d3dd75ec58cead0eb88ae7998f5494b6c`.
- Final audited worker: `ff4f19dcfb262b1a840fec2bcaccef5d874339113ff7fd20216426fb4d6d4221`.

## Full 256K Task2

The first candidate completed all 33 requests at a final prompt of 257,958 tokens; all 32 follow-ups had positive actual cached tokens. Prefill totaled 417.413 s (617.99 effective new-input tokens/s). Native decode was 54.14 tokens/s for tool rounds and 55.25 tokens/s for final code. The final answer used 2,305 tokens, below the 4,096 cap. Sampled device peak was 14,603.48 MiB; running RSS peak 8,583.24 MiB; process swap peak zero.

The final audited worker also passed all 33 requests and all 32 cached follow-ups, with the same final prompt/output counts. Prefill totaled **417.222 s (618.27 effective tokens/s)**; native tool/final-code decode was **54.42 / 55.54 tokens/s**. Device peak was **14,603.50 MiB**, running RSS peak **8,545.16 MiB**, and process swap peak **zero**. The final audit added native engine-owner validation and corrected capture counters for already-parked input promotion; it did not change the selection algorithm or model. Full measurements and artifact identities are in [the JSON summary](radiance-shared-snapshots-20261106.json).

Task quality is separate: both runs acknowledged 10 of 31 intermediate batches as requested, and final Python syntax passed. Full program semantic correctness was not established. Passing cache/resource checks does not mean all task instructions were followed.

## Correctness and audit evidence

- Common C++17 snapshot tests: sparse span mappings, exact coverage, private growing tails, immutable source preservation, shared physical quota, cold reads, malformed plans, quota failure, concurrent same-origin publication and lifetime release. ASan/UBSan passed on the final source.
- Native GPU tests: BF16 and packed formats, recurrent/draft state, RAM/SSD recovery, byte comparison against full-copy capture, shared complete blocks, private tails, rewind with reused logical ID/version, and rejection of a foreign engine owner. Storage recovery and page-transfer tests passed.
- Final-worker SSD test: 32 MiB host allowance, 8 GiB SSD, budget 4,096/reserve 512; real cold Query/replay, tool pressure, cancellation and subsequent recovery passed. The document question returned the planted `LANTERN-5829` value.
- Final-worker concurrency test: two active requests, six A/B pairs, shared-prefix selection, old-parent return and tool continuation passed, with native cached-token evidence.
- 89 Radiance Python tests, native CLI boundary checks, JavaScript syntax and the monitor browser regression passed.

Raw results, request/response fixtures, worker/kernel/model hashes, launch arguments, monitor snapshots and memory samples are retained under `shared-snapshot-20261106` in the test deployment. A local compressed evidence copy accompanies this task: 1,805,201 bytes, SHA-256 `9cdc65ad70176e9cdeac5df2dce78e60cc635cd4274738b78e557dfa38748d42`. The JSON summary records the measured artifacts rather than inferring them from a later Git commit.

## Delivery boundary

This change reuses common `HostKvStorage/HostKvRecord/StorageIoQueue` and adds a common immutable snapshot manifest. Radiance still owns GPU pages and native state. Existing session/publication transactions and common content-group policy remain in use.

Restore staging is still the **full selected execution view**, explicitly reported by capability `restore_staging=full_selected_view`. Bounded restoration batches, I/O priorities, independent metadata quotas, NInfer/PolyStrata migration, multi-GPU runtime qualification and concurrency fairness/tail-latency measurements remain future work. The two-request test establishes functional overlap, not a fairness performance claim.
