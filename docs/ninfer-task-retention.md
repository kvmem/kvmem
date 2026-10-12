# NInfer sparse task retention
Here B means the selected-token budget, R means the generation reserve, and
W=B+R is the physical window. The rolling recent-tail setting is independent
of R (for the long replay, R=4096 but recent=5120).


The NInfer integration previously required sink, image and recent pages. A short
human task between an old answer and a long tool trajectory could become an
ordinary retrieval candidate. A later User-role runtime snapshot could also
become the scoring query. High scores for old history could then exclude the
human task even though transport, token counts and cache matching were valid.

The prepared prompt now carries independent, provenance-backed instruction and
active User ranges. No message or model token is removed or rewritten.

| Region | Retention |
|---|---|
| System/Developer blocks, including rendered tool schemas | Complete blocks |
| Each real User in the open tool chain | Last 512 tokens; complete when shorter |
| Images | Existing complete-image pins |
| Sink and rolling recent tail | Existing configured page counts |
| Scoring query | Existing last-real-User policy; separate from these pins |

A User here means a provenance-backed User-role input, including runtime
snapshots; this is not human-versus-clock classification. A non-tool Assistant
closes the previous group. Explicit continuation of the
final Assistant keeps its current group open. User-role tool-response wrappers,
Tool contents and quoted ChatML do not become task provenance. Unsupported
custom serialization without proven User block boundaries cannot acquire exact
User pins. Raw-token prompts have no invented chat semantics.

Pins use original logical page coordinates, count overlapping pages once, and
are clipped to the computed frontier. Admission evaluates the exact pin/recent union through the admitted target
frontier. Fully resident requests may use B+R; MTP admission also leaves room
for its fixed verification/draft mappings. Requests that can overflow must fit
the union within B; an oversized required union is rejected
before device mutation instead of silently trimming instructions. Pins may reduce
the space available for scored history. They do not enlarge the configured GPU
window or make sparse attention equivalent to full-history attention.

Cache restoration replaces the previous request's pin metadata. If its restored
view lacks currently computed required pages, it reselects before continuing.
The same window preparation is used for rolling prefill, query replay and decode,
including the shared native target/MTP transfer. Media expansion and both graft
forms translate the provenance offsets along with the rest of the prompt.

## Regression checks

| Check | Actual assertions |
|---|---|
| `backends/ninfer/tests/models/qwen3_5/test_memory_context.cpp` | Synthetic token coordinates/frontiers passed to the production selector: old-history scores win, yet current-task/control and partial-tail pages must remain within 576 pages. The old sink/recent-only control must lose the task. Separately checks an exact 512-token-tail page set, overlap accounting, independently computed request pins, frontier clipping and capacity rejection. This does not run actual prefill/decode or concurrent lanes. |
| `test_frontend.cpp::test_memory_active_task_provenance` | Old task + completed answer + current task + runtime User + assistant tool call + 20 KB Tool/quoted fake User; decoded active ranges must equal `{CURRENT_TASK, snapshot}`, then a subsequent completed turn must contain only `NEXT_TASK`. |
| `test_frontend.cpp::test_memory_context_tokenization_invariance` | Text encoded token IDs are identical with/without context boundaries. Expanded-image preparation carries nonempty instruction and one in-bounds User range; this is not an image token-invariance or GPU Vision test. |
| `test_frontend.cpp::test_prompt_graft` / `test_direct_graft_context_cache` | Query begin and first active User begin move by the graft prefix length; active User range counts agree. |

The Linux host CI prepares and verifies the pinned backend and runs
`python3 tests/test_ninfer_context_contract.py`. That check compiles the production
page-selection helper and selector without CUDA or a model. Frontend/media/graft
checks run in the dedicated `ninfer_qwen3_5_memory_context_frontend_test` target
(`ninfer_tests ninfer_qwen3_5_frontend_test --memory-context-only`); this runs eight
frontend functions. The default full `ninfer_qwen3_5_frontend_test` is retained.
Frontend checks are not implied by the model-free host CI result. Real-model
execution and any default-suite failure must be reported separately.

This patch does not add DSH-specific clock recognition to NInfer's scoring query,
change the integration's admitted speculative backends, or establish factual
answer quality. Long User contents outside their protected tail remain ordinary
retrieval candidates. Full external-tool trajectories require their own harness
replay and cannot be qualified by these host tests.

## Side effects and capacity boundaries

The number of GPU rows is unchanged. Required pages can displace scored history
or cause different transfers; sparse attention remains approximate and neither
answer equivalence nor throughput improvement is claimed. A legal custom ChatML
template may merge formatting newlines in BPE: instruction pins end at the
closing special token, not a potentially merged trailing newline.

`test_admitted_frontier_charge()` checks overlapping and disjoint recent tails,
B < F <= B+R, first overflow, non-page-aligned speculative scratch limits, and
1,000 deterministic fixtures against an independent page-membership oracle at
every earlier frontier. A fully resident forced checkpoint restore restores all
computed rows with {B+R, 0, B+R}; later append pressure restores the normal B/R
split. The ordinary sparse selection path remains unchanged.

The CUDA container job trusts only the exact checked-out workspace before
preparing the pinned llama.cpp backend. It does not use a safe.directory wildcard
or change the engine pin. This addresses the directory-owner preparation failure
seen in the base workflow as well as the first revision of this PR.
