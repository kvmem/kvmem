# NInfer sparse task retention
Here B means the selected-token budget, R means the generation reserve, and
W=B+R is the physical window. The rolling recent-tail setting is independent
of R (for example, R=4096 and recent=5120 are independent settings).


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
| Selected task before a completed Assistant, followed only by sourced reminders | Same bounded task tail; reminders do not create a replacement task |
| Images | Existing complete-image pins |
| Sink and rolling recent tail | Existing configured page counts |
| Scoring query | Last eligible real User; explicitly sourced runtime reminders are skipped |

A User pin here means a provenance-backed User-role input, including runtime
snapshots. Pins do not classify content as human or clock text. A non-tool Assistant
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
| `test_frontend.cpp::test_memory_runtime_query_origin` | Multiple explicitly marked reminders leave the exact query at `CURRENT_TASK`; the same unmarked text remains eligible. Marking preserves tokens, positions, control/User ranges and checkpoint; next task replaces the query and runtime-only input has no trusted query. Invalid typed origins are rejected. |
| `test_openai_schema.cpp::test_runtime_reminder_origin` | Chat Completions wire origin reaches typed prompt preparation without changing content. Null/unknown/wrong-type values and non-User origins return HTTP 400. |
| `test_frontend.cpp::test_memory_context_tokenization_invariance` | Text encoded token IDs are identical with/without context boundaries. Expanded-image preparation carries nonempty instruction and one in-bounds User range; this is not an image token-invariance or GPU Vision test. |
| `test_frontend.cpp::test_prompt_graft` / `test_direct_graft_context_cache` | Query begin and first active User begin move by the graft prefix length; active User range counts agree. |

The Linux host CI prepares and verifies the pinned backend and runs
`python3 tests/test_ninfer_context_contract.py`. That check compiles the production
page-selection helper and selector without CUDA or a model. Frontend/media/graft
checks run in the dedicated `ninfer_qwen3_5_memory_context_frontend_test` target
(`ninfer_tests ninfer_qwen3_5_frontend_test --memory-context-only`); this runs nine
frontend functions. The default full `ninfer_qwen3_5_frontend_test` is retained.
Frontend checks are not implied by the model-free host CI result. Real-model
execution and any default-suite failure must be reported separately.

Chat Completions supports the optional message field
`"ninfer_origin": "runtime_reminder"`, lowered to typed `MessageOrigin::RuntimeReminder`.
It is a caller assertion that a User message came from runtime instrumentation;
the server never guesses origin from substrings, quoted text or control tokens.
Only marked reminders are skipped when choosing the scoring query. Without a
mark, identical text keeps the previous last-User behavior. A runtime-only input
has no trusted query and uses the existing recency fallback. Invalid field values
or a non-User role fail explicitly. C++ callers may supply the same typed origin.
Clients must set it; existing unmarked requests do not acquire the improvement.
Responses and Anthropic adapters do not support the extension.

This changes neither Jinja input nor model tokens, thinking-history rendering,
or rewrite checkpoints. Within an open tool chain mandatory ranges are unchanged.
When sourced reminders follow a completed Assistant without a new real User, the
selected earlier task also keeps its bounded tail. That adds at most one bounded
User tail (plus page rounding) to the required union; it can displace scored
history or change admission at a capacity boundary. In particular, reminder contents
remain retained under #115, and complete System/Developer blocks, including rendered
tool schemas, still consume their
existing budget. Selecting a different query can change historical pages and
transfers; no latency or answer-quality improvement is established by span tests.
It does not change the integration's admitted speculative backends or establish factual
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
or change the engine pin. This permits preparation when the container checkout
and runner use different directory owners.

## Tool-output fixture modes and Basic framing

The legacy permissive-output fixtures `test_structured_tool_output`,
`test_tool_marker_after_quoted_marker` and `test_forced_tool_call` select
`Automatic` explicitly. Their omitted-empty, mismatch and recovery expectations
remain unchanged. The forced-call fixture includes an extra leading LF after the
prompt's existing LF, which is valid only on that permissive path.

The default `Basic` grammar and decoder remain tested by
`test_tool_constraints.cpp`. Canonical empty parameters have separate opening and
closing framing LF bytes. A single LF in the same position can be the beginning
of a nonempty untyped string, including literal parameter tags, even when the
schema declares Boolean. Inferring an empty value from its declared type would
destroy that legitimate mismatch string. The regression tests run the real grammar
and decoder in 1-byte, 7-byte and whole-call feed chunks, checking canonical empty
values followed by another argument and preservation of those literal bytes.
No single-LF empty repair is enabled; consumer schema validation still matters.

The constrained decoder also reports `marker_seen` and accumulates omitted-empty
and top-level type-mismatch normalization occurrences from completed, published
calls. Partial or unread calls do not contribute those counts. Repeated parameters
count each normalization occurrence even if a later value replaces it. These are
diagnostics, not complete schema validation, and parameter interpretation is
unchanged. `test_tool_diagnostics.cpp` covers 18 scenarios in three feed sizes,
including complete/partial and parallel calls, literal tags, strict codecs,
Automatic and forced continuation.

Chat schema tests follow the baseline's documented tool-selection contract:
selection is enforced during execution while prompt declarations remain stable;
required/named choices retain thinking and strict tool schemas remain strict.
Updating these older expectations changes no production tool-selection code.
