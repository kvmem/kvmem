# NInfer sparse task retention

An active User task can age out of the recent window while the assistant consumes
long tool results. A later User-role runtime reminder can also replace its retrieval
query. Old history can then win the scored selection and exclude the current task.

With KVMem enabled, source-backed instruction and active User ranges are mandatory
pages, independently of retrieval scores. B is the selected-token budget, R is the
generation reserve, and W=B+R is the physical window. The rolling recent setting is
independent of R. Mandatory and scored pages share B; this change adds no attention
capacity or StateImage kinds.

| Region | Policy |
|---|---|
| Caller System/Developer content | Complete rendered content, including custom role remapping |
| Template control text | System/Developer layout blocks |
| Exclusively sourced generated tool JSON | Eligible for scored retrieval; not mandatory |
| Nonempty User content in the open task | Complete rendered content, including long messages |
| Images, sink, recent tail | Existing required pages |
| Unmodified-client queries | First and latest eligible User in the open group, deduplicated; at most two |
| Explicit runtime-origin queries | Last eligible User, skipping explicitly sourced reminders |

A completed non-tool Assistant closes a task; explicit continuation keeps it open.
Every active User is protected even when it is not a query. Ordinary human corrections
remain eligible. Empty or whitespace-only Users add no mandatory task range. Tool
contents and quoted ChatML do not acquire User authority. The existing treatment of
User-role tool-response wrappers is retained. Raw-token prompts have no invented
chat-task semantics.

Custom templates may merge or repeat typed Users. Their proven rendered extent is
retained conservatively. Source-coordinate unions must cover the original text field,
allowing only leading/trailing Unicode whitespace to be trimmed. Incomplete coverage,
opaque transformations or invalid coordinates require the complete rendered input to
be mandatory. This can reject long custom-template requests. The guarantee concerns
rendered context; information discarded or changed by a template cannot be recovered.
The separated-query exclusion applies to the maintained contiguous chat layout;
custom repeated/spliced layouts may fall back to the primary query or use a coarse
rendered extent.

Pins use original logical page coordinates and overlapping pages are charged once.
Admission checks the required union through the maximum admitted generation frontier.
Fully resident requests can use W; requests that can overflow must fit B, including
speculative scratch requirements. An oversized mandatory set returns a context-length
error before device mutation. It is not silently shortened to a 512-token tail.

Queries are scored independently and their rankings interleaved within the same
historical budget. Tool/Assistant gaps do not enter query sums. The physical scratch
still has three allocated query slots; the default uses at most two. Changed query
sets or pins can reject late checkpoints and require earlier replay. Checkpoints
compare their immutable query ranges/counts, pin membership, geometry and probe phase,
rather than treating the current mutable history directory as the captured identity.

No client fields are required. Chat Completions optionally accepts
`messages[i].ninfer_origin="runtime_reminder"`, lowered to the typed origin. It is a
caller assertion, not content-based author detection. Unmarked reminders after a
completed answer and runtime text embedded within one User cannot be identified
reliably. Responses and Anthropic continue to use ordinary role structure.

The Linux host CI prepares and verifies the pinned backend, then runs:

```sh
python3 scripts/prepare-backends.py --backend ninfer
python3 scripts/prepare-backends.py --backend ninfer --check
python3 tests/test_ninfer_context_contract.py
```

This compiles the production page selector, source-proof interpreter and immutable
checkpoint helper without CUDA or model weights. The checks cover adversarial old
history scores; complete long-task pages at 511/512/513/4096-token and alignment
boundaries; overlap, rolling growth and one-page capacity deficits; source mutation
and serialization separators; and independent query/frontier/pin-membership oracles.

The prepared backend's Frontend checks run separately:

```sh
ninfer_tests ninfer_qwen3_5_frontend_test --memory-context-only
ninfer_tests ninfer_qwen3_5_frontend_test
```

They cover Frontend-to-page complete User retention, merged/repeated/partial/opaque
layouts, independent User and instruction fallbacks, whitespace admission, control
provenance, media/graft offsets, covering BPE/NFC boundaries and marker independence.
The default full Frontend suite retains the legacy `Automatic` recovery fixtures;
`Basic` canonical-empty and literal-tag framing remains tested independently.
The integrated upstream parser also accepts compact non-strict empty framing
without consuming a subsequent parameter. Strict raw strings keep their literal
markup. An ambiguous non-strict string beginning with a closing tag followed by
the next parameter framing is therefore interpreted as parameter boundaries;
This matches Automatic's existing boundary interpretation. A stop token inside
open reasoning closes that block and continues the
reply; a turn-level stop after reasoning still ends it. The focused Frontend modes
`--tool-output-only` and `--reasoning-stop-only` exercise these contracts.

For a running two-lane server configured as documented in the HTTP test:

```sh
python3 tests/test_ninfer_task_retention_http.py --server http://127.0.0.1:18087 --output retention-results.json
```

The 22 retrieval checks cover system/task instructions, reminder changes, human
corrections, historical fact mapping, semantic tool selection, local tool-result
continuation, cache reuse and lane markers. The harness validates generated tool names
and arguments, then appends a fixed result message; no tool is executed. Two cases
tracked separately as model-output selection problems can be included
with `--include-output-selection`; the fixed model also fails them with full attention.
They are excluded explicitly from retrieval qualification, not marked as passed.
Lane markers do not prove scheduling overlap; performance logs must establish it.

The synthetic performance entry uses the same server geometry and a fresh service
per run, with `--request-log-jsonl events.jsonl`. It primes two requests, measures
one lane, then runs two simultaneous 1,536-token requests:

```sh
python3 tests/test_ninfer_retention_performance_http.py --server http://127.0.0.1:18087 --request-log events.jsonl --stderr-log server.log --output performance.json
```

Run eight isolated processes in old/new/new/old, new/old/old/new order. The entry
requires five successful requests, exact synthetic request hashes, two running
lanes with average Decode batch at least 1.9, three full 1,536-token outputs,
at least two history swaps in both C1/C2 measurements and no request errors or
engine recoveries. The optional stderr check detects logged transfer failures;
it does not enable transfer verification or require exact-verification records.
`--audit-existing-results rows.json` verifies existing rows
and logs without new inference; omit `--server` in that mode. Small samples and
run-to-run variance do not establish statistical equivalence or a general speedup.

For caller-provided real inputs, use a fresh service per version with C2,
max-context=262144, FP8 KV, NoSpec, prefill=1024, B=32768/R=4096, recent=5120,
sink=128, Host=16384 MiB and transfer verification:

```sh
python3 tests/test_ninfer_real_replay_http.py --server http://127.0.0.1:18087 --request case1.json --request case2.json --output real-results.json
```

Inputs must request streaming and be expected to finish with stop. This replays
unchanged fields as canonical JSON, once fresh and once warm, and
checks transport, stop termination and nonzero warm reuse. Private inputs and
responses must stay private; semantic review must be bound to the actual response
hashes. The published numeric page shapes include both original JSON key order
and the sorted wire used in replay. Tool-parameter key order can change rendered
tokenization even when the JSON objects are semantically equivalent. These shapes
test page geometry, not reconstruct source attribution or prove GPU selections.

Sparse retrieval can miss a needed tool schema or historical fact. Complete tasks
can consume all remaining scored-history space or increase capacity rejections.
Mixed-source JSON loses its whole-tool tag conservatively and may also lose a native
ToolBoundary cache marker, reducing prefix reuse even with KVMem off. Message bytes
and model tokens are not intentionally rewritten, but cache metadata, replay, selected
attention pages and transfers can change. No universal answer equivalence or speedup
is promised by host tests.

This fixed integration supports KVMem ordinary generation and its existing MTP paths.
KVMem plus DFlash2 is rejected by both serving and Engine validation. Dense DFlash2
is a separate supported path. Sparse DFlash2 requires target position, verify-width
and restored draft-context adaptations; removing the guards does not implement them.
The original `tests/models/qwen3_5/test_engine_dflash2_real.cpp` can also be linked
as a standalone test against the prepared backend libraries. The qualified dense
configuration is:

```sh
NINFER_TEST_ARTIFACT="$MODEL" "$DF2_TEST" 7 1 1 2 int8 0 1
```

This tests K=7, one optimized CUDA graph, C2, Int8 KV, Vision off and state capture
on, including budgets, seed reproduction, graft/page boundaries, prefix reuse and
rollback. It covers short contexts (up to 2304 in that test), not sparse DF2 or
long-context speculative performance. A reported `state_h2d=0` does not qualify
actual Host-state H2D restoration.
Host checks do not establish GPU Vision, long-history speculative generation, full
external-tool workflows or factual correctness of real-world replay inputs.
