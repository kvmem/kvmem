# Native startup memory recovery recommendations

After a single-GPU NInfer KVMem capacity-admission failure, the console displays
three configurations: **INT8, NVFP4 and RK8V4**. The warning panel is immediately
below the service error, above runtime details. Parameters and the explicit
**Apply and restart** actions remain visible; estimation details are expandable.
No recommendation is applied automatically.

These console actions currently support Windows. Linux keeps its generated-revision
and systemd launch flow; the native planning report remains available to CLI callers.

The **Auto adjust** button beside Budget is a separate proactive action. Its
hover/focus `i` help explains that it keeps the selected reserve, KV type, prefill
and other settings, loads weights once, computes the maximum budget with headroom,
and fills only Budget. It does not start a server. Unlike OOM recovery, it does not
cap reserve at 4K or limit the result to the old budget. A zero/oversized current
budget is replaced only in the planning request by the legal sink/recent minimum.
The returned maximum remains bounded by Context minus the effective Reserve.
Planning uses the same effective Reserve and Prefill as Check settings and displays
any alignment or capping with the result. Blank fields use the normal launcher's
defaults (currently Reserve 1024 and Prefill 128). The action fills only Budget.
The UI shows elapsed time, rejects a result if settings changed while
waiting, and displays errors beside Budget. Running services and concurrent
planning block the action. Currently supported: ordinary single-GPU NInfer
Qwen3.5-family KVMem with device-resident weights.

`POST /api/launcher/adjust-budget` validates the normal launch plan, checks worker
`budget_planning_version: 1`, and starts the native `--plan-budget` action in an
owned Windows job with a 180-second worker timeout. It returns a validated
`KVMEM_BUDGET_PLAN` result and elapsed milliseconds without saving configuration or
replacing the failed launch. The browser applies/saves Budget only after checking
that its request still matches the editing state. Shutdown/stop cancels planning
and releases its worker. No API key is sent to this non-serving action.

Cards are ranked by the available candidates' planned budgets. Options reaching
**32K (32,768 tokens, inclusive)** come first, ordered by the chosen KV precision
priority **INT8 > RK8V4 > NVFP4**. Remaining available options are ordered by
descending budget, with the same precision priority breaking ties. Unavailable
options appear last and cannot be recommended. Thus, when every available budget
is below 32K, the largest budget wins. The first available card is highlighted,
marked **Recommended**, and explains the choice in one sentence. This is a UI
ranking policy; it does not alter native estimates or automatically apply settings.

## Native planning contract

The Qwen3.5-family Engine still owns the loaded Model when runtime capacity is
rejected. It invokes the existing `make_sequence_planner` separately for each
candidate format and budget, without constructing another Program, loading
weights, allocating candidate GPU buffers or running additional CUDA Graph probes.
The resulting reservation includes native persistent storage, draft/state pools,
workspace, bookkeeping and the candidate's static graph allowance.

`runtime/engine/memory_recovery.h` searches page-aligned budgets using those native
quotes. The policy is:

- Reserve = min(original effective Reserve, 4096); smaller reserves are retained.
- Prefill = min(original effective Prefill, new Reserve), aligned down to 128.
- Budget respects the native sink/recent minimum, 64-token pages, context and
  concurrency. It cannot exceed the original effective Budget.
- Keep max(128 MiB, 5% of post-weight available capacity) as additional headroom
  for every format. There is no extra penalty for changing formats.
- Preserve FastPrefill for INT8 and RK8V4; disable it for NVFP4.
- If a format cannot be planned or even its minimum budget does not fit, report
  that format as unavailable without reducing Reserve again.

The terminal admission exception appends a bounded `KVMEM_MEMORY_PLAN` JSON
report with schema version 1, original effective settings, failed required/free
bytes, three candidates, headroom and total planning microseconds. It remains
part of the same FATAL message. Recommendation errors do not replace the original
startup error. This path is limited to ordinary single-device Qwen3.5-family
KVMem admission; early weight-allocation failures may have no plan.

This replaces the console's old KV-payload subtraction heuristic. The console
does not derive storage sizes from model geometry or invent numbers when a
worker lacks the native report. The reader validates the report against the
terminal admission evidence and the launcher's effective settings, then checks
all three options atomically for bounds, alignment, format, reserve/prefill,
FastPrefill and required bytes plus headroom. Missing or invalid reports produce
an explanatory message. A recovered probe OOM cannot override a later unrelated
terminal failure.

## Retry ownership

`memory_diagnosis` is cached for one failed launch instance. The retry endpoint
accepts the instance, option ID and current configuration. It rejects stale
instances, edited settings, running workers and unavailable formats. It applies
only validated changes using normal inspection and launch validation, retaining
private API keys and unrelated settings. Clean/manual exits and failures after
readiness do not get startup recommendations.

The UI shows native planning time. It is the calculation time for all candidate
formats, excluding earlier model loading, existing startup graph probing,
process teardown and browser polling. Graph memory is a static allowance;
successful startup does not establish long-context peak-memory safety or quality.

## Validation (2026-10-29)

The final integration is based on master after the Linux launcher and DFlash2
layer-pipeline changes. It preserves the generated Linux launch flow and packages
`console_memory.py` on both platforms; Windows exposes the interactive actions.

- 103 Python checks cover proactive planning, recovery, the controller, model
  inspection and packaging. Focused launcher-plan, packaging and CI classifier
  checks also pass. Both browser suites pass, including the 32K ranking boundary,
  all three retry choices, settings changed during planning, blank reserve/prefill,
  unchanged explicit reserve/KV, failure display, languages and mobile layout.
- A real Windows launcher plan resolves omitted Reserve to 1024 and omitted Prefill
  to 128. A simulated GPU quote verifies successful budget-only responses for each
  omitted field and both together. Controller and browser tests also verify that
  explicit values are normalized for planning and that the form retains those inputs.
- The complete NInfer patch chain passes the project's verifier. The changed
  runtime, static planning entry point, serve main and host test compile and link
  against the existing same-toolchain libraries. The host test verifies the largest
  fitting page and rejection of the next page, context limits, concurrency, fixed
  reserve/format/prefill and insufficient capacity. No CUDA objects were rebuilt.
- Earlier real-device qualification on RTX 5060 Ti with
  `Qwen3.8-27B-IQ3_S-vision-mtp-dflash2-q3.ninfer`, DFlash2 fixed 5 and Prefill 256:
  proactive INT8/R=8192 returned B=34112 in 8.094 seconds, RK8V4/R=4096 returned
  B=50560 in 8.125 seconds, and NVFP4/R=4096 returned B=73280 in 8.203 seconds.
  RK8V4 then started and returned a short response. These check the capability;
  they are not a controlled format-performance comparison.
- Earlier OOM recovery qualification started all three recommended formats and
  completed short responses. An installed-console attempt while an external test
  occupied the GPUs returned no fitting budget and preserved the input. Once idle,
  the real button returned B=38208 in 8.3 seconds with Reserve 4096 and INT8 retained.
- The previous review found an unhandled missing-field error for blank Reserve or
  Prefill. The controller now validates explicit overrides against the canonical
  plan and accepts omitted defaults. Initial sandbox-only filesystem/loopback
  test failures passed under normal local permissions. A previous link attempt
  exposed an existing output-session source/library mismatch; the additive static
  planning entry point has a separate translation unit and reuses the compatible
  generation libraries.

The final integration was not subjected to another GPU campaign. The native
planner and its boundary-test sources match the earlier GPU-qualified implementation;
startup and short inference do not establish long-context peak-memory safety or
quality. The normal Ninja dry run requests CMake regeneration, so this focused
validation did not alter the existing build tree or its dependency records.

Reproduce the controller regressions with:

```sh
python -m unittest discover -s tests -p 'test_auto_budget.py'
python -m unittest discover -s tests -p 'test_console_memory.py'
```

Browser coverage is included in `npm test --prefix ui`. The native host target is
`ninfer_memory_recovery_test` after preparing the pinned NInfer patches.
