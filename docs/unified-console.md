# Unified local console

The one-click console serves launch configuration, chat and monitoring from the same
local origin. Its default addresses are `/monitor/` and `/chat/` on port 18199;
the inference API still runs separately, normally on port 18200. Use the console's
Chat tab after starting a model. Switching tabs preserves an active response.

The chat reads settings from the running service, displays reported values, and
stores explicit overrides per conversation and model. Thinking, its token budget,
reasoning effort and history preservation follow each backend's supported controls.
Unreported defaults are labeled as unreported. While the service is starting or
running, launch settings show its `active_config` and are read-only, including
model/GPU selection, native arguments and template actions. Tabs and help remain
available. A successful stop or process exit unlocks the last run's parameters
for editing and restarting; a failed stop keeps them locked. A pending start or
unconfirmed service state also keeps editing locked. Browser drafts cannot replace
the running configuration, and API keys are never returned by the status endpoint.

Completed messages show the native average decode rate, output count and stop
reason. A one-token reply or an absent/nonpositive native rate has no displayed
decode speed. The details show native prefill/reuse counts and client wait times.
Live message rates require an exact request match: PolyStrata supplies a client ID,
and llama.cpp/NInfer attach the native response ID to their monitor rows. Request
interval rates use native output-commit counters; global engine rates are never
substituted for a chat-request rate.

Histories remain in the browser. Import and backup support KVMem, llama-ui and
PolyStrata exports. Imported IDs are stable, branches and reasoning are retained,
and unsupported tool/attachment data remains available in the backup. The chat
does not execute imported tool workflows. The old origin's export page reads its
storage without deleting it. In KVMem mode, PolyStrata's bundled chat and monitor
routes are disabled; its inference APIs, settings and history export remain available.

All views share the console theme, density and language. The mobile history and
settings drawers support Escape, keyboard focus containment and focus return.
Recent-model selectors show filenames with a distinguishing parent suffix when
necessary; the full path remains in the value and tooltip.

The monitor keeps readings and exceptional states visible. Metric definitions are
available from the adjacent information buttons in readable popovers; runtime
settings and cache details expand on demand. Device-assignment warnings remain
visible when assignments are unknown or no GPU is used.

Context and default output limits share the resource row. Recall policy has a
dedicated tab for scoring, graph propagation, LRFU and retention budgets; scoring,
graph propagation and LRFU remain independently selectable. PolyStrata alone has
an **Elastic experts** tab with off, workspace reuse and VRAM borrowing modes.
These controls round-trip through the existing native JSON arguments without
discarding unrelated flags. Switching engines preserves their individual settings.
The monitor displays configured MTP draft limits and policy, with request-level
draft counts only when the backend reports them.

NInfer's **Recall policy → Resident & session retention** section also provides
**Checkpoint extra recompute limit · tokens**. The default 0 disables nearby
checkpoint optimization; a positive value permits extra recomputation from a
legal earlier checkpoint. Active Query replay states remain protected. This is
not a latency limit: future retries can recapture a full state image or read SSD.
Saved JSON uses `CheckpointExtraRecomputeTokens`; Windows scripts accept
`-CheckpointExtraRecomputeTokens`, and Linux exports preserve the corresponding
worker environment. Enabling it requires worker capability
`checkpoint_recompute_revision >= 1` and `KVMEM_CHECKPOINT_MIN_GAP=0`.
See [checkpoint policy](checkpoint-policy.md#ninfer-checkpoint-recompute-limit)
for capture, cleanup and budget semantics. Other engines do not expose this setting.

NInfer's **Hardware & acceleration** tab offers **Tensor parallelism · heads**
after selecting at least two GPUs. It is off by default; multiple GPUs otherwise
use layer splitting. Optional **TP weights** follow the displayed GPU UUID order,
for example `3,1`. Leave them empty for native automatic allocation. These are
relative weights, not exact head counts or memory limits. Moving a GPU moves its
weight, while changing the selected GPUs clears the weights. TP disables layer
counts, and budget auto-adjustment currently remains single-GPU only. Per-backend
drafts retain the TP choice and weights across reloads and backend switches.

The launcher always emits `--tensor-attention heads`; positions mode is not
exposed. Starting or exporting a TP configuration requires a matching worker
reporting integer `heads_tensor_parallel_revision >= 1`. Static capability
descriptions and a dry-run plan describe launcher support; they do not qualify an
older installed worker. Update the complete NInfer package if the capability
check fails. Runtime details display the service's actual mode, weights and KV
head ranges; missing actual allocation is not inferred from requested weights.

An output limit of **0 / unlimited** in the launcher removes the service's fixed
output cap for all three backends. Requests that omit a limit inherit that setting;
Chat Completions `max_tokens: -1` explicitly removes a fixed service default.
The effective budget is resolved after tokenizing the prompt, against the remaining
logical context. KVMem's resident Budget + Reserve window does not cap total output;
Pressure can recycle it during generation. Explicit positive request limits still
apply, and logical context and RAM/SSD history capacity checks remain in force.
NInfer also bounds an automatic output budget by the complete history that fits in
its configured RAM/SSD capacity; a shorter history budget does not reject an otherwise
valid short request merely because the maximum context is larger. Explicit positive
caps and prompts that cannot fit retain their capacity errors.
Thinking tokens count toward total output. Native API zero-value conventions remain
backend-specific; the launcher's zero is translated into each backend's no-cap setting.

## Backend logs

The **Backend logs** tab (`/monitor/#logs`) reads the controller's shared stdout /
stderr capture for llama.cpp, NInfer and PolyStrata. It supports text search,
in-place highlighting / plain text, line wrapping, pause, automatic following and export.
Scrolling up disables following; **Jump to latest** restores it. The view shares
the monitor language, theme and density. Log contents are always text, never HTML.
Original prefixes, indentation and blank lines stay in one text stream. Explicit level
labels, timestamps, module tags and performance metrics receive subtle highlighting;
no timestamp or severity is invented. ANSI control sequences and carriage-return
progress are normalized for display.

`GET /api/launcher/logs` returns at most 256 KiB of complete recent output, plus
process/file identity and a revision. A matching `?revision=` returns `text: null`.
The page displays at most 2,000 lines; export saves the current cached
original snapshot (including known-key redactions), not the entire file. It polls
once per second only while visible and unpaused, with no overlapping requests and
bounded retry backoff. It does not query the inference worker. A stopped process's
last unterminated line is retained; a live incomplete line waits for a delimiter.
Stopping retains the captured output and its backend identity. Starting a new
worker resets the viewer to the new file, matching the existing file lifecycle.
After restarting the controller, an old file is labeled as a previous run if its
engine identity is unavailable.

For an already-running older controller, the view falls back to the existing
`/api/launcher/log` endpoint and explicitly shows its **32 KiB** limit. This allows
static UI updates without stopping that controller's owned inference process.
The modern endpoint is used on the next console restart. A native worker serving
only the monitor page has no access to the controller's process output; the log
view explains that the local console is required.

Model-free validation: `python tests/test_console_logs.py`,
`node ui/test-log-parser.cjs`, and `node ui/test-logs.cjs` (Playwright; accepts
`PLAYWRIGHT_MODULE`, `PLAYWRIGHT_BROWSERS_PATH`, `CHROMIUM`, `SCREENSHOT_DIR`).

## Building and packaging

Build the lightweight UI with `python scripts/build-webui.py` before packaging.
The manifest must declare `console_chat_protocol: 1`. The full upstream UI is
still available through `--full-ui`, but is not the unified console chat bundle.
`package-console.ps1 -ChatUiDir <directory>` can supply the built lightweight UI;
worker packages can provide it in `workers/share/kvmem/ui`. The one-click check
rejects missing or incompatible chat assets before opening the console.

## Validation and limits

Core coverage includes cancellation during prefill, disconnect/error handling,
service identity, native settings and thinking translation, historical imports,
offline packaging, and built-browser checks for all three backend settings
contracts. CI runs these alongside the existing console and monitor tests.

Earlier runtime validation on an RTX 5060 Ti used Qwen3.5 0.8B for llama.cpp,
Qwen3.5 4B MTP Q4 for NInfer and Tiel Coder 35B-A3B Q4 for PolyStrata. Streaming,
thinking, follow-up requests and monitoring passed. A short warmed 64-token
native/console/console/native comparison observed decode changes of -1.2%, -0.02%
and +3.0%, respectively, with about 20–30 ms extra time to first output through
the console. This does not establish cold-start or sustained-load performance.

This UI change does not alter native C++ inference code. Expert-profile persistence
is a separate pending integration item; initial expert-cache adaptation and system
memory pressure can still cause slow starts. Batch MTP and multi-GPU PolyStrata
remain outside the current integration.

## Shared system disk telemetry

Install console dependencies with `python -m pip install -r scripts/requirements-console.txt`.
The psutil collector supports Windows, Linux and macOS; this does not imply that the
existing Windows service launcher is portable. Windows distributions bundle psutil.

`GET /api/launcher/hardware` serves one cached system-wide sample to all monitor clients
and all three backends, even when no inference service is running. The collector uses
cumulative psutil disk read/write bytes and monotonic elapsed time, approximately once
per second. Values are bytes/s, rendered as MiB/s. They include other processes, unlike
the separate backend cache I/O counters. Engine metrics remain backend-owned.
First samples, unavailable counters, reset counters and expired readings display a dash.
The API reports its scope, status, sample time and actual interval. Unsupported or
permission-restricted OS counters are unavailable, never fabricated zeroes.
Standalone/older workers without this console endpoint retain native disk readings.
New controller code takes effect on console restart; replacing UI files alone cannot
activate the collector inside an already-running Python process.

On Windows, the console also supplies CPU/RAM and AMD GPU telemetry when Radiance
has no native hardware sampler. GPU readings are joined by PCI address to the
physical HIP UUIDs captured at launch, with the worker's rank/ordinal report checked
against that mapping. A device name or high utilization alone never establishes a
binding. The read-only helper uses WDDM/DXGI/PDH, with supported ADLX metrics from
the driver DLL in System32; no additional SDK is required. These are whole-adapter
readings and include other processes. ADLX uses AMD's compatibility initialization
entry when available: some drivers return an empty GPU list from the normal entry.
The helper initializes ADLX once per process and checks metric support before reading.

The helper is isolated from HTTP handlers, restarts after a crash or timeout, and
exits when its owning console closes. Samples older than five seconds are discarded.
Missing metrics remain unavailable: in particular, WDDM's power percentage is not
converted into watts. Total board power is shown only when ADLX reports that metric.
The monitor indicates partial readings when a driver omits any of these GPU metrics.

## Native metric details

The monitor expands existing public metrics without estimating missing counters.
Cache details include cumulative disk read/write bytes, operation counts, errors,
history misses/swaps/transfers, checkpoints, resident pages and staging memory when
reported. Native diagnostics are collapsed by default and retain their open state
across polls. NInfer groups expose request/queue state, capacity, occupancy, CUDA
residency, context-cache transfers/pressure, scheduler, batches, cumulative counters,
mean throughput since startup, host work, uptime and optional expert statistics.
llama.cpp conversation details retain each slot's native scope without summing shared
conversation counters. PolyStrata request details show expert hit rate (native 0..1),
speculative proposals/accepts and timing fields. Missing groups stay hidden.

The chat bridge matches llama.cpp's native_request_id to the SSE response ID;
ambiguous matches are rejected. Cumulative decode_tps is mapped to the mean, and only
a fresh request-specific live window is shown as real time. NInfer's global 5-second
rate is never used as a chat-request rate. Native SSE timing and speculative counters
remain usable without a monitor-ID match. Speculative counts are not labeled MTP-only
because an engine can include other draft algorithms. Completed-message details show
native decode duration and draft proposals/accepts.

The serving adapters publish native response IDs, phase durations, finish reasons and
actual speculative counters directly into the bounded monitor records, including
requests from external API clients. NInfer's execution TTFT is admission-to-first-token;
`ttft_request_ms` separately retains the native preparation-and-queue-inclusive value.
No-token outcomes have unavailable TTFT. The prefill duration follows the native
`prompt_ms` phase convention. Live and final request means share NInfer's existing
completion-timing calculation, including its first-token interval boundary.

NInfer also reports a per-request approximately 5-second window from cumulative native
commits and native generation time. The monitor prefers this structured sample; older
workers can still supply an explicitly engine-scoped operational-log rate. llama.cpp
retains its approximately 3-second request window. First incomplete windows, stopped
requests and expired samples are unavailable. The main monitor and completed chat
messages show actual MTP accepts/proposals; NInfer copy drafts are accounted separately.
Transport failure overrides a prior normal generation result in monitoring.

Native worker and controller updates take effect on restart. A new NInfer worker can
be installed beside the active executable and DLLs and selected by the existing worker
map, so updating files does not interrupt an already-running service.
