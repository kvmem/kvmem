# Lightweight chat UI

This is a small Svelte entry point for the pinned llama.cpp UI. It reuses the upstream Markdown/code renderer (including sandboxed HTML preview), input components, SSE record helpers, and Dexie conversation storage. It does not initialize the upstream router/agent/MCP/stream-resumption lifecycle.

Build from the repository root with Node.js 22 and npm:

```bash
python3 scripts/build-webui.py
```

The script exports the pinned `llama.cpp/tools/ui` into ignored `build-ui/`, applies the local entry points there, checks types, and writes static files to `build/share/kvmem/ui/`. Source archives without Git metadata use their bundled UI source. The llama.cpp working tree is not modified. UI history/settings use a separate `kvmem-chat` browser namespace.

Rebuild the server, start the usual IQ3/IQ4 recipe, and open `http://127.0.0.1:18200/`. No Node.js is needed on machines running a package that includes the built page. `--ui-dir PATH` overrides the static directory and `--no-ui` disables it. `scripts/prebuilt/package.py --ui-dir PATH` can include separately built assets.

The settings panel leaves unmodified parameters out of requests. Template reasoning effort and thinking-token budget are separate. The output limit includes reasoning. Histories are browser-local; normal continuation sends the complete history without `cache_reset`. The page does not execute tools. Stop closes the request; reloading does not resume an interrupted stream.

The speed displayed is generated tokens divided by the server's generation duration, including reasoning tokens but excluding prefill and post-generation cache commits. Very short replies are not throughput benchmarks.

## Runtime monitor

Open `/monitor/` on the same server. The lightweight chat header opens it in a new tab,
so monitoring does not interrupt a streamed answer. The page follows the Strata layout,
defaults to 极夜, and offers 蓝图 and system themes.

`scripts/build-webui.py` includes `ui/monitor/` in both lightweight and full builds.
To update only monitoring assets in an existing built UI (no Node.js needed):

```bash
python scripts/build-webui.py --monitor-only --output build/share/kvmem/ui
```

llama.cpp serves these files through `--ui-dir`. The lightweight chat entry is
specific to llama.cpp; for NInfer, build `--full-ui` into a separate output directory,
configure `-DNINFER_WEBUI_DIR=/absolute/path/to/that/output` and build `ninfer-serve`;
the static files are embedded in its executable. Both backends must be rebuilt to
provide `/api/monitor/snapshot`. The endpoint follows the existing API-key gate and
uses `Cache-Control: no-store`; the static page itself is public. Enter the key in
the page when prompted. It is stored only in that tab's `sessionStorage`.

The monitor keeps up to 100 finished request metadata records in server memory,
plus up to 512 active records; excess records are counted explicitly. It stores no
prompt, reply, sampling parameters or key. Restarting the server resets the history.
Streaming request progress is sampled from accepted-token and prefill observations;
NInfer aggregate responses expose final token/timing statistics when complete.

For image-bearing input, all three backends publish the optional `has_media`
request flag. The model-status card and request row show a short image-input
hint while the request is processing/reading and has not started generating.
Queued requests retain their queue label. This is an input-processing hint,
not encoder progress, a percentage, or a remaining-time estimate; it can include
image preparation, prefix reuse and text prefill. Video frames use the same hint.
NInfer registers media requests before preparation on Chat Completions, Responses
and Messages; llama.cpp reports the flag on its shared chat/responses path.
PolyStrata exposes an in-flight preparation record and reuses its identity when
generation starts, retaining its existing execution timing boundary. Preparation
records leave engine prefill metrics unavailable so previous requests' readings
are not attributed to the new input.
Finishing or failing a request removes the active hint. A stale snapshot pauses
the indicator and retains the existing connection warning. Older servers without
the flag keep their ordinary status labels.
NInfer history follows its generation lifecycle: authentication, protocol parsing,
and unknown-model failures before that lifecycle are not recorded as generation requests.
Live readings are not a benchmark. CPU/RAM telemetry supports Windows and Linux.
CPU usage is shown in its own card below the GPU section, with a percentage and
usage bar. It uses the existing hardware sampler on all three backends and covers
the whole system, including other applications. Missing or stale samples show
an unavailable value and an empty bar; connection failures mark the card stale.
NVIDIA telemetry uses the optional driver NVML library when the CUDA SDK headers
are available at build time. Unsupported fields are null, never substituted with
demonstration values. Device identity follows inference placement; monitoring does not
change that placement. Cache disk throughput is
derived from KVMem's read-byte counters, not whole-system disk I/O.

The browser polls without overlapping calls, reduces polling in hidden tabs, and
marks the last readings stale on failures. Server restarts clear chart history.
Monitoring has its own short state mutex; hardware sampling runs on a separate
thread and does not acquire model execution locks.

Verification: `node ui/test-monitor.cjs` (Playwright, optionally `CHROMIUM` and
`PLAYWRIGHT_MODULE`), `ctest -R kvmem-monitor-test`, and
`python tests/test_webui_builder.py`. Static layout/font attribution is distributed
with the page as `Strata-LICENSE.txt` and `Outfit-OFL.txt`.

`scripts/test_monitor.py --binary PATH --model PATH --ui-dir PATH --gpu GPU-UUID
--log PATH [--backend ninfer]` starts and stops its own disposable server. It checks
static assets, API-key enforcement, polling during generation, usage agreement,
rejections, client disconnects, and telemetry on the selected NVIDIA device.

`/monitor/` shares the current service origin and API authentication. It shows request statistics,
then the inference GPUs bound to the loaded model: a wide single-device card or two side-by-side
cards (stacked on narrow screens). Other physical GPUs appear under About → Hardware. The model's
`/props.execution.monitor_devices` supplies CUDA UUIDs for NInfer and physical PCI identities for
llama.cpp; these match NVML hardware samples, never numeric NVML indices or instantaneous load.
Missing binding information is explicit and does not fall back to displaying an arbitrary card.

Service state follows active requests independently of the selected historical request. Decode,
prefill, logical context and reused input tokens are request metrics; the output meter shows token
budget use, not completion. GPU readings describe the whole card. RAM/SSD history capacity remains
visible, with session counts, history hits/evictions and cache disk reads in expandable details.
Themes are 极夜 (default), 蓝图, and system. Fonts and assets are bundled locally.
The header language picker switches between 中文 and English without reloading or
resetting request selection and chart history. Language preference is stored locally;
Chinese remains the default. Labels, statuses, errors and accessibility text are
translated locally; model/device identities and original license texts are preserved.
Navigation is ordered Launch configuration → Monitor → About, with Chat on the right.
The toolbar's **Layout: Comfortable / Compact** selection applies to monitoring and launch
configuration and persists per browser, independently of theme and language. Comfortable
is the default. Compact reduces spacing, controls, gauges and chart heights; long request,
template and log lists scroll internally. Common field help can be expanded, while backend
limitations and warnings remain visible. Changing density does not reload data, reset edits,
or restart inference. Desktop compact mode places the single-GPU gauges beside their trends.
Logical context, GPU utilization and whole-card VRAM use 270-degree gauges with
numeric readings. Decode speed and GPU utilization retain their time-series charts.
Missing readings show an empty arc and an unavailable label; zero has no colored arc.

`ui/test-monitor.cjs` supplies model-free browser checks; set `MONITOR_FIXTURE_ONLY=1` to serve its
fixture on localhost:18412 for manual browser checks. `MONITOR_FIXTURE_FILE` can point to a JSON
scenario using `dual`, `idle`, `missing`, `stale`, `unknown`, `cpu`, or `failure` switches.
`scripts/test_monitor.py` verifies disposable real-model services, including physical device
identity with CUDA-visible remapping. These fixtures never belong in the built monitor directory.

## Independent local console (Windows)

The desktop monitor now includes **Launch configuration**: six essential settings,
five common-setting tabs, and a collapsed engine-specific section. It uses the full
window width, with process state and lifecycle actions in a right-hand column.
Low-frequency history controls are collapsed. The monitor uses the same theme and
language controls. Small windows stack the columns; expanded details may scroll.

The right sidebar has one built-in template: **NInfer · 0.18 resources + C**, retaining the resource settings of
`scripts/windows/start-ninfer.ps1` after release update `fd77613` (2026-10-13).
It uses context 204800, GPU history 36864, reserve 16384, RAM history 12288 MiB,
prefill 256, output limit 16384, INT8 KV, one concurrent request and one retained session.
It enables adaptive MTP with 4 drafts, N-gram with 31 drafts/minimum match 12,
CPU-resident vision with 1024 tokens, and thinking budget 8192. Sink/recent are
12288/1024. Current source builds use graph recall C (Mean-K page softmax plus query graph propagation, accumulated attention under pressure). Legacy LRFU/max controls are removed; old draft/template ranking values are omitted from launch requests. Published 0.18 binaries retain their original ranking.
The original `v0.18.0` Git tag predates the sink/recent release update.
The recipe requires a model with Vision/MTP support. Other built-in recipes were removed;
personal templates are unaffected. Model/worker paths are retained per backend;
the current port and GPU order are preserved, as is same-backend manual layer placement.
Applying the template only edits and autosaves the draft, with Undo available. It never
starts or changes the running service. The template is disabled if NInfer is unavailable.

Start the independent host from the repository root:

On Windows, double-click `START-HERE-CN.bat`; it checks the console files, chooses the bundled
`runtime/python/python.exe` (or installed 64-bit Python 3.9+ in a source checkout), and opens
the browser. Repeated clicks reuse only the same installation/backend/state directory's
running console. Other applications using that port are left running and reported as a
conflict and the startup window asks for another port. Press Enter or type `q` to cancel;
invalid or unavailable values can be entered again. `START-HERE-CN.bat -Port 18198` selects
an initial console port; `-Check` only checks
console files and Python, and `-NoBrowser` suppresses opening the browser.
See [offline console packaging](../docs/windows-one-click.md) for release preparation.
Use `START-HERE-EN.bat` for English startup/port/shutdown prompts and an English UI.
It reuses the same console and services. Its language hint overrides a previously saved
Chinese UI preference on opening; the header picker can then change and remember it normally.

The manual entry remains available:

```powershell
python scripts/local-console.py
```

Open `http://127.0.0.1:18199/monitor/`. This host serves the UI before a model is loaded
and remains available after inference stops. It requires Python and Windows PowerShell;
there is no new compiled-server dependency. Opening `/monitor/` directly on an inference
server continues to work, but its configuration tab explains how to open the local console.

The console uses `scripts/windows/start-backend.ps1 -DryRun` from its backend checkout
for validation and argument translation. It then starts that exact argument vector,
without a shell. A separate mainline checkout or unpacked worker installation can be selected:

```powershell
python scripts/local-console.py --backend-root C:\path\to\backend-checkout
```

The model field has a **Choose model** button that opens the local Windows file dialog,
filtered to `.ninfer` or `.gguf` for the selected backend. It fills the absolute path;
the model is never uploaded or copied through the browser. Cancellation keeps the old
selection. The eight most recent picker selections per backend are stored in this
browser's local storage and offered in a dropdown. Manual file-path input remains
available. The dialog opens on the controller's desktop, with one picker at a time;
the selection is automatically saved as an edit and never starts inference. The authenticated local mutation endpoint
validates the selected file and extension, and closes a picker left open for five minutes.

**Find models** shows filenames, adaptive file sizes and full paths for the selected backend
(`.ninfer`, not `.infer`; otherwise `.gguf`). Choose a local folder or paste its full path
(quoted Windows paths are accepted). Explicitly scanned folders are remembered in
`<state-dir>/model-directories.json`, independently of the browser origin/console port.
Up to 16 folders can be remembered, selected again or forgotten; forgetting never deletes files.
The results can be filtered by name/path. Empty files are marked and cannot be selected.
Other-format counts explain when a folder contains GGUF while NInfer is selected, or vice versa.

The default scan checks the current model folder, remembered folders, files beside the UI and
worker executables, and named `model`/`models` folders up to four ancestors above those locations.
Executable/workspace roots are shallow; only model libraries are traversed recursively.
It never loads a model, uploads data, or starts a service. Scans are bounded to six
subdirectory levels, 10,000 directory entries, 200 results and approximately three seconds
(individual filesystem calls can still block). The search scope shows scanned, covered,
partial, unavailable and unvisited locations. Time/entry/result/depth limits are distinguished.
Closing the dialog cancels its browser request; late responses cannot replace a later scan.
Drive roots/network shares, child symlinks/junctions, hidden dependency folders, named
`mmproj`/projector files and non-first GGUF shards are excluded. Filename matching is only
candidate discovery, not architecture/quantization compatibility validation.
Under **Templates & vision > Vision assets**, the **Vision projection GGUF** field
accepts any matching GGUF filename. Windows users can use **Choose file**; other
hosts can paste a path. For PolyStrata, an explicit path takes priority. When empty,
the launcher first checks `<model-stem>-mmproj.gguf` and `mmproj-<model-stem>.gguf`,
then falls back to a single adjacent GGUF whose name contains `mmproj` (case-insensitive).
Missing or multiple candidates require an explicit selection; `mmproj-BF16.gguf`
has no special priority. Static inspection uses the same selection rules.

The portable package includes `scripts/windows/select-model.ps1` for model/folder/template/projector
selection. Missing/failed/invalid native picker results produce readable guidance with path
pasting as a fallback; raw PowerShell errors are not shown. After copying new UI files over
a running older controller, restart guidance is shown instead of sending incompatible scan requests.

This console supports NInfer and llama.cpp. Only these two backends can be selected,
even when another checkout's launcher declares experimental engines. Unsupported sampling/template controls
are explicitly unavailable. Common controls include a single API key, served model name,
log level, custom chat-template file, default thinking mode and concurrency. They map to
each backend's existing native flags; NInfer's thinking-on override remains a chat-request
setting. llama.cpp currently exposes concurrency 1–4 on one GPU, NInfer 1–8.
Native options keep their original spellings, with a small
allowlist of numeric thread/GPU-offload options; arbitrary shell text and duplicate common
options are rejected. Both engines use the same GPU cards. Switching engines preserves the
selected UUIDs and their order; it never silently chooses another card. Multi-GPU launch is
currently exposed for NInfer layer splitting. When switching a multi-GPU selection to
llama.cpp, the cards keep that selection and explain that this launcher currently requires
one GPU for llama.cpp. Validation blocks preview/start until the user selects one card.
This is a limit of the current launcher integration, not a claim that llama.cpp lacks multi-GPU support.

Common resource, cache and service settings follow the current edits when switching engines,
including GPU selection, context/B/R/H, prefill/output budgets, port/listen address, concurrency,
SSD and LRFU settings, model alias and log level. Different supported ranges remain subject
to validation; a switch does not clamp a value or reset it to a default. Model/worker paths,
KV format, chat templates, feature overrides and native arguments keep separate in-page
drafts per backend, including incomplete native edits. Returning to NInfer reorders saved
layer counts by GPU UUID if only the order changed, or clears them with a message if the
selected devices changed. Controls have the same grouping and labels; native formats,
supported capabilities and explanatory limits can differ.

For llama.cpp, Cache & history has separate **K cache format** and **V cache format**
selectors. Both offer `q8_0`, `q5_0`, `q4_0`, `f16` and `f32`.
The two selectors sit side by side in the same grid cell occupied by NInfer's combined
KV selector. Format guidance uses the existing help popover; only invalid pairs show an inline error.
Quantized types can be combined with each other; the two floating-point types can be combined with each other.
The native KVMem worker rejects float/quantized mixing, so the UI explains and blocks
that combination without changing either selection. Independent values use `KvTypeK`
and `KvTypeV`, translated to `--cache-type-k` and `--cache-type-v`; no combined
`--kv-dtype` flag is emitted alongside them. The preview lists both formats.
Older drafts and templates with only `KvType` initialize both selectors with that value.
The launcher still accepts legacy `KvType` and uses it as the base when only one side
is overridden. NInfer keeps its native combined format selector and rejects independent
K/V fields. After updating an already running controller, restart it and refresh the UI
to enable the new fields; older controllers show restart guidance.

The adapter checkout can differ from the executable installation. Register absolute
executable paths by backend in `<state-dir>/worker-paths.json`, for example
`{"ninfer":"C:\\installed\\ninfer-serve.exe","llamacpp":"C:\\installed\\llama-kvmem-server.exe"}`.
An empty Worker field uses this registration. Without registration, the console searches
the selected backend root, its `bin/`, then `workers/<backend>/`; if the console installation
is different, it searches the same three locations there next. NInfer filenames are
`ninfer-kvmem-server.exe` (preferred) and `ninfer-serve.exe`; llama.cpp uses
`llama-kvmem-server.exe`. Placing an executable beside `START-HERE-CN.bat` therefore needs no
manual path. Search is limited to these names/locations and does not execute candidates.
The UI shows the detected path. An explicit Worker always takes precedence; a missing
explicit or registered file fails clearly, without silently choosing another build.

Edits are automatically saved in this browser, including incomplete inputs and GPU order.
The heading shows saving/saved status and offers retry on storage failure. Reloading restores
the browser draft; clearing site data removes it. The server-side v1 draft remains the
initial fallback and is available through the existing save API.
`Save as template` stores a named snapshot with an optional description under **My templates**.
Personal templates include machine-specific paths and settings except API keys, and support apply/undo,
rename, explicit update from the current draft, and confirmed deletion. They are stored in
this browser alongside edits, not shared across browsers. Built-in recipes remain separate.
`Check configuration` opens a dialog with model, GPU order/layers, cache budgets, resolved
executable and port availability. Errors link back to relevant fields where identifiable.
The canonical PowerShell command is collapsed by default and can be copied. Checking does
not load a model or prove model compatibility/resource fit; it never starts inference.
`Start service` validates again, rejects an occupied port or a second managed process,
and displays loading/ready/exited states and the log tail. `Stop service` terminates only
the owned Windows Job process tree, unloading the model and its in-memory history. The
job also closes on controller exit. A gated child is assigned to the job before spawning
the worker, so wrappers cannot create unowned descendants. No existing service is adopted
or stopped, and there is no implicit restart. Draft and active configuration are separate.
The service card shows an indeterminate progress bar while the launch request is pending
and while waiting for backend health readiness. It shows completion only after the health
check succeeds; failures/exits stop the animation and expose the error or log guidance.
No loading percentage is inferred. The Chat link remains in the top-right toolbar and
is disabled until the managed service is ready, so availability changes do not move navigation.
The console proxies only four read-only monitoring endpoints; chat opens on the worker's
own origin. The controller remains loopback-only, validates Host/Origin/fetch-site, requires
a per-run token on mutations, and serves only an explicit static-file allowlist.
The inference service's Listen address can be `127.0.0.1` (this computer, default) or
`0.0.0.0` (all IPv4 interfaces). With all interfaces selected, other devices use this
computer's IP address and the configured port. Preview and launch check availability on
the selected listen address. Health checks, monitor proxying and the local Chat link
continue to use `127.0.0.1`; the console itself still listens only on loopback.

The persisted v1 console format is `{schema_version: 1, parameters: {...}}`, using typed
launcher parameter names. It is a bridge to the existing launcher, not an implementation
of the separate proposed `common/backends` public configuration schema. Unexposed settings
are omitted so native defaults survive; unsupported keys/types fail rather than being
silently ignored. API keys remain only in page/controller/worker memory: drafts, personal
templates and launch-plan files exclude them; previews and log responses redact them.
Reloading clears the input, so restarting requires re-entry. Monitoring automatically
authenticates to the owned service; external clients supply the same key themselves.

The primary resource row contains context, GPU history budget B and generation reserve R;
RAM history budget H is under Cache & history. R is append KV workspace per active request,
not an output-length setting. Both backends reuse that workspace with resident-only
Pressure. Retained sessions counts idle histories only.
Help appears from the small information buttons on hover or keyboard focus. Click to pin,
press Escape or click elsewhere to dismiss; opening help does not move the form.

Cache & history → Advanced exposes the Query scorer (page-softmax or Mean-Q
cosine), GQA mean/max aggregation, graph strength, LRFU, and the Query/replay
token limits. These controls are independent; selecting a scorer does not toggle
graph propagation or history. Alpha zero disables propagation. The default C
recipe uses page-softmax/mean, alpha 0.2 and no LRFU. Legacy signed-cosine
templates migrate to explicit controls; disabled LRFU remains explicitly false.
History decay and reward fraction retain their draft values when inactive.

Check configuration queries the selected worker's model-free
`--kvmem-capabilities` contract with GPU visibility disabled. Older workers or
launch scripts produce an update message before model startup. An offline
package checks its controller, script and page hashes together on startup, so
partial file replacement cannot silently mix console versions.

Runtime details and monitoring show the native startup report and whether it
matches the launch plan, including the actual scorer, graph strength, LRFU and
Query/replay limits. Editing a draft cannot change this report. When native
logging is disabled or the report is incomplete, the UI waits for confirmation.

Feature switches disable only dependent controls. Disabled values stay in browser drafts
and personal templates but are omitted from preview/start payloads. SSD controls its path
and budget; LRFU controls decay and candidate fraction, while score aggregation stays
independent. Vision controls image budget and encoder placement; thinking-off disables
the thinking budget. NInfer also exposes default reasoning effort and preservation of
closed-turn assistant reasoning. Thinking-off disables and omits the effort override
while retaining its draft value; preserving past reasoning is independent of the current
thinking switch. These two controls are currently NInfer-only. The JSON bridge preserves
typed native arrays; its native-option allowlist is unchanged by the direct script's
`WorkerArgsFile` support.

Draft method selects Off, MTP or DFlash2 according to the backend capability
profile. Disabled draft settings retain their values while being omitted from
preview/start payloads. Existing `MtpDrafts`/`AdaptiveMtp` templates migrate to
`SpeculativeMode`/`DraftTokens`/`DraftPolicy`. The profile controls draft limits,
fixed/adaptive support, concurrency and required companion assets; a compatible
model is still required. N-gram zero disables minimum match. Fast prefill depends
on KV format, and layer allocation needs
at least two GPUs. llama.cpp concurrency above one disables SSD while keeping its saved
values. Controls unsupported by a backend remain unavailable when other switches change.
The SSD enable switch is browser state only; draft controls map to native worker options.

Checks: `python tests/test_local_console.py`, `python tests/test_webui_builder.py`,
`node ui/test-monitor.cjs`, and `node ui/test-launcher.cjs`. The launcher browser check
uses a disposable controller and state directory, and loads `ui/test-launcher-settings.cjs`
for isolated, model-free checks of help, dependencies, active preview payloads, credentials
and draft/template restoration. These settings checks fixture device discovery and preview
responses; native argument translation is checked separately by
`powershell -File scripts/windows/test-backend-launcher.ps1` and the console unit tests.
Optional `TEST_MODEL`, `TEST_WORKER`,
and `TEST_GPU` enable a real llama.cpp start/generate/monitor/stop check. `BACKEND_ROOT`
selects its adapter checkout, and `CONSOLE_TEST_PORT` defaults to 18431 (worker uses +1).
`CHROMIUM` and `PLAYWRIGHT_MODULE` select the locally installed browser/runtime.

## Chat regression checks

After a lightweight build, `node ui/test-chat-props.cjs` checks the built page
against both native `/props` contracts without loading a model. `UI_DIST` may
select another built directory. NInfer uses the standard model alias and generation
settings when llama.cpp extensions are absent; unreported thinking defaults remain
server-owned. This browser check also runs in the lightweight CI job.
Chat throughput uses the native `predicted_per_second` value. Single-token replies
and missing or nonpositive rates display only the output token count, since a
clamped duration cannot establish a valid decode rate.

Run against a disposable test service, not an active conversation. The HTTP check needs a model that can answer simple arithmetic, with a generation reserve of at least 128. The browser check uses up to 512 generated tokens and requires Playwright's Chromium or `CHROMIUM` pointing to a compatible installation.

```bash
KVMEM_UI_TEST_URL=http://127.0.0.1:18400 python3 scripts/test_webui.py
KVMEM_UI_TEST_URL=http://127.0.0.1:18400 node ui/test-browser.mjs
python3 scripts/test_start_server.py
```

Browser tests check real generation, local history after reload, code rendering, stop/continuation, and the absence of unsupported backend requests. Local image upload/follow-up was also checked on IQ3 27B with GPU Q8 vision and MTP3/ReplaySSM.
