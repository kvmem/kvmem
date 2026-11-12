#Requires -Version 5.1
# Add the offline, one-click console to a worker package (or a new console-only folder).
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$OutputDir,
    [Parameter(Mandatory)][string]$PythonArchive,
    [Parameter(Mandatory)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$PythonSha256,
    [Parameter(Mandatory)][string]$PsutilWheel,
    [Parameter(Mandatory)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$PsutilSha256,
    [string]$SourceDir,
    [string]$ChatUiDir,
    [ValidateSet('all','ninfer','radiance')][string]$Backend='all'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
# Get-FileHash is a module function in Windows PowerShell. Load this host's own
# module when invoked from Python or PowerShell 7 with an inherited module path.
Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Utility/Microsoft.PowerShell.Utility.psd1') -Force
if (!$SourceDir) { $SourceDir = Join-Path $PSScriptRoot '../..' }
$source = (Resolve-Path -LiteralPath $SourceDir).Path
$archive = (Resolve-Path -LiteralPath $PythonArchive).Path
$output = [IO.Path]::GetFullPath($OutputDir)
$installedUi = Join-Path $output 'workers/share/kvmem/ui'
$copyChat = !(Test-Path -LiteralPath (Join-Path $installedUi 'index.html'))
if (!$copyChat) { $ChatUiDir = $installedUi }
elseif (!$ChatUiDir) {
    $ChatUiDir = Join-Path $source 'build/share/kvmem/ui'
    if (!(Test-Path -LiteralPath (Join-Path $ChatUiDir 'index.html'))) { $ChatUiDir = Join-Path $source 'ui/dist' }
}
$chatManifest = Join-Path $ChatUiDir 'ui-build.json'
if (!(Test-Path -LiteralPath $chatManifest) -or (Get-Content -LiteralPath $chatManifest -Raw | ConvertFrom-Json).console_chat_protocol -ne 1) {
    throw 'Unified chat assets are required. Build with scripts/build-webui.py and supply -ChatUiDir.'
}
if ($copyChat -and (Test-Path -LiteralPath (Join-Path $output 'ui/dist'))) { throw 'Chat destination already exists: ui/dist' }
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ine $PythonSha256) {
    throw 'Python archive SHA-256 does not match the release publisher supplied checksum'
}
$files = @('START-HERE-CN.bat', 'START-HERE-EN.bat', 'scripts/local-console.py', 'scripts/console_chat.py', 'scripts/console_radiance.py', 'scripts/console_state.py', 'scripts/windows/start-ui.ps1',
           'scripts/windows/describe-launch-json.ps1', 'scripts/windows/start-backend.ps1',
           'scripts/windows/start-ninfer.ps1', 'scripts/windows/start-dflash2.ps1','scripts/windows/start-polystrata.ps1','tools/polystrata/server.py','tools/polystrata/assets.py',
           'scripts/windows/native-process.ps1', 'scripts/windows/query-scoring.ps1', 'scripts/windows/token-settings.ps1', 'scripts/windows/select-model.ps1', 'scripts/speculative-capabilities.json')
$files += 'scripts/model_health.py', 'scripts/console_hardware.py', 'scripts/console_memory.py', 'scripts/requirements-console.txt'
$files += 'scripts/launcher-capabilities.json'
$files += 'scripts/check-recall-worker.py', 'scripts/windows/recall-strategy.ps1'
$files += 'backends/radiance/tools/inspect_rad.py', 'backends/radiance/tools/draft_control.py'
foreach ($name in '__init__.py', 'config.py', 'plan.py', 'radiance.py', 'export.py', 'systemd.py', 'worker.py', 'recall.py') {
    $files += 'scripts/launcher/' + $name
}
foreach ($name in '__init__.py', 'reader.py', 'schema.py', 'layouts.py', 'formats.py', 'framing.py', 'file_io.py', 'LICENSE', 'PROVENANCE.json') {
    $files += 'scripts/model_health_ninfer/' + $name
}
foreach ($name in 'index.html', 'monitor.css', 'monitor.js', 'i18n.js', 'launcher.css', 'launcher.js',
                  'Strata-LICENSE.txt', 'Outfit-OFL.txt', 'outfit-latin-wght.woff2', 'console-shell.js', 'console-shell.css', 'history-export.html', 'logs.js', 'logs.css') {
    $files += 'ui/monitor/' + $name
}
if ($Backend -eq 'ninfer') {
    $files = @($files | Where-Object { $_ -notmatch '(^tools/polystrata/|/start-(dflash2|polystrata)\.ps1$)' })
}
# Validate all inputs and reserved destinations before adding anything to a worker package.
foreach ($relative in $files) {
    if (!(Test-Path -LiteralPath (Join-Path $source $relative) -PathType Leaf)) { throw "Missing source: $relative" }
    $destination = Join-Path $output $relative
    if (Test-Path -LiteralPath $destination) {
        if (!(Test-Path -LiteralPath $destination -PathType Leaf) -or
            (Get-FileHash -LiteralPath $destination).Hash -ne (Get-FileHash -LiteralPath (Join-Path $source $relative)).Hash) {
            throw "Refusing to replace an existing package file: $relative"
        }
    }
}
foreach ($relative in 'runtime/python', 'console-build.json', 'START-HERE.txt', 'licenses/Python-LICENSE.txt') {
    if (Test-Path -LiteralPath (Join-Path $output $relative)) { throw "Destination already exists: $relative" }
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($archive)
try {
    $names = @($zip.Entries | ForEach-Object FullName)
    if ('python.exe' -notin $names -or 'LICENSE.txt' -notin $names -or
        !@($names | Where-Object { $_ -match '^python\d+\.zip$' }).Count -or
        !@($names | Where-Object { $_ -match '^python\d+\._pth$' }).Count) {
        throw 'Supply the official Windows x64 embeddable Python ZIP, not an installer or virtual environment'
    }
    foreach ($entry in $zip.Entries) {
        # The official archive is flat. Reject links/traversal and unexpected nested payloads.
        if ($entry.FullName -match '[/\\:]' -or $entry.FullName -in @('.', '..')) {
            throw 'Unexpected path in the embeddable Python archive'
        }
    }
} finally { $zip.Dispose() }
$runtime = Join-Path $output 'runtime/python'
$null = New-Item -ItemType Directory -Path (Split-Path $runtime) -Force
[IO.Compression.ZipFile]::ExtractToDirectory($archive, $runtime)
$python = Join-Path $runtime 'python.exe'
$version = @(& $python -I -S -c 'import sys,ctypes,json,http.server,urllib.request,webbrowser; assert sys.version_info >= (3,10); assert sys.maxsize > 2**32; print(sys.version.split()[0])')
if ($LASTEXITCODE -ne 0) { throw 'Bundled Python failed its x64 standard-library check' }
& $python -I -S (Join-Path $source 'scripts/install-console-dependency.py') --wheel $PsutilWheel --sha256 $PsutilSha256 --runtime $runtime
if ($LASTEXITCODE -ne 0) { throw 'Cannot install the verified psutil wheel' }
& $python -I -c 'import psutil; assert psutil.__version__.split(chr(46)) == list(map(str, (7, 2, 2))); print(psutil.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'Bundled psutil failed its import check' }
foreach ($relative in $files) {
    $destination = Join-Path $output $relative
    $null = New-Item -ItemType Directory -Path (Split-Path $destination) -Force
    if (!(Test-Path -LiteralPath $destination)) { Copy-Item -LiteralPath (Join-Path $source $relative) -Destination $destination }
}
$null = New-Item -ItemType Directory -Path (Join-Path $output 'licenses') -Force
if ($copyChat) {
    $null = New-Item -ItemType Directory -Path (Join-Path $output 'ui') -Force
    Copy-Item -LiteralPath $ChatUiDir -Destination (Join-Path $output 'ui/dist') -Recurse
}
Copy-Item -LiteralPath (Join-Path $runtime 'LICENSE.txt') -Destination (Join-Path $output 'licenses/Python-LICENSE.txt')
$readme = @'
KVMem - start here

Double-click START-HERE-EN.bat for English startup instructions and an English UI.
START-HERE-CN.bat keeps the original Chinese startup instructions.
Both entries ask for the console IP and port; press Enter to keep the defaults.
Use 127.0.0.1 for this computer, or 0.0.0.0 for LAN access through this computer's IP.
Matching address/port and installation reuse the existing console and its services.
Python is bundled; no Python, pip, Node.js or build-tool installation is needed.
Keep the console window open. Closing it stops services started by this console.
The window prints shutdown instructions in the selected startup language.
Ctrl+C also closes it. Stop service in the UI stops inference while keeping the console
available; closing only the browser leaves services running. Separately started backends
are unaffected.
Repeated clicks reopen this installation's running console without restarting its model.

In Launch configuration, choose a backend, a local model and a GPU. Check configuration,
then Start service. Open Chat when the service is ready. Edits do not change a running model.
Models and inference workers must be included separately; the console does not download them.
Put ninfer-kvmem-server.exe, ninfer-serve.exe or llama-kvmem-server.exe beside START-HERE-CN.bat,
or in bin or workers/<backend>, with their runtime dependencies and chat UI.
Leave Backend executable empty for detection. Manual paths and worker-paths.json take
precedence. Configuration checks query the selected worker's model-free launch contract.
If no default worker is found, choose its executable in Engine-specific settings.
Use Find models to scan nearby model libraries, or choose/paste a model folder.
Explicitly scanned folders are remembered locally, including across console port changes.
The dialog shows search locations, incomplete scans, other formats and empty files.
Filter results by name or path, then select a model. This does not start inference.
If the Windows folder picker is unavailable, paste a full folder path instead.
Both engines use the same GPU cards. Switching engines preserves common resource,
cache and service settings; model assets, KV formats and native options remain separate.
This launcher currently exposes single-GPU llama.cpp and multi-GPU NInfer configuration.
A multi-GPU selection is preserved on switch to llama.cpp, with guidance to choose one
GPU before launching. The launcher does not silently choose a different card.
For llama.cpp, Cache & history provides independent K and V cache format selectors.
Quantized q8_0/q5_0/q4_0 types can be combined, as can f16/f32 types. Float/quantized
mixing is rejected. Existing combined KV selections initialize both selectors.
After updating files over a running console, close its startup window, restart it,
then refresh the browser to use the updated controller settings.
Enable MTP under Hardware & acceleration to use speculative decoding on supported
backends/models. Turning it off retains draft length, adaptive MTP and N-gram choices
while keeping them inactive. Re-enabling restores them; an initial draft length is 4.
Use an English path for the installation and models. Install the driver and VC++ runtime
required by the selected worker. Read the worker's validation report for supported GPUs.

Default console: http://127.0.0.1:18199/monitor/
Default inference service: http://127.0.0.1:18200/
For another console port: START-HERE-CN.bat -Port 18198
English entry with another console port: START-HERE-EN.bat -Port 18198
If the port is unavailable, the startup window asks for another port. Enter a number
from 1 to 65535, or press Enter/type q to cancel. The browser opens the selected address.
This only changes the console port; the inference port remains a separate UI setting.
For a diagnostic check: START-HERE-CN.bat -Check
To skip opening a browser: START-HERE-CN.bat -NoBrowser

The console controls only services it starts. It never replaces a running worker.
API keys are not stored in drafts or templates. Browser settings remain browser-local.
'@
if ($Backend -eq 'ninfer') {
$readme = @'
KVMem - NInfer

Double-click START-HERE-EN.bat (English) or START-HERE-CN.bat (Chinese).
The package includes Python, the console and Chat. No Python or Node.js installation is required.
Select a local .ninfer Qwen dense model and a GPU, check the configuration, then Start service.
Open Chat after the service is ready. Models are supplied separately.
Keep the startup window open. Closing it stops only the services it started.
Use START-HERE-EN.bat -Check for diagnostics, or -Port 18198 for another console port.

双击 START-HERE-CN.bat，选择本机 .ninfer Qwen 稠密模型和显卡，检查配置后启动。
服务就绪后打开聊天页面。Python、控制台、聊天界面均已内置，模型另行提供。
请保持启动窗口开启；关闭窗口会停止它启动的服务。
诊断：START-HERE-CN.bat -Check；更换控制台端口：START-HERE-CN.bat -Port 18198。

GPU and model qualification: provenance/validation.json.
'@
}
$utf8 = [Text.UTF8Encoding]::new($false)
if ($Backend -eq 'radiance') {
$readme = @'
KVMem - Radiance Windows console

Double-click START-HERE-EN.bat (English) or START-HERE-CN.bat (Chinese).
Select radiance-kvmem-server.exe, a .rad model, an AMD GPU, RadianceHome and ROCm path.
The runtime and model are supplied separately. Set ROCM_PATH to the installed SDK before
opening the console. Windows text inference is verified on RX 9070 XT (gfx1201), ordinary
Qwen3.8-27B IQ3R v4, DFlash7, one GPU and one sequence, FP8 KV and F32 state.
Keep the startup window open. Closing it stops only services it started.
Diagnostics: START-HERE-EN.bat -Check

双击 START-HERE-CN.bat，选择 Radiance 引擎、.rad 模型、AMD 显卡及运行库路径。
模型和原生运行库另行提供。启动控制台前设置 ROCM_PATH。Windows 已验证 RX 9070 XT 单卡单会话文本推理，
配置为普通 Qwen3.8-27B IQ3R v4、DFlash7、FP8 KV 和 F32 状态。
'@
}
[IO.File]::WriteAllText((Join-Path $output 'START-HERE.txt'), $readme, $utf8)
$hashes = @{}
foreach ($relative in $files) { $hashes[$relative] = (Get-FileHash -LiteralPath (Join-Path $output $relative)).Hash.ToLowerInvariant() }
$manifest = @{ schema_version=1; launch_contract_version=3; python_version=[string]$version[-1];
    console_chat_protocol=1; chat_manifest_sha256=(Get-FileHash -LiteralPath $chatManifest).Hash.ToLowerInvariant();
    python_archive_sha256=$PythonSha256.ToLowerInvariant(); psutil_wheel_sha256=$PsutilSha256.ToLowerInvariant(); offline_console=$true;
    models_included=$false; controller_files=$hashes;
    supported_backends=[string[]]$(if($Backend -ne 'all'){@($Backend)}else{@('llamacpp','ninfer','polystrata','radiance')}) }
[IO.File]::WriteAllText((Join-Path $output 'console-build.json'), ($manifest | ConvertTo-Json -Depth 5), $utf8)
Write-Output "One-click console added: $output"
