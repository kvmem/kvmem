#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$LlamaWorker,
    [Parameter(Mandatory)][string]$NinferWorker,
    [Parameter(Mandatory)][string]$NinferSource,
    [Parameter(Mandatory)][string]$OutputDir,
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$VcpkgInstalled,
    [string]$ValidationReport,
    [string]$PythonArchive,
    [string]$PythonSha256,
    [string]$PsutilWheel,
    [string]$PsutilSha256,
    [string]$LlamaUiDir,
    [string]$PolystrataWorker,
    [string]$PolystrataVisionWorker,
    [string]$PolystrataPythonRuntime,
    [string]$PolystrataGgmlSource,
    [string]$PolystrataValidationReport,
    [string]$LlamaSource,
    [ValidateSet('all','ninfer')][string]$Backend='all'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native-process.ps1')
$null = Get-Command dumpbin -ErrorAction Stop
$source = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if ($Backend -eq 'all' -and !$LlamaWorker) { throw 'The all-backend package requires LlamaWorker' }
if ($Backend -eq 'ninfer' -and ($LlamaWorker -or $PolystrataWorker)) { throw 'NInfer-only packages accept only NinferWorker' }
if ($Backend -eq 'ninfer' -and $PythonArchive -and !$LlamaUiDir) { throw 'Supply LlamaUiDir with the built unified chat assets' }
if(!$LlamaSource) { $LlamaSource=Join-Path $source 'backends/llamacpp' }
$LlamaSource=(Resolve-Path -LiteralPath $LlamaSource).Path
if(!(Test-Path -LiteralPath (Join-Path $LlamaSource 'LICENSE') -PathType Leaf)) {throw 'LlamaSource must identify the source used to build LlamaWorker, including its license'}
if ([bool]$PythonArchive -ne [bool]$PythonSha256) { throw 'Supply both PythonArchive and PythonSha256 for the offline console' }
if ($PythonArchive -and (!$PsutilWheel -or !$PsutilSha256)) { throw 'Supply PsutilWheel and PsutilSha256 for the offline console' }
if (!$PolystrataWorker -and ($PolystrataVisionWorker -or $PolystrataPythonRuntime -or $PolystrataGgmlSource -or $PolystrataValidationReport)) {throw 'PolyStrata component options require PolystrataWorker'}
if ($PolystrataWorker) {
    if ($PythonArchive) {throw 'With PolyStrata, use PolystrataPythonRuntime (including frontend dependencies) instead of the standard-library PythonArchive'}
    if (!$PolystrataPythonRuntime) {throw 'PolystrataWorker requires PolystrataPythonRuntime with frontend dependencies'}
    foreach($path in @($PolystrataWorker,(Join-Path $PolystrataPythonRuntime 'python.exe'))) {
        if(!(Test-Path -LiteralPath $path -PathType Leaf)) {throw "Missing PolyStrata worker or portable runtime: $path"}
    }
}
if (($PythonArchive -or $PolystrataWorker) -and !$LlamaUiDir) {
    $LlamaUiDir = Join-Path (Split-Path (Split-Path ([IO.Path]::GetFullPath($LlamaWorker)))) 'share/kvmem/ui'
}
if ($LlamaUiDir) {
    if (!(Test-Path -LiteralPath (Join-Path $LlamaUiDir 'index.html') -PathType Leaf)) { throw 'Supply LlamaUiDir with built chat assets for the llama.cpp worker' }
    if ($PythonArchive -or $PolystrataWorker) {
        $uiManifest = Join-Path $LlamaUiDir 'ui-build.json'
        if (!(Test-Path -LiteralPath $uiManifest) -or (Get-Content -LiteralPath $uiManifest -Raw | ConvertFrom-Json).console_chat_protocol -ne 1) { throw 'Rebuild the unified console chat assets with scripts/build-webui.py' }
    }
    $LlamaUiDir = (Resolve-Path -LiteralPath $LlamaUiDir).Path
}
$NinferSource = (Resolve-Path -LiteralPath $NinferSource).Path
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $OutputDir) { throw 'OutputDir must be new' }
if (!$CudaPath -or !(Test-Path -LiteralPath (Join-Path $CudaPath 'EULA.txt'))) { throw 'CUDA Toolkit and EULA required' }
$workerEntries = [Collections.Generic.List[object]]::new()
if ($Backend -eq 'all') { $workerEntries.Add(@('llamacpp', $LlamaWorker, 'llama-kvmem-server.exe')) }
$workerEntries.Add(@('ninfer', $NinferWorker, 'ninfer-kvmem-server.exe'))
foreach ($entry in $workerEntries) {
    if (!(Test-Path -LiteralPath $entry[1] -PathType Leaf)) { throw "Missing worker: $($entry[1])" }
}
foreach ($dir in @('scripts/windows', 'licenses', 'provenance') + @($workerEntries | ForEach-Object { "workers/$($_[0])" })) {
    $null = New-Item -ItemType Directory -Path (Join-Path $OutputDir $dir) -Force
}
$workerInfo = @{}
foreach ($entry in $workerEntries) {
    $workerBackend = $entry[0]; $inputFile = (Resolve-Path -LiteralPath $entry[1]).Path
    $destination = Join-Path $OutputDir "workers/$workerBackend"
    $executable = Join-Path $destination $entry[2]
    Copy-Item -LiteralPath $inputFile -Destination $executable
    $search = @((Split-Path $inputFile), (Join-Path $CudaPath 'bin'), (Join-Path $CudaPath 'bin/x64'))
    if ($VcpkgInstalled) { $search += Join-Path $VcpkgInstalled 'x64-windows/bin' }
    $search += $env:PATH.Split(';')
    $queue = [Collections.Generic.Queue[string]]::new(); $queue.Enqueue($executable)
    $seen = @{}
    # Backends may load local plugins at runtime, outside the PE import table.
    foreach ($dll in Get-ChildItem -LiteralPath (Split-Path $inputFile) -Filter '*.dll' -File) {
        $copy = Join-Path $destination $dll.Name
        Copy-Item -LiteralPath $dll.FullName -Destination $copy
        $seen[$dll.Name] = $true; $queue.Enqueue($copy)
    }
    while ($queue.Count) {
        $file = $queue.Dequeue()
        $dependencies = & dumpbin /nologo /dependents $file
        if ($LASTEXITCODE -ne 0) { throw "dumpbin failed: $file" }
        foreach ($line in $dependencies) {
            if ($line -notmatch '^\s+([\w.+-]+\.dll)\s*$') { continue }
            $name = $Matches[1]
            if ($seen.ContainsKey($name)) { continue }; $seen[$name] = $true
            if ($name -match '^(api-ms-|ext-ms-)' -or
                (Test-Path -LiteralPath (Join-Path $env:SystemRoot "System32/$name"))) { continue }
            $found = $null
            foreach ($dir in $search) {
                if (!$dir) { continue }
                $candidate = Join-Path $dir $name
                if (Test-Path -LiteralPath $candidate -PathType Leaf) { $found = $candidate; break }
            }
            if (!$found) { throw "Unresolved dependency $name from $file" }
            $copy = Join-Path $destination $name
            Copy-Item -LiteralPath $found -Destination $copy; $queue.Enqueue($copy)
        }
    }
    $info = New-KVMemProcessInfo $executable @('--help')
    $info.EnvironmentVariables['PATH'] = "$env:SystemRoot/System32;$env:SystemRoot"
    $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
    $process = [Diagnostics.Process]::Start($info)
    $stdout = $process.StandardOutput.ReadToEndAsync(); $stderr = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    [IO.File]::WriteAllText((Join-Path $OutputDir "provenance/$workerBackend-help.txt"), $stdout.Result + $stderr.Result)
    if ($process.ExitCode -ne 0) { throw "$workerBackend failed with a clean runtime PATH" }
    $process.Dispose()
    $info = New-KVMemProcessInfo $executable @('--kvmem-capabilities')
    $info.EnvironmentVariables['PATH'] = "$env:SystemRoot/System32;$env:SystemRoot"
    $info.EnvironmentVariables['CUDA_VISIBLE_DEVICES'] = '-1'
    $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
    $process = [Diagnostics.Process]::Start($info)
    $stdout = $process.StandardOutput.ReadToEndAsync(); $stderr = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw "$workerBackend does not support the console launch contract" }
    $contract = $stdout.Result | ConvertFrom-Json
    if ($contract.schema_version -ne 1 -or $contract.backend -ne $workerBackend -or
        $contract.query_scoring_revision -lt 2 -or !$contract.independent_graph_lrfu -or !$contract.signed_lrfu_hits) {
        throw "$workerBackend has an incompatible console launch contract"
    }
    [IO.File]::WriteAllText((Join-Path $OutputDir "provenance/$workerBackend-capabilities.json"), $stdout.Result)
    $process.Dispose()
    $workerInfo[$workerBackend] = @{ sha256=(Get-FileHash -LiteralPath $executable).Hash.ToLowerInvariant();
        path="workers/$workerBackend/$($entry[2])"; clean_path_help_passed=$true; launch_contract=$contract }
}
$packageScripts = @('start-backend.ps1', 'start-ninfer.ps1', 'native-process.ps1', 'query-scoring.ps1', 'token-settings.ps1')
if ($Backend -eq 'all') { $packageScripts += @('start-dflash2.ps1','start-polystrata.ps1') }
foreach ($script in $packageScripts) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $script) -Destination (Join-Path $OutputDir 'scripts/windows')
}
if ($Backend -eq 'all') {
    New-Item -ItemType Directory -Path (Join-Path $OutputDir 'tools/polystrata') -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $source 'tools/polystrata/server.py') -Destination (Join-Path $OutputDir 'tools/polystrata/server.py')
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../speculative-capabilities.json') -Destination (Join-Path $OutputDir 'scripts/speculative-capabilities.json')
Copy-Item -LiteralPath (Join-Path $source 'docs') -Destination (Join-Path $OutputDir 'docs') -Recurse
Copy-Item -LiteralPath (Join-Path $source 'VERSION') -Destination (Join-Path $OutputDir 'VERSION')
$packageReadme = @'
# KVMem Windows workers

This package contains llama.cpp and ninfer workers, their runtime dependencies and
the common launcher. Models are supplied separately. Install the NVIDIA driver
and Microsoft Visual C++ 2015-2022 x64 Runtime before starting a worker.

See [launch and backend options](docs/multi-backend.md) and
[active RAM/SSD storage](docs/active-kv-tiering.md). Use `-Describe` to inspect
launcher capabilities, and `-DryRun` to inspect the exact worker arguments.

```powershell
./scripts/windows/start-backend.ps1 -Backend ninfer -Model 'D:/models/model.ninfer' -DryRun
```

`-HostMiB`, `-DiskPath` and `-DiskMiB` configure shared native KV RAM and process-local
SSD storage for either backend. SSD contents live only for the worker lifetime;
they cannot restore a session after restart or transfer it between backends.

`BUILD-INFO.json` records the source versions and dependency-loading checks.
`provenance/validation.json`, when present, records the actual runtime acceptance
scope. Dependency packaging alone does not qualify every model or configuration.
'@
if ($Backend -eq 'ninfer') {
$packageReadme = @'
# KVMem Windows - NInfer

This package contains only the NInfer worker and its runtime dependencies.
Use converted .ninfer Qwen dense model artifacts, supplied separately.
Install a compatible NVIDIA driver and Microsoft Visual C++ 2015-2022 x64 Runtime.
The included console exposes only NInfer. See START-HERE.txt for startup and
provenance/validation.json for the qualified GPU, build and model scope.

本包仅包含 NInfer 后端。模型另行提供，使用转换后的 .ninfer Qwen 稠密模型包。
安装兼容的 NVIDIA 驱动和 Microsoft Visual C++ 2015-2022 x64 运行库后，
双击 START-HERE-CN.bat。Python、控制台和聊天界面均随包提供。

TP restrictions and deferred features are recorded in
docs/ninfer-upstream-63428ac2-integration.md. Models are not downloaded automatically.
'@
}
[IO.File]::WriteAllText((Join-Path $OutputDir 'README.md'), $packageReadme, [Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath (Join-Path $CudaPath 'EULA.txt') -Destination (Join-Path $OutputDir 'licenses/CUDA-EULA.txt')
Copy-Item -LiteralPath (Join-Path $LlamaSource 'LICENSE') -Destination (Join-Path $OutputDir 'licenses/llama-MIT.txt')
Copy-Item -LiteralPath (Join-Path $source 'README.md') -Destination (Join-Path $OutputDir 'licenses/KVMem-README.md')
Copy-Item -LiteralPath (Join-Path $NinferSource 'LICENSE') -Destination (Join-Path $OutputDir 'licenses/ninfer-LICENSE')
foreach ($tree in @(@('ninfer-third-party', (Join-Path $NinferSource 'third_party')),
                   @('llama-vendor', (Join-Path $LlamaSource 'vendor')))) {
    foreach ($file in Get-ChildItem -LiteralPath $tree[1] -Recurse -File) {
        if ($file.Name -notmatch '^(LICENSE|COPYING|NOTICE)') { continue }
        $relative = $file.FullName.Substring($tree[1].Length).TrimStart('\')
        $dest = Join-Path $OutputDir "licenses/$($tree[0])/$relative"
        $null = New-Item -ItemType Directory -Path (Split-Path $dest) -Force
        Copy-Item -LiteralPath $file.FullName -Destination $dest
    }
}
if ($VcpkgInstalled) {
    foreach ($file in Get-ChildItem -LiteralPath (Join-Path $VcpkgInstalled 'x64-windows/share') -Filter copyright -Recurse -File) {
        $dest = Join-Path $OutputDir ('licenses/vcpkg/' + $file.Directory.Name + '.txt')
        $null = New-Item -ItemType Directory -Path (Split-Path $dest) -Force
        Copy-Item -LiteralPath $file.FullName -Destination $dest
    }
}
if ($ValidationReport) { Copy-Item -LiteralPath $ValidationReport -Destination (Join-Path $OutputDir 'provenance/validation.json') }
if ($LlamaUiDir) {
    # llama.cpp resolves share/kvmem/ui relative to the worker's grandparent directory.
    $null = New-Item -ItemType Directory -Path (Join-Path $OutputDir 'workers/share/kvmem') -Force
    Copy-Item -LiteralPath $LlamaUiDir -Destination (Join-Path $OutputDir 'workers/share/kvmem/ui') -Recurse
}
if ($PythonArchive) {
    & (Join-Path $PSScriptRoot 'package-console.ps1') -OutputDir $OutputDir -PythonArchive $PythonArchive -PythonSha256 $PythonSha256 -PsutilWheel $PsutilWheel -PsutilSha256 $PsutilSha256 -Backend $Backend
    [IO.File]::AppendAllText((Join-Path $OutputDir 'README.md'), "`n## Desktop console`n`nDouble-click START-HERE-EN.bat for English, or START-HERE-CN.bat for Chinese startup instructions. Python is bundled. See START-HERE.txt for details.`n", [Text.UTF8Encoding]::new($false))
}
if ($PolystrataWorker) {
    & (Join-Path $PSScriptRoot 'package-polystrata.ps1') -ComponentOnly -OutputDir $OutputDir `
        -Worker $PolystrataWorker -VisionWorker $PolystrataVisionWorker -PythonRuntime $PolystrataPythonRuntime `
        -CudaPath $CudaPath -GgmlSource $PolystrataGgmlSource -ValidationReport $PolystrataValidationReport
    $polyExe=Join-Path $OutputDir 'workers/polystrata/strata-poly.exe'
    $info=New-KVMemProcessInfo $polyExe @('--help')
    $info.EnvironmentVariables['PATH']="$env:SystemRoot/System32;$env:SystemRoot"
    $info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
    $process=[Diagnostics.Process]::Start($info)
    $stdout=$process.StandardOutput.ReadToEndAsync();$stderr=$process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    [IO.File]::WriteAllText((Join-Path $OutputDir 'provenance/polystrata-help.txt'),$stdout.Result+$stderr.Result)
    # The pinned native CLI deliberately returns 2 for --help as well as usage errors.
    $polyHelpExit=$process.ExitCode
    if($polyHelpExit -notin @(0,2) -or ($stdout.Result+$stderr.Result) -notmatch 'usage: strata-poly --model' -or ($stdout.Result+$stderr.Result) -notmatch '--kvmem-budget') {throw 'PolyStrata failed with a clean runtime PATH'}
    $process.Dispose()
    $workerInfo['polystrata']=@{sha256=(Get-FileHash -LiteralPath $polyExe).Hash.ToLowerInvariant();path='workers/polystrata/strata-poly.exe';clean_path_help_passed=$true;help_exit_code=$polyHelpExit;architecture='qwen35moe';batch_mtp=$false}
    [IO.File]::AppendAllText((Join-Path $OutputDir 'README.md'), "`n## PolyStrata`n`nThis package also contains the single-GPU qwen35moe PolyStrata backend and its portable Python frontend. Start the common console with START-HERE-EN.bat or START-HERE-CN.bat. Select a model GGUF and GPU; an omitted Tokenizer is exported locally from that GGUF on first start. No model download occurs. The model's matching mmproj is required for vision. See provenance/polystrata-validation.json for measured scope and known limitations.`n", [Text.UTF8Encoding]::new($false))
}
$manifest = @{ version=(Get-Content -LiteralPath (Join-Path $source 'VERSION') -Raw).Trim();
    workers=$workerInfo; models_included=$false; kv_cross_backend_transfer=$false;
    offline_console_included=([bool]$PythonArchive -or [bool]$PolystrataPythonRuntime);
    backend_sources=(Get-Content -LiteralPath (Join-Path $source 'backends/versions.json') -Raw | ConvertFrom-Json);
    runtime_validation='See attached validation report; packaging certifies dependency loading only' }
if ($Backend -eq 'ninfer') {
    $manifest.backend='ninfer'
    $manifest.backend_sources=@{ninfer=$manifest.backend_sources.ninfer}
}
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText((Join-Path $OutputDir 'BUILD-INFO.json'), ($manifest | ConvertTo-Json -Depth 32), $utf8)
$hashes = foreach ($file in Get-ChildItem -LiteralPath $OutputDir -Recurse -File | Sort-Object FullName) {
    (Get-FileHash -LiteralPath $file.FullName).Hash.ToLowerInvariant() + '  ' +
        $file.FullName.Substring($OutputDir.Length + 1).Replace('\', '/')
}
[IO.File]::WriteAllText((Join-Path $OutputDir 'SHA256SUMS'), ($hashes -join "`n") + "`n", $utf8)
Write-Output "Prepared unpacked package: $OutputDir"
