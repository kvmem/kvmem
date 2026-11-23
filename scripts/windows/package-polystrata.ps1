#Requires -Version 5.1
[CmdletBinding()]
param([Parameter(Mandatory)][string]$Worker,[string]$VisionWorker,
      [Parameter(Mandatory)][string]$OutputDir,[string]$PythonRuntime,
      [string]$CudaPath=$env:CUDA_PATH,[string]$ValidationReport,[string]$GgmlSource,
      [string]$ChatUiDir,
      [switch]$ComponentOnly)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$poly=Join-Path $root 'backends/polystrata'
$out=[IO.Path]::GetFullPath($OutputDir)
if((Test-Path -LiteralPath $out) -and !$ComponentOnly) { throw 'OutputDir must be new' }
if (!$ComponentOnly) {
    if (!$ChatUiDir) { $ChatUiDir = Join-Path $root 'build/share/kvmem/ui' }
    $chatManifest = Join-Path $ChatUiDir 'ui-build.json'
    if (!(Test-Path -LiteralPath $chatManifest) -or (Get-Content -LiteralPath $chatManifest -Raw | ConvertFrom-Json).console_chat_protocol -ne 1) { throw 'Supply -ChatUiDir with the unified console chat build' }
}
if($ComponentOnly) {
    foreach($reserved in @('workers/polystrata','backends/polystrata','runtime/python')) {
        if(Test-Path -LiteralPath (Join-Path $out $reserved)) {throw "PolyStrata component destination exists: $reserved"}
    }
}
if(!(Get-Command dumpbin -ErrorAction SilentlyContinue)) { throw 'Run from the x64 Visual Studio developer shell' }
foreach($path in @($Worker)+@($VisionWorker | Where-Object {$_})) { if(!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing worker: $path" } }
if(!$GgmlSource) {
    $cache=Join-Path (Split-Path ([IO.Path]::GetFullPath($Worker))) 'CMakeCache.txt'
    if(Test-Path -LiteralPath $cache) { $line=Select-String -LiteralPath $cache -Pattern '^STRATA_GGML_DIR:PATH=(.+)$';if($line) {$GgmlSource=$line.Matches[0].Groups[1].Value} }
}
if(!$GgmlSource -or !(Test-Path -LiteralPath (Join-Path $GgmlSource 'LICENSE') -PathType Leaf)) {throw 'Supply GgmlSource with the pinned llama.cpp license used to build the worker'}
if($PythonRuntime -and !(Test-Path -LiteralPath (Join-Path $PythonRuntime 'python.exe') -PathType Leaf)) { throw 'PythonRuntime must be a portable Python folder with python.exe and its installed frontend dependencies' }
foreach($dir in @('workers/polystrata','backends/polystrata','tools/polystrata','scripts/windows','licenses','provenance')) { New-Item -ItemType Directory -Path (Join-Path $out $dir) -Force | Out-Null }
$workers=@(,@($Worker,'strata-poly.exe'))
if($VisionWorker) { $workers+=,@($VisionWorker,'strata-vision.exe') }
foreach($entry in $workers) {
    $destination=Join-Path $out ('workers/polystrata/'+$entry[1])
    Copy-Item -LiteralPath $entry[0] -Destination $destination
    $queue=[Collections.Generic.Queue[string]]::new();$queue.Enqueue($destination);$seen=@{}
    $search=@((Split-Path ([IO.Path]::GetFullPath($entry[0]))))
    if($CudaPath) { $search+=@((Join-Path $CudaPath 'bin'),(Join-Path $CudaPath 'bin/x64')) }
    $search+=$env:PATH.Split(';')
    while($queue.Count) {
        $file=$queue.Dequeue();$imports=& dumpbin /nologo /dependents $file
        if($LASTEXITCODE) { throw "dumpbin failed: $file" }
        foreach($line in $imports) {
            if($line -notmatch '^\s+([\w.+-]+\.dll)\s*$') {continue}
            $dll=$Matches[1]
            if($seen.ContainsKey($dll)) {continue};$seen[$dll]=$true
            if($dll -match '^(api-ms-|ext-ms-)' -or (Test-Path -LiteralPath (Join-Path $env:SystemRoot "System32/$dll"))) {continue}
            $found=$null;foreach($folder in $search) {if($folder -and (Test-Path -LiteralPath (Join-Path $folder $dll) -PathType Leaf)) {$found=Join-Path $folder $dll;break}}
            if(!$found) {throw "Unresolved dependency $dll from $file"}
            $copy=Join-Path $out "workers/polystrata/$dll";Copy-Item -LiteralPath $found -Destination $copy;$queue.Enqueue($copy)
        }
    }
}
# The integration serves chat/monitor in the shared console; retain only the old-origin export helper.
foreach($file in Get-ChildItem -LiteralPath (Join-Path $poly 'serve') -Recurse -File) {
    if($file.FullName -match '[\\/]__pycache__[\\/]' -or $file.Extension -eq '.pyc') {continue}
    if($file.FullName -match '[\\/]serve[\\/]web[\\/]' -and $file.Name -ne 'history-export.html') {continue}
    $relative=$file.FullName.Substring($poly.Length+1)
    $dest=Join-Path $out "backends/polystrata/$relative"
    New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $dest
}
$consoleInspection = @('scripts/model_health.py', 'scripts/console_hardware.py', 'scripts/console_memory.py', 'scripts/requirements-console.txt')
$consoleInspection += 'scripts/windows_amd_hardware.py', 'scripts/windows_gpu_hardware.py'
foreach ($name in '__init__.py', 'reader.py', 'schema.py', 'layouts.py', 'formats.py', 'framing.py', 'file_io.py', 'LICENSE', 'PROVENANCE.json') {
    $consoleInspection += 'scripts/model_health_ninfer/' + $name
}
foreach($relative in (@('tools/polystrata/server.py','tools/polystrata/assets.py','scripts/local-console.py','scripts/console_chat.py','scripts/speculative-capabilities.json',
 'scripts/windows/start-backend.ps1','scripts/windows/start-polystrata.ps1','scripts/windows/native-process.ps1',
 'scripts/windows/query-scoring.ps1', 'scripts/windows/token-settings.ps1','scripts/windows/start-ui.ps1','scripts/windows/describe-launch-json.ps1',
 'scripts/windows/select-model.ps1','START-HERE-CN.bat','START-HERE-EN.bat','backends/versions.json') + $consoleInspection)) {
    $dest=Join-Path $out $relative;New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
    if((Test-Path -LiteralPath $dest) -and (Get-FileHash -LiteralPath $dest).Hash -ne (Get-FileHash -LiteralPath (Join-Path $root $relative)).Hash) {throw "Conflicting common package file: $relative"}
    Copy-Item -LiteralPath (Join-Path $root $relative) -Destination $dest
}
foreach($file in Get-ChildItem -LiteralPath (Join-Path $root 'ui/monitor') -Recurse -File) {
    $relative=$file.FullName.Substring($root.Length+1);$dest=Join-Path $out $relative
    New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
    if((Test-Path -LiteralPath $dest) -and (Get-FileHash -LiteralPath $dest).Hash -ne (Get-FileHash -LiteralPath $file.FullName).Hash) {throw "Conflicting common UI file: $relative"}
    Copy-Item -LiteralPath $file.FullName -Destination $dest
}
New-Item -ItemType Directory -Path (Join-Path $out 'backends/polystrata/tools') -Force | Out-Null
foreach($name in @('strata_tokenizer.py','gguf_reader.py')) { Copy-Item -LiteralPath (Join-Path $poly "tools/$name") -Destination (Join-Path $out "backends/polystrata/tools/$name") }
Copy-Item -LiteralPath (Join-Path $poly 'requirements.txt') -Destination (Join-Path $out 'backends/polystrata/requirements.txt')
Copy-Item -LiteralPath (Join-Path $poly 'LICENSE') -Destination (Join-Path $out 'licenses/PolyStrata-LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $root 'README.md') -Destination (Join-Path $out 'licenses/KVMem-README.md')
Copy-Item -LiteralPath (Join-Path $GgmlSource 'LICENSE') -Destination (Join-Path $out 'licenses/llama.cpp-MIT.txt')
if($CudaPath) { Copy-Item -LiteralPath (Join-Path $CudaPath 'EULA.txt') -Destination (Join-Path $out 'licenses/CUDA-EULA.txt') }
if($PythonRuntime) {
    New-Item -ItemType Directory -Path (Join-Path $out 'runtime') -Force | Out-Null
    Copy-Item -LiteralPath $PythonRuntime -Destination (Join-Path $out 'runtime/python') -Recurse
    & (Join-Path $out 'runtime/python/python.exe') -I -c 'import numpy,jinja2,regex,yaml,PIL,psutil; from pathlib import Path; print("frontend dependencies available")'
    if($LASTEXITCODE) { throw 'Relocated Python runtime or frontend dependencies failed validation' }
}
if($ValidationReport) { Copy-Item -LiteralPath $ValidationReport -Destination (Join-Path $out $(if($ComponentOnly){'provenance/polystrata-validation.json'}else{'provenance/validation.json'})) }
if($ComponentOnly) { Write-Output "Added PolyStrata component: $out";return }
Copy-Item -LiteralPath $ChatUiDir -Destination (Join-Path $out 'ui/dist') -Recurse
$manifest=@{backend='polystrata';architecture='qwen35moe';models_included=$false;offline_python_included=[bool]$PythonRuntime;
    source=(Get-Content -LiteralPath (Join-Path $root 'backends/versions.json') -Raw|ConvertFrom-Json).polystrata;
    capability_scope=@{gpu_count=1;kv=@('f16','q8_0');max_concurrency=8;solo_mtp_drafts=@(0..3);batch_mtp=$false};
    validation='provenance/validation.json records configurations actually qualified; packaging alone is not runtime qualification'}
$manifest|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $out 'BUILD-INFO.json') -Encoding utf8
$hashes=foreach($file in Get-ChildItem -LiteralPath $out -Recurse -File|Sort-Object FullName) { (Get-FileHash -LiteralPath $file.FullName).Hash.ToLowerInvariant()+'  '+$file.FullName.Substring($out.Length+1).Replace('\','/') }
[IO.File]::WriteAllText((Join-Path $out 'SHA256SUMS'),($hashes -join "`n")+"`n",[Text.UTF8Encoding]::new($false))
Write-Output "Prepared PolyStrata package: $out"
