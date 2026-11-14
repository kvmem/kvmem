#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 18199,
    [string]$ListenHost = '127.0.0.1',
    [switch]$PromptAddress,
    [ValidateSet('zh-CN', 'en')][string]$Language = 'zh-CN',
    [string]$StateDir,
    [switch]$NoBrowser,
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))

if (Test-Path -LiteralPath (Join-Path $root 'bin/amdhip64_7.dll') -PathType Leaf) {
    $env:ROCM_PATH = $root
    $env:HIP_PATH = $root
    $env:PATH = (Join-Path $root 'bin') + [IO.Path]::PathSeparator + $env:PATH
    if (!$env:RADIANCE_LOAD_STAGING_MIB) { $env:RADIANCE_LOAD_STAGING_MIB = '128' }
    if (!$env:RADIANCE_LOAD_WORKERS) { $env:RADIANCE_LOAD_WORKERS = '1' }
    if (!$env:RADIANCE_LOAD_WINDOW_MIB) { $env:RADIANCE_LOAD_WINDOW_MIB = '16' }
    if (!$env:RADIANCE_LOAD_RELEASE_PAGES) { $env:RADIANCE_LOAD_RELEASE_PAGES = '1' }
}
try {
    foreach ($relative in 'scripts/local-console.py', 'scripts/console_chat.py', 'scripts/console_state.py', 'scripts/model_health.py', 'scripts/console_hardware.py', 'scripts/console_memory.py',
                          'scripts/windows_gpu_hardware.py', 'scripts/windows_amd_hardware.py',
                          'scripts/launcher/__init__.py', 'scripts/launcher/config.py', 'scripts/launcher/plan.py',
                          'scripts/launcher/export.py', 'scripts/launcher/systemd.py', 'scripts/launcher/worker.py', 'scripts/launcher-capabilities.json',
                          'scripts/launcher/recall.py', 'scripts/check-recall-worker.py', 'scripts/windows/recall-strategy.ps1',
                          'scripts/model_health_ninfer/__init__.py', 'scripts/model_health_ninfer/reader.py',
                          'scripts/model_health_ninfer/schema.py', 'scripts/model_health_ninfer/layouts.py',
                          'scripts/model_health_ninfer/formats.py', 'scripts/model_health_ninfer/framing.py',
                          'scripts/model_health_ninfer/file_io.py', 'scripts/windows/start-backend.ps1',
                          'scripts/windows/describe-launch-json.ps1', 'scripts/windows/native-process.ps1', 'scripts/windows/query-scoring.ps1', 'scripts/windows/token-settings.ps1',
                          'scripts/windows/select-model.ps1',
                          'ui/monitor/index.html', 'ui/monitor/launcher.js', 'ui/monitor/monitor.js',
                          'ui/monitor/console-shell.js', 'ui/monitor/console-shell.css') {
        if (!(Test-Path -LiteralPath (Join-Path $root $relative) -PathType Leaf)) {
            throw "Incomplete KVMem package: missing $relative. Extract the complete ZIP first."
        }
    }
    $chatRoot = @('ui/dist', 'workers/share/kvmem/ui', 'build/share/kvmem/ui') |
        ForEach-Object { Join-Path $root $_ } |
        Where-Object { Test-Path -LiteralPath (Join-Path $_ 'index.html') -PathType Leaf } |
        Select-Object -First 1
    if (!$chatRoot -or !(Test-Path -LiteralPath (Join-Path $chatRoot 'ui-build.json') -PathType Leaf)) {
        throw 'Unified chat files are missing. Extract the complete KVMem package first.'
    }
    $chatManifest = Get-Content -LiteralPath (Join-Path $chatRoot 'ui-build.json') -Raw | ConvertFrom-Json
    if (!$chatManifest.PSObject.Properties['console_chat_protocol'] -or $chatManifest.console_chat_protocol -ne 1) {
        throw 'The chat bundle does not match this console. Extract the complete KVMem package first.'
    }
    # Avoid Windows Store aliases and automatic installs by either Python launcher.
    $env:PYLAUNCHER_ALLOW_INSTALL = $null
    $env:PYLAUNCHER_ALWAYS_INSTALL = $null
    $env:PYTHON_MANAGER_AUTOMATIC_INSTALL = 'false'
    $probe = 'import sys,ctypes,json,http.server,urllib.request,webbrowser,psutil; assert sys.version_info >= (3,10); assert sys.maxsize > 2**32; print(sys.executable)'
    $embedded = Join-Path $root 'runtime/python/python.exe'
    $python = $null
    if (Test-Path -LiteralPath $embedded -PathType Leaf) {
        $output = & $embedded -I -c $probe 2>&1
        if ($LASTEXITCODE -ne 0) { throw 'The bundled Python runtime is incomplete. Extract a fresh copy of the KVMem ZIP.' }
        $python = $embedded
    } else {
        foreach ($name in 'python.exe', 'py.exe') {
            $candidate = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if (!$candidate -or $candidate.Source -like '*\Microsoft\WindowsApps\*') { continue }
            $arguments = @('-I', '-c', $probe)
            if ($name -eq 'py.exe') { $arguments = @('-3') + $arguments }
            try {
                $output = @(& $candidate.Source @arguments 2>$null)
                if ($LASTEXITCODE -eq 0 -and $output -and (Test-Path -LiteralPath ([string]$output[-1]) -PathType Leaf)) {
                    $python = [string]$output[-1]; break
                }
            } catch { continue }
        }
        if (!$python) {
            throw 'Python with psutil was not found. Use a complete KVMem package, or install 64-bit Python 3.10+ and run: python -m pip install -r scripts/requirements-console.txt'
        }
    }
    if ($Check) {
        Write-Output "KVMem UI files and Python runtime OK: $python"
        Write-Output 'This check does not load a model or validate GPU/worker dependencies.'
        exit 0
    }
    if (!$StateDir) { $StateDir = Join-Path $root 'build/local-console' }
    $arguments = @('-X', 'utf8', '-I', (Join-Path $root 'scripts/local-console.py'), '--host', $ListenHost, '--port', "$Port", '--state-dir', $StateDir, '--prompt-port', '--language', $Language)
    if ($PromptAddress) { $arguments += '--prompt-address' }
    $bundledWorkers = @(
        'ninfer-kvmem-server.exe', 'ninfer-serve.exe', 'llama-kvmem-server.exe',
        'bin/radiance-kvmem-server.exe', 'bin/ninfer-kvmem-server.exe', 'bin/ninfer-serve.exe', 'bin/llama-kvmem-server.exe',
        'workers/ninfer/ninfer-kvmem-server.exe', 'workers/ninfer/ninfer-serve.exe',
        'workers/llamacpp/llama-kvmem-server.exe', 'workers/polystrata/strata-poly.exe'
    ) | ForEach-Object { Join-Path $root $_ } | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
    if ($bundledWorkers) { $arguments += '--prefer-bundled-worker' }
    if (!$NoBrowser) { $arguments += '--open-browser' }
    Write-Output 'KVMem - local launch console'
    & $python @arguments
    exit $LASTEXITCODE
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
