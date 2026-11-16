param(
    [Parameter(Position=0)][string]$Model = '',
    [ValidateSet('off','dflash2')][string]$Drafter = 'off',
    [ValidateSet('bf16','fp8')][string]$KV = 'bf16',
    [int]$Budget = 0,
    [int]$Recent = -1,
    [ValidateRange(1,65535)][int]$Port = 18380
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
if (-not $Model) {
    $models = @(Get-ChildItem -LiteralPath (Join-Path $root 'models') -Filter '*.rad' -File -ErrorAction SilentlyContinue)
    if ($models.Count -eq 1) { $Model = $models[0].FullName }
    else { $Model = Read-Host 'Enter the full path of the .rad model (or drag it here)' }
}
$Model = $Model.Trim().Trim([char]34)
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Model file not found: $Model" }
$Model = (Resolve-Path -LiteralPath $Model).Path
if ($Drafter -eq 'dflash2') {
    if ($Budget -le 0) { $Budget = 3072 }
    if ($Recent -lt 0) { $Recent = 2176 }
    if ($Recent -lt 2176) { throw 'DFlash2 requires recent >= 2176 for this tested model.' }
    $depth = 7
} else {
    if ($Budget -le 0) { $Budget = 1024 }
    if ($Recent -lt 0) { $Recent = 256 }
    $depth = 0
}
if ($Budget -lt ($Recent + 128)) { throw 'Budget must be at least recent + 128 sink tokens.' }
$homeDir = Join-Path $root 'bin\radiance_home'
$exe = Join-Path $root 'bin\radiance-kvmem-server.exe'
foreach ($file in @($exe,(Join-Path $homeDir 'kernels.rkb'),(Join-Path $homeDir 'architectures\qwen35_w8a8.dll'),(Join-Path $homeDir 'kernels\libr3d.dll'),(Join-Path $homeDir 'kernels\libr3diqr.dll'),(Join-Path $root 'bin\amdhip64_7.dll'))) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Package file missing: $file" }
}
# Match the qualified launcher's environment without modifying the user's system.
foreach ($variable in @(Get-ChildItem Env: | Where-Object { $_.Name -match '^(RADIANCE_|KVMEM_|KV_DIAG_)' })) {
    Remove-Item -LiteralPath ('Env:' + $variable.Name)
}
$env:PATH = (Join-Path $root 'bin') + ';' + $env:PATH
$env:ROCM_PATH = $root
$env:HIP_PATH = $root
$env:KVMEM_CAPTURE_EARLY_SEAL = '1'
$env:OMP_NUM_THREADS = '2'
$arguments = @('--model',$Model,'--radiance-home',$homeDir,'--host','127.0.0.1','--port',"$Port",
    '--served-model-name','radiance-local','--kernels','libr3diqr,libr3d','--tp','1','--max-num-seqs','1',
    '--max-model-len','8192','--max-num-batched-tokens','256','--gpu-headroom-mib','160',
    '--kv-cache-dtype',$KV,'--state-cache-dtype','f32','--num-speculative-tokens',"$depth",'--mm-max-patches','0',
    '--no-prefix-cache','--prefix-cache-mib','0','--deterministic','--kvmem-budget',"$Budget",
    '--kvmem-block-tokens','128','--kvmem-replay-tokens','512','--kvmem-query-tokens','256',
    '--kvmem-query-replay-tokens','512','--kvmem-host-mib','2048','--kvmem-sink-tokens','128',
    '--kvmem-recent-tokens',"$Recent",'--kvmem-rolling-generation','--kvmem-drafter',$Drafter)
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$log = Join-Path $logs ('server-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '.log')
Write-Host "Model: $Model"
Write-Host "Drafter: $Drafter | KV: $KV | budget: $Budget | recent: $Recent"
Write-Host "Server: http://127.0.0.1:$Port | Log: $log"
Write-Host 'Leave this window open. When the server is ready, open chat.cmd. Ctrl+C stops the server.'
$ErrorActionPreference = 'Continue'
& $exe @arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $log
$status = $LASTEXITCODE
Write-Host "Server exited with code $status."
exit $status
