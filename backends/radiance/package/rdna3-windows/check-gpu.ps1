$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$env:PATH = (Join-Path $root 'bin') + ';' + $env:PATH
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$info = Join-Path $logs ("gpu-info-$stamp.txt")
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,VideoProcessor | Format-List | Out-File -FilePath $info -Encoding UTF8
Get-Content -LiteralPath $info
Write-Host 'Run this check with the GPU idle and the model server stopped.'
$exe = Join-Path $root 'bin\r3di_iqr_gpu_test.exe'
$dll = Join-Path $root 'bin\radiance_home\kernels\libr3diqr.dll'
if (-not (Test-Path -LiteralPath $exe)) { throw "Missing $exe" }
$failed = $false
foreach ($mode in @('iqr','dflash')) {
    $log = Join-Path $logs ("gpu-$mode-$stamp.log")
    Write-Host "Starting $mode numeric tests; log: $log"
    $ErrorActionPreference = 'Continue'
    if ($mode -eq 'iqr') { & $exe $dll 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $log }
    else { & $exe $dll --dflash 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $log }
    $status = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($status -ne 0) { $failed = $true; Write-Host "$mode failed (exit $status)." }
    else { Write-Host "$mode passed." }
}
Write-Host "Logs saved in $logs"
if ($failed) { exit 1 }
