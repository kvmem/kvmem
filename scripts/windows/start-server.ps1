#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('iq3', 'iq4')][string]$Recipe,
    [string]$Model = $env:MODEL,
    [string]$Mmproj = $env:MMPROJ,
    [string]$BuildDir = $env:BUILD_DIR,
    [string]$Gpu = $env:CUDA_VISIBLE_DEVICES,
    [ValidateRange(1, 65535)][int]$Port = 18200,
    [string]$ListenHost = $env:HOST,
    [string]$ApiKey,
    [string]$ApiKeyFile = $env:LLAMA_ARG_API_KEY_FILE,
    [ValidateRange(1, 2147483647)][int]$BlockTokens = 128,
    [ValidateRange(0, 2147483647)][int]$SinkTokens = 12288,
    [ValidateRange(0, 2147483647)][int]$RecentTokens = 1024,
    [string]$ReasoningEffort,
    [ValidateSet('f16', 'f32', 'q8_0', 'q5_0', 'q4_0')][string]$CacheTypeK,
    [ValidateSet('f16', 'f32', 'q8_0', 'q5_0', 'q4_0')][string]$CacheTypeV,
    [ValidateRange(0, 2147483647)][int]$ReasoningBudget = 4096,
    [ValidateRange(1, 5)][int]$Mtp = 3,
    [ValidateSet('legacy', 'layerwise')][string]$RecallStrategy = 'legacy',
    [ValidateSet('upstream-global', 'first-layer')][string]$LayerwiseMtpHistory,
    [ValidateRange(0, 2147483647)][int]$LayerwiseActivationRamMiB,
    [ValidateRange(0, 2147483647)][int]$LayerwiseActivationDiskMiB,
    [ValidateRange(1, 2147483647)][int]$LayerwiseStagingMiB,
    [ValidateRange(1, 2147483647)][int]$LayerwiseMetadataMiB,
    [ValidateRange(1, 2147483647)][int]$LayerwiseActivationMetadataMiB,
    [ValidateSet('page-softmax', 'meanq-cosine')][string]$QueryScorer = 'page-softmax',
    [ValidateRange(0.0, 1.0)][double]$GraphAlpha,
    [ValidateRange(1, 2147483647)][int]$QueryMaxTokens,
    [ValidateRange(1, 2147483647)][int]$QueryReplayMaxTokens,
    [switch]$Lrfu,
    [switch]$NoLrfu,
    [ValidateRange(0.0, 1.0)][double]$LrfuLambda = 0.5,
    [ValidateScript({ $_ -gt 0 -and $_ -le 1 })][double]$LrfuTopFraction = 0.9,
    [ValidateSet('mean', 'max')][string]$ScoreAggregation = 'mean',
    [ValidateSet('cpu', 'gpu')][string]$VisionDevice,
    [switch]$NoVision,
    [string]$ChatTemplateFile,
    [string]$ChatTemplateKwargs,
    [string]$UiDir,
    [switch]$NoUi,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native-process.ps1')
. (Join-Path $PSScriptRoot 'query-scoring.ps1')
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$BuildDir) {
    if (Test-Path -LiteralPath (Join-Path $root 'bin/llama-kvmem-server.exe')) { $BuildDir = $root }
    else { $BuildDir = Join-Path $root 'build-win' }
}
$binary = Join-Path $BuildDir 'bin/llama-kvmem-server.exe'
$requiredFiles=@(@('server', $binary), @('Model', $Model))
if(!$NoVision){$requiredFiles+=,@('Mmproj',$Mmproj)}
foreach ($item in $requiredFiles) {
    if (!$item[1] -or !(Test-Path -LiteralPath $item[1] -PathType Leaf)) {
        throw "Missing $($item[0]) file: $($item[1]). Pass -Model / -Mmproj or set MODEL / MMPROJ."
    }
}
$binary = (Resolve-Path -LiteralPath $binary).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
if(!$NoVision){$Mmproj = (Resolve-Path -LiteralPath $Mmproj).Path}
if (!$Gpu) {
    $rows = @(& nvidia-smi --query-gpu=uuid,name --format=csv,noheader)
    if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed; select a GPU with -Gpu' }
    $preferred = @($rows | Where-Object { $_ -match 'RTX 5060 Ti' })
    if ($preferred.Count -eq 1) { $Gpu = ($preferred[0] -split ',')[0].Trim() }
    elseif ($rows.Count -eq 1) { $Gpu = ($rows[0] -split ',')[0].Trim() }
    else { throw 'Multiple GPUs found; select one with -Gpu or CUDA_VISIBLE_DEVICES' }
}
if ([string]::IsNullOrWhiteSpace($Gpu) -or $Gpu -eq '-1') { throw 'Select an enabled GPU' }
$budget = 36864; $reserve = 16384; $kv = 'q8_0'
if ($Recipe -eq 'iq4') { $budget = 32768; $reserve = 12288; $kv = 'q5_0' }
if ([string]::IsNullOrWhiteSpace($ListenHost)) { $ListenHost = $env:LLAMA_ARG_HOST }
if ([string]::IsNullOrWhiteSpace($ListenHost)) { $ListenHost = '127.0.0.1' }
if (!$VisionDevice) { $VisionDevice = 'cpu' }
$visionFlag = '--mmproj-offload'
if ($VisionDevice -eq 'cpu') { $visionFlag = '--no-mmproj-offload' }
$serverArgs = @('-m', $Model, '--host', $ListenHost, '--port', "$Port",
    '-c', '262144', '-n', "$reserve", '--kvmem-budget', "$budget", '--kvmem-gen-reserve', "$reserve",
    '--kv-dtype', $kv, '--spec-type', 'draft-mtp', '--spec-draft-n-max', "$Mtp",
    '--kvmem-block-tokens', "$BlockTokens", '--kvmem-query-policy', 'user',
    '--kvmem-sink-tokens', "$SinkTokens", '--kvmem-recent-tokens', "$RecentTokens",
    '--enable-thinking', '--reasoning-budget', "$ReasoningBudget")
if(!$NoVision){$serverArgs=@($serverArgs[0],$serverArgs[1],'--mmproj',$Mmproj,$visionFlag,'--image-max-tokens','512')+@($serverArgs | Select-Object -Skip 2)}
$serverArgs += @(Get-KVMemQueryScoringArgs $PSBoundParameters)
$serverArgs = @(Remove-KVMemImplicitRecallBands $serverArgs $PSBoundParameters)
if ($CacheTypeK) { $serverArgs += @('--cache-type-k', $CacheTypeK) }
if ($CacheTypeV) { $serverArgs += @('--cache-type-v', $CacheTypeV) }
if ($ApiKey) { $serverArgs += @('--api-key', $ApiKey) }
if ($ApiKeyFile) { $serverArgs += @('--api-key-file', $ApiKeyFile) }
if ($ReasoningEffort) { $serverArgs += @('--reasoning-effort', $ReasoningEffort) }
if ($ChatTemplateFile) {
    $template = (Resolve-Path -LiteralPath $ChatTemplateFile).Path
    if (!(Test-Path -LiteralPath $template -PathType Leaf) -or [string]::IsNullOrWhiteSpace([IO.File]::ReadAllText($template))) {
        throw 'ChatTemplateFile must be a nonempty file'
    }
    $serverArgs += @('--chat-template-file', $template)
}
if ($ChatTemplateKwargs) {
    $parsed = ConvertFrom-Json $ChatTemplateKwargs
    if ($null -eq $parsed -or $parsed -isnot [System.Management.Automation.PSCustomObject]) {
        throw 'ChatTemplateKwargs must be a JSON object'
    }
    $serverArgs += @('--chat-template-kwargs', $ChatTemplateKwargs)
}
if ($UiDir) {
    if (!(Test-Path -LiteralPath (Join-Path $UiDir 'index.html') -PathType Leaf)) { throw 'UiDir must contain index.html' }
    $serverArgs += @('--ui-dir', (Resolve-Path -LiteralPath $UiDir).Path)
}
if ($NoUi) { $serverArgs += '--no-ui' }
$effective=@{};foreach($key in $PSBoundParameters.Keys){if($key -notin @('ApiKey','ApiKeyFile')){$effective[$key]=$PSBoundParameters[$key]}}
$effective.Backend='llamacpp';$effective.Worker=$binary;$effective.Model=$Model;$effective.Gpu=$Gpu
$effective.Budget=$budget;$effective.Reserve=$reserve;$effective.BlockTokens=$BlockTokens;$effective.Concurrency=1
$effective.KvType=$kv;$effective.SpeculativeMode='mtp';$effective.DraftTokens=$Mtp;$effective.DraftPolicy='fixed'
$effective.Vision=!$NoVision;$effective.VisionResidency=if($VisionDevice -eq 'cpu'){'cpu'}else{'resident'}
if($CacheTypeK){$effective.KvTypeK=$CacheTypeK};if($CacheTypeV){$effective.KvTypeV=$CacheTypeV}
if ($DryRun) {
    @{ argv = @($binary) + $serverArgs; effective_parameters=$effective; environment = @{ CUDA_VISIBLE_DEVICES = $Gpu; CUDA_DEVICE_ORDER = 'PCI_BUS_ID' } } |
        ConvertTo-Json -Depth 5
    return
}
Invoke-KVMemRecallWorkerCheck $binary $effective $root
# Check for an existing listener without terminating any process. The server
# still owns the final bind, so a race results in a normal startup failure.
$probe = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)
$probe.Server.ExclusiveAddressUse = $true
try { $probe.Start() } finally { $probe.Stop() }
$info = New-KVMemProcessInfo $binary $serverArgs
$info.EnvironmentVariables['CUDA_VISIBLE_DEVICES'] = $Gpu
$info.EnvironmentVariables['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
$info.EnvironmentVariables['KVMEM_VISION_DEVICE'] = $VisionDevice
$child = New-Object System.Diagnostics.Process
$child.StartInfo = $info
$started = $false
$uiHost = if ($ListenHost -in @('0.0.0.0', '::')) { '127.0.0.1' } else { $ListenHost }
try {
    Write-Host "Starting $Recipe on http://$uiHost`:$Port/ (foreground; Ctrl+C to stop)"
    $started = $child.Start()
    while (!$child.WaitForExit(250)) {}
    $code = $child.ExitCode
} finally {
    # Own process handle only; no PID-file or port-based process termination.
    if ($started -and !$child.HasExited) { $child.Kill(); $child.WaitForExit() }
    $child.Dispose()
}
exit $code
