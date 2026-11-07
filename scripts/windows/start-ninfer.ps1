#Requires -Version 5.1
# Shared source/package shortcut for the 0.18 NInfer recipe.
# Put native options last after -WorkerArgs. For exact argv across shells,
# including empty strings and embedded quotes, use -WorkerArgsFile.
[CmdletBinding(PositionalBinding=$false)]
param(
    [string]$Model,
    [string]$Worker,
    [string]$Gpu = $(if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { '0' }),
    [uint32[]]$StageLayers = @(),
    [switch]$TensorParallel,
    [double[]]$TensorSplit = @(),
    [ValidateRange(1, 65535)][int]$Port = 18200,
    [string]$ListenHost = '127.0.0.1',
    [string]$ApiKey,
    [string]$ModelAlias,
    [ValidateSet('off', 'error', 'warning', 'info', 'trace', 'debug')][string]$LogLevel,
    [string]$ChatTemplateFile,
    [ValidateSet('default', 'on', 'off')][string]$ThinkingMode = 'default',
    [ValidateSet('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')][string]$ReasoningEffort,
    [switch]$PreserveThinking,
    [ValidateRange(256, 2147483647)][int]$Context = 204800,
    [ValidateRange(1, 2147483647)][int]$Budget = 36864,
    [ValidateRange(64, 2147483647)][int]$Reserve = 16384,
    [ValidateRange(1, 1048576)][int]$HostMiB = 12288,
    [ValidateRange(128, 2147483647)][int]$Prefill = 256,
    [ValidateRange(1, 2147483647)][int]$MaxTokens = 16384,
    [string]$KvType = 'int8',
    [ValidateRange(0, 15)][int]$MtpDrafts = 4,
    [switch]$AdaptiveMtp = $true,
    [ValidateSet('none', 'mtp', 'dflash2')][string]$SpeculativeMode,
    [ValidateRange(0, 15)][int]$DraftTokens,
    [ValidateSet('fixed', 'adaptive')][string]$DraftPolicy,
    [string]$DraftModel,
    [switch]$FastPrefill,
    [ValidateRange(0, 63)][int]$NgramDrafts = 31,
    [ValidateRange(4, 64)][int]$NgramMinMatch = 12,
    [ValidateRange(1, 8)][int]$Concurrency = 1,
    [ValidateRange(1, 16)][int]$RetainedSessions = 1,
    [switch]$Vision = $true,
    [switch]$NoVision,
    [ValidateSet('resident', 'cpu')][string]$VisionResidency = 'cpu',
    [ValidateRange(0, 2147483647)][int]$CpuVisionIdleSeconds = 300,
    [ValidateRange(1, 16384)][int]$VisionTokens = 1024,
    [ValidateSet('off', 'auto')][string]$DeviceProfile = 'off',
    [ValidateRange(0, 2147483647)][int]$ThinkingBudget = 8192,
    [ValidateRange(0, 2147483647)][int]$SinkTokens = 12288,
    [ValidateRange(0, 2147483647)][int]$RecentTokens = 1024,
    [ValidateSet('legacy', 'layerwise')][string]$RecallStrategy = 'legacy',
    [ValidateSet('upstream-global', 'first-layer')][string]$LayerwiseMtpHistory,
    [ValidateRange(0, 2147483647)][int]$LayerwiseActivationRamMiB,
    [ValidateRange(0, 2147483647)][int]$LayerwiseActivationDiskMiB,
    [ValidateRange(1, 2147483647)][int]$LayerwiseStagingMiB,
    [ValidateRange(1, 2147483647)][int]$LayerwiseMetadataMiB,
    [ValidateRange(1, 2147483647)][int]$LayerwiseCheckpointTarget,
    [ValidateRange(1, 2147483647)][int]$LayerwiseActivationMetadataMiB,
    [ValidateSet('page-softmax', 'meanq-cosine')][string]$QueryScorer = 'page-softmax',
    [ValidateRange(0.0, 1.0)][double]$GraphAlpha,
    [ValidateRange(1, 2147483647)][int]$QueryMaxTokens,
    [ValidateRange(1, 2147483647)][int]$QueryReplayMaxTokens,
    [ValidateRange(0, 2147483647)][int]$CheckpointExtraRecomputeTokens = 0,
    [switch]$Lrfu,
    [switch]$NoLrfu,
    [ValidateRange(0.0, 1.0)][double]$LrfuLambda = 0.5,
    [ValidateScript({ $_ -gt 0 -and $_ -le 1 })][double]$LrfuTopFraction = 0.9,
    [ValidateSet('mean', 'max')][string]$ScoreAggregation = 'mean',
    [string]$DiskPath,
    [ValidateRange(0, 1048576)][int]$DiskMiB = 4096,
    [AllowEmptyCollection()][AllowEmptyString()][string[]]$WorkerArgs = @(),
    [string]$WorkerArgsFile,
    [switch]$Describe,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments=$true)][AllowEmptyCollection()][AllowEmptyString()][string[]]$WorkerArgsTail = @()
)
$ErrorActionPreference = 'Stop'
if ($RecallStrategy -eq 'layerwise') {
    # A new strategy starts with ordinary text generation. Preserve every
    # explicitly supplied draft/vision choice for the actual worker gate.
    foreach ($entry in @{MtpDrafts=0;AdaptiveMtp=$false;NgramDrafts=0;Vision=$false}.GetEnumerator()) {
        if (!$PSBoundParameters.ContainsKey($entry.Key)) { Set-Variable -Name $entry.Key -Value $entry.Value }
    }
    $requestedMode = if ($SpeculativeMode) { $SpeculativeMode } elseif ($MtpDrafts -gt 0) { 'mtp' } else { 'none' }
    if ($requestedMode -ne 'none') {
        if (!$PSBoundParameters.ContainsKey('DraftTokens') -and !$PSBoundParameters.ContainsKey('MtpDrafts')) {
            throw 'Layerwise speculative decoding requires an explicit draft width'
        }
        $requestedWidth = if ($PSBoundParameters.ContainsKey('DraftTokens')) { $DraftTokens } else { $MtpDrafts }
        if ($requestedWidth -lt 1) { throw 'Layerwise speculative draft width must be positive' }
    }
}
. (Join-Path $PSScriptRoot 'native-process.ps1')
$resolvedArgs = Resolve-KVMemWorkerArgs $PSBoundParameters $WorkerArgsTail
$forward = @{}
foreach ($entry in $PSBoundParameters.GetEnumerator()) { $forward[$entry.Key] = $entry.Value }
foreach ($name in @('WorkerArgs', 'WorkerArgsFile', 'WorkerArgsTail')) { $null = $forward.Remove($name) }
$forward['WorkerArgs'] = $resolvedArgs
$forward['Backend'] = 'ninfer'
foreach ($parameterName in @('Gpu', 'Context', 'Budget', 'Reserve', 'HostMiB', 'Prefill', 'MaxTokens', 'KvType', 'MtpDrafts', 'AdaptiveMtp', 'NgramDrafts', 'NgramMinMatch', 'Concurrency', 'RetainedSessions', 'Vision', 'VisionResidency', 'CpuVisionIdleSeconds', 'VisionTokens', 'DeviceProfile', 'ThinkingBudget', 'SinkTokens', 'RecentTokens')) {
    if ($RecallStrategy -eq 'layerwise' -and !$PSBoundParameters.ContainsKey($parameterName) -and
        $parameterName -in @('Budget','Reserve','HostMiB','SinkTokens','RecentTokens')) { continue }
    if (!$forward.ContainsKey($parameterName)) { $forward[$parameterName] = Get-Variable -Name $parameterName -ValueOnly }
}
$null = $forward.Remove('NoVision')
$forward['Vision'] = [bool]$Vision -and !$NoVision
$forward['VisionResidency'] = if ($forward['Vision']) { $VisionResidency } else { 'resident' }
if (!$MtpDrafts) {
    if (!$PSBoundParameters.ContainsKey('AdaptiveMtp')) { $forward['AdaptiveMtp'] = $false }
    if (!$PSBoundParameters.ContainsKey('NgramDrafts')) { $forward['NgramDrafts'] = 0 }
}
if ($PSBoundParameters.ContainsKey('SpeculativeMode')) {
    foreach ($legacy in @('MtpDrafts','AdaptiveMtp','NgramDrafts')) {
        if (!$PSBoundParameters.ContainsKey($legacy)) { $null = $forward.Remove($legacy) }
    }
}
if ($PSBoundParameters.ContainsKey('DraftTokens') -and !$PSBoundParameters.ContainsKey('MtpDrafts')) { $null = $forward.Remove('MtpDrafts') }
if ($PSBoundParameters.ContainsKey('DraftPolicy') -and !$PSBoundParameters.ContainsKey('AdaptiveMtp')) { $null = $forward.Remove('AdaptiveMtp') }
if (!$forward.ContainsKey('Model')) { $forward['Model'] = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../model.ninfer')) }
& (Join-Path $PSScriptRoot 'start-backend.ps1') @forward
