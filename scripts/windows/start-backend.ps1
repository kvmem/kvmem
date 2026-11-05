#Requires -Version 5.1
[CmdletBinding(PositionalBinding=$false)]
param(
    [Parameter(Mandatory)][ValidateSet('llamacpp', 'ninfer', 'polystrata')][string]$Backend,
    [string]$Model,
    [string]$Worker,
    [string]$Tokenizer,
    [string]$PolystrataSource,
    [string]$Python,
    [string]$VisionWorker,
    [string]$Mmproj,
    [string]$Lora,
    [double]$LoraScale = 1,
    [ValidateRange(256,65536)][int]$VramReserveMiB=256,
    [ValidateRange(1,65536)][int]$ReplayMaxTokens=2048,
    [string]$Gpu = $env:CUDA_VISIBLE_DEVICES,
    [uint32[]]$StageLayers = @(),
    [switch]$TensorParallel,
    [double[]]$TensorSplit = @(),
    [ValidateRange(1, 65535)][int]$Port = 18200,
    [string]$ListenHost = '127.0.0.1',
    [ValidateLength(1, 512)][ValidatePattern('^[A-Za-z0-9._~+/-]+=*$')][string]$ApiKey,
    [string]$ModelAlias,
    [ValidateSet('off', 'error', 'warning', 'info', 'trace', 'debug')][string]$LogLevel,
    [string]$ChatTemplateFile,
    [ValidateSet('default', 'on', 'off')][string]$ThinkingMode = 'default',
    [ValidateSet('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')][string]$ReasoningEffort,
    [switch]$PreserveThinking,
    [ValidateRange(256, 2147483647)][int]$Context = 262144,
    [ValidateRange(128, 2147483647)][int]$Budget = 32768,
    [ValidateSet(32,64)][int]$BlockTokens = 32,
    [ValidateRange(32, 2147483647)][int]$Reserve = 1024,
    [ValidateRange(1, 1048576)][int]$HostMiB = 16384,
    [ValidateRange(1, 2147483647)][int]$Prefill = 128,
    [ValidateRange(0, 2147483647)][int]$MaxTokens = 1024,
    [string]$KvType,
    [string]$KvTypeK,
    [string]$KvTypeV,
    [ValidateRange(0, 15)][int]$MtpDrafts = 0,
    [switch]$AdaptiveMtp,
    [ValidateSet('none', 'mtp', 'dflash2')][string]$SpeculativeMode,
    [ValidateRange(0, 15)][int]$DraftTokens,
    [ValidateSet('fixed', 'adaptive')][string]$DraftPolicy,
    [string]$DraftModel,
    [switch]$FastPrefill,
    [ValidateRange(0, 63)][int]$NgramDrafts = 0,
    [ValidateRange(4, 64)][int]$NgramMinMatch = 12,
    [ValidateRange(1, 8)][int]$Concurrency = 1,
    [ValidateRange(1, 16)][int]$RetainedSessions = 4,
    [switch]$Vision,
    [ValidateSet('resident', 'cpu')][string]$VisionResidency = 'resident',
    [ValidateRange(0, 2147483647)][int]$CpuVisionIdleSeconds = 300,
    [ValidateRange(1, 16384)][int]$VisionTokens = 1024,
    [ValidateSet('off', 'auto')][string]$DeviceProfile = 'off',
    [ValidateRange(0, 2147483647)][int]$ThinkingBudget = 0,
    [ValidateRange(0, 2147483647)][int]$SinkTokens = 12288,
    [ValidateRange(0, 2147483647)][int]$RecentTokens = 1024,
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
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native-process.ps1')
. (Join-Path $PSScriptRoot 'query-scoring.ps1')
. (Join-Path $PSScriptRoot 'token-settings.ps1')
$WorkerArgs = Resolve-KVMemWorkerArgs $PSBoundParameters $WorkerArgsTail
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$specProfiles = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../speculative-capabilities.json') -Raw | ConvertFrom-Json
$capabilities = @{
    backend = $Backend; model_format = if ($Backend -eq 'ninfer') { '.ninfer' } else { '.gguf' }
    text_chat = $true; streaming = $true; history_reuse = $true
    speculative_decoding = $true; multimodal = $true
    concurrent_requests = $true; disk_restore = $true
    active_kv_tiering = $true; ssd_persistence = 'process'
    cross_backend_kv_transfer = $false
    speculative = @{ schema_version = 1; modes = $specProfiles.$Backend }
}
if ($Backend -eq 'ninfer') {
    $capabilities.disk_restore = $false
    $capabilities.checkpoint_recompute_min_revision = 1
    $capabilities.supported_combinations = @{
        text = @{ kv = @('bf16', 'int8', 'fp8', 'nvfp4', 'k8v4', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8'); max_concurrency = 8; mtp_drafts = @(0..15) }
        images = @{ kv = @('bf16', 'int8', 'fp8', 'nvfp4', 'k8v4', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8'); max_concurrency = 8; residency = @('resident', 'cpu') }
        rk8v4 = @{ max_concurrency = 8; text_only = $false }
        cold_disk = @{ enabled = $false; reason = 'replaced_by_process_local_tiering' }
        active_disk = @{ enabled = $true; shared_active_idle_quota = $true; max_concurrency = 8 }
        ngram = @{ max_concurrency = 8; text_only = $false; image_kv = @('bf16', 'int8', 'fp8', 'nvfp4', 'k8v4', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8'); requires_speculative = @('mtp', 'dflash2'); max_drafts = 63; concurrent_max_drafts = 15; wide_max_concurrency = 1; concurrent_width_policy = 'clamp_to_15' }
        lookup_ngram = @{ enabled = $false; reason = 'legacy_path_disabled'; replacement = '--ngram-draft-tokens --ngram-min-match' }
        adaptive_mtp = @{ max_concurrency = 8; text_only = $false; ngram = $true; ngram_max_concurrency = 8; ngram_concurrent_max_drafts = 15; cold_disk = $false }
        fast_prefill = @{ kv = @('int8', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8'); text_only = $false; max_concurrency = 8; residency = @('resident', 'cpu'); cold_disk = $false }
        multi_gpu = @{ mode = 'layer'; modes = @('layer', 'tensor'); tensor_attention = @('heads'); tensor_split = $true;
            heads_tensor_parallel_min_revision = 1; max_devices = 8; vision_residency = @('resident', 'cpu'); mtp = $true; host_history = 'shared' }
        video = $false
    }
    if (@($WorkerArgs | Where-Object { $_ -match '^--lookup-ngram(=|$)' }).Count) {
        throw 'KVMem disables legacy --lookup-ngram; use -NgramDrafts and -NgramMinMatch'
    }
}
if ($Backend -eq 'polystrata') {
    $capabilities.block_tokens = @{default=32;supported=@(32,64);startup_fixed=$true}
    $capabilities.supported_combinations=@{
        text=@{architectures=@('qwen35moe','gemma4');kv=@('f16','q8_0','int8','int8k');max_concurrency=8;max_devices=1}
        mtp=@{drafts=@(0..3);solo=$true;batch=$false;batch_fallback='ordinary_decode'}
        storage=@{scope='process';shared_active_idle_quota=$true;ssd_persistence='process'}
        query=@{scorers=@('page-softmax','meanq-cosine');bounded_replay=$true;tool_continuation='resident_pressure'}
    }
}
if ($Describe) { $capabilities | ConvertTo-Json -Depth 4; return }
if ($Backend -ne 'llamacpp' -and ($Lora -or $PSBoundParameters.ContainsKey('LoraScale'))) { throw 'Lora configures llama.cpp only' }
if ($Backend -ne 'ninfer' -and $PSBoundParameters.ContainsKey('CheckpointExtraRecomputeTokens')) {
    throw '-CheckpointExtraRecomputeTokens configures NInfer only'
}
if ($Backend -eq 'ninfer' -and $CheckpointExtraRecomputeTokens -gt 0 -and
    $env:KVMEM_CHECKPOINT_MIN_GAP -and $env:KVMEM_CHECKPOINT_MIN_GAP.Trim() -notin @('', '0')) {
    throw 'CheckpointExtraRecomputeTokens requires KVMEM_CHECKPOINT_MIN_GAP=0'
}
if ($Backend -ne 'polystrata' -and $PSBoundParameters.ContainsKey('BlockTokens')) {
    throw '-BlockTokens configures PolyStrata in this launcher'
}
if ($Backend -eq 'polystrata' -and !$PSBoundParameters.ContainsKey('Prefill')) { $Prefill=512 }
$resolvedBlock = if ($Backend -eq 'llamacpp') { 32 } else { $BlockTokens }
# A native logical-batch override must not change the independently emitted ubatch.
$llamaPhysicalPrefill=$Prefill
$tokenInputs = Resolve-KVMemTokenOverrides $Backend $WorkerArgs $Context $Prefill $resolvedBlock
$WorkerArgs = @($tokenInputs.WorkerArgs)
$resolvedBlock = $tokenInputs.BlockTokens
$resolvedContext = $tokenInputs.Context
$tokenPlan = Resolve-KVMemTokenSettings $Backend $resolvedBlock $Budget $Reserve $tokenInputs.Prefill $SinkTokens $RecentTokens $resolvedContext
$Budget=[int]$tokenPlan.effective.Budget; $Reserve=[int]$tokenPlan.effective.Reserve
$Prefill=[int]$tokenPlan.effective.Prefill; $SinkTokens=[int]$tokenPlan.effective.SinkTokens; $RecentTokens=[int]$tokenPlan.effective.RecentTokens
foreach ($change in $tokenPlan.adjustments) {
    Write-Warning "KVMEM_CONFIG $($change.parameter): $($change.requested) -> $($change.effective) tokens ($($change.reason))"
}
if ($Backend -eq 'polystrata') {
    & (Join-Path $PSScriptRoot 'start-polystrata.ps1') -Parameters $PSBoundParameters -Root $root -WorkerArgs $WorkerArgs -TokenPlan $tokenPlan
    return
}
# Canonical settings take precedence over defaults, while conflicting explicit
# aliases fail before the worker starts. Old saved MTP configurations still work.
if (!$SpeculativeMode) { $SpeculativeMode = if ($MtpDrafts -gt 0) { 'mtp' } else { 'none' } }
$specCap = $specProfiles.$Backend.$SpeculativeMode
if ($PSBoundParameters.ContainsKey('DraftTokens')) {
    if ($PSBoundParameters.ContainsKey('MtpDrafts') -and $MtpDrafts -ne $DraftTokens) { throw 'DraftTokens conflicts with MtpDrafts' }
    $MtpDrafts = $DraftTokens
} elseif ($SpeculativeMode -ne 'none' -and !$MtpDrafts) { $MtpDrafts = if ($SpeculativeMode -eq 'dflash2') { 5 } else { 4 } }
if ($DraftPolicy) {
    if ($PSBoundParameters.ContainsKey('AdaptiveMtp') -and [bool]$AdaptiveMtp -ne ($DraftPolicy -eq 'adaptive')) { throw 'DraftPolicy conflicts with AdaptiveMtp' }
    $AdaptiveMtp = $DraftPolicy -eq 'adaptive'
}
if ($SpeculativeMode -eq 'none' -and ($MtpDrafts -or $AdaptiveMtp -or $NgramDrafts -or $DraftModel)) { throw 'Disabled speculative decoding cannot have active draft options' }
if ($SpeculativeMode -ne 'none') {
    if ($MtpDrafts -lt 1 -or $MtpDrafts -gt $specCap.max_draft_tokens) { throw "$Backend $SpeculativeMode draft count must be 1..$($specCap.max_draft_tokens)" }
    if ($AdaptiveMtp -and !$specCap.adaptive) { throw "$Backend $SpeculativeMode does not support adaptive drafts" }
    if ($NgramDrafts -and !$specCap.ngram) { throw "$Backend $SpeculativeMode does not support ngram drafts" }
    if ($Concurrency -gt $specCap.max_concurrency -or $Gpu.Split(',').Count -gt $specCap.max_devices) { throw "$Backend $SpeculativeMode does not support this concurrency or GPU count" }
    if ($specCap.draft_asset -eq 'embedded' -and $DraftModel) { throw 'NInfer uses the draft component embedded in the selected model' }
    if ($specCap.draft_asset -eq 'required_gguf' -and !$DraftModel) { throw 'DFlash2 on llama.cpp requires -DraftModel with a compatible draft GGUF' }
}
if ($DraftModel) {
    if (!(Test-Path -LiteralPath $DraftModel -PathType Leaf) -or [IO.Path]::GetExtension($DraftModel) -ne '.gguf') { throw 'DraftModel must be an existing GGUF file' }
    $DraftModel = (Resolve-Path -LiteralPath $DraftModel).Path
}
if (@('SpeculativeMode','DraftTokens','DraftPolicy','DraftModel') | Where-Object { $PSBoundParameters.ContainsKey($_) }) {
    if (@($WorkerArgs | Where-Object { $_ -match '^--(spec|spec-type|draft-tokens|spec-draft-n-max|adaptive-mtp|spec-draft-model)(=|$)' }).Count) { throw 'Use named speculative settings without conflicting native options in WorkerArgs' }
}
if (@($WorkerArgs | Where-Object { $_ -match '^--kvmem-disk-(path|mib)(=|$)' }).Count) {
    throw 'Use -DiskPath and -DiskMiB to configure shared active/idle SSD storage'
}
if (($PSBoundParameters.ContainsKey('DiskPath') -or ($PSBoundParameters.ContainsKey('DiskMiB') -and $DiskMiB -gt 0)) -and
    [string]::IsNullOrWhiteSpace($DiskPath)) {
    throw 'SSD storage requires a nonempty -DiskPath'
}
if ($DiskPath -and $DiskMiB -eq 0) { throw 'Omit -DiskPath when -DiskMiB is 0 (SSD disabled)' }
if (@('KvType', 'KvTypeK', 'KvTypeV') | Where-Object { $PSBoundParameters.ContainsKey($_) }) {
    if (@($WorkerArgs | Where-Object { $_ -match '^(--kv-dtype|--cache-type-[kv]|-ct[kv])(=|$)' }).Count) {
        throw 'Use -KvType/-KvTypeK/-KvTypeV without conflicting KV options in WorkerArgs'
    }
}
foreach ($setting in @(
    @('ApiKey', '^--api-key(-file)?(=|$)'),
    @('ModelAlias', '^(-a|--alias|--model-id)(=|$)'),
    @('LogLevel', '^(-lv|--verbosity|--log-verbosity|--log-level)(=|$)'),
    @('ChatTemplateFile', '^--chat-template(-file)?(=|$)'),
    @('ThinkingMode', '^--(enable-thinking|no-think(ing)?)(=|$)'),
    @('ReasoningEffort', '^--(default-reasoning-effort|reasoning-effort)(=|$)'),
    @('ThinkingBudget', '^--(default-thinking-budget|reasoning-budget)(=|$)'),
    @('MaxTokens', '^(-n|--n-predict|--predict|--default-max-tokens)(=|$)'),
    @('Vision', '^--(mmproj|mmproj-offload|no-mmproj-offload|mmproj-device|image-max-tokens)(=|$)'),
    @('Mmproj', '^--mmproj(=|$)'),
    @('Lora', '^--lora(-scale)?(=|$)'),
    @('PreserveThinking', '^--(preserve-thinking|chat-template-kwargs)(=|$)'),
    @('CpuVisionIdleSeconds', '^--cpu-vision-idle-seconds(=|$)'),
    @('Concurrency', '^(-np|--parallel|--max-concurrency)(=|$)')
)) {
    if ($PSBoundParameters.ContainsKey($setting[0]) -and @($WorkerArgs | Where-Object { $_ -match $setting[1] }).Count) {
        throw "Use -$($setting[0]) without a conflicting native option in WorkerArgs"
    }
}
if ($Lora -or $PSBoundParameters.ContainsKey('LoraScale')) {
    if (!$Lora -or !(Test-Path -LiteralPath $Lora -PathType Leaf) -or [IO.Path]::GetExtension($Lora) -ne '.gguf') {
        throw 'Lora requires an existing adapter GGUF'
    }
    if ([double]::IsNaN($LoraScale) -or [math]::Abs($LoraScale) -gt [float]::MaxValue) { throw 'LoraScale must be a finite float' }
    $Lora = (Resolve-Path -LiteralPath $Lora).Path
}
if ($ChatTemplateFile) {
    if (!(Test-Path -LiteralPath $ChatTemplateFile -PathType Leaf)) { throw 'ChatTemplateFile must be an existing template file' }
    $ChatTemplateFile = (Resolve-Path -LiteralPath $ChatTemplateFile).Path
}
if ($Backend -eq 'ninfer' -and $ThinkingMode -eq 'on') {
    throw 'NInfer has no native thinking-on startup override; use default or off (requests can select thinking)'
}
if ($Backend -eq 'ninfer' -and $ThinkingMode -eq 'off' -and
    ($ReasoningEffort -or @($WorkerArgs | Where-Object { $_ -match '^--default-reasoning-effort(=|$)' }).Count)) {
    throw 'ReasoningEffort conflicts with ThinkingMode off'
}
if (@($WorkerArgs | Where-Object { $_ -match '^--(no-)?kvmem($|-(budget|gen-reserve|host-mib|sessions|sink-tokens|recent-tokens|lrfu|lrfu-lambda|lrfu-top-frac|score-agg|query-scorer|graph-alpha)(=|$))' }).Count) {
    throw 'KVMem common options are owned by the launcher; use its named parameters instead of WorkerArgs'
}
if (!$Worker) {
    $file = if ($Backend -eq 'ninfer') { 'ninfer-kvmem-server.exe' } else { 'llama-kvmem-server.exe' }
    $Worker = Join-Path $root "workers/$Backend/$file"
}
foreach ($entry in @(@('Worker', $Worker), @('Model', $Model))) {
    if (!$entry[1] -or !(Test-Path -LiteralPath $entry[1] -PathType Leaf)) {
        throw "Missing $($entry[0]) file: $($entry[1])"
    }
}
$Worker = (Resolve-Path -LiteralPath $Worker).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
if ([IO.Path]::GetExtension($Model) -ne $capabilities.model_format) {
    throw "$Backend requires a $($capabilities.model_format) model"
}
if (!$Gpu -or $Gpu.Trim() -eq '-1') { throw 'Select a GPU UUID with -Gpu or CUDA_VISIBLE_DEVICES' }
if ($Backend -eq 'ninfer') {
    $gpuIds = @($Gpu.Split(',') | ForEach-Object { $_.Trim() })
    if ($gpuIds.Count -gt 8 -or @($gpuIds | Where-Object { !$_ -or $_ -eq '-1' }).Count -or
        @($gpuIds | Select-Object -Unique).Count -ne $gpuIds.Count) {
        throw 'ninfer requires 1..8 distinct GPU UUIDs in -Gpu, in stage order'
    }
    $Gpu = $gpuIds -join ','
    if ($StageLayers.Count -and ($gpuIds.Count -lt 2 -or $StageLayers.Count -ne $gpuIds.Count -or
        @($StageLayers | Where-Object { $_ -eq 0 }).Count)) {
        throw 'StageLayers requires one positive layer count per GPU and at least two GPUs'
    }
    if ($TensorParallel -and ($gpuIds.Count -lt 2 -or $StageLayers.Count)) {
        throw 'TensorParallel requires at least two GPUs and cannot be combined with StageLayers'
    }
    if ($TensorSplit.Count -and (!$TensorParallel -or $TensorSplit.Count -ne $gpuIds.Count -or
        @($TensorSplit | Where-Object { [double]::IsNaN($_) -or [double]::IsInfinity($_) -or $_ -le 0 }).Count)) {
        throw 'TensorSplit requires TensorParallel and one finite positive weight per GPU in UUID order'
    }
    $totalTensorWeight = [double]0
    foreach ($weight in $TensorSplit) { $totalTensorWeight += $weight }
    if ([double]::IsInfinity($totalTensorWeight)) { throw 'TensorSplit total weight must be finite' }
    if (@($WorkerArgs | Where-Object { $_ -match '^--(device|devices|stage-layers|tensor-parallel|tensor-split|tensor-attention)(=|$)' }).Count) {
        throw 'Use -Gpu, -StageLayers, -TensorParallel and -TensorSplit to configure ninfer devices'
    }
    if (@($WorkerArgs | Where-Object { $_ -match '^--default-thinking-budget(=|$)' }).Count) {
        throw 'Use -ThinkingBudget to configure the ninfer thinking budget'
    }
}
if (!$KvType) { $KvType = if ($Backend -eq 'ninfer') { 'int8' } else { 'q8_0' } }
if ($Backend -eq 'ninfer') {
    if ($PSBoundParameters.ContainsKey('KvTypeK') -or $PSBoundParameters.ContainsKey('KvTypeV')) {
        throw 'KvTypeK and KvTypeV configure llama.cpp only; use KvType for ninfer'
    }
    $requestedNgramDrafts = $NgramDrafts
    if ($Concurrency -gt 1 -and $NgramDrafts -gt 15) { $NgramDrafts = 15 }
    if ($KvType -notin @('bf16', 'int8', 'fp8', 'nvfp4', 'k8v4', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8')) {
        throw 'ninfer KvType must be bf16/int8/fp8/nvfp4/k8v4/rk8v4/rk4v4/rk4v4-e8/rk2v4-e8'
    }
    if ($AdaptiveMtp -and !$MtpDrafts) {
        throw 'Adaptive MTP requires MTP; image, ngram and disk limits still apply'
    }
    if ($FastPrefill -and ($KvType -notin @('int8', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8'))) {
        throw 'Fast prefill requires INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8'
    }
    if ($VisionResidency -ne 'resident' -and !$Vision) { throw 'CPU Vision requires -Vision' }
    if ($RecentTokens -lt 64) { throw 'ninfer -RecentTokens must be at least 64 to keep the newest append page' }
    $sinkBlocks = if ($SinkTokens -lt 64) { 1 } else { [int][Math]::Floor($SinkTokens / 64) }
    $recentBlocks = [int][Math]::Floor($RecentTokens / 64)
    if (($sinkBlocks + $recentBlocks) * 64 -gt $Budget) {
        throw 'ninfer sink and recent tokens exceed -Budget after rounding down to 64-token pages'
    }
    if ($NgramDrafts -and (!$MtpDrafts -or
        ([Math]::Max($MtpDrafts, $NgramDrafts) + $MtpDrafts) -gt $Reserve)) {
        throw 'ngram requires MTP and reserve >= max(MTP drafts, effective ngram drafts) + MTP drafts'
    }
    if ($requestedNgramDrafts -ne $NgramDrafts) {
        Write-Warning "NgramDrafts reduced from $requestedNgramDrafts to $NgramDrafts for Concurrency $Concurrency"
    }
    $nativeArgs = @($Model, '--host', $ListenHost, '--port', "$Port", '--max-context', "$resolvedContext",
        '--max-concurrency', "$Concurrency", '--prefill-chunk', "$Prefill",
        '--kvmem-budget', "$Budget", '--kvmem-gen-reserve', "$Reserve", '--kvmem-host-mib', "$HostMiB",
        '--kvmem-sink-tokens', "$SinkTokens", '--kvmem-recent-tokens', "$RecentTokens",
        '--kvmem-sessions', "$RetainedSessions", '--kv-dtype', $KvType, '--device-profile', $DeviceProfile)
    if ($MaxTokens -gt 0) { $nativeArgs += @('--default-max-tokens', "$MaxTokens") }
    if ($gpuIds.Count -gt 1) { $nativeArgs += @('--devices', ((0..($gpuIds.Count - 1)) -join ',')) }
    else { $nativeArgs += @('--device', '0') }
    if ($StageLayers.Count) { $nativeArgs += @('--stage-layers', ($StageLayers -join ',')) }
    if ($TensorParallel) {
        $nativeArgs += @('--tensor-parallel', '--tensor-attention', 'heads')
        if ($TensorSplit.Count) {
            $weights = @($TensorSplit | ForEach-Object { $_.ToString('G17', [Globalization.CultureInfo]::InvariantCulture) })
            $nativeArgs += @('--tensor-split', ($weights -join ','))
        }
    }
    if ($MtpDrafts) { $nativeArgs += @('--spec', $SpeculativeMode, '--draft-tokens', "$MtpDrafts",
        '--ngram-draft-tokens', "$NgramDrafts", '--ngram-min-match', "$NgramMinMatch") }
    if ($AdaptiveMtp) { $nativeArgs += '--adaptive-mtp' }
    if ($FastPrefill) { $nativeArgs += '--fast-prefill-kernel' }
    if ($Vision) { $nativeArgs += @('--vision', '--vision-residency', $VisionResidency, '--vision-max-merged', "$VisionTokens") }
    if ($Vision -and $VisionResidency -eq 'cpu') { $nativeArgs += @('--cpu-vision-idle-seconds', "$CpuVisionIdleSeconds") }
    if ($ThinkingBudget -gt 0) { $nativeArgs += @('--default-thinking-budget', "$ThinkingBudget") }
    if ($ReasoningEffort) { $nativeArgs += @('--default-reasoning-effort', $ReasoningEffort) }
    if ($PreserveThinking) { $nativeArgs += '--preserve-thinking' } # false/unset inherits the model template
} else {
    # Legacy KvType still sets both sides. Explicit K/V values override that base;
    # emit separate native flags when either independent value was requested.
    $separateKv = $PSBoundParameters.ContainsKey('KvTypeK') -or $PSBoundParameters.ContainsKey('KvTypeV')
    $effectiveK = if ($KvTypeK) { $KvTypeK } else { $KvType }
    $effectiveV = if ($KvTypeV) { $KvTypeV } else { $KvType }
    $aliases = @{ fp16='f16'; fp32='f32'; q8='q8_0'; q5='q5_0'; q4='q4_0' }
    if ($aliases.ContainsKey($effectiveK)) { $effectiveK = $aliases[$effectiveK] }
    if ($aliases.ContainsKey($effectiveV)) { $effectiveV = $aliases[$effectiveV] }
    $quantized = @('q8_0', 'q5_0', 'q4_0')
    $supported = $quantized + @('f16', 'f32')
    if ($effectiveK -cnotin $supported -or $effectiveV -cnotin $supported) {
        throw 'llama.cpp K/V types must use q8_0, q5_0, q4_0, f16 or f32'
    }
    if (($effectiveK -in $quantized) -ne ($effectiveV -in $quantized)) {
        throw 'llama.cpp cannot mix floating-point and quantized K/V; choose q8_0/q5_0/q4_0 for both, or f16/f32 for both'
    }
    foreach ($name in @('StageLayers', 'TensorParallel', 'TensorSplit', 'FastPrefill', 'NgramDrafts', 'NgramMinMatch', 'DeviceProfile', 'CpuVisionIdleSeconds')) {
        if ($PSBoundParameters.ContainsKey($name)) { throw "-$name configures ninfer; use llama native options in -WorkerArgs" }
    }
    if ($SpeculativeMode -ne 'none') {
        $nativeArgsSpec = @('--spec-type', $(if ($SpeculativeMode -eq 'mtp') { 'draft-mtp' } else { 'draft-dflash' }), '--spec-draft-n-max', "$MtpDrafts")
        if ($DraftModel) { $nativeArgsSpec += @('--spec-draft-model', $DraftModel) }
    } else { $nativeArgsSpec = @() }
    $nativeArgs = @('-m', $Model, '--host', $ListenHost, '--port', "$Port", '-c', "$resolvedContext",
        '-n', $(if ($MaxTokens -eq 0) { '-1' } else { "$MaxTokens" }), '--kvmem-budget', "$Budget", '--kvmem-gen-reserve', "$Reserve",
        '-b', "$Prefill", '-ub', "$llamaPhysicalPrefill",
        '--kvmem-sink-tokens', "$SinkTokens", '--kvmem-recent-tokens', "$RecentTokens")
    if ($separateKv) { $nativeArgs += @('--cache-type-k', $effectiveK, '--cache-type-v', $effectiveV) }
    else { $nativeArgs += @('--kv-dtype', $KvType) }
    $nativeArgs += @('--kvmem-host-mib', "$HostMiB", '--kvmem-sessions', "$RetainedSessions")
    if ($PSBoundParameters.ContainsKey('Concurrency')) { $nativeArgs += @('--parallel', "$Concurrency") }
    if ($PSBoundParameters.ContainsKey('ThinkingBudget')) {
        $nativeArgs += @('--reasoning-budget', $(if ($ThinkingBudget -eq 0) { '-1' } else { "$ThinkingBudget" }))
    }
    if ($ReasoningEffort) { $nativeArgs += @('--reasoning-effort', $ReasoningEffort) }
    if ($PSBoundParameters.ContainsKey('PreserveThinking')) {
        $nativeArgs += @('--chat-template-kwargs', (@{preserve_thinking=[bool]$PreserveThinking} | ConvertTo-Json -Compress))
    }
    if ($Lora) {
        $nativeArgs += @('--lora', $Lora, '--lora-scale', $LoraScale.ToString('R', [Globalization.CultureInfo]::InvariantCulture))
    }
    if ($Vision) {
        if (!$Mmproj) {
            $folder = Split-Path -Parent $Model
            $stem = [IO.Path]::GetFileNameWithoutExtension($Model)
            $exact = @(@((Join-Path $folder "$stem-mmproj.gguf"), (Join-Path $folder "mmproj-$stem.gguf")) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
            $candidates = @(if ($exact.Count) { $exact } else { Get-ChildItem -LiteralPath $folder -File -Filter '*mmproj*.gguf' | ForEach-Object { $_.FullName } })
            if ($candidates.Count -ne 1) { throw 'Vision requires -Mmproj: no unique adjacent projector was found; select the matching GGUF explicitly' }
            $Mmproj = $candidates[0]
        }
        if (!(Test-Path -LiteralPath $Mmproj -PathType Leaf) -or [IO.Path]::GetExtension($Mmproj) -ne '.gguf') { throw 'Mmproj must be an existing matching GGUF file' }
        $nativeArgs += @('--mmproj', (Resolve-Path -LiteralPath $Mmproj).Path,
            $(if ($VisionResidency -eq 'cpu') { '--no-mmproj-offload' } else { '--mmproj-offload' }), '--image-max-tokens', "$VisionTokens")
    }
}
if ($ApiKey) { $nativeArgs += @('--api-key', $ApiKey) }
if ($ModelAlias) { $nativeArgs += @($(if ($Backend -eq 'ninfer') { '--model-id' } else { '--alias' }), $ModelAlias) }
if ($ChatTemplateFile) { $nativeArgs += @($(if ($Backend -eq 'ninfer') { '--chat-template' } else { '--chat-template-file' }), $ChatTemplateFile) }
if ($LogLevel) {
    if ($Backend -eq 'ninfer') { $nativeArgs += @('--log-level', $LogLevel) }
    else { $nativeArgs += @('--verbosity', "$(@{off=0; error=1; warning=2; info=3; trace=4; debug=5}[$LogLevel])") }
}
if ($ThinkingMode -eq 'off') { $nativeArgs += $(if ($Backend -eq 'ninfer') { '--no-thinking' } else { '--no-think' }) }
if ($ThinkingMode -eq 'on') { $nativeArgs += '--enable-thinking' }
$nativeArgs += '--kvmem'
if ($Backend -eq 'llamacpp') { $nativeArgs += @('--kvmem-block-tokens', "$resolvedBlock") }
if ($Backend -eq 'llamacpp') { $nativeArgs += $nativeArgsSpec }
if ($DiskPath) { $nativeArgs += @('--kvmem-disk-path', $DiskPath, '--kvmem-disk-mib', "$DiskMiB") }
$nativeArgs += @(Get-KVMemQueryScoringArgs $PSBoundParameters)
$nativeArgs += $WorkerArgs
$launchEnvironment = @{ CUDA_VISIBLE_DEVICES = $Gpu; CUDA_DEVICE_ORDER = 'PCI_BUS_ID' }
if ($Backend -eq 'ninfer') {
    $launchEnvironment['KVMEM_CHECKPOINT_EXTRA_RECOMPUTE_TOKENS'] = "$CheckpointExtraRecomputeTokens"
}
if ($DryRun) {
    $displayArgs = @($nativeArgs)
    for ($i = 0; $i -lt $displayArgs.Count; $i++) {
        if ($displayArgs[$i] -eq '--api-key' -and $i + 1 -lt $displayArgs.Count) {
            $i++; $displayArgs[$i] = '<redacted>'
        } elseif ($displayArgs[$i] -match '^--api-key=') { $displayArgs[$i] = '--api-key=<redacted>' }
    }
    @{ backend = $Backend; argv = @($Worker) + $displayArgs; capabilities = $capabilities
       token_settings = $tokenPlan
       environment = $launchEnvironment } |
        ConvertTo-Json -Depth 6
    return
}
if ($Lora -or $TensorParallel -or $CheckpointExtraRecomputeTokens -gt 0) {
    $probeInfo = New-KVMemProcessInfo $Worker @('--kvmem-capabilities')
    $probeInfo.RedirectStandardOutput = $true
    $probeInfo.RedirectStandardError = $true
    $probeInfo.EnvironmentVariables['CUDA_VISIBLE_DEVICES'] = '-1'
    $probe = [Diagnostics.Process]::new()
    $probe.StartInfo = $probeInfo
    try {
        $null = $probe.Start()
        $stdout = $probe.StandardOutput.ReadToEndAsync()
        $stderr = $probe.StandardError.ReadToEndAsync()
        if (!$probe.WaitForExit(10000)) { $probe.Kill(); $probe.WaitForExit(); throw 'NInfer capability probe timed out' }
        $contract = $stdout.Result | ConvertFrom-Json
        $revision = if ($contract) { $contract.PSObject.Properties['heads_tensor_parallel_revision'] } else { $null }
        if ($probe.ExitCode -ne 0) { throw 'NInfer capability probe failed' }
        $loraCapability = if ($contract) { $contract.PSObject.Properties['fixed_lora'] } else { $null }
        if ($Lora -and (!$loraCapability -or $loraCapability.Value -isnot [bool] -or !$loraCapability.Value)) {
            throw 'Worker does not support fixed LoRA; update the llama.cpp worker'
        }
        if ($TensorParallel -and (!$revision -or
            ($revision.Value -isnot [int] -and $revision.Value -isnot [long]) -or $revision.Value -lt 1)) {
            throw 'Worker does not support KVMem heads tensor parallelism; update the complete NInfer package'
        }
        $checkpointRevision = if ($contract) { $contract.PSObject.Properties['checkpoint_recompute_revision'] } else { $null }
        if ($CheckpointExtraRecomputeTokens -gt 0 -and (!$checkpointRevision -or
            ($checkpointRevision.Value -isnot [int] -and $checkpointRevision.Value -isnot [long]) -or $checkpointRevision.Value -lt 1)) {
            throw 'Worker does not support checkpoint recompute limits; update the complete NInfer package'
        }
    } finally { $probe.Dispose() }
}
$info = New-KVMemProcessInfo $Worker $nativeArgs
foreach ($entry in $launchEnvironment.GetEnumerator()) { $info.EnvironmentVariables[$entry.Key] = $entry.Value }
$child = New-Object System.Diagnostics.Process
$child.StartInfo = $info
$started = $false
try {
    Write-Host "Starting $Backend at http://$ListenHost`:$Port/ (Ctrl+C to stop)"
    $started = $child.Start()
    while (!$child.WaitForExit(250)) {}
    if ($child.ExitCode -ne 0) { throw "$Backend exited with code $($child.ExitCode)" }
} finally {
    if ($started -and !$child.HasExited) { $child.Kill(); $child.WaitForExit() }
    $child.Dispose()
}
