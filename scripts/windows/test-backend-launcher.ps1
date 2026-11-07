#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$launcher = Join-Path $PSScriptRoot 'start-backend.ps1'
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$testDir = Join-Path $tempRoot ('kvmem-backend-test-' + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path $testDir
try {
    foreach ($name in 'worker.exe', 'model with spaces.ninfer', 'model.gguf') {
        [IO.File]::WriteAllText((Join-Path $testDir $name), 'dry-run fixture')
    }
    $common = @{ Worker = (Join-Path $testDir 'worker.exe'); Gpu = 'GPU-test'; DryRun = $true; WarningAction = 'SilentlyContinue' }
    $template = Join-Path $testDir 'chat template.jinja'
    [IO.File]::WriteAllText($template, 'test template')
    foreach ($backend in 'llamacpp', 'ninfer') {
        $model = Join-Path $testDir $(if ($backend -eq 'ninfer') { 'model with spaces.ninfer' } else { 'model.gguf' })
        $plan = & $launcher @common -Backend $backend -Model $model -ApiKey 'test-key-123' `
            -ModelAlias 'Friendly model' -ChatTemplateFile $template -LogLevel debug -ThinkingMode off -Concurrency 2 | ConvertFrom-Json
        $aliasFlag = if ($backend -eq 'ninfer') { '--model-id' } else { '--alias' }
        $templateFlag = if ($backend -eq 'ninfer') { '--chat-template' } else { '--chat-template-file' }
        $parallelFlag = if ($backend -eq 'ninfer') { '--max-concurrency' } else { '--parallel' }
        $thinkingFlag = if ($backend -eq 'ninfer') { '--no-thinking' } else { '--no-think' }
        foreach ($pair in @(@($aliasFlag, 'Friendly model'), @($templateFlag, $template), @($parallelFlag, '2'), @('--api-key', '<redacted>'))) {
            if ($plan.argv[[array]::IndexOf($plan.argv, $pair[0]) + 1] -ne $pair[1]) { throw 'Common UI field lost its native mapping' }
        }
        if ($plan.argv -notcontains $thinkingFlag -or $plan.argv -contains 'test-key-123') { throw 'Thinking mapping or preview redaction failed' }
        foreach ($level in 'off', 'error', 'warning', 'info', 'trace', 'debug') {
            $levels = & $launcher @common -Backend $backend -Model $model -LogLevel $level | ConvertFrom-Json
            $flag = if ($backend -eq 'ninfer') { '--log-level' } else { '--verbosity' }
            $expected = if ($backend -eq 'ninfer') { $level } else { "$(@{off=0; error=1; warning=2; info=3; trace=4; debug=5}[$level])" }
            if ($levels.argv[[array]::IndexOf($levels.argv, $flag) + 1] -ne $expected) { throw 'Log level mapping changed native levels' }
        }
        $defaults = & $launcher @common -Backend $backend -Model $model -ThinkingMode default | ConvertFrom-Json
        if ($defaults.argv -contains '--kvmem-lrfu' -or $defaults.argv[[array]::IndexOf($defaults.argv,'--kvmem-query-scorer')+1] -ne 'page-softmax') { throw 'Default scorer changed' }
        foreach ($variant in @(@{QueryScorer='meanq-cosine'},@{QueryScorer='meanq-cosine';NoLrfu=$true},@{ScoreAggregation='max'},@{GraphAlpha=0})) {
            $chosen = & $launcher @common -Backend $backend -Model $model @variant | ConvertFrom-Json
            $cosine = $variant['QueryScorer'] -eq 'meanq-cosine'
            $expectedHistory = $cosine -and !$variant['NoLrfu']
            if (($chosen.argv -contains '--kvmem-lrfu') -ne $expectedHistory) { throw 'Scorer history default or override lost' }
            $agg = if ($cosine -or $variant['ScoreAggregation'] -eq 'max') { 'max' } else { 'mean' }
            if ($chosen.argv[[array]::IndexOf($chosen.argv,'--kvmem-score-agg')+1] -ne $agg) { throw 'GQA aggregation default lost' }
        }
        foreach ($scorer in 'page-softmax','meanq-cosine') {
            foreach ($history in $false,$true) { foreach ($alpha in 0,0.2,1) {
                $chosen = & $launcher @common -Backend $backend -Model $model -QueryScorer $scorer -Lrfu:$history -GraphAlpha $alpha | ConvertFrom-Json
                if (($chosen.argv -contains '--kvmem-lrfu') -ne $history -or
                    [double]$chosen.argv[[array]::IndexOf($chosen.argv,'--kvmem-graph-alpha')+1] -ne $alpha) {
                    throw 'Independent Query controls were lost in the launch plan'
                }
            } }
        }
        foreach ($conflict in @(@{Lrfu=$true;NoLrfu=$true}, @{GraphAlpha=-0.1}, @{LrfuLambda=0.3})) {
            $rejected = $false
            try { $null = & $launcher @common -Backend $backend -Model $model @conflict } catch { $rejected = $true }
            if (!$rejected) { throw 'Accepted an incompatible scoring combination' }
        }
        if (@($defaults.argv | Where-Object { $_ -match '^--(enable-thinking|no-think(ing)?)$' }).Count) { throw 'Default thinking must inherit the native default' }
        $limits = & $launcher @common -Backend $backend -Model $model -QueryMaxTokens 160 -QueryReplayMaxTokens 320 | ConvertFrom-Json
        foreach ($pair in @(@('--kvmem-query-max-tokens','160'), @('--kvmem-query-replay-max-tokens','320'))) {
            if ($limits.argv[[array]::IndexOf($limits.argv,$pair[0])+1] -ne $pair[1]) { throw 'Query/replay limit lost in launch plan' }
        }
        foreach ($case in @(@{Concurrency=9}, @{ApiKey='two,keys'}, @{ChatTemplateFile=(Join-Path $testDir 'missing.jinja')},
            @{ModelAlias='name'; WorkerArgs=@($aliasFlag, 'duplicate')})) {
            $rejected = $false
            try { $null = & $launcher @common -Backend $backend -Model $model @case } catch { $rejected = $true }
            if (!$rejected) { throw 'Invalid or conflicting new service setting accepted' }
        }
    }
    $thinkingOn = & $launcher @common -Backend llamacpp -Model (Join-Path $testDir 'model.gguf') -ThinkingMode on -Concurrency 8 | ConvertFrom-Json
    foreach ($threshold in 0,16,2147483647) {
        $checkpoint = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') -CheckpointExtraRecomputeTokens $threshold | ConvertFrom-Json
        if ($checkpoint.environment.KVMEM_CHECKPOINT_EXTRA_RECOMPUTE_TOKENS -ne "$threshold") { throw 'Checkpoint recompute environment lost' }
    }
    $checkpointDefault = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') | ConvertFrom-Json
    if ($checkpointDefault.environment.KVMEM_CHECKPOINT_EXTRA_RECOMPUTE_TOKENS -ne '0') { throw 'Checkpoint recompute default changed' }
    foreach ($case in @(@{Backend='ninfer';CheckpointExtraRecomputeTokens=-1}, @{Backend='llamacpp';CheckpointExtraRecomputeTokens=0})) {
        $rejected = $false
        try { $null = & $launcher @common -Model (Join-Path $testDir 'model with spaces.ninfer') @case } catch { $rejected = $true }
        if (!$rejected) { throw 'Invalid checkpoint recompute setting accepted' }
    }
    if ($thinkingOn.argv -notcontains '--enable-thinking') { throw 'llama thinking-on mapping failed' }
    $splitKv = & $launcher @common -Backend llamacpp -Model (Join-Path $testDir 'model.gguf') -KvTypeK q8_0 -KvTypeV q4_0 | ConvertFrom-Json
    if ($splitKv.argv -contains '--kv-dtype' -or
        $splitKv.argv[[array]::IndexOf($splitKv.argv, '--cache-type-k') + 1] -ne 'q8_0' -or
        $splitKv.argv[[array]::IndexOf($splitKv.argv, '--cache-type-v') + 1] -ne 'q4_0') {
        throw 'Independent K/V settings lost their separate native flags'
    }
    foreach ($case in @(
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); ThinkingMode='on'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); KvTypeK='q8_0'},
        @{Backend='llamacpp'; Model=(Join-Path $testDir 'model.gguf'); KvTypeK='q8_0'; KvTypeV='f16'},
        @{Backend='llamacpp'; Model=(Join-Path $testDir 'model.gguf'); KvTypeK='q8_0'; WorkerArgs=@('--cache-type-v', 'q5_0')},
        @{Backend='llamacpp'; Model=(Join-Path $testDir 'model.gguf'); Concurrency=9}
    )) {
        $rejected = $false
        try { $null = & $launcher @common @case } catch { $rejected = $true }
        if (!$rejected) { throw 'Unsupported thinking or KV combination accepted' }
    }
    $native = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -WorkerArgs @('--chat-template', 'template with spaces.jinja') | ConvertFrom-Json
    if ($native.backend -ne 'ninfer' -or $native.argv[1] -ne (Join-Path $testDir 'model with spaces.ninfer') -or
        $native.argv[-1] -ne 'template with spaces.jinja' -or
        $native.environment.CUDA_VISIBLE_DEVICES -ne 'GPU-test' -or !$native.capabilities.multimodal) {
        throw 'ninfer dry-run lost model, argv, device or capability boundaries'
    }
    if ($native.capabilities.disk_restore -or $native.capabilities.supported_combinations.cold_disk.enabled -or
        $native.capabilities.supported_combinations.adaptive_mtp.cold_disk -or
        $native.argv -contains '--kvmem-disk-path' -or $native.argv -contains '--kvmem-disk-mib') {
        throw 'ninfer advertised or enabled disabled cold snapshots'
    }
    if ($native.capabilities.supported_combinations.lookup_ngram.enabled -or
        $native.capabilities.supported_combinations.lookup_ngram.reason -ne 'legacy_path_disabled') {
        throw 'ninfer advertised disabled legacy lookup drafting'
    }
    foreach ($lookupArgs in @(
        @('--lookup-ngram', '4'), @('--lookup-ngram=4'),
        @('--lookup-ngram', '0'), @('--lookup-ngram=0'), @('--lookup-ngram')
    )) {
        foreach ($drafts in @(0, 3)) {
            $disabled = $false
            try {
                $null = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                    -MtpDrafts $drafts -NgramDrafts $(if ($drafts) { 15 } else { 0 }) -WorkerArgs $lookupArgs
            } catch {
                $disabled = $_.Exception.Message -match 'disables legacy --lookup-ngram.*-NgramDrafts'
            }
            if (!$disabled) { throw 'Launcher allowed disabled legacy lookup drafting' }
        }
    }
    $dualArgs = $common.Clone()
    $dualArgs.Gpu = ' GPU-second , GPU-first '
    $dual = & $launcher @dualArgs -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -StageLayers 30,34 -Vision -VisionResidency cpu -MtpDrafts 3 | ConvertFrom-Json
    if ($dual.environment.CUDA_VISIBLE_DEVICES -ne 'GPU-second,GPU-first' -or
        $dual.argv -contains '--device' -or
        $dual.argv[[array]::IndexOf($dual.argv, '--devices') + 1] -ne '0,1' -or
        $dual.argv[[array]::IndexOf($dual.argv, '--stage-layers') + 1] -ne '30,34' -or
        $dual.capabilities.supported_combinations.multi_gpu.mode -ne 'layer') {
        throw 'Multi-GPU launcher lost UUID order, stage split or capability'
    }
    $tensor = & $launcher @dualArgs -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -TensorParallel -TensorSplit 3,1 | ConvertFrom-Json
    if ($tensor.environment.CUDA_VISIBLE_DEVICES -ne 'GPU-second,GPU-first' -or
        $tensor.argv -notcontains '--tensor-parallel' -or $tensor.argv -contains '--stage-layers' -or
        $tensor.argv[[array]::IndexOf($tensor.argv, '--tensor-attention') + 1] -ne 'heads' -or
        $tensor.argv[[array]::IndexOf($tensor.argv, '--tensor-split') + 1] -ne '3,1') {
        throw 'Heads TP launcher lost UUID order or weights'
    }
    $automaticTensor = & $launcher @dualArgs -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -TensorParallel | ConvertFrom-Json
    if ($automaticTensor.argv -contains '--tensor-split') { throw 'Automatic TP weights must be omitted' }
    foreach ($badTensor in @(
        @{TensorParallel=$true;StageLayers=@(30,34)}, @{TensorSplit=@(3,1)},
        @{TensorParallel=$true;TensorSplit=@(3)}, @{TensorParallel=$true;TensorSplit=@(0,1)},
        @{TensorParallel=$true;TensorSplit=@([double]::NaN,1)},
        @{TensorParallel=$true;TensorSplit=@([double]::PositiveInfinity,1)},
        @{TensorParallel=$true;TensorSplit=@(1e308,1e308)}
    )) {
        $rejected = $false
        try { $null = & $launcher @dualArgs -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') @badTensor }
        catch { $rejected = $_.Exception.Message -match 'TensorParallel|TensorSplit' }
        if (!$rejected) { throw 'Invalid TP configuration accepted' }
    }
    foreach ($nativeTensor in @(@('--tensor-attention','positions'), @('--tensor-split=3,1'), @('--tensor-parallel'))) {
        $rejected = $false
        try { $null = & $launcher @dualArgs -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') -WorkerArgs $nativeTensor }
        catch { $rejected = $_.Exception.Message -match 'Use -Gpu, -StageLayers, -TensorParallel and -TensorSplit' }
        if (!$rejected) { throw 'Native arguments bypassed the heads-only TP controls' }
    }
    foreach ($badGpu in @('GPU-test,GPU-test', 'GPU-test,', ',GPU-test', '-1,GPU-test', '0,1,2,3,4,5,6,7,8')) {
        $argsBad = $common.Clone(); $argsBad.Gpu = $badGpu
        $rejected = $false
        try { $null = & $launcher @argsBad -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') }
        catch { $rejected = $_.Exception.Message -match 'distinct GPU UUIDs' }
        if (!$rejected) { throw 'Invalid multi-GPU list accepted' }
    }
    foreach ($badSplit in @(@(64), @(0,64), @(20,20,24))) {
        $rejected = $false
        try { $null = & $launcher @dualArgs -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') -StageLayers $badSplit }
        catch { $rejected = $_.Exception.Message -match 'one positive layer count per GPU' }
        if (!$rejected) { throw 'Invalid stage layer counts accepted' }
    }
    $llamaModel = Join-Path $testDir 'model.gguf'
    $llamaDual = & $launcher @dualArgs -Backend llamacpp -Model $llamaModel -MtpDrafts 3 -TensorSplit 11,9 | ConvertFrom-Json
    if ($llamaDual.environment.CUDA_VISIBLE_DEVICES -ne 'GPU-second,GPU-first' -or
        $llamaDual.argv[[array]::IndexOf($llamaDual.argv, '--split-mode') + 1] -ne 'layer' -or
        $llamaDual.argv[[array]::IndexOf($llamaDual.argv, '--gpu-layers') + 1] -ne 'all' -or
        $llamaDual.argv[[array]::IndexOf($llamaDual.argv, '--device') + 1] -ne 'CUDA0,CUDA1' -or
        $llamaDual.argv[[array]::IndexOf($llamaDual.argv, '--tensor-split') + 1] -ne '11,9' -or
        $llamaDual.argv[[array]::IndexOf($llamaDual.argv, '--kvmem-mtp-state') + 1] -ne 'replay' -or
        $llamaDual.argv -contains '--devices' -or $llamaDual.argv -contains '--stage-layers') {
        throw 'llama.cpp layer split lost UUID order, weights or MTP history'
    }
    $llamaAuto = & $launcher @dualArgs -Backend llamacpp -Model $llamaModel | ConvertFrom-Json
    if ($llamaAuto.argv -contains '--tensor-split' -or $llamaAuto.argv -contains '--kvmem-mtp-state') {
        throw 'Automatic llama.cpp split weights and unused MTP history must be omitted'
    }
    $llamaSingle = & $launcher @common -Backend llamacpp -Model $llamaModel -MtpDrafts 3 | ConvertFrom-Json
    if ($llamaSingle.argv -contains '--split-mode' -or $llamaSingle.argv -contains '--device' -or
        $llamaSingle.argv -contains '--kvmem-mtp-state') {
        throw 'Single-GPU llama.cpp must keep its native device defaults'
    }
    foreach ($badLlama in @(
        @{Gpu='GPU-first'; TensorSplit=@(11)}, @{Gpu='GPU-first'; TensorSplit=@(11,9)},
        @{TensorSplit=@(11)}, @{TensorSplit=@(11,0)}, @{TensorSplit=@([double]::NaN,1)},
        @{TensorSplit=@(11,9); TensorParallel=$true}, @{TensorSplit=@(11,9); StageLayers=@(40,24)}
    )) {
        $argsBad = $dualArgs.Clone(); $rejected = $false
        if ($badLlama.ContainsKey('Gpu')) { $argsBad.Gpu = $badLlama.Gpu; $badLlama.Remove('Gpu') | Out-Null }
        try { $null = & $launcher @argsBad -Backend llamacpp -Model $llamaModel @badLlama }
        catch { $rejected = $_.Exception.Message -match 'llama\.cpp TensorSplit|configures ninfer' }
        if (!$rejected) { throw "Invalid llama.cpp split configuration accepted: $($badLlama | Out-String)" }
    }
    foreach ($nativeDevice in @(@('--device', 'CUDA1'), @('--split-mode=layer'), @('--gpu-layers', 'all'), @('--tensor-split', '11,9'))) {
        $rejected = $false
        try { $null = & $launcher @dualArgs -Backend llamacpp -Model $llamaModel -WorkerArgs $nativeDevice }
        catch { $rejected = $_.Exception.Message -match 'Use -Gpu and -TensorSplit' }
        if (!$rejected) { throw 'Native arguments bypassed the llama.cpp device controls' }
    }
    foreach ($diskCase in @(
        @{DiskPath=''}, @{DiskMiB=4096},
        @{WorkerArgs=@('--kvmem-disk-path', 'cache')},
        @{WorkerArgs=@('--kvmem-disk-mib', '0')},
        @{WorkerArgs=@('--kvmem-disk-path=cache')}
    )) {
        $disabled = $false
        try { $null = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') @diskCase }
        catch { $disabled = $_.Exception.Message -match 'DiskPath|DiskMiB' }
        if (!$disabled) { throw 'Launcher allowed ambiguous SSD configuration' }
    }
    foreach ($backend in 'ninfer', 'llamacpp') {
        $model = if ($backend -eq 'ninfer') { 'model with spaces.ninfer' } else { 'model.gguf' }
        $disk = & $launcher @common -Backend $backend -Model (Join-Path $testDir $model) `
            -DiskPath 'cache with spaces' -DiskMiB 1024 -HostMiB 512 | ConvertFrom-Json
        if ($disk.argv[[array]::IndexOf($disk.argv, '--kvmem-disk-path') + 1] -ne 'cache with spaces' -or
            $disk.argv[[array]::IndexOf($disk.argv, '--kvmem-disk-mib') + 1] -ne '1024' -or
            $disk.argv[[array]::IndexOf($disk.argv, '--kvmem-host-mib') + 1] -ne '512' -or
            !$disk.capabilities.active_kv_tiering -or $disk.capabilities.ssd_persistence -ne 'process') {
            throw 'Shared H/D launcher mapping changed ownership or path boundaries'
        }
    }
    foreach ($backend in 'llamacpp', 'ninfer') {
        foreach ($count in 2, 8) {
            $model = Join-Path $testDir $(if ($backend -eq 'ninfer') {'model with spaces.ninfer'} else {'model.gguf'})
            $ssd = & $launcher @common -Backend $backend -Model $model -Concurrency $count `
                -DiskPath 'cache' -DiskMiB 1024 -HostMiB 512 | ConvertFrom-Json
            if ($ssd.argv -notcontains '--kvmem-disk-path' -or $ssd.argv -notcontains '--kvmem-disk-mib') {
                throw 'Concurrent public SSD parameters were lost'
            }
        }
    }
    $fastDisk = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -FastPrefill -DiskPath 'cache' -MtpDrafts 3 -AdaptiveMtp | ConvertFrom-Json
    if ($fastDisk.argv -notcontains '--kvmem-disk-path' -or $fastDisk.argv -notcontains '--fast-prefill-kernel') {
        throw 'SSD mapping incorrectly rejected independent prefill configuration'
    }
    $mtp = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -MtpDrafts 3 -Concurrency 2 -RetainedSessions 8 | ConvertFrom-Json
    if ($mtp.argv -notcontains '--ngram-draft-tokens' -or $mtp.argv -notcontains '--kvmem-sessions' -or
        $mtp.capabilities.supported_combinations.text.max_concurrency -ne 8) { throw 'MTP/concurrency mapping failed' }
    foreach ($drafts in @(1, 4, 5, 15)) {
        $fixed = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
            -MtpDrafts $drafts | ConvertFrom-Json
        if ($fixed.argv[[array]::IndexOf($fixed.argv, '--draft-tokens') + 1] -ne "$drafts" -or
            $fixed.capabilities.supported_combinations.text.mtp_drafts -notcontains $drafts) {
            throw 'Fixed MTP draft count mapping failed'
        }
    }
    $adaptive = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -MtpDrafts 15 -AdaptiveMtp | ConvertFrom-Json
    if ($adaptive.argv -notcontains '--adaptive-mtp') { throw 'Adaptive MTP mapping failed' }
    $fast = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -KvType rk8v4 -Concurrency 4 -FastPrefill -MtpDrafts 15 -AdaptiveMtp | ConvertFrom-Json
    if ($fast.argv -notcontains '--fast-prefill-kernel') { throw 'Fast RK8V4 concurrency mapping failed' }
    foreach ($format in 'int8', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8') {
        foreach ($lanes in 1..8) {
            foreach ($residency in 'resident', 'cpu') {
                $fastImage = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                    -KvType $format -Concurrency $lanes -FastPrefill -Vision -VisionResidency $residency `
                    -MtpDrafts 3 -AdaptiveMtp -NgramDrafts 31 | ConvertFrom-Json
                if ($fastImage.argv -notcontains '--fast-prefill-kernel' -or
                    $fastImage.argv -notcontains '--vision' -or
                    $fastImage.argv[[array]::IndexOf($fastImage.argv, '--kv-dtype') + 1] -ne $format -or
                    $fastImage.argv[[array]::IndexOf($fastImage.argv, '--ngram-draft-tokens') + 1] -ne "$(if($lanes -gt 1){15}else{31})" -or
                    $fastImage.capabilities.supported_combinations.fast_prefill.text_only -or
                    $fastImage.capabilities.supported_combinations.fast_prefill.max_concurrency -ne 8 -or
                    $fastImage.capabilities.supported_combinations.fast_prefill.kv -notcontains $format) {
                    throw 'Fast image/concurrency/ngram mapping or capability failed'
                }
            }
        }
    }
    $cpu = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -KvType rk8v4 -Vision -VisionResidency cpu -MtpDrafts 15 -AdaptiveMtp | ConvertFrom-Json
    if ($cpu.argv[[array]::IndexOf($cpu.argv, '--vision-residency') + 1] -ne 'cpu') { throw 'CPU Vision mapping failed' }
    foreach ($residency in 'resident', 'cpu') {
        foreach ($lanes in 1..8) {
            $imageLanes = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                -Vision -VisionResidency $residency -Concurrency $lanes -MtpDrafts 3 -AdaptiveMtp | ConvertFrom-Json
            if ($imageLanes.argv[[array]::IndexOf($imageLanes.argv, '--max-concurrency') + 1] -ne "$lanes" -or
                $imageLanes.argv[[array]::IndexOf($imageLanes.argv, '--vision-max-merged') + 1] -ne '1024' -or
                $imageLanes.argv[[array]::IndexOf($imageLanes.argv, '--ngram-draft-tokens') + 1] -ne '0' -or
                $imageLanes.capabilities.supported_combinations.images.max_concurrency -ne 8) {
                throw 'Concurrent Vision mapping or default preprocessing limit failed'
            }
        }
    }
    $copy = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -MtpDrafts 3 -NgramDrafts 31 -KvType rk8v4 | ConvertFrom-Json
    if ($copy.argv[[array]::IndexOf($copy.argv, '--ngram-draft-tokens') + 1] -ne '31') {
        throw 'ngram mapping failed'
    }
    if ($copy.argv -contains '--default-thinking-budget') { throw 'thinking budget was enabled by default' }
    $budget = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -ThinkingBudget 8192 | ConvertFrom-Json
    if ($budget.argv[[array]::IndexOf($budget.argv, '--default-thinking-budget') + 1] -ne '8192') {
        throw 'thinking budget mapping failed'
    }
    $budgetRejected = $false
    try {
        $null = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
            -WorkerArgs @('--default-thinking-budget', '8192')
    } catch {
        $budgetRejected = $_.Exception.Message -match 'Use -ThinkingBudget'
    }
    if (!$budgetRejected) { throw 'native thinking-budget argument was accepted' }
    if ($copy.argv[[array]::IndexOf($copy.argv, '--kvmem-sink-tokens') + 1] -ne '12288' -or
        $copy.argv[[array]::IndexOf($copy.argv, '--kvmem-recent-tokens') + 1] -ne '1024') {
        throw 'default sink/recent windows were not 12288/1024 tokens'
    }
    $windows = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -SinkTokens 128 -RecentTokens 127 | ConvertFrom-Json
    if ($windows.argv[[array]::IndexOf($windows.argv, '--kvmem-sink-tokens') + 1] -ne '128' -or
        $windows.argv[[array]::IndexOf($windows.argv, '--kvmem-recent-tokens') + 1] -ne '64') {
        throw 'sink/recent mapping failed'
    }
    foreach ($recent in @(0, 1, 63)) {
        $recentRejected = $false
        try {
            $null = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                -RecentTokens $recent
        } catch { $recentRejected = $true }
        if (!$recentRejected) { throw 'recent smaller than one 64-token page was accepted' }
    }
    $windowsRejected = $false
    try {
        $null = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
            -WorkerArgs @('--kvmem-recent-tokens', '128')
    } catch {
        $windowsRejected = $_.Exception.Message -match 'KVMem common options are owned by the launcher'
    }
    if (!$windowsRejected) { throw 'native sink/recent argument was accepted' }
    $windowsFit = $false
    try {
        $null = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
            -Budget 128 -RecentTokens 128
    } catch {
        $windowsFit = $_.Exception.Message -match 'SinkTokens \+ RecentTokens must fit Budget'
    }
    if (!$windowsFit) { throw 'sink plus recent larger than the budget was accepted' }
    $llama = & $launcher @common -Backend llamacpp -Model (Join-Path $testDir 'model.gguf') `
        -WorkerArgs @('--spec-type', 'draft-mtp') | ConvertFrom-Json
    if ($llama.argv -notcontains 'draft-mtp' -or !$llama.capabilities.speculative_decoding -or
        !$llama.capabilities.disk_restore -or $llama.argv -notcontains '--kvmem-host-mib') { throw 'llama native options changed' }
    foreach ($backend in 'ninfer', 'llamacpp') {
        $model = if ($backend -eq 'ninfer') { 'model with spaces.ninfer' } else { 'model.gguf' }
        $disabled = & $launcher @common -Backend $backend -Model (Join-Path $testDir $model) `
            -RetainedSessions 2 -NoLrfu -DiskMiB 0 | ConvertFrom-Json
        if ($disabled.argv -notcontains '--kvmem' -or
            $disabled.argv -contains '--kvmem-lrfu' -or $disabled.argv -contains '--kvmem-disk-path' -or
            $disabled.argv[[array]::IndexOf($disabled.argv, '--kvmem-sessions') + 1] -ne '2') {
            throw 'Common KVMem enablement, sessions or off switches were not exported consistently'
        }
        foreach ($flag in '--kvmem', '--no-kvmem', '--kvmem-budget', '--kvmem-sessions', '--no-kvmem-lrfu') {
            $rejected = $false
            try { $null = & $launcher @common -Backend $backend -Model (Join-Path $testDir $model) -WorkerArgs @($flag) }
            catch { $rejected = $_.Exception.Message -match 'KVMem common options' }
            if (!$rejected) { throw 'Duplicate common KVMem argument accepted' }
        }
    }
    foreach ($residency in 'resident', 'cpu') {
        foreach ($adaptive in $false, $true) {
            $imageCopy = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                -MtpDrafts 4 -AdaptiveMtp:$adaptive -NgramDrafts 31 -Vision -VisionResidency $residency | ConvertFrom-Json
            if ($imageCopy.argv -notcontains '--vision' -or
                $imageCopy.capabilities.supported_combinations.ngram.text_only -or
                $imageCopy.capabilities.supported_combinations.ngram.image_kv -notcontains 'k8v4' -or
                $imageCopy.capabilities.supported_combinations.ngram.image_kv -notcontains 'bf16' -or
                $imageCopy.capabilities.supported_combinations.ngram.image_kv -notcontains 'rk8v4' -or
                $imageCopy.capabilities.supported_combinations.ngram.image_kv -notcontains 'int8' -or
                $imageCopy.capabilities.supported_combinations.ngram.image_kv -notcontains 'nvfp4' -or
                $imageCopy.argv[[array]::IndexOf($imageCopy.argv, '--vision-residency') + 1] -ne $residency -or
                $imageCopy.argv[[array]::IndexOf($imageCopy.argv, '--ngram-draft-tokens') + 1] -ne '31' -or
                (($imageCopy.argv -contains '--adaptive-mtp') -ne $adaptive)) { throw 'Image/ngram mapping failed' }
        }
    }
    foreach ($format in 'nvfp4', 'k8v4') {
        foreach ($drafts in 0..4) {
            foreach ($lanes in 1..4) {
                $nvfp4 = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                    -KvType $format -MtpDrafts $drafts -Concurrency $lanes | ConvertFrom-Json
                if ($nvfp4.argv[[array]::IndexOf($nvfp4.argv, '--kv-dtype') + 1] -ne $format -or
                    $nvfp4.capabilities.supported_combinations.text.kv -notcontains $format -or
                    ($drafts -and $nvfp4.argv[[array]::IndexOf($nvfp4.argv, '--ngram-draft-tokens') + 1] -ne '0')) {
                    throw 'NVFP4/K8V4 text/MTP mapping or disabled ngram settings failed'
                }
            }
            $image = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                -KvType $format -MtpDrafts $drafts -Vision | ConvertFrom-Json
            if ($image.argv -notcontains '--vision' -or $image.capabilities.supported_combinations.images.kv -notcontains $format) {
                throw 'NVFP4/K8V4 resident-image mapping failed'
            }
        }
    }
    foreach ($lanes in 2..8) {
        foreach ($kv in 'bf16', 'int8', 'fp8', 'nvfp4', 'k8v4', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8') {
            foreach ($width in 0, 1, 15, 16, 31, 63) {
                foreach ($adaptive in $false, $true) {
                    $copyLanes = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                        -Concurrency $lanes -KvType $kv -MtpDrafts 3 -AdaptiveMtp:$adaptive -NgramDrafts $width | ConvertFrom-Json
                    if ($copyLanes.argv[[array]::IndexOf($copyLanes.argv, '--ngram-draft-tokens') + 1] -ne "$([Math]::Min($width, 15))" -or
                        $copyLanes.capabilities.supported_combinations.ngram.max_concurrency -ne 8 -or
                        $copyLanes.capabilities.supported_combinations.ngram.concurrent_max_drafts -ne 15 -or
                        $copyLanes.capabilities.supported_combinations.ngram.concurrent_width_policy -ne 'clamp_to_15') {
                        throw 'Concurrent ngram mapping or capability failed'
                    }
                    foreach ($residency in 'resident', 'cpu') {
                        $image = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                            -Concurrency $lanes -KvType $kv -MtpDrafts 3 -AdaptiveMtp:$adaptive -NgramDrafts $width `
                            -Vision -VisionResidency $residency | ConvertFrom-Json
                        if ($image.argv[[array]::IndexOf($image.argv, '--kv-dtype') + 1] -ne $kv -or
                            $image.argv[[array]::IndexOf($image.argv, '--ngram-draft-tokens') + 1] -ne "$([Math]::Min($width, 15))" -or
                            $image.capabilities.supported_combinations.ngram.image_kv -notcontains $kv) {
                            throw 'Concurrent image/ngram KV mapping failed'
                        }
                    }
                }
            }
        }
    }
    $clamped = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -MtpDrafts 3 -NgramDrafts 31 -Concurrency 8 -WarningVariable normalizationWarnings | ConvertFrom-Json
    if ($clamped.argv[[array]::IndexOf($clamped.argv, '--ngram-draft-tokens') + 1] -ne '15' -or
        $normalizationWarnings.Count -ne 1 -or
        "$($normalizationWarnings[0])" -notmatch '31 to 15.*Concurrency 8') {
        throw 'Concurrent ngram clamp must report the requested and effective widths once'
    }
    foreach ($backend in 'ninfer','llamacpp') {
        $model = Join-Path $testDir $(if ($backend -eq 'ninfer') { 'model with spaces.ninfer' } else { 'model.gguf' })
        $rounded = & $launcher @common -Backend $backend -Model $model -Budget 10001 -Reserve 1001 -Prefill 999 `
            -SinkTokens 101 -RecentTokens 101 -WarningVariable alignmentWarnings | ConvertFrom-Json
        $effective=$rounded.token_settings.effective
        $expectedReserve=if ($backend -eq 'ninfer') { 960 } else { 992 }
        if ($effective.Budget -ne 9984 -or $effective.Reserve -ne $expectedReserve) { throw 'budget/reserve alignment failed' }
        if ($effective.Prefill -ne $(if ($backend -eq 'ninfer') { 896 } else { 999 })) { throw 'prefill policy mismatch' }
        if ($backend -eq 'llamacpp' -and $rounded.argv[[Array]::IndexOf($rounded.argv,'--kvmem-block-tokens')+1] -ne '32') { throw 'llama default block size must be explicit in the launch plan' }
        if ($alignmentWarnings.Count -ne $rounded.token_settings.adjustments.Count) { throw 'adjustments must be logged once' }
        foreach ($item in $rounded.token_settings.adjustments) {
            if ($item.effective -gt $item.requested) { throw 'ordinary rounding increased a requested budget' }
        }
    }
    # Values that look like token options must remain opaque to the launcher.
    foreach ($backend in 'ninfer','llamacpp') {
        $model=Join-Path $testDir $(if ($backend -eq 'ninfer') { 'model with spaces.ninfer' } else { 'model.gguf' })
        $nameFlag=if ($backend -eq 'ninfer') { '--model-id' } else { '--alias' }
        $messageFlag=if ($backend -eq 'ninfer') { '--thinking-budget-message' } else { '--reasoning-budget-message' }
        $optionValues=if ($backend -eq 'ninfer') { @('--max-context=524288','--prefill-chunk') } else { @('--ctx-size=524288','-b','--kvmem-block-tokens=128') }
        foreach ($value in $optionValues) {
            $opaqueArgs=@($nameFlag,$value,'--chat-template',$value,$messageFlag,$value)
            $opaque = & $launcher @common -Backend $backend -Model $model -WorkerArgs $opaqueArgs | ConvertFrom-Json
            if ($opaque.token_settings.effective.Context -ne 262144 -or
                $opaque.token_settings.effective.Prefill -ne 128 -or
                $opaque.token_settings.block_tokens -ne $(if ($backend -eq 'ninfer') { 64 } else { 32 })) {
                throw 'An opaque native string value changed token settings'
            }
            for ($at=0; $at -lt $opaqueArgs.Count; $at++) {
                if ($opaque.argv[$opaque.argv.Count-$opaqueArgs.Count+$at] -cne $opaqueArgs[$at]) {
                    throw 'Native option/value boundaries were lost'
                }
            }
        }
        # Real overrides after both string values and zero-value switches still apply.
        $mixedArgs=if ($backend -eq 'ninfer') {
            @('--model-id','--max-context=256','--no-thinking','--max-context','524288','--greedy','--prefill-chunk','999')
        } else {
            @('-a','--ctx-size=256','--metrics','--ctx-size','524288','--no-mmap','--batch-size','777')
        }
        $mixed = & $launcher @common -Backend $backend -Model $model -WorkerArgs $mixedArgs | ConvertFrom-Json
        $expectedPrefill=if ($backend -eq 'ninfer') { 896 } else { 777 }
        if ($mixed.token_settings.effective.Context -ne 524288 -or
            $mixed.token_settings.effective.Prefill -ne $expectedPrefill -or
            $mixed.argv[-3] -cne $mixedArgs[1] -or $mixed.argv[-2] -cne $mixedArgs[2] -or $mixed.argv[-1] -cne $mixedArgs[5]) {
            throw 'String values or switches hid a subsequent real token override'
        }
    }
    foreach ($rawPrefill in 256,999,0) {
        $override = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
            -Budget 10001 -Reserve 1001 -Prefill 999 -SinkTokens 101 -RecentTokens 101 `
            -WorkerArgs @('--prefill-chunk','128',"--prefill-chunk=$rawPrefill",'--chat-template','template with spaces.jinja') `
            -WarningVariable overrideWarnings | ConvertFrom-Json
        $expected=if ($rawPrefill -eq 256) { 256 } else { 896 }
        if ($override.token_settings.requested.Prefill -ne $rawPrefill -or
            $override.token_settings.effective.Prefill -ne $expected -or
            @($override.argv | Where-Object { $_ -match '^--prefill-chunk(=|$)' }).Count -ne 1 -or
            $override.argv[[Array]::IndexOf($override.argv,'--prefill-chunk')+1] -ne "$expected" -or
            $override.argv[-1] -ne 'template with spaces.jinja' -or
            $overrideWarnings.Count -ne $override.token_settings.adjustments.Count) {
            throw 'Native prefill override must be normalized once and match preview, warnings and argv'
        }
    }
    foreach ($backend in 'ninfer','llamacpp') {
        $model=Join-Path $testDir $(if ($backend -eq 'ninfer') { 'model with spaces.ninfer' } else { 'model.gguf' })
        $flag=if ($backend -eq 'ninfer') { '--max-context' } else { '-c' }
        $alias=if ($backend -eq 'ninfer') { '--max-context' } else { '--ctx-size' }
        $expanded = & $launcher @common -Backend $backend -Model $model -Context 256 -Budget 300000 `
            -WorkerArgs @($flag,'1024',"$alias=524288") | ConvertFrom-Json
        if ($expanded.token_settings.effective.Context -ne 524288 -or
            $expanded.argv[[Array]::IndexOf($expanded.argv,$flag)+1] -ne '524288' -or
            @($expanded.argv | Where-Object { $_ -match '^(--max-context|--ctx-size|-c)(=|$)' }).Count -ne 1) {
            throw 'Context overrides must be resolved before capacity validation and emitted once'
        }
        $rejected=$false
        try { $null = & $launcher @common -Backend $backend -Model $model -WorkerArgs @($alias,'256') }
        catch { $rejected=$_.Exception.Message -match 'Budget \+ Reserve must fit Context' }
        if (!$rejected) { throw 'Capacity validation ignored the smaller native context override' }
    }
    $batchOverride = & $launcher @common -Backend llamacpp -Model (Join-Path $testDir 'model.gguf') -Prefill 999 `
        -WorkerArgs @('-b','256','--batch-size=777') | ConvertFrom-Json
    if ($batchOverride.token_settings.effective.Prefill -ne 777 -or
        $batchOverride.argv[[Array]::IndexOf($batchOverride.argv,'-b')+1] -ne '777' -or
        $batchOverride.argv[[Array]::IndexOf($batchOverride.argv,'-ub')+1] -ne '999' -or
        @($batchOverride.argv | Where-Object { $_ -match '^(-b|--batch-size)(=|$)' }).Count -ne 1) {
        throw 'Logical batch override changed physical batch or disagreed with preview'
    }
    foreach ($invalid in @(@('--prefill-chunk'), @('--prefill-chunk','bad'), @('--prefill-chunk','-1'),
        @('--prefill-chunk=2147483648'), @('--max-context=0'))) {
        $rejected=$false
        try { $null = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') -WorkerArgs $invalid }
        catch { $rejected=$_.Exception.Message -match 'requires an integer' }
        if (!$rejected) { throw 'Invalid native token override accepted' }
    }
    $customBlock = & $launcher @common -Backend llamacpp -Model (Join-Path $testDir 'model.gguf') `
        -Budget 10001 -Reserve 1001 -SinkTokens 101 -RecentTokens 101 `
        -WorkerArgs @('--kvmem-block-tokens','64','--kvmem-block-tokens=128') | ConvertFrom-Json
    if ($customBlock.token_settings.block_tokens -ne 128 -or $customBlock.token_settings.effective.Reserve -ne 896 -or
        $customBlock.token_settings.effective.SinkTokens -ne 128 -or $customBlock.token_settings.effective.RecentTokens -ne 0 -or
        @($customBlock.argv | Where-Object { $_ -match '^--kvmem-block-tokens(=|$)' }).Count -ne 1) {
        throw 'explicit block size must use the native last-value rule and preserve minimum sink protection'
    }
    $bad = @(
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); MtpDrafts=16},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); MtpDrafts=-1},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model.gguf')},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); Budget=385},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); Reserve=64},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); Vision=$true; Concurrency=9},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); AdaptiveMtp=$true; MtpDrafts=0},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); FastPrefill=$true; KvType='k8v4'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); FastPrefill=$true; KvType='bf16'; Vision=$true},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); FastPrefill=$true; KvType='fp8'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); FastPrefill=$true; KvType='nvfp4'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); KvType='nvfp4'; Vision=$true; Concurrency=9},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); KvType='nvfp4'; NgramDrafts=31},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); KvType='unknown-kv'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); KvType='nvpf4'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); NgramDrafts=31},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); MtpDrafts=3; NgramDrafts=64; Concurrency=2},
        @{Backend='llamacpp'; Model=(Join-Path $testDir 'model.gguf'); DiskMiB=1024}
    )
    foreach ($argsCase in $bad) {
        $rejected = $false
        try { $null = & $launcher @common @argsCase } catch { $rejected = $true }
        if (!$rejected) { throw 'Invalid backend configuration was accepted' }
    }
    foreach ($backend in 'ninfer', 'llamacpp') {
        $description = & $launcher -Backend $backend -Describe | ConvertFrom-Json
        if ($description.backend -ne $backend -or !$description.text_chat) { throw 'Capability query failed' }
    }
    Write-Output 'Backend launcher PASS (argv, paths, capability selection, invalid combinations)'
} finally {
    $resolved = [IO.Path]::GetFullPath($testDir)
    if (!$resolved.StartsWith($tempRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($resolved) -notlike 'kvmem-backend-test-*') { throw 'Unexpected fixture directory' }
    Get-ChildItem -LiteralPath $resolved -File | ForEach-Object { Remove-Item -LiteralPath $_.FullName }
    Remove-Item -LiteralPath $resolved
}
