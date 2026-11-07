# Consume token overrides in native argv order before normalization. Emit only the
# final canonical values, so a trailing raw option cannot undo the launch plan.
function Resolve-KVMemTokenOverrides([string]$Backend, [string[]]$Arguments,
    [int]$Context, [int]$Prefill, [int]$BlockTokens) {
    $values = @{Context=$Context; Prefill=$Prefill; BlockTokens=$BlockTokens}
    $aliases = @{}
    if ($Backend -eq 'ninfer') {
        $aliases = @{'--max-context'='Context'; '--prefill-chunk'='Prefill'}
    } elseif ($Backend -eq 'llamacpp') {
        $aliases = @{'-c'='Context'; '--ctx-size'='Context'; '-b'='Prefill'; '--batch-size'='Prefill'; '--kvmem-block-tokens'='BlockTokens'}
    }
    if (!$aliases.Count) { $values.WorkerArgs=$Arguments; return $values }
    # Zero-value options from the pinned native parsers (serve_options.cpp,
    # llama-kvmem-server.cpp, kvmem-server-options.h and cli_options.h).
    # All other native options take one value. Keep that value opaque, even
    # when it starts with '-' or contains '='. Update this list with backend CLI changes.
    $switches = @('--help','-h','--kvmem','--no-kvmem','--kvmem-lrfu','--no-kvmem-lrfu')
    if ($Backend -eq 'ninfer') {
        $switches += @(
            '--version','--models-autoload','--no-models-autoload','--kvmem-verify-transfers',
            '--fast-prefill-kernel','--use-alt-prefix-caching','--use-original-prefix-caching',
            '--disk-kv-restore','--disk-kv-directstorage','--release-diverged-checkpoints',
            '--concurrent-prefill','--recover-invariant-failures','--thorough-admission-search',
            '--recency-eviction','--value-aware-demote','--kv-lease-growth','--auto-long-anchors',
            '--branch-anchors','--no-endpoint-anchors','--tensor-parallel','--ngram-native-sessions',
            '--vision','--vision-cpu','--no-cuda-graph','--no-prefix-reuse','--lenient-assistant-history',
            '--lenient-image-detail','--derive-session-keys','--auto-save-evicted','--auto-prefix-grid',
            '--lm-head-draft','--full-draft-head','--adaptive-mtp','--lm-head-q4','--lm-head-q6',
            '--embedding-q4','--embedding-q6','--mtp-experts-q4','--gdn-state-fp16','--rope-yarn',
            '--wddm-evictable-budget','--model-suspend','--no-auto-resume','--mlp-a8-decode',
            '--no-prefill-a8','--nvfp4-w4a8-int','--nvfp4-w4a16','--prefill-cublas',
            '--no-prefill-cublas-projections','--no-thinking','--preserve-thinking','--cors',
            '--no-webui','--webui-mcp-proxy','--structured-output','--unconstrained-response-format',
            '--assistant-prefill','--usage-chunk-choice','--greedy','--post-thinking'
        )
    } else {
        $switches += @(
            '--usage','--mmproj-offload','--no-mmproj-offload','--no-ui','--no-webui','--ui','--webui',
            '--kvmem-harvest-v','--jinja','--no-jinja','--enable-thinking','--no-think','--metrics',
            '--kvmem-trace','--no-kvmem-trace','--list-devices','--mmap','--no-mmap','--mlock'
        )
    }
    $remaining = [Collections.Generic.List[string]]::new()
    for ($i=0; $i -lt $Arguments.Count; $i++) {
        $parts = $Arguments[$i] -split '=', 2
        $flag = $parts[0]
        if ($aliases.Keys -cnotcontains $flag) {
            $remaining.Add($Arguments[$i])
            if ($parts.Count -eq 1 -and $switches -cnotcontains $flag -and $i+1 -lt $Arguments.Count) {
                $i++; $remaining.Add($Arguments[$i])
            }
            continue
        }
        if ($parts.Count -eq 2) { $raw=$parts[1] }
        else {
            if ($i+1 -ge $Arguments.Count) { throw "$flag requires an integer value" }
            $i++; $raw=$Arguments[$i]
        }
        $key=$aliases[$flag]; $value=0
        $minimum=if ($Backend -eq 'ninfer' -and $key -eq 'Prefill') { 0 } else { 1 }
        if (![int]::TryParse($raw, [ref]$value) -or $value -lt $minimum) {
            throw "$flag requires an integer in [$minimum, 2147483647]"
        }
        $values[$key]=$value
    }
    $values.WorkerArgs=$remaining.ToArray()
    return $values
}

# Resolve the public launch plan before either preview or process creation.
function Resolve-KVMemTokenSettings([string]$Backend, [int]$BlockTokens,
    [long]$Budget, [long]$Reserve, [long]$Prefill, [long]$SinkTokens, [long]$RecentTokens, [long]$Context,
    [string]$RecallStrategy='legacy', $ExplicitParameters=@{}) {
    $block = if ($Backend -eq 'ninfer') { 64 } else { $BlockTokens }
    $values = [ordered]@{Budget=$Budget; Reserve=$Reserve; SinkTokens=$SinkTokens; RecentTokens=$RecentTokens; Prefill=$Prefill; Context=$Context}
    $requested = [ordered]@{}; foreach ($key in $values.Keys) { $requested[$key]=$values[$key] }
    $adjustments = [Collections.Generic.List[object]]::new()
    if($RecallStrategy -eq 'layerwise') {
        if($Budget -le 0 -or $Budget % $block -or $Reserve -lt $block -or $Reserve % $block) {throw 'Layerwise Budget/Reserve require block alignment and Reserve >= one block'}
        foreach($key in @('SinkTokens','RecentTokens')) {
            if(!$ExplicitParameters.ContainsKey($key)) {$requested[$key]=$null;$values[$key]=$null}
            elseif($values[$key] -lt 0 -or $values[$key] % $block) {throw "Layerwise $key requires block alignment; explicit zero is preserved"}
        }
        if($null -ne $values.SinkTokens -and $null -ne $values.RecentTokens -and $values.SinkTokens+$values.RecentTokens -gt $Budget) {throw 'SinkTokens + RecentTokens must fit layerwise startup Budget'}
        if($Budget+$Reserve -gt $Context -or $Prefill -le 0) {throw 'Layerwise Budget + Reserve must fit Context and Prefill must be positive'}
        return @{requested=$requested;effective=$values;block_tokens=$block;adjustments=@()}
    }
    foreach ($key in @('Budget','Reserve','SinkTokens','RecentTokens')) {
        $minimum = switch ($key) {
            Budget { if ($Backend -eq 'llamacpp') { $block } else { 128 } }
            Reserve { $block }
            SinkTokens { if ($Backend -eq 'polystrata') { 0 } else { $block } }
            RecentTokens { if ($Backend -eq 'llamacpp') { 0 } else { $block } }
        }
        $value = [long][Math]::Floor($values[$key] / $block) * $block
        if (($Backend -eq 'llamacpp' -and $key -in @('Budget','Reserve')) -or
            ($Backend -ne 'polystrata' -and $key -eq 'SinkTokens')) { $value=[Math]::Max($minimum,$value) }
        if ($value -lt $minimum) { throw "$key must retain at least $minimum tokens after alignment" }
        $values[$key]=$value
    }
    if ($Backend -in @('ninfer','polystrata')) {
        # Native ninfer's zero sentinel selects 1024 before applying the KVMem R limit.
        if ($Backend -eq 'ninfer' -and $Prefill -eq 0) { $Prefill=1024 }
        $alignment=if ($Backend -eq 'ninfer') { 128 } else { 1 }
        $values.Prefill=[long][Math]::Floor([Math]::Min($Prefill,$values.Reserve)/$alignment)*$alignment
        if ($values.Prefill -lt $alignment) { throw "Prefill and Reserve must allow at least $alignment prefill tokens" }
    }
    if ($values.SinkTokens + $values.RecentTokens -gt $values.Budget) { throw 'SinkTokens + RecentTokens must fit Budget after alignment' }
    if ($values.Budget + $values.Reserve -gt $Context) { throw 'Budget + Reserve must fit Context after alignment' }
    foreach ($key in $values.Keys) {
        if ($requested[$key] -ne $values[$key]) {
            $reason=if ($key -eq 'Prefill') { 'prefill alignment and Reserve limit' } else { "alignment to $block tokens" }
            $adjustments.Add(@{parameter=$key; requested=$requested[$key]; effective=$values[$key]; reason=$reason})
        }
    }
    return @{requested=$requested; effective=$values; block_tokens=$block; adjustments=@($adjustments.ToArray())}
}
