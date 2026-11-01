# Data-only adapter from the shared launcher to PolyStrata's Python frontend.
param([Parameter(Mandatory)][System.Collections.IDictionary]$Parameters,
      [Parameter(Mandatory)][string]$Root,[string[]]$WorkerArgs=@(), [hashtable]$TokenPlan)
$ErrorActionPreference='Stop'
if ($TokenPlan) {
    $copy=@{}; foreach ($key in $Parameters.Keys) { $copy[$key]=$Parameters[$key] }
    foreach ($key in $TokenPlan.effective.Keys) { $copy[$key]=$TokenPlan.effective[$key] }
    $Parameters=$copy
}
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'query-scoring.ps1')
if($Parameters.ContainsKey('ReplayMaxTokens')) {
    if($Parameters.ContainsKey('QueryReplayMaxTokens') -and $Parameters.QueryReplayMaxTokens -ne $Parameters.ReplayMaxTokens) { throw 'ReplayMaxTokens conflicts with QueryReplayMaxTokens' }
    $Parameters['QueryReplayMaxTokens']=$Parameters.ReplayMaxTokens
}
function Setting($name,$default) { if($Parameters.ContainsKey($name)) { $Parameters[$name] } else { $default } }
foreach($name in @('StageLayers','TensorParallel','TensorSplit','FastPrefill','NgramDrafts','NgramMinMatch','DeviceProfile','KvTypeK','KvTypeV','AdaptiveMtp','LogLevel')) {
    if($Parameters.ContainsKey($name)) { throw "PolyStrata does not expose -$name in this integration" }
}
if((Setting 'ReasoningEffort' '') -notin @('','none','low','medium','high','xhigh')) { throw 'PolyStrata reasoning effort must be none, low, medium or high' }
if(((Setting 'ThinkingMode' 'default') -eq 'off' -and (Setting 'ReasoningEffort' '') -notin @('','none')) -or ((Setting 'ThinkingMode' 'default') -eq 'on' -and (Setting 'ReasoningEffort' '') -eq 'none')) { throw 'ThinkingMode conflicts with ReasoningEffort' }
$mode=Setting 'SpeculativeMode' $(if((Setting 'MtpDrafts' 0) -gt 0){'mtp'}else{'none'})
$count=Setting 'DraftTokens' (Setting 'MtpDrafts' $(if($mode -eq 'mtp'){2}else{0}))
if($Parameters.ContainsKey('MtpDrafts') -and $Parameters.ContainsKey('DraftTokens') -and $Parameters.MtpDrafts -ne $count) { throw 'DraftTokens conflicts with MtpDrafts' }
if($mode -notin @('none','mtp') -or ($mode -eq 'none' -and $count) -or ($mode -eq 'mtp' -and ($count -lt 1 -or $count -gt 3)) -or (Setting 'DraftPolicy' 'adaptive') -notin @('fixed','adaptive') -or (Setting 'DraftModel' '')) { throw 'PolyStrata supports native adaptive embedded MTP with a 1..3 draft ceiling in solo execution; native batch uses ordinary decode' }
$gpu=Setting 'Gpu' $env:CUDA_VISIBLE_DEVICES
if($gpu -notmatch '^GPU-[0-9a-fA-F-]{8,64}$') { throw 'PolyStrata requires one GPU UUID' }
$worker=Setting 'Worker' (Join-Path $Root 'workers/polystrata/strata-poly.exe')
if(!$worker) { $worker=Join-Path $Root 'workers/polystrata/strata-poly.exe' }
$model=Setting 'Model' ''
$tree=Setting 'PolystrataSource' (Join-Path $Root 'backends/polystrata')
if(!$tree) { $tree=Join-Path $Root 'backends/polystrata' }
$tokenizer=Setting 'Tokenizer' ''
foreach($entry in @(@('Worker',$worker),@('Model',$model),@('frontend',(Join-Path $tree 'serve/server.py')))) {
    if(!$entry[1] -or !(Test-Path -LiteralPath $entry[1] -PathType Leaf)) { throw "Missing PolyStrata $($entry[0]): $($entry[1])" }
}
$python=Setting 'Python' (Join-Path $Root 'runtime/python/python.exe')
if(!$python) { $python=Join-Path $Root 'runtime/python/python.exe' }
if(!(Test-Path -LiteralPath $python -PathType Leaf)) {
    if(Setting 'Python' '') {throw "Missing Python executable: $python"}
    $python=(Get-Command python -ErrorAction Stop).Source
}
$worker=(Resolve-Path -LiteralPath $worker).Path;$model=(Resolve-Path -LiteralPath $model).Path;$python=(Resolve-Path -LiteralPath $python).Path
$arguments=@((Join-Path $Root 'tools/polystrata/server.py'),'--source',$tree,'--worker',$worker,'--model',$model,'--gpu',$gpu)
$arguments+=@('--kvmem-block-tokens',[string](Setting 'BlockTokens' 32))
$arguments+=@('--thinking-mode',[string](Setting 'ThinkingMode' 'default'))
if($Parameters.ContainsKey('ThinkingBudget')) { $arguments+=@('--thinking-budget',[string]$Parameters.ThinkingBudget) }
if(Setting 'ReasoningEffort' '') { $arguments+=@('--reasoning-effort',$(if($Parameters.ReasoningEffort -eq 'xhigh'){'high'}else{[string]$Parameters.ReasoningEffort})) }
if(Setting 'PreserveThinking' $true) { $arguments+='--preserve-thinking' } else { $arguments+='--no-preserve-thinking' }
if($tokenizer) {
    if(!(Test-Path -LiteralPath (Join-Path $tokenizer 'tokenizer.json') -PathType Leaf)) {throw "Missing PolyStrata tokenizer: $tokenizer"}
    $arguments+=@('--tokenizer',(Resolve-Path -LiteralPath $tokenizer).Path)
}
$map=@{Host='ListenHost';Port='Port';Context='Context';Budget='Budget';Reserve='Reserve';'host-mib'='HostMiB';Prefill='Prefill';'max-tokens'='MaxTokens';'vram-reserve-mib'='VramReserveMiB';Sink='SinkTokens';Recent='RecentTokens';'retained-sessions'='RetainedSessions';Concurrency='Concurrency';'model-alias'='ModelAlias';'chat-template-file'='ChatTemplateFile';'disk-path'='DiskPath';'disk-mib'='DiskMiB'}
$defaults=@{ListenHost='127.0.0.1';Port=18200;Context=262144;Budget=32768;Reserve=1024;HostMiB=16384;Prefill=512;MaxTokens=1024;VramReserveMiB=256;SinkTokens=12288;RecentTokens=1024;RetainedSessions=[Math]::Max(4,1+[int](Setting 'Concurrency' 1));Concurrency=1;QueryMaxTokens=512;ReplayMaxTokens=2048}
foreach($key in $map.Keys) {
    $name=$map[$key]
    if($Parameters.ContainsKey($name)) { $arguments+=@(('--'+$key.ToLowerInvariant()),[string]$Parameters[$name]) }
    elseif($defaults.ContainsKey($name)) { $arguments+=@(('--'+$key.ToLowerInvariant()),[string]$defaults[$name]) }
}
$kv=Setting 'KvType' 'f16'
if($kv -notin @('f16','q8_0')) { throw 'PolyStrata KV format must be f16 or q8_0' }
$arguments+=@('--kv-type',$kv,'--mtp-drafts',[string]$count)
# Reuse the shared default/validation contract, translating only flag names.
$scoring=@(Get-KVMemQueryScoringArgs $Parameters)
foreach($value in $scoring) {
    if($value -eq '--no-kvmem-lrfu') { $arguments+='--no-lrfu' }
    elseif($value -eq '--kvmem-query-replay-max-tokens') { $arguments+='--replay-max-tokens' }
    elseif($value -like '--kvmem-*') { $arguments+=($value -replace '^--kvmem-','--') }
    else { $arguments+=$value }
}
if(Setting 'Vision' $false) {
    $visionWorker=Setting 'VisionWorker' (Join-Path (Split-Path $worker) 'strata-vision.exe')
    $mmproj=Setting 'Mmproj' ''
    if(!$visionWorker) { $visionWorker=Join-Path (Split-Path $worker) 'strata-vision.exe' }
    if(!$mmproj) {
        $folder=Split-Path $model
        $stem=[IO.Path]::GetFileNameWithoutExtension($model)
        $exact=@(@((Join-Path $folder "$stem-mmproj.gguf"),(Join-Path $folder "mmproj-$stem.gguf")) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
        $candidates=@(if($exact.Count) { $exact } else { Get-ChildItem -LiteralPath $folder -File -Force | Where-Object { $_.Name -match 'mmproj' -and $_.Extension -eq '.gguf' -and $_.FullName -ne $model } | ForEach-Object { $_.FullName } })
        if($candidates.Count -ne 1) { throw 'Vision requires -Mmproj: no unique adjacent projector was found; select the matching GGUF in Templates & vision > Vision assets' }
        $mmproj=$candidates[0]
    }
    if([IO.Path]::GetExtension($mmproj) -ne '.gguf') { throw 'Mmproj must be an existing matching GGUF file' }
    foreach($asset in @($visionWorker,$mmproj)) { if(!(Test-Path -LiteralPath $asset -PathType Leaf)) { throw "Missing PolyStrata vision asset: $asset" } }
    $mmproj=(Resolve-Path -LiteralPath $mmproj).Path
    $arguments+=@('--vision-worker',$visionWorker,'--mmproj',$mmproj,'--vision-tokens',[string](Setting 'VisionTokens' 1024),'--vision-device',$(if((Setting 'VisionResidency' 'resident') -eq 'cpu'){'cpu'}else{'gpu'}))
}
if($WorkerArgs.Count) { $arguments+=@('--')+$WorkerArgs }
$environment=@{CUDA_DEVICE_ORDER='PCI_BUS_ID'}
if(Setting 'ApiKey' '') { $environment.STRATA_API_KEY=$Parameters.ApiKey }
if(Setting 'DryRun' $false) {
    $safeEnv=@{};foreach($key in $environment.Keys) { $safeEnv[$key]=if($key -eq 'STRATA_API_KEY'){'<redacted>'}else{$environment[$key]} }
    $checkInfo=New-KVMemProcessInfo $python (@($arguments[0],'--dry-run')+@($arguments | Select-Object -Skip 1))
    foreach($key in $environment.Keys) { $checkInfo.EnvironmentVariables[$key]=$environment[$key] }
    $checkInfo.RedirectStandardOutput=$true;$checkInfo.RedirectStandardError=$true
    $check=[Diagnostics.Process]::Start($checkInfo)
    $outputTask=$check.StandardOutput.ReadToEndAsync();$errorTask=$check.StandardError.ReadToEndAsync()
    try {
        if(!$check.WaitForExit(30000)) { $check.Kill();throw 'PolyStrata configuration check timed out' }
        $checkedOutput=$outputTask.Result;$checkedError=$errorTask.Result
        if($check.ExitCode) {
            $reason=($checkedError -split "`n" | Where-Object { $_ -match 'error:' } | Select-Object -Last 1) -replace '^.*?error:\s*',''
            if(!$reason) { $reason=$checkedError.Trim() }
            throw "[KVMEM_CONFIG] $reason"
        }
        $resolved=$checkedOutput | ConvertFrom-Json
        $safeEnv.STRATA_FATTN=$resolved.env.STRATA_FATTN
        $safeEnv.STRATA_FDEC=$resolved.env.STRATA_FDEC
    } finally { $check.Dispose() }
    @{token_settings=$TokenPlan;effective_settings=@{block_tokens=[int](Setting 'BlockTokens' 32);draft_policy=$(if($count){'adaptive'}else{'fixed'});draft_tokens=$count;model_alias=$resolved.model_name;max_tokens=$resolved.kvmem_default_max_tokens;preserve_thinking=$resolved.preserve_thinking;vram_reserve_mib=$resolved.kvmem_vram_reserve_mib};backend='polystrata';native_argv=@($resolved.exe)+@($resolved.args);argv=@($python)+$arguments;environment=$safeEnv;capabilities=@{architecture='qwen35moe';gpu_count=1;batch_mtp=$false;solo_mtp=($count -gt 0);concurrency=(Setting 'Concurrency' 1);kv_type=$kv;storage_quota='process'}} | ConvertTo-Json -Depth 8
    return
}
$info=New-KVMemProcessInfo $python $arguments
foreach($key in $environment.Keys) { $info.EnvironmentVariables[$key]=$environment[$key] }
$child=[Diagnostics.Process]::Start($info)
try { while(!$child.WaitForExit(250)) {} ; if($child.ExitCode) { throw "PolyStrata exited with code $($child.ExitCode)" } }
finally { if(!$child.HasExited) { $child.Kill();$child.WaitForExit() };$child.Dispose() }

