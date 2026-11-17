function Get-KVMemRecallArgs($Bound, [string]$Backend = '', [string]$Mode = '') {
    if (!$Backend) { $Backend=if($Bound.ContainsKey('Backend')){$Bound.Backend}else{'llamacpp'} }
    if (!$Mode) {
        $Mode=if($Bound.ContainsKey('SpeculativeMode')){$Bound.SpeculativeMode}elseif(($Bound.ContainsKey('MtpDrafts') -and $Bound.MtpDrafts -gt 0) -or $Bound.ContainsKey('Recipe')){'mtp'}else{'none'}
    }
    $strategy=if($Bound.ContainsKey('RecallStrategy')){$Bound.RecallStrategy}else{'legacy'}
    $resources=[ordered]@{
        LayerwiseActivationRamMiB=@('--kvmem-layerwise-activation-ram-mib',256)
        LayerwiseActivationDiskMiB=@('--kvmem-layerwise-activation-disk-mib',0)
        LayerwiseStagingMiB=@('--kvmem-layerwise-staging-mib',16)
        LayerwiseMetadataMiB=@('--kvmem-layerwise-metadata-mib',256)
        LayerwiseActivationMetadataMiB=@('--kvmem-layerwise-activation-metadata-mib',128)
    }
    if ($strategy -notin @('legacy','layerwise')) {throw 'RecallStrategy must be legacy or layerwise'}
    $history=if($Bound.ContainsKey('LayerwiseMtpHistory')){$Bound.LayerwiseMtpHistory}else{''}
    if ($history -and ($history -notin @('upstream-global','first-layer') -or $strategy -ne 'layerwise' -or $Mode -ne 'mtp')) {throw 'LayerwiseMtpHistory requires layerwise recall and OWN MTP'}
    if ($Mode -eq 'guess' -and ($strategy -ne 'layerwise' -or $Backend -ne 'polystrata')) {throw 'Guess is the layerwise PolyStrata Gemma shared-target mode'}
    if ($Bound.ContainsKey('LayerwiseCheckpointTarget') -and $strategy -ne 'layerwise') {
        throw 'LayerwiseCheckpointTarget requires layerwise recall'
    }
    if ($strategy -eq 'legacy') {
        foreach($key in $resources.Keys) {if($Bound.ContainsKey($key)){throw 'Layerwise resource limits require RecallStrategy=layerwise'}}
        return
    }
    $values=@{}
    foreach($key in $resources.Keys) {
        if(!$Bound.ContainsKey($key)){continue}
        $value=[long]$Bound[$key]
        if($value -lt 0 -or $value -gt 2147483647 -or ($key -notin @('LayerwiseActivationRamMiB','LayerwiseActivationDiskMiB') -and !$value)) {throw "$key requires a finite positive MiB budget"}
        $values[$key]=$value
    }
    if($values.ContainsKey('LayerwiseActivationRamMiB') -and $values.ContainsKey('LayerwiseActivationDiskMiB') -and
        !$values.LayerwiseActivationRamMiB -and !$values.LayerwiseActivationDiskMiB) {throw 'Layerwise activation RAM or SSD budget must be positive'}
    @('--kvmem-recall-strategy','layerwise')
    foreach($key in $resources.Keys) {if($values.ContainsKey($key)){@($resources[$key][0],[string]$values[$key])}}
    if($Bound.ContainsKey('LayerwiseCheckpointTarget')) {@('--kvmem-layerwise-checkpoint-target',[string]$Bound.LayerwiseCheckpointTarget)}
    if($history) {@('--kvmem-layerwise-mtp-history',$history)}
    if($Backend -eq 'llamacpp' -and $Mode -eq 'mtp') {@('--kvmem-mtp-state','replay')}
}

function Remove-KVMemImplicitRecallBands([string[]]$Arguments,$Bound) {
    if(!$Bound.ContainsKey('RecallStrategy') -or $Bound.RecallStrategy -ne 'layerwise') {return $Arguments}
    for($i=0;$i -lt $Arguments.Count;$i++) {
        $key=if($Arguments[$i] -eq '--kvmem-sink-tokens'){'SinkTokens'}elseif($Arguments[$i] -eq '--kvmem-recent-tokens'){'RecentTokens'}else{''}
        if($key -and !$Bound.ContainsKey($key)) {$i++;continue}
        $Arguments[$i]
    }
}

function Invoke-KVMemRecallWorkerCheck([string]$Worker,$Parameters,[string]$Root) {
    if((!$Parameters.ContainsKey('RecallStrategy') -or $Parameters.RecallStrategy -ne 'layerwise') -and
       (!$Parameters.ContainsKey('CacheMinTokens') -or !$Parameters.CacheMinTokens)) {return}
    $python=if($Parameters.ContainsKey('Python')){$Parameters.Python}else{''}
    $pythonArgs=@()
    if(!$python) {
        $bundled=Join-Path $Root 'runtime/python/python.exe'
        if(Test-Path -LiteralPath $bundled -PathType Leaf) {$python=$bundled}
        elseif(Get-Command python -ErrorAction SilentlyContinue) {$python=(Get-Command python).Source}
        else {$python=(Get-Command py -ErrorAction Stop).Source;$pythonArgs=@('-3')}
    }
    $safe=@{};foreach($key in $Parameters.Keys) {if($key -ne 'ApiKey'){$safe[$key]=$Parameters[$key]}}
    $info=New-KVMemProcessInfo $python ($pythonArgs+@('-B',(Join-Path $Root 'scripts/check-recall-worker.py'),'--worker',$Worker,'--backend',[string]$safe.Backend))
    $info.RedirectStandardInput=$true;$info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
    $check=[Diagnostics.Process]::Start($info)
    try {
        $output=$check.StandardOutput.ReadToEndAsync();$errors=$check.StandardError.ReadToEndAsync()
        $check.StandardInput.Write(($safe | ConvertTo-Json -Depth 10 -Compress));$check.StandardInput.Close()
        if(!$check.WaitForExit(30000)) {$check.Kill();$check.WaitForExit();throw 'Layerwise worker capability check timed out'}
        if($check.ExitCode) {throw $errors.Result.Trim()}
        $null=$output.Result | ConvertFrom-Json
    } finally {$check.Dispose()}
}
