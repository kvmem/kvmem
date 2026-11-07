# Shared launch-time Query scoring contract; worker validates the same combinations.
. (Join-Path $PSScriptRoot 'recall-strategy.ps1')
function Get-KVMemQueryScoringArgs($Bound) {
    $recall=@(Get-KVMemRecallArgs $Bound)
    if($Bound.ContainsKey('RecallStrategy') -and $Bound.RecallStrategy -eq 'layerwise') {return $recall}
    $recall
    $scorer = if ($Bound.ContainsKey('QueryScorer')) { $Bound['QueryScorer'] } else { 'page-softmax' }
    $cosine = $scorer -eq 'meanq-cosine'
    $agg = if ($Bound.ContainsKey('ScoreAggregation')) { $Bound['ScoreAggregation'] } elseif ($cosine) { 'max' } else { 'mean' }
    $alpha = if ($Bound.ContainsKey('GraphAlpha')) { [double]$Bound['GraphAlpha'] } elseif ($cosine) { 0.0 } else { 0.2 }
    if ($Bound.ContainsKey('Lrfu') -and $Bound.ContainsKey('NoLrfu')) { throw 'Specify Lrfu or NoLrfu, not both' }
    $history = if ($Bound.ContainsKey('Lrfu')) { [bool]$Bound['Lrfu'] } elseif ($Bound.ContainsKey('NoLrfu')) { ![bool]$Bound['NoLrfu'] } else { $cosine }
    if ([double]::IsNaN($alpha) -or [double]::IsInfinity($alpha) -or $alpha -lt 0 -or $alpha -gt 1) { throw 'GraphAlpha must be finite and in [0,1]' }
    if (!$history -and ($Bound.ContainsKey('LrfuLambda') -or $Bound.ContainsKey('LrfuTopFraction'))) { throw 'LRFU tuning requires enabled LRFU' }
    $culture = [Globalization.CultureInfo]::InvariantCulture
    @('--kvmem-query-scorer', $scorer, '--kvmem-score-agg', $agg, '--kvmem-graph-alpha', $alpha.ToString('R',$culture))
    if ($history) {
        '--kvmem-lrfu'
        $decay = if ($Bound.ContainsKey('LrfuLambda')) { [double]$Bound['LrfuLambda'] } else { 0.5 }
        $fraction = if ($Bound.ContainsKey('LrfuTopFraction')) { [double]$Bound['LrfuTopFraction'] } else { 0.9 }
        @('--kvmem-lrfu-lambda', $decay.ToString('R',$culture), '--kvmem-lrfu-top-frac', $fraction.ToString('R',$culture))
    } else { '--no-kvmem-lrfu' }
    if ($Bound.ContainsKey('QueryMaxTokens')) { @('--kvmem-query-max-tokens', "$($Bound['QueryMaxTokens'])") }
    if ($Bound.ContainsKey('QueryReplayMaxTokens')) { @('--kvmem-query-replay-max-tokens', "$($Bound['QueryReplayMaxTokens'])") }
}
