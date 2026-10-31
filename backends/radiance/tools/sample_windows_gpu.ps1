param(
    [Parameter(Mandatory)][string]$OutputPath,
    [Parameter(Mandatory)][string]$StopPath,
    [ValidateRange(1, 86400)][int]$MaxSeconds = 1800
)

# Adapter-wide WDDM counters include other processes. Preserve every LUID;
# never label their sum as this model's device allocation or process VRAM.
$ErrorActionPreference = 'Stop'
$output = [IO.Path]::GetFullPath($OutputPath)
$stop = [IO.Path]::GetFullPath($StopPath)
$stream = [IO.FileStream]::new($output, [IO.FileMode]::CreateNew,
    [IO.FileAccess]::Write, [IO.FileShare]::Read)
$writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false))
$timer = [Diagnostics.Stopwatch]::StartNew()
$count = 0
$failures = 0
try {
    $writer.WriteLine('timestamp_utc,elapsed_s,adapter_path,memory_kind,used_bytes,status')
    while ($timer.Elapsed.TotalSeconds -lt $MaxSeconds -and !(Test-Path -LiteralPath $stop)) {
        try {
            $samples = (Get-Counter @('\GPU Adapter Memory(*)\Dedicated Usage',
                '\GPU Adapter Memory(*)\Shared Usage') -MaxSamples 1).CounterSamples
            foreach ($sample in $samples) {
                # PDH_CSTATUS_VALID_DATA and PDH_CSTATUS_NEW_DATA are both valid.
                if ($sample.Status -notin @(0, 1) -or $sample.CookedValue -lt 0) {
                    $failures++
                    continue
                }
                $stamp = [DateTime]::UtcNow.ToString('o')
                $elapsed = $timer.Elapsed.TotalSeconds.ToString('F3', [Globalization.CultureInfo]::InvariantCulture)
                $kind = if ($sample.Path.EndsWith('\shared usage')) { 'shared' } else { 'dedicated' }
                $writer.WriteLine("$stamp,$elapsed,$($sample.Path),$kind,$([long]$sample.CookedValue),$($sample.Status)")
                $count++
            }
            $writer.Flush()
        } catch {
            $failures++
            if ($failures -le 3) { Write-Warning $_.Exception.Message }
        }
        Start-Sleep -Seconds 1
    }
} finally {
    $writer.Dispose()
}
Write-Output "valid_samples=$count failures=$failures"
if ($count -eq 0) { exit 2 }
