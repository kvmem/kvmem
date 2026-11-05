#Requires -Version 5.1
param([switch]$DryRunOnly)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native-process.ps1')
$temp = Join-Path ([IO.Path]::GetTempPath()) ('kvmem-launcher-' + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path (Join-Path $temp 'bin') -Force
function Check([bool]$Ok, [string]$Message) { if (!$Ok) { throw $Message } }
try {
    # A real argv echo executable catches PS 5.1 quoting errors that dry runs miss.
    $echoExe = Join-Path $temp 'bin/llama-kvmem-server.exe'
    if (!$DryRunOnly) {
      Add-Type -TypeDefinition @'
using System;
using System.Text;
public class ArgvEcho {
    public static void Main(string[] args) {
        if (args.Length == 1 && args[0] == "--kvmem-capabilities") {
            Console.WriteLine(Environment.GetEnvironmentVariable("KVMEM_TEST_LORA_CAPABILITY") ?? "{}");
            return;
        }
        foreach (string a in args) Console.WriteLine(Convert.ToBase64String(Encoding.UTF8.GetBytes(a)));
    }
}
'@ -OutputAssembly $echoExe -OutputType ConsoleApplication
    $expected = @('', 'a b', 'C:\space path\', '{"reasoning_effort":"medium"}', 'quote"slash\', '$x; & echo nope')
    $info = New-KVMemProcessInfo $echoExe $expected
    $info.RedirectStandardOutput = $true
    $child = [Diagnostics.Process]::Start($info)
    $actual = @($child.StandardOutput.ReadToEnd().Split([char]10) | Select-Object -SkipLast 1 |
        ForEach-Object { [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_.TrimEnd([char]13))) })
    $child.WaitForExit(); $child.Dispose()
    Check ($actual.Count -eq $expected.Count) 'argv count'
    for ($i = 0; $i -lt $expected.Count; $i++) { Check ($actual[$i] -ceq $expected[$i]) "argv mismatch $i" }
    } else {
        [IO.File]::WriteAllText($echoExe, 'not executed')
        Write-Output 'SKIP: executable argv roundtrip (DryRunOnly)'
    }
    $model = Join-Path $temp 'model with spaces.gguf'
    $mmproj = Join-Path $temp 'mmproj.gguf'
    [IO.File]::WriteAllText($model, 'fixture'); [IO.File]::WriteAllText($mmproj, 'fixture')
    if (!$DryRunOnly) {
        try {
            foreach ($case in @(
                @{json='{"fixed_lora":true}'; accepted=$true},
                @{json='{"fixed_lora":1}'; accepted=$false},
                @{json='{"fixed_lora":"true"}'; accepted=$false},
                @{json='{"fixed_lora":false}'; accepted=$false},
                @{json='{}'; accepted=$false}
            )) {
                $env:KVMEM_TEST_LORA_CAPABILITY = $case.json
                $accepted = $false
                try {
                    & (Join-Path $PSScriptRoot 'start-backend.ps1') -Backend llamacpp -Model $model -Worker $echoExe -Lora $mmproj -Gpu 'GPU-00000000-0000-0000-0000-000000000001' -DiskMiB 0 | Out-Null
                    $accepted = $true
                } catch {
                    if ($case.accepted) { throw }
                    Check ($_.Exception.Message -match 'does not support fixed LoRA') 'LoRA capability rejection'
                }
                Check ($accepted -eq $case.accepted) 'LoRA capability must be boolean true'
            }
        } finally {
            Remove-Item Env:KVMEM_TEST_LORA_CAPABILITY -ErrorAction SilentlyContinue
        }
    }
    foreach ($recipe in 'iq3', 'iq4') {
        $data = & (Join-Path $PSScriptRoot "start-$recipe.ps1") -BuildDir $temp -Model $model -Mmproj $mmproj `
            -Gpu 0 -BlockTokens 32 -ReasoningEffort medium -ChatTemplateKwargs '{"enable_thinking":true}' -DryRun | ConvertFrom-Json
        $a = $data.argv
        Check ($a[[Array]::IndexOf($a, '-m') + 1] -eq $model) 'model path'
        Check ($a[[Array]::IndexOf($a, '--kvmem-block-tokens') + 1] -eq '32') 'block tokens'
        Check ($a[[Array]::IndexOf($a, '--spec-draft-n-max') + 1] -eq '3') 'MTP default'
        Check ($a[[Array]::IndexOf($a, '--chat-template-kwargs') + 1] -eq '{"enable_thinking":true}') 'template JSON'
        $limit = '16384'; $budget = '36864'; $vision = '--no-mmproj-offload'; $kv = 'q8_0'
        if ($recipe -eq 'iq4') { $limit = '12288'; $budget = '32768'; $vision = '--no-mmproj-offload'; $kv = 'q5_0' }
        Check ($a[[Array]::IndexOf($a, '--kvmem-gen-reserve') + 1] -eq $limit) 'generation reserve'
        Check ($a[[Array]::IndexOf($a, '--kvmem-budget') + 1] -eq $budget) 'retrieval budget'
        Check ($a[[Array]::IndexOf($a, '--kv-dtype') + 1] -eq $kv) 'KV type'
        Check ($a -contains $vision) 'vision placement'
        Check ($a -notcontains '--mmproj-offload') 'default vision must not use GPU'
        $gpuVision = & (Join-Path $PSScriptRoot "start-$recipe.ps1") -BuildDir $temp -Model $model -Mmproj $mmproj `
            -Gpu 0 -VisionDevice gpu -DryRun | ConvertFrom-Json
        Check ($gpuVision.argv -contains '--mmproj-offload') 'explicit GPU vision override'
        Check ($gpuVision.argv -notcontains '--no-mmproj-offload') 'no conflicting CPU vision flag'
        $mixed = & (Join-Path $PSScriptRoot "start-$recipe.ps1") -BuildDir $temp -Model $model -Mmproj $mmproj `
            -Gpu 0 -CacheTypeK q8_0 -CacheTypeV q4_0 -DryRun | ConvertFrom-Json
        $a = $mixed.argv
        Check ($a[[Array]::IndexOf($a, '--cache-type-k') + 1] -eq 'q8_0') 'mixed K type'
        Check ($a[[Array]::IndexOf($a, '--cache-type-v') + 1] -eq 'q4_0') 'mixed V type'
        Check ([Array]::IndexOf($a, '--cache-type-k') -gt [Array]::IndexOf($a, '--kv-dtype')) 'K overrides recipe'
        Check ([Array]::IndexOf($a, '--cache-type-v') -gt [Array]::IndexOf($a, '--kv-dtype')) 'V overrides recipe'
    }
    $failed = $false
    try { & (Join-Path $PSScriptRoot 'start-server.ps1') -Recipe iq3 -BuildDir $temp -Model $model -Mmproj $mmproj -Gpu 0 -ChatTemplateKwargs '[]' -DryRun }
    catch { $failed = $true }
    Check $failed 'invalid template JSON must fail'
    # An occupied port must remain untouched and the model must never start.
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    $listener.Start()
    try {
        $failed = $false
        try { & (Join-Path $PSScriptRoot 'start-server.ps1') -Recipe iq3 -BuildDir $temp -Model $model -Mmproj $mmproj -Gpu 0 -Port $listener.LocalEndpoint.Port }
        catch { $failed = $true }
        Check $failed 'occupied port must fail'
        Check $listener.Server.IsBound 'existing listener must survive'
    } finally { $listener.Stop() }
    Write-Output 'Windows launcher: recipes, validation and port ownership passed'
} finally {
    $resolved = [IO.Path]::GetFullPath($temp)
    $base = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (!$resolved.StartsWith($base, [StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetFileName($resolved) -notlike 'kvmem-*') { throw 'Unsafe cleanup path' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
