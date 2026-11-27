# Data-only bridge: values from JSON are splatted, never evaluated as PowerShell code.
param([Parameter(Mandatory)][string]$BackendRoot, [Parameter(Mandatory)][string]$ConfigFile)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$inputConfig = Get-Content -LiteralPath $ConfigFile -Raw -Encoding UTF8 | ConvertFrom-Json
$parameters = @{}
foreach ($property in $inputConfig.PSObject.Properties) {
    $parameters[$property.Name] = $property.Value
}
$parameters.DryRun = $true
$output = @(& (Join-Path $BackendRoot 'scripts/windows/start-backend.ps1') @parameters 3>&1)
$warnings = @($output | Where-Object { $_ -is [System.Management.Automation.WarningRecord] } | ForEach-Object Message)
$text = ($output | Where-Object { $_ -isnot [System.Management.Automation.WarningRecord] }) -join "`n"
$jsonStart = $text.IndexOf('{')
if ($jsonStart -lt 0) { throw 'Launcher did not return a launch plan' }
if ($jsonStart -gt 0) { $warnings += $text.Substring(0, $jsonStart).Trim() }
$plan = $text.Substring($jsonStart) | ConvertFrom-Json
if ($warnings.Count) { $plan | Add-Member -NotePropertyName warnings -NotePropertyValue ($warnings -join "`n") }
$port = if ($parameters.ContainsKey('Port')) { [int]$parameters.Port } else { 18200 }
$listenHost = if ($parameters.ContainsKey('ListenHost')) { [string]$parameters.ListenHost } else { '127.0.0.1' }
$probe = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Parse($listenHost), $port)
$probe.Server.ExclusiveAddressUse = $true
$available = $false
try { $probe.Start(); $available = $true } catch [Net.Sockets.SocketException] { } finally { $probe.Stop() }
$plan | Add-Member -NotePropertyName preflight -NotePropertyValue @{port=$port; listen_host=$listenHost; port_available=$available}
$plan | ConvertTo-Json -Depth 16
