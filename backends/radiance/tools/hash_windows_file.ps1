param([Parameter(Mandatory=$true)][string]$LiteralPath)
$ErrorActionPreference = 'Stop'
$item = Get-Item -LiteralPath $LiteralPath
if ($item.PSIsContainer) { throw 'A file is required' }
$beforeLength = $item.Length
$beforeTicks = $item.LastWriteTimeUtc.Ticks
$stream = [IO.File]::Open($item.FullName, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
$sha = [Security.Cryptography.SHA256]::Create()
try {
    $hash = [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
    $item.Refresh()
    if ($beforeLength -ne $item.Length -or $beforeTicks -ne $item.LastWriteTimeUtc.Ticks) {
        throw 'File changed during hashing'
    }
    @{sha256=$hash; bytes=$item.Length; modified_ticks=$item.LastWriteTimeUtc.Ticks} | ConvertTo-Json -Compress
} finally {
    $sha.Dispose()
    $stream.Dispose()
}
