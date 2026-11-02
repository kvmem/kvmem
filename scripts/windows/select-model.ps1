# Open a local Windows file picker. Inputs are data, never executable text.
param(
    [Parameter(Mandatory)][ValidateSet('ninfer', 'llamacpp', 'polystrata')][string]$Backend,
    [string]$InitialDirectory = '',
    [switch]$Folder,
    [switch]$Template,
    [switch]$Projector
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Windows.Forms
[Windows.Forms.Application]::EnableVisualStyles()
$dialog = if ($Folder) { New-Object Windows.Forms.FolderBrowserDialog } else { New-Object Windows.Forms.OpenFileDialog }
$owner = New-Object Windows.Forms.Form
try {
    $owner.TopMost = $true
    $owner.ShowInTaskbar = $false
    if ($Folder) {
        $dialog.Description = 'KVMem - Select model folder to scan'
        $dialog.ShowNewFolderButton = $false
        if ($InitialDirectory -and (Test-Path -LiteralPath $InitialDirectory -PathType Container)) {
            $dialog.SelectedPath = $InitialDirectory
        }
    } else {
    $dialog.Title = if ($Template) { 'KVMem - Select chat template' } elseif ($Projector) { 'KVMem - Select vision projector' } else { 'KVMem - Select model' }
    $dialog.Filter = if ($Template) { 'Chat template (*.jinja;*.j2;*.txt)|*.jinja;*.j2;*.txt|All files (*.*)|*.*' } elseif ($Projector) { 'Vision projector (*.gguf)|*.gguf' } elseif ($Backend -eq 'ninfer') { 'NInfer model (*.ninfer)|*.ninfer' } else { 'GGUF model (*.gguf)|*.gguf' }
    $dialog.CheckFileExists = $true
    $dialog.CheckPathExists = $true
    $dialog.Multiselect = $false
    $dialog.RestoreDirectory = $true
    if ($InitialDirectory -and (Test-Path -LiteralPath $InitialDirectory -PathType Container)) {
        $dialog.InitialDirectory = $InitialDirectory
    }
    }
    $selected = $null
    if ($dialog.ShowDialog($owner) -eq [Windows.Forms.DialogResult]::OK) {
        $selected = if ($Folder) { $dialog.SelectedPath } else { $dialog.FileName }
    }
    @{ path = $selected } | ConvertTo-Json -Compress
} finally {
    $dialog.Dispose()
    $owner.Dispose()
}
