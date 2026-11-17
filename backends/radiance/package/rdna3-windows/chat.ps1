param([ValidateRange(1,65535)][int]$Port = 18380, [string]$Prompt = '', [int]$MaxTokens = 256)
$ErrorActionPreference = 'Stop'
$url = "http://127.0.0.1:$Port"
$messages = New-Object 'System.Collections.Generic.List[object]'
function Send-Prompt([string]$text) {
    $messages.Add(@{role='user'; content=$text})
    $payload = @{model='radiance-local'; messages=@($messages.ToArray()); max_tokens=$MaxTokens; temperature=0;
                 chat_template_kwargs=@{enable_thinking=$false}}
    $body = [Text.Encoding]::UTF8.GetBytes((ConvertTo-Json -InputObject $payload -Depth 20 -Compress))
    $result = Invoke-RestMethod -Method Post -Uri ($url+'/v1/chat/completions') -ContentType 'application/json; charset=utf-8' -Body $body -TimeoutSec 600
    $answer = [string]$result.choices[0].message.content
    $messages.Add(@{role='assistant'; content=$answer})
    Write-Host "`nAssistant: $answer`n"
    if ($result.timings) {
        Write-Host ("Decode: {0:N2} token/s | Generated: {1} | Drafted: {2} | Accepted: {3}" -f $result.timings.predicted_per_second,$result.timings.predicted_n,$result.timings.draft_n,$result.timings.draft_n_accepted)
    }
}
try {
    $null = Invoke-RestMethod -Method Get -Uri ($url+'/v1/models') -TimeoutSec 10
    if ($Prompt) { Send-Prompt $Prompt; exit 0 }
    Write-Host 'Connected. Type /exit to quit. Keep the server window open.'
    while ($true) {
        $line = Read-Host 'You'
        if ($line -eq '/exit') { break }
        if ($line.Trim()) { Send-Prompt $line }
    }
} catch {
    Write-Host "Chat failed: $($_.Exception.Message)"
    Write-Host 'Start the server first and wait until it is ready. Check the server log if startup fails.'
    exit 1
}
