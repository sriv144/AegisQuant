$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$existing = Get-CimInstance Win32_Process |
    Where-Object {
        $_.CommandLine -match "main_us_v2\.py" -and
        $_.CommandLine -match "--loop"
    }

if ($existing) {
    Write-Output "AegisQuant v2 loop already running."
    exit 0
}

$logDir = Join-Path $repoRoot ".cache\sleeve_snapshots"
New-Item -ItemType Directory -Force $logDir | Out-Null

$stdout = Join-Path $logDir "v2_loop.out.log"
$stderr = Join-Path $logDir "v2_loop.err.log"

$process = Start-Process `
    -FilePath "python" `
    -ArgumentList @("main_us_v2.py", "--loop") `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

Write-Output "Started AegisQuant v2 loop pid=$($process.Id)."
