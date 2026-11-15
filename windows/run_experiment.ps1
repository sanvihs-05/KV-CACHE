<#
.SYNOPSIS
  Run the full measurement loop: SMART before -> WSL run -> sync/flush -> SMART after -> CSV.

.DESCRIPTION
  For every repetition and policy:
    1. read the SSD's lifetime host-write counter (SMART)
    2. run one policy in WSL (python -m kvwear.runner)
    3. sync inside Linux, flush the Windows volume cache, wait for the SSD counter to settle
    4. read SMART again; the difference is what the run wrote (plus Windows background noise)
    5. Linux's own /proc/diskstats count comes back in the runner's result
  Idle runs (same steps, `sleep` instead of the LLM) measure background writes so the
  analysis can subtract them. Policy order is shuffled per repetition to spread drift.
  All rows go to results\experiments.csv; then run:  python -m kvwear.analysis report

  Run in an elevated PowerShell (SMART needs admin). Pause OneDrive sync first, close
  browsers and other apps, and plug the laptop in.

.EXAMPLE
  .\windows\run_experiment.ps1 -Reps 3
  .\windows\run_experiment.ps1 -Policies spill,reuse,reuse_int4 -Reps 5 -Set store.ram_mb=512
  .\windows\run_experiment.ps1 -Engine flexgen -Config configs/flexgen.yaml -Policies flexgen -Reps 3
#>
param(
    [string[]]$Policies = @("write_through", "spill", "spill_int8", "spill_int4", "reuse", "reuse_int8", "reuse_int4", "oracle", "no_ssd"),
    [int]$Reps = 3,
    [string]$Distro = "Ubuntu-22.04",
    [string]$Config = "configs/default.yaml",
    [ValidateSet("hf", "sim", "flexgen")][string]$Engine = "hf",
    [string[]]$Set = @(),
    [string[]]$TrainSet = @(),   # shift test: workload the reuse predictor is trained on
    [int]$IdleSeconds = 300,
    [int]$IdleReps = 2,
    [int]$SettleSeconds = 20,
    [string]$Device = "",
    [string]$OutCsv = "",
    [switch]$NoShuffle,
    [switch]$SkipWarmup
)
$ErrorActionPreference = "Stop"
Import-Module "$PSScriptRoot\SmartLogger.psm1" -Force
Import-Module "$PSScriptRoot\KvwearWsl.psm1" -Force

$root = (Resolve-Path "$PSScriptRoot\..").Path
if (-not $OutCsv) { $OutCsv = Join-Path $root "results\experiments.csv" }
New-Item -ItemType Directory -Force (Split-Path $OutCsv) | Out-Null

# ---------------------------------------------------------------- pre-flight
if (-not (Test-IsAdmin)) { throw "Run this from an elevated PowerShell (Run as Administrator): SMART needs it." }
Test-WslReady -Distro $Distro
$proj = Get-WslProjectPath -Distro $Distro -WindowsPath $root
$probe = Get-SsdWriteCounter -Device $Device
if (-not $Device) { $Device = $probe.Device }
Write-Host ("SSD: {0} ({1}), lifetime host writes {2:N1} GB, wear used {3}%" -f $probe.Model, $Device, ($probe.HostBytesWritten / 1e9), $probe.PercentageUsed)
Write-Host "WSL project dir: $proj"
if (Get-Process OneDrive -ErrorAction SilentlyContinue) {
    Write-Warning "OneDrive is running. Pause syncing (tray icon > Pause syncing) or its writes become noise."
}

$rows = New-Object System.Collections.Generic.List[object]
if (Test-Path $OutCsv) { Import-Csv $OutCsv | ForEach-Object { $rows.Add($_) } }

function Save-Rows {
    $cols = New-Object System.Collections.Generic.List[string]
    foreach ($r in $rows) { foreach ($n in $r.PSObject.Properties.Name) { if (-not $cols.Contains($n)) { $cols.Add($n) } } }
    $rows | Select-Object -Property $cols.ToArray() | Export-Csv -Path $OutCsv -NoTypeInformation
}

function Measure-Window([scriptblock]$Body) {
    $before = Get-SsdWriteCounter -Device $Device
    $payload = & $Body
    Sync-WslDisk -Distro $Distro
    Invoke-HostFlush
    Start-Sleep -Seconds $SettleSeconds
    $after = Get-SsdWriteCounter -Device $Device
    [pscustomobject]@{
        Before  = $before; After = $after; Payload = $payload
        Window  = ([datetime]::Parse($after.Timestamp) - [datetime]::Parse($before.Timestamp)).TotalSeconds
        Delta   = [int64]$after.HostBytesWritten - [int64]$before.HostBytesWritten
    }
}

function Add-Row($kind, $policy, $rep, $runId, $m, $result, $exitCode) {
    $row = [ordered]@{
        run_id = $runId; policy = $policy; rep = $rep; kind = $kind; engine = $Engine; exit_code = $exitCode
        window_s = [math]::Round($m.Window, 2)
        smart_bytes_before = $m.Before.HostBytesWritten; smart_bytes_after = $m.After.HostBytesWritten
        smart_bytes_delta = $m.Delta; smart_source = $m.After.Source; ssd_model = $m.After.Model
        smart_started = $m.Before.Timestamp; smart_finished = $m.After.Timestamp
    }
    if ($result) { foreach ($p in $result.PSObject.Properties) { if (-not $row.Contains($p.Name)) { $row[$p.Name] = $p.Value } } }
    $rows.Add([pscustomobject]$row)
    Save-Rows
    Write-Host ("  -> SMART delta {0:N3} GB over {1:N0} s" -f ($m.Delta / 1e9), $m.Window) -ForegroundColor Cyan
}

function Invoke-Idle($rep) {
    $id = "{0:yyyyMMdd-HHmmss}-idle-r{1}" -f (Get-Date), $rep
    Write-Host "[idle $rep] $IdleSeconds s with WSL running but no LLM" -ForegroundColor Yellow
    $m = Measure-Window { & wsl.exe -d $Distro -- sleep $IdleSeconds; $null }
    Add-Row "idle" "idle" $rep $id $m $null 0
}

$setArgs = ($Set | ForEach-Object { "--set '$_'" }) -join " "
if ($TrainSet.Count) { $setArgs += " " + (($TrainSet | ForEach-Object { "--train-set '$_'" }) -join " ") }

# ---------------------------------------------------------------- warm-up (downloads the model; not measured)
if (-not $SkipWarmup) {
    Write-Host "[warm-up] downloading/loading the model once (not measured)" -ForegroundColor Yellow
    $warmPolicy = if ($Engine -eq "flexgen") { "flexgen" } else { "no_ssd" }
    $wa = "--config '$Config' --engine $Engine --policy '$warmPolicy' --run-id warmup --quiet $setArgs"
    if ($Engine -ne "flexgen") { $wa += " --set workload.num_conversations=2" }
    $w = Invoke-KvwearRunner -Distro $Distro -ProjectDir $proj -Arguments $wa
    if ($w.ExitCode -ne 0) { throw "warm-up run failed (exit $($w.ExitCode)); fix that before measuring" }
}

# ---------------------------------------------------------------- measurements
# Resume: rows already in the CSV (from an interrupted earlier call) are not repeated.
function Get-IdleCount { @($rows | Where-Object { $_.kind -eq "idle" }).Count }
function Test-RunDone($policy, $rep) {
    @($rows | Where-Object { $_.kind -eq "llm" -and $_.policy -eq $policy -and [string]$_.rep -eq [string]$rep -and [string]$_.exit_code -eq "0" }).Count -gt 0
}
if ($rows.Count -gt 0) { Write-Host "Resuming: $($rows.Count) rows already in $OutCsv" -ForegroundColor Yellow }

for ($i = (Get-IdleCount) + 1; $i -le $IdleReps; $i++) { Invoke-Idle $i }

for ($rep = 1; $rep -le $Reps; $rep++) {
    $order = if ($NoShuffle) { $Policies } else { $Policies | Get-Random -Count $Policies.Count }
    foreach ($p in $order) {
        if (Test-RunDone $p $rep) { Write-Host "[rep $rep/$Reps] policy $p already done, skipping"; continue }
        $safe = $p -replace '[^A-Za-z0-9_.-]', '_'
        $id = "{0:yyyyMMdd-HHmmss}-{1}-r{2}" -f (Get-Date), $safe, $rep
        Write-Host "[rep $rep/$Reps] policy $p  (run $id)" -ForegroundColor Green
        $a = "--config '$Config' --engine $Engine --policy '$p' --run-id '$id' --quiet $setArgs"
        $m = Measure-Window { Invoke-KvwearRunner -Distro $Distro -ProjectDir $proj -Arguments $a }
        $r = $m.Payload
        if ($r.ExitCode -ne 0 -or -not $r.Result) { Write-Warning "run $id failed (exit $($r.ExitCode)); recorded with exit_code" }
        Add-Row "llm" $p $rep $id $m $r.Result $r.ExitCode
    }
}

for ($i = (Get-IdleCount) + 1; $i -le 2 * $IdleReps; $i++) { Invoke-Idle $i }

Write-Host "`nDone. Rows in $OutCsv" -ForegroundColor Green
Write-Host "Next: py -3.10 -m kvwear.analysis report --experiments `"$OutCsv`""
