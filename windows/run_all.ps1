<#
.SYNOPSIS
  The whole SMART-measured campaign, unattended (~3.5 h). Run elevated:
    powershell -ExecutionPolicy Bypass -File windows\run_all.ps1

.DESCRIPTION
  0. keep the PC awake (for this process only), stop OneDrive (restarted at the end)
  1. validation: week-1 check + a tiny SMART-bracketed experiment; stop if SMART looks wrong
  2. multi-turn sweep, RAM tier 1 GB          -> results\final\experiments_ram1024.csv
  3. multi-turn sweep, RAM tier 256 MB        -> results\final\experiments_ram256.csv
  4. shorter context (1000 tokens), 256 MB    -> results\final\experiments_ram256_ctx1000.csv
  5. FlexLLMGen, KV 100% on disk              -> results\final\experiments_flexgen.csv
  6. FlexLLMGen with --compress-cache         -> results\final\experiments_flexgen_compress.csv
  Progress: results\final\status.txt ; full log: results\final\run_all_<time>.log
#>
param(
    [int]$Reps = 3,
    [int]$IdleSeconds = 180,
    [string[]]$Stages = @("validate", "ram1024", "ram256", "ctx1000", "flexgen", "flexgen_compress")   # extra stages (run explicitly): qwen, qwen_long, shift
)
$ErrorActionPreference = "Stop"
$Stages = @($Stages | ForEach-Object { $_ -split ',' } | Where-Object { $_ })   # -File passes "a,b" as one string
$root = (Resolve-Path "$PSScriptRoot\..").Path
$out = Join-Path $root "results\final"
New-Item -ItemType Directory -Force $out | Out-Null
$status = Join-Path $out "status.txt"
function Set-Status($msg) { "{0:yyyy-MM-dd HH:mm:ss}  {1}" -f (Get-Date), $msg | Add-Content $status; Write-Host "### $msg" -ForegroundColor Magenta }

Start-Transcript -Path (Join-Path $out ("run_all_{0:yyyyMMdd-HHmmss}.log" -f (Get-Date))) | Out-Null
Import-Module "$PSScriptRoot\SmartLogger.psm1" -Force
if (-not (Test-IsAdmin)) { Set-Status "FAILED: not elevated"; throw "Run elevated." }

# keep the system awake while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
Add-Type -Namespace KvWear -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][KvWear.Power]::SetThreadExecutionState([uint32]"0x80000001")

$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($battery -and $battery.BatteryStatus -ne 2) { Write-Warning "Laptop seems to be on battery; plug in the charger for stable speed numbers." }

$oneDrive = Get-Process OneDrive -ErrorAction SilentlyContinue | Select-Object -First 1
$oneDrivePath = if ($oneDrive) { $oneDrive.Path } else { $null }
if ($oneDrive) { Set-Status "stopping OneDrive for the measurements (restarted at the end)"; Get-Process OneDrive | Stop-Process -Force }

$common = @{ Reps = $Reps; IdleSeconds = $IdleSeconds; IdleReps = 1 }
$plan = [ordered]@{
    ram1024          = @{ OutCsv = "$out\experiments_ram1024.csv" }
    ram256           = @{ OutCsv = "$out\experiments_ram256.csv"; Set = @("store.ram_mb=256") }
    ctx1000          = @{ OutCsv = "$out\experiments_ram256_ctx1000.csv"; Set = @("store.ram_mb=256", "workload.max_context_tokens=1000");
                          Policies = @("write_through", "spill", "reuse", "reuse_int8", "oracle", "no_ssd") }
    shift            = @{ OutCsv = "$out\experiments_ram256_shift.csv"; Set = @("store.ram_mb=256");
                          TrainSet = @("workload.mix={doc_qa: 0.1, debug: 0.1, general: 0.8}", "workload.turns={doc_qa: [1, 2], debug: [1, 2], general: [1, 3]}", "workload.num_shared_docs=8");
                          Policies = @("reuse", "reuse_int8") }
    qwen             = @{ OutCsv = "$out\experiments_qwen.csv"; Config = "configs/qwen.yaml";
                          Policies = @("write_through", "spill", "spill_int8", "reuse", "reuse_int8", "oracle", "no_ssd") }
    qwen_long        = @{ OutCsv = "$out\experiments_qwen_long.csv"; Config = "configs/qwen_long.yaml";
                          Policies = @("write_through", "spill", "reuse", "reuse_int8", "oracle", "no_ssd") }
    flexgen          = @{ OutCsv = "$out\experiments_flexgen.csv"; Engine = "flexgen"; Config = "configs/flexgen.yaml";
                          Policies = @("flexgen"); SkipWarmup = $true }
    flexgen_compress = @{ OutCsv = "$out\experiments_flexgen_compress.csv"; Engine = "flexgen"; Config = "configs/flexgen.yaml";
                          Policies = @("flexgen_compress"); Set = @("engine.flexgen_compress_cache=true"); SkipWarmup = $true }
}

try {
    if ($Stages -contains "validate") {
        Set-Status "validate: week-1 check"
        & "$PSScriptRoot\week1_check.ps1"
        Set-Status "validate: tiny SMART-bracketed experiment"
        $vcsv = "$out\validation.csv"
        if (Test-Path $vcsv) { Remove-Item $vcsv }
        & "$PSScriptRoot\run_experiment.ps1" -Config configs/smoke.yaml -Policies write_through, no_ssd -Reps 1 `
            -IdleSeconds 60 -IdleReps 1 -SettleSeconds 20 -OutCsv $vcsv -SkipWarmup
        $v = Import-Csv $vcsv
        $wt = $v | Where-Object { $_.policy -eq "write_through" } | Select-Object -First 1
        if (-not $wt -or $wt.exit_code -ne "0" -or [int64]$wt.smart_bytes_delta -le 0) {
            Set-Status "FAILED validation: write_through row missing, failed, or SMART did not move"; throw "validation failed"
        }
        Set-Status ("validate OK: write_through SMART delta {0:N3} GB, app {1:N3} GB, linux {2:N3} GB" -f `
            ([int64]$wt.smart_bytes_delta / 1e9), ([double]$wt.app_bytes_written / 1e9), ([double]$wt.linux_bytes_written / 1e9))
    }
    foreach ($name in $plan.Keys) {
        if ($Stages -notcontains $name) { continue }
        $args1 = $common.Clone()
        foreach ($k in $plan[$name].Keys) { $args1[$k] = $plan[$name][$k] }
        Set-Status "stage $name started"
        try {
            & "$PSScriptRoot\run_experiment.ps1" @args1
            Set-Status "stage $name done"
        } catch {
            Set-Status "stage $name FAILED: $_"
        }
    }
    Set-Status "ALL DONE"
} finally {
    if ($oneDrivePath) {
        # restart OneDrive un-elevated (via explorer), as the normal user
        Start-Process explorer.exe -ArgumentList "`"$oneDrivePath`""
        Set-Status "OneDrive restarted"
    }
    [void][KvWear.Power]::SetThreadExecutionState([uint32]"0x80000000")
    Stop-Transcript | Out-Null
}
