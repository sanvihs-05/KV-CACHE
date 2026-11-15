<#
.SYNOPSIS
  Week-1 check: does the KV cache really reach the SSD?

.DESCRIPTION
  Brackets linux/check_offload.sh (FlexLLMGen with 100% of the KV cache on disk, then the
  multi-turn engine with write_through) with two SMART reads, and compares three numbers:
    - how big the offload folder got        (Linux, sampled every second)
    - what Linux wrote to its virtual disk  (/proc/diskstats)
    - what the SSD says it received         (SMART host writes)
  If the folder grows but SMART barely moves, the KV never left the page cache.
  Run elevated, with OneDrive paused.
#>
param(
    [string]$Distro = "Ubuntu-22.04",
    [string]$Device = "",
    [int]$SettleSeconds = 20,
    [switch]$SkipFlexgen
)
$ErrorActionPreference = "Stop"
Import-Module "$PSScriptRoot\SmartLogger.psm1" -Force
Import-Module "$PSScriptRoot\KvwearWsl.psm1" -Force

if (-not (Test-IsAdmin)) { throw "Run from an elevated PowerShell (SMART needs admin)." }
Test-WslReady -Distro $Distro
$root = (Resolve-Path "$PSScriptRoot\..").Path
$proj = Get-WslProjectPath -Distro $Distro -WindowsPath $root

$before = Get-SsdWriteCounter -Device $Device
Write-Host ("SMART before: {0:N3} GB  ({1})" -f ($before.HostBytesWritten / 1e9), $before.Model)

$flag = if ($SkipFlexgen) { "--skip-flexgen" } else { "" }
$r = Invoke-WslBash -Distro $Distro -Command "cd '$proj' && bash linux/check_offload.sh $flag"
Sync-WslDisk -Distro $Distro
Invoke-HostFlush
Start-Sleep -Seconds $SettleSeconds
$after = Get-SsdWriteCounter -Device $before.Device
$smart = [int64]$after.HostBytesWritten - [int64]$before.HostBytesWritten

$linux = 0; $peak = 0
foreach ($l in $r.Lines) {
    if ($l -match '^CHECK_LINUX_BYTES=(\d+)') { $linux += [int64]$Matches[1] }
    if ($l -match '^CHECK_PEAK_BYTES=(\d+)') { $peak = [math]::Max($peak, [int64]$Matches[1]) }
}
Write-Host ""
Write-Host ("offload folder peak size : {0,10:N3} GB" -f ($peak / 1e9))
Write-Host ("Linux wrote (diskstats)  : {0,10:N3} GB" -f ($linux / 1e9))
Write-Host ("SSD received (SMART)     : {0,10:N3} GB  (includes Windows background writes)" -f ($smart / 1e9))
if ($r.ExitCode -ne 0) { Write-Warning "check_offload.sh failed (exit $($r.ExitCode)); read its output above." }
elseif ($linux -le 0) { Write-Warning "FAIL: Linux reported no writes. Check that the offload dir is on the WSL ext4 disk (~/kvwear_offload)." }
elseif ($smart -lt 0.5 * $linux) { Write-Warning "SUSPICIOUS: SMART moved much less than Linux wrote. Re-run with a longer -SettleSeconds; some drives update the counter lazily." }
else { Write-Host "PASS: KV writes reached the SSD. virtual-disk overhead ~ x$([math]::Round($smart / $linux, 2)) (upper bound: includes background noise)" -ForegroundColor Green }
