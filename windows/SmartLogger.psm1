# SmartLogger.psm1 - read the SSD's lifetime host-write counter (SMART) on Windows.
#
# Uses smartctl (smartmontools, https://www.smartmontools.org) because it reads the real
# NVMe "Data Units Written" counter. Install:  winget install smartmontools.smartmontools
# Must run in an elevated (Administrator) PowerShell.
#
#   NVMe : data_units_written x 512,000 bytes   (NVMe spec: 1 unit = 1000 x 512-byte sectors)
#   SATA : attribute 241 Total_LBAs_Written x unit (default 512 bytes; some vendors use 32 MiB
#          or 1 GiB units - check your drive's datasheet and pass -AtaUnitBytes)
#
# The counter counts HOST writes (what the PC sent). Internal NAND write amplification is
# on top of that and usually isn't exposed.

Set-StrictMode -Version 2.0

function Test-IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    return ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Find-Smartctl {
    $cmd = Get-Command smartctl -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @("$env:ProgramFiles\smartmontools\bin\smartctl.exe",
                     "${env:ProgramFiles(x86)}\smartmontools\bin\smartctl.exe")) {
        if ($p -and (Test-Path $p)) { return $p }
    }
    return $null
}

function Get-SmartDevices {
    $smartctl = Find-Smartctl
    if (-not $smartctl) { throw "smartctl not found. Install: winget install smartmontools.smartmontools" }
    $scan = (& $smartctl --scan -j) -join "`n" | ConvertFrom-Json
    return @($scan.devices | ForEach-Object { [pscustomobject]@{ Name = $_.name; Type = $_.type; Info = $_.info_name } })
}

function Get-SsdWriteCounter {
    [CmdletBinding()]
    param(
        [string]$Device = "",           # e.g. /dev/nvme0 or /dev/sda ; default: first scanned device
        [double]$AtaUnitBytes = 512
    )
    $smartctl = Find-Smartctl
    if (-not $smartctl) {
        throw "smartctl not found. Install it with: winget install smartmontools.smartmontools (then reopen PowerShell)"
    }
    if (-not (Test-IsAdmin)) { throw "Reading SMART needs an elevated PowerShell (Run as Administrator)." }
    $devs = Get-SmartDevices
    if (-not $Device) {
        if (-not $devs) { throw "smartctl found no drives" }
        $Device = $devs[0].Name
    }
    $match = $devs | Where-Object { $_.Name -eq $Device } | Select-Object -First 1
    $smartArgs = @('-a', '-j')
    if ($match -and $match.Type) { $smartArgs += @('-d', $match.Type) }   # e.g. '-d nvme' for /dev/sda
    $raw = (& $smartctl @smartArgs $Device) -join "`n"
    $j = $raw | ConvertFrom-Json

    $bytes = $null; $units = $null; $source = $null; $pctUsed = $null; $poh = $null
    $props = $j.PSObject.Properties.Name
    if ($props -contains "nvme_smart_health_information_log") {
        $log = $j.nvme_smart_health_information_log
        $units = [int64]$log.data_units_written
        $bytes = $units * 512000
        $pctUsed = $log.percentage_used
        $poh = $log.power_on_hours
        $source = "smartctl-nvme"
    } elseif ($props -contains "ata_smart_attributes") {
        $attr = $j.ata_smart_attributes.table | Where-Object { $_.id -eq 241 } | Select-Object -First 1
        if ($attr) {
            $units = [int64]$attr.raw.value
            $bytes = [int64]($units * $AtaUnitBytes)
            $source = "smartctl-ata241"
        }
        $pohAttr = $j.ata_smart_attributes.table | Where-Object { $_.id -eq 9 } | Select-Object -First 1
        if ($pohAttr) { $poh = $pohAttr.raw.value }
    }
    if ($null -eq $bytes) { throw "Drive $Device exposes no host-write counter that this script understands." }

    $model = if ($props -contains "model_name") { $j.model_name } else { "" }
    [pscustomobject]@{
        Timestamp        = (Get-Date).ToString("o")
        Device           = $Device
        Model            = $model
        Source           = $source
        HostBytesWritten = [int64]$bytes
        RawUnits         = $units
        PercentageUsed   = $pctUsed
        PowerOnHours     = $poh
    }
}

function Invoke-HostFlush {
    # Push Windows' own write cache (incl. the WSL ext4.vhdx file) to the SSD before reading SMART.
    # Default: every local fixed volume (the vhdx may live on D: while Windows is on C:).
    param([string[]]$DriveLetter = @())
    if (-not $DriveLetter) {
        $DriveLetter = @(Get-Volume | Where-Object { $_.DriveType -eq 'Fixed' -and $_.DriveLetter } |
                         ForEach-Object { [string]$_.DriveLetter })
    }
    foreach ($d in $DriveLetter) {
        try { Write-VolumeCache -DriveLetter $d -ErrorAction Stop } catch { Write-Warning "Write-VolumeCache $d failed: $_" }
    }
}

Export-ModuleMember -Function Test-IsAdmin, Find-Smartctl, Get-SmartDevices, Get-SsdWriteCounter, Invoke-HostFlush
