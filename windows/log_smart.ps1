<#
.SYNOPSIS
  Print the SSD's lifetime host-write counter and append it to a CSV log.
.EXAMPLE
  .\windows\log_smart.ps1 -Label before
  .\windows\log_smart.ps1 -ListDevices
#>
param(
    [string]$Label = "",
    [string]$Device = "",
    [string]$OutCsv = "$PSScriptRoot\..\results\smart_log.csv",
    [double]$AtaUnitBytes = 512,
    [switch]$ListDevices
)
$ErrorActionPreference = "Stop"
Import-Module "$PSScriptRoot\SmartLogger.psm1" -Force

if ($ListDevices) { Get-SmartDevices | Format-Table; return }

$c = Get-SsdWriteCounter -Device $Device -AtaUnitBytes $AtaUnitBytes
$c | Add-Member -NotePropertyName Label -NotePropertyValue $Label
New-Item -ItemType Directory -Force (Split-Path $OutCsv) | Out-Null
$c | Export-Csv -Path $OutCsv -Append -NoTypeInformation
"{0}  {1} ({2})  host writes = {3:N3} GB   wear used = {4}%" -f $c.Timestamp, $c.Model, $c.Device, ($c.HostBytesWritten / 1e9), $c.PercentageUsed
