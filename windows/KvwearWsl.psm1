# KvwearWsl.psm1 - helpers for running the Linux side of the experiment from Windows.

Set-StrictMode -Version 2.0

$script:VenvActivate = '~/.venvs/kvwear/bin/activate'

function Get-WslProjectPath {
    param([string]$Distro, [string]$WindowsPath)
    $p = (& wsl.exe -d $Distro -- wslpath -a ($WindowsPath -replace '\\', '/')) | Select-Object -First 1
    if (-not $p) { throw "Could not translate $WindowsPath into a WSL path (is the '$Distro' distro installed?)" }
    return $p.Trim()
}

function Test-WslReady {
    param([string]$Distro)
    $names = (& wsl.exe -l -q) -replace "`0", "" | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() }
    if ($names -notcontains $Distro) {
        throw "WSL distro '$Distro' not found (have: $($names -join ', ')). Install: wsl --install -d Ubuntu-22.04, then pass -Distro Ubuntu-22.04"
    }
    & wsl.exe -d $Distro -- bash -lc "test -f $script:VenvActivate"
    if ($LASTEXITCODE -ne 0) { throw "Python venv missing in WSL. Run inside WSL: bash linux/setup_wsl.sh" }
}

function Invoke-WslBash {
    # Runs a bash command in WSL. Shows output live and returns the stdout lines.
    # Avoid double quotes inside $Command (Windows PowerShell 5.1 mangles them for native exes).
    param([string]$Distro, [string]$Command, [switch]$AsRoot)
    $wslArgs = @('-d', $Distro)
    if ($AsRoot) { $wslArgs += @('-u', 'root') }
    $wslArgs += @('--', 'bash', '-lc', $Command)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $captured = @()
    try {
        & wsl.exe @wslArgs | Tee-Object -Variable captured | Out-Host
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $prev }
    [pscustomobject]@{ ExitCode = $code; Lines = @($captured) }
}

function Invoke-KvwearRunner {
    param([string]$Distro, [string]$ProjectDir, [string]$Arguments)
    $cmd = "cd '$ProjectDir' && source $script:VenvActivate && export PYTHONDONTWRITEBYTECODE=1 && python -m kvwear.runner $Arguments"
    $r = Invoke-WslBash -Distro $Distro -Command $cmd
    $line = $r.Lines | Where-Object { $_ -like 'KVWEAR_RESULT *' } | Select-Object -Last 1
    $result = $null
    if ($line) { $result = $line.Substring('KVWEAR_RESULT '.Length) | ConvertFrom-Json }
    [pscustomobject]@{ ExitCode = $r.ExitCode; Result = $result; Lines = $r.Lines }
}

function Sync-WslDisk {
    param([string]$Distro)
    & wsl.exe -d $Distro -u root -- sync
}

Export-ModuleMember -Function Get-WslProjectPath, Test-WslReady, Invoke-WslBash, Invoke-KvwearRunner, Sync-WslDisk
