# Run the scouting-app bridge at logon and restart it if it dies.
# Run from the checkout you want deployed, using the virtual environment's Python:
#   powershell -ExecutionPolicy Bypass -File scripts\install_bridge_task.ps1 -Python C:\path\.venv\Scripts\python.exe
# It runs as you, in your session, because YouTube cookies come from your browser profile.
param(
    [Parameter(Mandatory = $true)][string]$Python,
    [string]$Data = "",
    [string]$TaskName = "AutoScout Bridge"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path $Python)) { throw "No Python at $Python" }

$launcher = Join-Path $repo "scripts\run_bridge.cmd"
$lines = @("@echo off", "cd /d `"$repo`"")
if ($Data) { $lines += "set AUTOSCOUT_DATA=$Data" }
$lines += "`"$Python`" -m autoscout.cli bridge >> `"$repo\bridge.log`" 2>&1"
Set-Content -Path $launcher -Value $lines -Encoding ascii

$action = New-ScheduledTaskAction -Execute $launcher -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Output "Installed '$TaskName'. Start it now with: Start-ScheduledTask -TaskName '$TaskName'"
