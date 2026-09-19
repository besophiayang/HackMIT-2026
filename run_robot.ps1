param(
    [Parameter(Mandatory = $true)]
    [string]$RobotIp
)

$parsedIp = $null
if (-not [System.Net.IPAddress]::TryParse($RobotIp, [ref]$parsedIp)) {
    throw "Invalid robot IP address: $RobotIp"
}

$linuxProject = (wsl.exe -d Ubuntu-24.04 -- wslpath -a $PSScriptRoot).Trim()
if (-not $linuxProject) {
    throw "Could not translate the project folder into a WSL path."
}

$ubuntuCommand = "cd '$linuxProject' && source `$HOME/dimos-app/.venv/bin/activate && uv run python robot_program.py --robot-ip '$RobotIp'"
wsl.exe -d Ubuntu-24.04 -- bash -lc $ubuntuCommand

if ($LASTEXITCODE -ne 0) {
    throw "The Go2 program exited with code $LASTEXITCODE."
}
