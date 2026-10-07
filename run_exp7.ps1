param(
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonCommand = Get-Command $PythonExecutable -ErrorAction SilentlyContinue

if ($null -eq $pythonCommand) {
    throw "Python executable '$PythonExecutable' was not found. Install Python 3.10-3.12 or pass its path with -PythonExecutable."
}

Push-Location $projectRoot
try {
    & $pythonCommand.Source -m src.train run
    if ($LASTEXITCODE -ne 0) {
        throw "The retraining pipeline failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
