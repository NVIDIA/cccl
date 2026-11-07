param([Parameter(Mandatory = $true)][string]$TestExecutable)
$ErrorActionPreference = 'Stop'
& $TestExecutable
exit $LASTEXITCODE
