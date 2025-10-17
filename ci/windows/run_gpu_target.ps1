Param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PassthroughArgs
)

$ErrorActionPreference = "Stop"

$CURRENT_PATH = Split-Path $pwd -leaf
if ($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

if ($null -eq $PassthroughArgs) {
    $PassthroughArgs = @()
}

Import-Module "$PSScriptRoot/build_common.psm1"

$ciDirWindows = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoDirWindows = (Resolve-Path (Join-Path $ciDirWindows "..")).Path

$ciDir = $ciDirWindows -replace "\\", "/"
$repoDir = $repoDirWindows -replace "\\", "/"
$utilScript = "$ciDir/util/build_and_test_targets.sh"

$argString = if ($PassthroughArgs.Count -gt 0) { " " + ($PassthroughArgs -join " ") } else { "" }

$bashCommand = "cd $repoDir; $utilScript$argString"
Write-Host $bashCommand -ForegroundColor Blue

& bash -lc $bashCommand
$exitCode = $LASTEXITCODE

if ($CURRENT_PATH -ne "ci") {
    popd
}

if ($exitCode -ne 0) {
    exit $exitCode
}
