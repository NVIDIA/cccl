Param(
    [Parameter(Mandatory = $false)]
    [Alias("std")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(11, 14, 17, 20)]
    [int]$CXX_STANDARD = 17,
    [Parameter(Mandatory = $false)]
    [Alias("cpu-only")]
    [switch]$CPU_ONLY = $false
)

$ErrorActionPreference = "Stop"

# if not cpu-only, emit an error. GPU tests are not yet supported.
if (-not $CPU_ONLY) {
    Write-Error "Thrust tests require the -cpu-only flag"
    exit 1
}

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

# Execute the build script:
$build_command = "$PSScriptRoot/build_thrust.ps1 -std $CXX_STANDARD"
Write-Host "Executing: $build_command"
Invoke-Expression $build_command

Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList $CXX_STANDARD

$PRESET = "thrust-cpu-cpp$CXX_STANDARD"

test_preset "Thrust" "$PRESET"

If($CURRENT_PATH -ne "ci") {
    popd
}
