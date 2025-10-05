Param(
    [Parameter(Mandatory = $false)]
    [Alias("std")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(17, 20)]
    [int]$CXX_STANDARD = 17,
    [Parameter(Mandatory = $false)]
    [Alias("arch")]
    [string]$CUDA_ARCH = ""
)

$ErrorActionPreference = "Stop"

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

# Build first
$build_command = "$PSScriptRoot/build_cudax.ps1 -std $CXX_STANDARD -arch `"$CUDA_ARCH`""
Write-Host "Executing: $build_command"
Invoke-Expression $build_command

Import-Module -Name "$PSScriptRoot/build_common.psm1" -ArgumentList $CXX_STANDARD, $CUDA_ARCH

$PRESET = "cudax-cpp$CXX_STANDARD"
test_preset "CUDA Experimental" "$PRESET"

If($CURRENT_PATH -ne "ci") {
    popd
}
