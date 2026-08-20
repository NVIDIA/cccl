Param(
    [Parameter(Mandatory = $false)]
    [Alias("std")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(20)]
    [int]$CXX_STANDARD = 20,
    [Parameter(Mandatory = $false)]
    [Alias("arch")]
    [string]$CUDA_ARCH = "",
    [Parameter(Mandatory = $false)]
    [Alias("cmake-options")]
    [string]$CMAKE_OPTIONS = "",
    [Parameter(Mandatory = $false)]
    [Alias("enable-tile")]
    [switch]$ENABLE_TILE = $false
)

$ErrorActionPreference = "Stop"

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

# Build first
$buildCmd = "$PSScriptRoot/build_cudax.ps1 -std $CXX_STANDARD -arch '$CUDA_ARCH' -cmake-options '$CMAKE_OPTIONS' -ENABLE_TILE:$ENABLE_TILE"
Write-Host "Running: $buildCmd"
Invoke-Expression $buildCmd

Import-Module -Name "$PSScriptRoot/build_common.psm1" -ArgumentList @($CXX_STANDARD, $CUDA_ARCH, $CMAKE_OPTIONS, $ENABLE_TILE)

$PRESET = "cudax"
test_preset "CUDA Experimental" "$PRESET"

If($CURRENT_PATH -ne "ci") {
    popd
}
