Param(
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

Remove-Module -Name build_common -ErrorAction SilentlyContinue
Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList @(20, $CUDA_ARCH, $CMAKE_OPTIONS, $ENABLE_TILE)

$PRESET = "cccl-c-parallel"
$LOCAL_CMAKE_OPTIONS = ""

configure_and_build_preset "CCCL C Parallel" $PRESET $LOCAL_CMAKE_OPTIONS

If($CURRENT_PATH -ne "ci") {
    popd
}
