Param(
    [Parameter(Mandatory = $false)]
    [Alias("arch")]
    [string]$CUDA_ARCH = "",
    [Parameter(Mandatory = $false)]
    [Alias("cmake-options")]
    [string]$CMAKE_OPTIONS = ""
)

$ErrorActionPreference = "Stop"

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

# Build first
$buildCmd = "$PSScriptRoot/build_cccl_c_parallel.ps1 -arch '$CUDA_ARCH' -cmake-options '$CMAKE_OPTIONS'"
Write-Host "Running: $buildCmd"
Invoke-Expression $buildCmd

Remove-Module -Name build_common -ErrorAction SilentlyContinue
Import-Module -Name "$PSScriptRoot/build_common.psm1" -ArgumentList @(20, $CUDA_ARCH, $CMAKE_OPTIONS)

$PRESET = "cccl-c-parallel"
test_preset "CCCL C Parallel" "$PRESET"

If($CURRENT_PATH -ne "ci") {
    popd
}
