
Param(
    [Parameter(Mandatory = $false)]
    [Alias("std")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(20)]
    [int]$CXX_STANDARD = 20,
    [Parameter(Mandatory = $false)]
    [ValidateNotNullOrEmpty()]
    [Alias("arch")]
    [int]$CUDA_ARCH = 0
)

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

Remove-Module -Name build_common
Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList $CXX_STANDARD, $CUDA_ARCH

$PRESET = "cudax-cpp$CXX_STANDARD"
$CMAKE_OPTIONS = ""

configure_and_build_preset "CUDA Experimental" "$PRESET" "$CMAKE_OPTIONS"

If($CURRENT_PATH -ne "ci") {
    popd
}
