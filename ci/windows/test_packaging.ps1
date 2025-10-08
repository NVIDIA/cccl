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

Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList $CXX_STANDARD, "$CUDA_ARCH"

$PRESET = "packaging"
$CMAKE_OPTIONS = ""

if ($env:GITHUB_SHA) {
    $CMAKE_OPTIONS += "-DCCCL_EXAMPLE_CPM_TAG=$env:GITHUB_SHA "
}

configure_preset "Packaging" $PRESET "$CMAKE_OPTIONS"
test_preset "Packaging" $PRESET

If($CURRENT_PATH -ne "ci") {
    popd
}
