Param(
    [Parameter(Mandatory = $false)]
    [Alias("std")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(17, 20)]
    [int]$CXX_STANDARD = 17,
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

Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList @($CXX_STANDARD, $CUDA_ARCH, $CMAKE_OPTIONS)

$PRESET = "cub"
$LOCAL_CMAKE_OPTIONS = "-DCMAKE_CXX_STANDARD=$CXX_STANDARD -DCMAKE_CUDA_STANDARD=$CXX_STANDARD"

if ($CL_VERSION -lt [version]"19.20") {
    $LOCAL_CMAKE_OPTIONS = "$LOCAL_CMAKE_OPTIONS -DCCCL_IGNORE_DEPRECATED_COMPILER=ON"
}

configure_and_build_preset "CUB" $PRESET $LOCAL_CMAKE_OPTIONS

if ($env:GITHUB_ACTIONS) {
    Write-Host "Packaging test artifacts..."
    & bash "./upload_cub_test_artifacts.sh"
}

If($CURRENT_PATH -ne "ci") {
    popd
}
