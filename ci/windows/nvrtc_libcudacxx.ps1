Param(
    [Parameter(Mandatory = $false)]
    [Alias("std")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(11, 14, 17, 20)]
    [int]$CXX_STANDARD = 17,
    [Parameter(Mandatory = $false)]
    [ValidateNotNullOrEmpty()]
    [Alias("arch")]
    [int]$CUDA_ARCH = 0
)

$ErrorActionPreference = "Stop"

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList $CXX_STANDARD, $CUDA_ARCH

$PRESET = "libcudacxx-nvrtc-cpp${CXX_STANDARD}"
$CMAKE_OPTIONS = "-DLIBCUDACXX_EXECUTOR='NoopExecutor()'"

configure_and_build_preset "libcudacxx NVRTC" "$PRESET" "$CMAKE_OPTIONS"

test_preset "libcudacxx NVRTC" "$PRESET"

If($CURRENT_PATH -ne "ci") {
    popd
}
