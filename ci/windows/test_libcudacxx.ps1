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

# Build first
$buildCmd = "$PSScriptRoot/build_libcudacxx.ps1 -std $CXX_STANDARD -arch '$CUDA_ARCH' -cmake-options '$CMAKE_OPTIONS'"
Write-Host "Running: $buildCmd"
Invoke-Expression $buildCmd

Import-Module -Name "$PSScriptRoot/build_common.psm1" -ArgumentList @($CXX_STANDARD, $CUDA_ARCH, $CMAKE_OPTIONS)

# Run ctest-based and lit-based test presets like on Linux
test_preset "libcudacxx (CTest)" "libcudacxx-ctest-cpp${CXX_STANDARD}"
test_preset "libcudacxx (lit)"   "libcudacxx-lit-cpp${CXX_STANDARD}"

If($CURRENT_PATH -ne "ci") {
    popd
}
