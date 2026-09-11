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

Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList @($CXX_STANDARD, $CUDA_ARCH, $CMAKE_OPTIONS, $ENABLE_TILE)

$PRESET = "libcudacxx"
$LOCAL_CMAKE_OPTIONS = "-DCMAKE_CXX_STANDARD=$CXX_STANDARD -DCMAKE_CUDA_STANDARD=$CXX_STANDARD"

$uploadTestArtifacts = $false
if ($env:GITHUB_ACTIONS) {
    & bash "./util/workflow/has_consumers.sh"
    $uploadTestArtifacts = $LASTEXITCODE -eq 0
    if ($uploadTestArtifacts) {
        $env:LIT_OPTS = "$env:LIT_OPTS -Dtest_executable_mode=build".Trim()
    }
}

configure_and_build_preset "libcudacxx" $PRESET $LOCAL_CMAKE_OPTIONS

if ($uploadTestArtifacts) {
    Write-Host "Packaging test artifacts..."
    Invoke-Checked {
        & bash "./upload_libcudacxx_test_artifacts.sh"
    } "Packaging test artifacts failed"
}

If($CURRENT_PATH -ne "ci") {
    popd
}
