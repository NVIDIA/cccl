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

Import-Module -Name "$PSScriptRoot/build_common.psm1" -ArgumentList @($CXX_STANDARD, $CUDA_ARCH, $CMAKE_OPTIONS, $ENABLE_TILE)

if ($env:GITHUB_ACTIONS) {
    $producerId = & bash "./util/workflow/get_producer_id.sh"
    if ($LASTEXITCODE -ne 0) {
        throw "Finding the producer job failed (exit code $LASTEXITCODE)"
    }
    $producerId = "$producerId".Trim()
    $artifactName = "z_libcudacxx-test-artifacts-$env:DEVCONTAINER_NAME-$producerId"
    Write-Host "Unpacking artifact '$artifactName'"
    Invoke-Checked {
        & bash "./util/artifacts/download_packed.sh" "$artifactName" "../"
    } "Downloading test artifacts failed"
} else {
    $buildCmd = "$PSScriptRoot/build_libcudacxx.ps1 -std $CXX_STANDARD -arch '$CUDA_ARCH' -cmake-options '$CMAKE_OPTIONS' -ENABLE_TILE:$ENABLE_TILE"
    Write-Host "Running: $buildCmd"
    Invoke-Expression $buildCmd
}

if ($env:GITHUB_ACTIONS) {
    test_preset "libcudacxx (CTest)" "libcudacxx-ctest"
    $env:LIT_OPTS = "$env:LIT_OPTS -Dtest_executable_mode=replay".Trim()
    test_preset "libcudacxx (lit replay)" "libcudacxx-lit"
} else {
    test_preset "libcudacxx (CTest)" "libcudacxx-ctest-cpp${CXX_STANDARD}"
    test_preset "libcudacxx (lit)"   "libcudacxx-lit-cpp${CXX_STANDARD}"
}

If($CURRENT_PATH -ne "ci") {
    popd
}
