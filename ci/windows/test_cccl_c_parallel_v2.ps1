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

Remove-Module -Name build_common -ErrorAction SilentlyContinue
Import-Module -Name "$PSScriptRoot/build_common.psm1" -ArgumentList @(20, $CUDA_ARCH, $CMAKE_OPTIONS)

$PRESET = "cccl-c-parallel-v2"
$LOCAL_CMAKE_OPTIONS = ""

configure_and_build_preset "CCCL C Parallel v2 (HostJIT)" $PRESET $LOCAL_CMAKE_OPTIONS

# TEMPORARY (do not merge): repeat count for hunting NVIDIA/cccl#10802. Kept at
# 1 so each job is exactly the shape of the real CI lane, which makes any
# reproduction unambiguously representative. test_preset throws on failure, so
# the job stops at the first reproduction with its logs intact.
$repeats = 1
if ($env:CCCL_C_PARALLEL_V2_TEST_REPEATS) {
    $repeats = [int]$env:CCCL_C_PARALLEL_V2_TEST_REPEATS
}
for ($i = 1; $i -le $repeats; $i++) {
    Write-Host "=== CCCL C Parallel v2 test pass $i of $repeats ==="
    test_preset "CCCL C Parallel v2 (HostJIT) pass $i" "$PRESET"
}

If($CURRENT_PATH -ne "ci") {
    popd
}
