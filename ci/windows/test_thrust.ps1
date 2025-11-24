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
    [Alias("cpu-only")]
    [switch]$CPU_ONLY = $false,
    [Parameter(Mandatory = $false)]
    [Alias("gpu-only")]
    [switch]$GPU_ONLY = $false,
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

if ($CPU_ONLY) {
    $PRESET = "thrust-cpu"
    $artifactTag = "test_cpu"
} elseif ($GPU_ONLY) {
    $PRESET = "thrust-gpu"
    $artifactTag = "test_gpu"
} else {
    $PRESET = "thrust"
    $artifactTag = ""
}

if ($env:GITHUB_ACTIONS -and $artifactTag) {
    $producerId = (& bash "./util/workflow/get_producer_id.sh").Trim()
    $artifactName = "z_thrust-test-artifacts-$env:DEVCONTAINER_NAME-$producerId-$artifactTag"
    Write-Host "Unpacking artifact '$artifactName'"
    & bash "./util/artifacts/download_packed.sh" "$artifactName" "../"
} else {
    $cmd = "$PSScriptRoot/build_thrust.ps1 -std $CXX_STANDARD -arch '$CUDA_ARCH' -cmake-options '$CMAKE_OPTIONS'"
    Write-Host "Running: $cmd"
    Invoke-Expression $cmd
}

Import-Module -Name "$PSScriptRoot/build_common.psm1" -ArgumentList @($CXX_STANDARD, $CUDA_ARCH, $CMAKE_OPTIONS)

test_preset "Thrust ($PRESET)" "$PRESET"

If($CURRENT_PATH -ne "ci") {
    popd
}
