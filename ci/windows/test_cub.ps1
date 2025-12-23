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
    [Alias("no-lid")]
    [switch]$NO_LID_SWITCH = $false,
    [Parameter(Mandatory = $false)]
    [Alias("lid0")]
    [switch]$LID0_SWITCH = $false,
    [Parameter(Mandatory = $false)]
    [Alias("lid1")]
    [switch]$LID1_SWITCH = $false,
    [Parameter(Mandatory = $false)]
    [Alias("lid2")]
    [switch]$LID2_SWITCH = $false,
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

Import-Module -Name "$PSScriptRoot/build_common.psm1" -ArgumentList @($CXX_STANDARD, $CUDA_ARCH, $CMAKE_OPTIONS)

$PRESET = "cub"
$artifactTag = ""
if ($NO_LID_SWITCH) {
    $artifactTag = "no_lid"
    $PRESET = "cub-nolid"
} elseif ($LID0_SWITCH) {
    $artifactTag = "lid_0"
    $PRESET = "cub-lid0"
} elseif ($LID1_SWITCH) {
    $artifactTag = "lid_1"
    $PRESET = "cub-lid1"
} elseif ($LID2_SWITCH) {
    $artifactTag = "lid_2"
    $PRESET = "cub-lid2"
}

if ($env:GITHUB_ACTIONS -and $artifactTag) {
    $producerId = (& bash "./util/workflow/get_producer_id.sh").Trim()
    $artifactName = "z_cub-test-artifacts-$env:DEVCONTAINER_NAME-$producerId-$artifactTag"
    Write-Host "Unpacking artifact '$artifactName'"
    & bash "./util/artifacts/download_packed.sh" "$artifactName" "../"
} else {
    $buildCmd = "$PSScriptRoot/build_cub.ps1 -std $CXX_STANDARD -arch '$CUDA_ARCH' -cmake-options '$CMAKE_OPTIONS'"
    Write-Host "Running: $buildCmd"
    Invoke-Expression $buildCmd
}

test_preset "CUB ($PRESET)" "$PRESET"

If($CURRENT_PATH -ne "ci") {
    popd
}
