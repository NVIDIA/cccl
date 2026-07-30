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
    [switch]$LID2_SWITCH = $false
)

$ErrorActionPreference = "Stop"

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList @($CXX_STANDARD, $CUDA_ARCH, $CMAKE_OPTIONS)

$PRESET = "cub"
$artifactTags = @()

if ($NO_LID_SWITCH) {
    $artifactTags += "no_lid"
    $PRESET = "cub-nolid"
} elseif ($LID0_SWITCH) {
    $artifactTags += "lid_0"
    $PRESET = "cub-lid0"
} elseif ($LID1_SWITCH) {
    $artifactTags += "lid_1"
    $PRESET = "cub-lid1"
} elseif ($LID2_SWITCH) {
    $artifactTags += "lid_2"
    $PRESET = "cub-lid2"
}
$LOCAL_CMAKE_OPTIONS = "-DCMAKE_CXX_STANDARD=$CXX_STANDARD -DCMAKE_CUDA_STANDARD=$CXX_STANDARD"

if ($CL_VERSION -lt [version]"19.20") {
    $LOCAL_CMAKE_OPTIONS = "$LOCAL_CMAKE_OPTIONS -DCCCL_IGNORE_DEPRECATED_COMPILER=ON"
}

configure_and_build_preset "CUB" $PRESET $LOCAL_CMAKE_OPTIONS

if ($env:GITHUB_ACTIONS) {
    Write-Host "Packaging test artifacts..."
    if ($artifactTags.Count -gt 0) {
        & bash "./upload_cub_test_artifacts.sh" @artifactTags
    } else {
        & bash "./upload_cub_test_artifacts.sh"
    }
}

If($CURRENT_PATH -ne "ci") {
    popd
}
