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
    [switch]$LID2_SWITCH = $false
)

$ErrorActionPreference = "Stop"

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

$artifactTags = @()
if ($NO_LID_SWITCH) { $artifactTags += "no_lid" }
if ($LID0_SWITCH) { $artifactTags += "lid_0" }
if ($LID1_SWITCH) { $artifactTags += "lid_1" }
if ($LID2_SWITCH) { $artifactTags += "lid_2" }

Import-Module -Name "$PSScriptRoot/build_common.psm1" -ArgumentList $CXX_STANDARD, $CUDA_ARCH

$PRESET = "cub-cpp$CXX_STANDARD"
if ($NO_LID_SWITCH) { $PRESET = "cub-nolid-cpp$CXX_STANDARD" }
elseif ($LID0_SWITCH) { $PRESET = "cub-lid0-cpp$CXX_STANDARD" }
elseif ($LID1_SWITCH) { $PRESET = "cub-lid1-cpp$CXX_STANDARD" }
elseif ($LID2_SWITCH) { $PRESET = "cub-lid2-cpp$CXX_STANDARD" }

if ($env:GITHUB_ACTIONS -and $artifactTags.Count -gt 0) {
    $producerId = (& bash "./util/workflow/get_producer_id.sh").Trim()
    foreach ($tag in $artifactTags) {
        $artifactName = "z_cub-test-artifacts-$env:DEVCONTAINER_NAME-$producerId-$tag"
        Write-Host "Unpacking artifact '$artifactName'"
        & bash "./util/artifacts/download_packed.sh" "$artifactName" "../"
    }
} else {
    $build_command = "$PSScriptRoot/build_cub.ps1 -std $CXX_STANDARD -arch `"$CUDA_ARCH`""
    Write-Host "Executing: $build_command"
    Invoke-Expression $build_command
}

test_preset "CUB ($PRESET)" "$PRESET"

If($CURRENT_PATH -ne "ci") {
    popd
}
