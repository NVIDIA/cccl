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

# Execute the build script first:
$build_command = "$PSScriptRoot/build_cub.ps1 -std $CXX_STANDARD -arch `"$CUDA_ARCH`""
Write-Host "Executing: $build_command"
Invoke-Expression $build_command

Import-Module -Name "$PSScriptRoot/build_common.psm1" -ArgumentList $CXX_STANDARD, $CUDA_ARCH

$PRESET = "cub-cpp$CXX_STANDARD"
if ($NO_LID_SWITCH) { $PRESET = "cub-nolid-cpp$CXX_STANDARD" }
elseif ($LID0_SWITCH) { $PRESET = "cub-lid0-cpp$CXX_STANDARD" }
elseif ($LID1_SWITCH) { $PRESET = "cub-lid1-cpp$CXX_STANDARD" }
elseif ($LID2_SWITCH) { $PRESET = "cub-lid2-cpp$CXX_STANDARD" }

test_preset "CUB ($PRESET)" "$PRESET"

If($CURRENT_PATH -ne "ci") {
    popd
}
