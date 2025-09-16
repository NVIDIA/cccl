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
    [switch]$GPU_ONLY = $false
)

$ErrorActionPreference = "Stop"

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

# Execute the build script:
$build_command = "$PSScriptRoot/build_thrust.ps1 -std $CXX_STANDARD -arch `"$CUDA_ARCH`""
Write-Host "Executing: $build_command"
Invoke-Expression $build_command

Import-Module -Name "$PSScriptRoot/build_common.psm1" -ArgumentList $CXX_STANDARD, $CUDA_ARCH

if ($CPU_ONLY) {
    $PRESETS = @("thrust-cpu-cpp$CXX_STANDARD")
} elseif ($GPU_ONLY) {
    $PRESETS = @("thrust-gpu-cpp$CXX_STANDARD")
} else {
    $PRESETS = @("thrust-cpp$CXX_STANDARD")
}

foreach ($PRESET in $PRESETS) {
    test_preset "Thrust ($PRESET)" "$PRESET"
}

If($CURRENT_PATH -ne "ci") {
    popd
}
