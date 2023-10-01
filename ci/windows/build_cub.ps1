
Param(
    [Parameter(Mandatory = $true)]
    [Alias("cxx")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(11, 14, 17, 20)]
    [int]$CXX_STANDARD = 17
)

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

Remove-Module -Name build_common
Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList $CXX_STANDARD

$PRESET = "cub-cpp$CXX_STANDARD"

# Override the preset 60;70;80, since cuda::atomic can't be used on msvc+sm60:
$CMAKE_OPTIONS = "-DCMAKE_CUDA_ARCHITECTURES=70;80"

configure_and_build_preset "CUB" "$PRESET" "$CMAKE_OPTIONS"

If($CURRENT_PATH -ne "ci") {
    popd
}
