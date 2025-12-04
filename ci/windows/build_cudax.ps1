Param(
    [Parameter(Mandatory = $false)]
    [Alias("std")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(20)]
    [int]$CXX_STANDARD = 20,
    [Parameter(Mandatory = $false)]
    [Alias("arch")]
    [string]$CUDA_ARCH = "",
    [Parameter(Mandatory = $false)]
    [Alias("cmake-options")]
    [string]$CMAKE_OPTIONS = ""
)

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

Remove-Module -Name build_common -ErrorAction SilentlyContinue
Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList @($CXX_STANDARD, $CUDA_ARCH, $CMAKE_OPTIONS)

$PRESET = "cudax"
$LOCAL_CMAKE_OPTIONS = "-DCMAKE_CXX_STANDARD=$CXX_STANDARD -DCMAKE_CUDA_STANDARD=$CXX_STANDARD"

configure_and_build_preset "CUDA Experimental" $PRESET $LOCAL_CMAKE_OPTIONS

If($CURRENT_PATH -ne "ci") {
    popd
}
