
Param(
    [Parameter(Mandatory = $true)]
    [Alias("cxx")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(11, 14, 17, 20)]
    [int]$CXX_STANDARD = 17,
    [Parameter(Mandatory = $true)]
    [Alias("archs")]
    [ValidateNotNullOrEmpty()]
    [string]$GPU_ARCHS = "70"
)

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

Remove-Module -Name build_common
Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList $CXX_STANDARD, $GPU_ARCHS

$ENABLE_DIALECT_CPP11 = If ($CXX_STANDARD -ne 11) {"OFF"} Else {"ON"}
$ENABLE_DIALECT_CPP14 = If ($CXX_STANDARD -ne 14) {"OFF"} Else {"ON"}
$ENABLE_DIALECT_CPP17 = If ($CXX_STANDARD -ne 17) {"OFF"} Else {"ON"}
$ENABLE_DIALECT_CPP20 = If ($CXX_STANDARD -ne 20) {"OFF"} Else {"ON"}

$CMAKE_OPTIONS = @(
    "-DCCCL_ENABLE_THRUST=OFF"
    "-DCCCL_ENABLE_LIBCUDACXX=OFF"
    "-DCCCL_ENABLE_CUB=ON"
    "-DCCCL_ENABLE_TESTING=OFF"
    "-DCUB_ENABLE_DIALECT_CPP11=$ENABLE_DIALECT_CPP11"
    "-DCUB_ENABLE_DIALECT_CPP14=$ENABLE_DIALECT_CPP14"
    "-DCUB_ENABLE_DIALECT_CPP17=$ENABLE_DIALECT_CPP17"
    "-DCUB_ENABLE_DIALECT_CPP20=$ENABLE_DIALECT_CPP20"
    "-DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT=ON"
    "-DCUB_IGNORE_DEPRECATED_CPP_DIALECT=ON"
    "-DCUB_ENABLE_RDC_TESTS=OFF"
)

configure_and_build "CUB" $CMAKE_OPTIONS

If($CURRENT_PATH -ne "ci") {
    popd
}
