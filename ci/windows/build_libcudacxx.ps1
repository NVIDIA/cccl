
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

$CMAKE_OPTIONS = @(
    "-DCCCL_ENABLE_THRUST=OFF"
    "-DCCCL_ENABLE_LIBCUDACXX=ON"
    "-DCCCL_ENABLE_CUB=OFF"
    "-DCCCL_ENABLE_TESTING=OFF"
    "-DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=ON"
)

$LIT_OPTIONS = @(
    "-v"
    "--no-progress-bar"
    "-Dexecutor=""NoopExecutor()"""
    "-Dcompute_archs=$GPU_ARCHS"
    "-Dstd=c++$CXX_STANDARD"
    "$BUILD_DIR/libcudacxx/test"
)

configure $CMAKE_OPTIONS

pushd $BUILD_DIR/libcudacxx/

sccache_stats('Start')
lit $LIT_OPTIONS
$test_result = $LastExitCode
sccache_stats('Stop')

popd
If($CURRENT_PATH -ne "ci") {
    popd
}

If ($test_result -ne 0) {
    throw 'Step Failed'
}