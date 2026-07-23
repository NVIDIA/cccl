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
    [string]$CMAKE_OPTIONS = ""
)

# Build the CCCL samples on Windows. Samples are a standalone CMake project
# (samples/CMakeLists.txt), so we bypass CCCL's preset infrastructure and
# drive cmake directly. Windows CI is build-only - there are no Windows GPU
# runners in the CCCL matrix.

$CURRENT_PATH = Split-Path $pwd -leaf
If ($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

Remove-Module -Name build_common -ErrorAction SilentlyContinue
Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList @($CXX_STANDARD, $CUDA_ARCH, $CMAKE_OPTIONS)

$REPO_ROOT = (Resolve-Path "..").Path
$SAMPLES_SRC_DIR = Join-Path $REPO_ROOT "samples"
$SAMPLES_BUILD_DIR = Join-Path $BUILD_DIR "samples"

Write-Host "SAMPLES_SRC_DIR=$SAMPLES_SRC_DIR"
Write-Host "SAMPLES_BUILD_DIR=$SAMPLES_BUILD_DIR"

$configure_command = "cmake -S `"$SAMPLES_SRC_DIR`" -B `"$SAMPLES_BUILD_DIR`""
$configure_command += " -DCMAKE_CXX_STANDARD=$CXX_STANDARD"
$configure_command += " -DCMAKE_CUDA_STANDARD=$CXX_STANDARD"
$configure_command += " -DCCCL_SOURCE_DIR=`"$REPO_ROOT`""
if ($CUDA_ARCH) {
    $configure_command += " -DCMAKE_CUDA_ARCHITECTURES=`"$CUDA_ARCH`""
}
if ($CMAKE_OPTIONS) {
    $configure_command += " $CMAKE_OPTIONS"
}

Write-Host "=== Configure ==="
Write-Host $configure_command
Invoke-Expression $configure_command
if ($LastExitCode -ne 0) {
    throw "CCCL Samples configure Failed"
}

Write-Host "=== Build ==="
cmake --build "$SAMPLES_BUILD_DIR" --parallel $env:CMAKE_BUILD_PARALLEL_LEVEL
if ($LastExitCode -ne 0) {
    throw "CCCL Samples build Failed"
}

Write-Host "=== Install ==="
cmake --install "$SAMPLES_BUILD_DIR"
if ($LastExitCode -ne 0) {
    throw "CCCL Samples install Failed"
}

If ($CURRENT_PATH -ne "ci") {
    popd
}
