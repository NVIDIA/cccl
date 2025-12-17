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

$ErrorActionPreference = "Stop"

# We need the full path to cl because otherwise cmake will replace CMAKE_CXX_COMPILER with the full path
# and keep CMAKE_CUDA_HOST_COMPILER at "cl" which breaks our cmake script
$script:HOST_COMPILER  = (Get-Command "cl").source -replace '\\','/'
$script:PARALLEL_LEVEL = $env:NUMBER_OF_PROCESSORS

Write-Host "=== Docker Container Resource Info ==="
Write-Host "Number of Processors: $script:PARALLEL_LEVEL"
Get-WmiObject Win32_OperatingSystem | ForEach-Object {
    Write-Host ("Memory: total={0:N1} GB, free={1:N1} GB" -f ($_.TotalVisibleMemorySize / 1MB), ($_.FreePhysicalMemory / 1MB))
}
Write-Host "======================================"

# Extract the CL version for export to build scripts:
$script:CL_VERSION_STRING = & cl.exe /?
if ($script:CL_VERSION_STRING -match "Version (\d+\.\d+)\.\d+") {
    $CL_VERSION = [version]$matches[1]
    Write-Host "Detected cl.exe version: $CL_VERSION"
}

$script:GLOBAL_CMAKE_OPTIONS = $CMAKE_OPTIONS
if ($CUDA_ARCH) {
    $script:GLOBAL_CMAKE_OPTIONS += ' "-DCMAKE_CUDA_ARCHITECTURES={0}"' -f $CUDA_ARCH
}

if (-not $env:CCCL_BUILD_INFIX) {
    $env:CCCL_BUILD_INFIX = ""
}

# Presets will be configured in this directory:
$BUILD_DIR = "../build/$env:CCCL_BUILD_INFIX"

If(!(test-path -PathType container "../build")) {
    New-Item -ItemType Directory -Path "../build"
}

# The most recent build will always be symlinked to cccl/build/latest
New-Item -ItemType Directory -Path "$BUILD_DIR" -Force

# Convert to an absolute path:
$BUILD_DIR = (Get-Item -Path "$BUILD_DIR").FullName

# Prepare environment for CMake:
$env:CMAKE_BUILD_PARALLEL_LEVEL = $PARALLEL_LEVEL
$env:CTEST_PARALLEL_LEVEL = 1
$env:CUDAHOSTCXX = $script:HOST_COMPILER
$env:CXX = $script:HOST_COMPILER

Write-Host "========================================"
Write-Host "Begin build"
Write-Host "pwd=$pwd"
Write-Host "BUILD_DIR=$BUILD_DIR"
Write-Host "CXX_STANDARD=$CXX_STANDARD"
Write-Host "CXX=$env:CXX"
Write-Host "CUDACXX=$env:CUDACXX"
Write-Host "CUDAHOSTCXX=$env:CUDAHOSTCXX"
Write-Host "TBB_ROOT=$env:TBB_ROOT"
Write-Host "NVCC_VERSION=$NVCC_VERSION"
Write-Host "CMAKE_BUILD_PARALLEL_LEVEL=$env:CMAKE_BUILD_PARALLEL_LEVEL"
Write-Host "CTEST_PARALLEL_LEVEL=$env:CTEST_PARALLEL_LEVEL"
Write-Host "CCCL_BUILD_INFIX=$env:CCCL_BUILD_INFIX"
Write-Host "GLOBAL_CMAKE_OPTIONS=$script:GLOBAL_CMAKE_OPTIONS"
Write-Host "Current commit is:"
Write-Host "$(git log -1 --format=short)"
Write-Host "========================================"

cmake --version
ctest --version

function configure_preset {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$BUILD_NAME,
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$PRESET,
        [Parameter(Mandatory = $false)]
        [string]$LOCAL_CMAKE_OPTIONS = ""
    )

    $step = "$BUILD_NAME (configure)"

    # CMake must be invoked in the same directory as the presets file:
    pushd ".."

    # Echo and execute command to stdout:
    $configure_command = "cmake --preset $PRESET --log-level VERBOSE"
    if ($LOCAL_CMAKE_OPTIONS) {
        $configure_command += " $LOCAL_CMAKE_OPTIONS"
    }
    if ($script:GLOBAL_CMAKE_OPTIONS) {
        $configure_command += " $script:GLOBAL_CMAKE_OPTIONS"
    }

    Write-Host $configure_command
    Invoke-Expression $configure_command
    $test_result = $LastExitCode

    If ($test_result -ne 0) {
        throw "$step Failed"
    }

    popd
    Write-Host "$step complete."
}

function build_preset {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$BUILD_NAME,
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$PRESET
    )

    $step = "$BUILD_NAME (build)"

    # CMake must be invoked in the same directory as the presets file:
    pushd ".."

    sccache -z >$null

    cmake --build --preset $PRESET -v
    $test_result = $LastExitCode

    $preset_dir = "${BUILD_DIR}/${PRESET}"
    $sccache_json = "${preset_dir}/sccache_stats.json"

    sccache --show-adv-stats
    sccache --show-adv-stats --stats-format=json > "${sccache_json}"

    echo "$step complete"

    If ($test_result -ne 0) {
         throw "$step Failed"
    }

    popd
}

function test_preset {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$BUILD_NAME,
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$PRESET
    )

    $step = "$BUILD_NAME (test)"

    # CTest must be invoked in the same directory as the presets file:
    pushd ".."

    sccache -z >$null

    ctest --preset $PRESET
    $test_result = $LastExitCode

    sccache --show-adv-stats

    echo "$step complete"

    If ($test_result -ne 0) {
         throw "$step Failed"
    }

    popd
}

function configure_and_build_preset {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$BUILD_NAME,
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$PRESET,
        [Parameter(Mandatory = $false)]
        [string]$LOCAL_CMAKE_OPTIONS = ""
    )

    configure_preset $BUILD_NAME $PRESET $LOCAL_CMAKE_OPTIONS
    build_preset $BUILD_NAME $PRESET
}

Export-ModuleMember -Function configure_preset, build_preset, test_preset, configure_and_build_preset
Export-ModuleMember -Variable BUILD_DIR, CL_VERSION
