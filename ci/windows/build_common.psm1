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

Write-Host "::group::Environment Information"
Write-Host "Number of Processors: $script:PARALLEL_LEVEL"
Get-WmiObject Win32_OperatingSystem | ForEach-Object {
    Write-Host ("Memory: total={0:N1} GB, free={1:N1} GB" -f ($_.TotalVisibleMemorySize / 1MB), ($_.FreePhysicalMemory / 1MB))
}

# Extract the CL version for export to build scripts:
$CL_VERSION = cmd /c "`"$script:HOST_COMPILER`" /? 2>&1" | Select-String "Compiler Version"
$CL_VERSION -match ".*Compiler Version ([0-9]+\.[0-9]+)\..*"
$CL_VERSION = [version]$matches[1]
Write-Host "Detected cl.exe version: $CL_VERSION"

$CUDA_VERSION = cmd /c "nvcc --version 2>&1" | Select-String "Cuda compilation tools, release (\d+\.\d+)"
$CUDA_VERSION -match ".*Cuda compilation tools, release ([0-9]+\.[0-9]+),.*"
$CUDA_VERSION = [version]$matches[1]
Write-Host "Detected nvcc version: $CUDA_VERSION"

# If both versions are set and CCCL_BUILD_INFIX is not defined, set it to cudaXX.Y-clXX.YY
if ($CL_VERSION -and $CUDA_VERSION -and -not $env:CCCL_BUILD_INFIX) {
    $env:CCCL_BUILD_INFIX = "cuda{0}-cl{1}" -f $CUDA_VERSION, $CL_VERSION
}

$script:GLOBAL_CMAKE_OPTIONS = $CMAKE_OPTIONS
if ($CUDA_ARCH) {
    $script:GLOBAL_CMAKE_OPTIONS += ' "-DCMAKE_CUDA_ARCHITECTURES={0}"' -f $CUDA_ARCH
}

# Presets will be configured in this directory:
$BUILD_DIR = "../build/$env:CCCL_BUILD_INFIX"

# Create the build dir and symlink it to build/latest:
$latest_link = "../build/latest"
[System.IO.Directory]::CreateDirectory($BUILD_DIR) | Out-Null
$BUILD_DIR = (Get-Item -Path "$BUILD_DIR").FullName # to absolute path
Remove-Item -Path $latest_link -Force -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType SymbolicLink -Path $latest_link -Target $BUILD_DIR | Out-Null

# Prepare environment for CMake:
$env:CMAKE_BUILD_PARALLEL_LEVEL = $PARALLEL_LEVEL
$env:CTEST_PARALLEL_LEVEL = 1
$env:CUDAHOSTCXX = $script:HOST_COMPILER
$env:CXX = $script:HOST_COMPILER

Write-Host "pwd=$pwd"
Write-Host "BUILD_DIR=$BUILD_DIR"
Write-Host "CXX_STANDARD=$CXX_STANDARD"
Write-Host "CXX=$env:CXX"
Write-Host "CUDACXX=$env:CUDACXX"
Write-Host "CUDAHOSTCXX=$env:CUDAHOSTCXX"
Write-Host "CL_VERSION=$CL_VERSION"
Write-Host "CUDA_VERSION=$CUDA_VERSION"
Write-Host "TBB_ROOT=$env:TBB_ROOT"
Write-Host "NVCC_VERSION=$NVCC_VERSION"
Write-Host "CMAKE_BUILD_PARALLEL_LEVEL=$env:CMAKE_BUILD_PARALLEL_LEVEL"
Write-Host "CTEST_PARALLEL_LEVEL=$env:CTEST_PARALLEL_LEVEL"
Write-Host "CCCL_BUILD_INFIX=$env:CCCL_BUILD_INFIX"
Write-Host "GLOBAL_CMAKE_OPTIONS=$script:GLOBAL_CMAKE_OPTIONS"
Write-Host "Current commit is:"
Write-Host "$(git log -1 --format=short)"
Write-Host "$(sccache --version)"
Write-Host "$(cmake --version)"
Write-Host "$(ctest --version)"
Write-Host "::endgroup::"

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

    Write-Host "::group::$step - $PRESET"
    $now = Get-Date

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

    Write-Host "::endgroup::"
    $end = Get-Date
    $elapsed = "{0:hh\:mm\:ss}" -f ($end - $now)
    Write-Host "$step complete in $elapsed" -ForegroundColor Blue
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

    Write-Host "::group::$step - $PRESET"
    $now = Get-Date

    # CMake must be invoked in the same directory as the presets file:
    pushd ".."

    sccache -z >$null

    cmake --build --preset $PRESET -v
    $test_result = $LastExitCode

    $preset_dir = "${BUILD_DIR}/${PRESET}"
    $sccache_json = "${preset_dir}/sccache_stats.json"

    sccache --show-adv-stats
    sccache --show-adv-stats --stats-format=json > "${sccache_json}"

    Write-Host "::endgroup::"
    $end = Get-Date
    $elapsed = "{0:hh\:mm\:ss}" -f ($end - $now)
    Write-Host "$step complete in $elapsed" -ForegroundColor Blue

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

    Write-Host "::group::$step - $PRESET"
    $now = Get-Date

    # CTest must be invoked in the same directory as the presets file:
    pushd ".."

    sccache -z >$null

    ctest --preset $PRESET
    $test_result = $LastExitCode

    sccache --show-adv-stats

    Write-Host "::endgroup::"
    $end = Get-Date
    $elapsed = "{0:hh\:mm\:ss}" -f ($end - $now)
    Write-Host "$step complete in $elapsed" -ForegroundColor Blue

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
