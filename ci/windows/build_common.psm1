Param(
    [Parameter(Mandatory = $false)]
    [Alias("std")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(11, 14, 17, 20)]
    [int]$CXX_STANDARD = 17
)

$ErrorActionPreference = "Stop"

# We need the full path to cl because otherwise cmake will replace CMAKE_CXX_COMPILER with the full path
# and keep CMAKE_CUDA_HOST_COMPILER at "cl" which breaks our cmake script
$script:HOST_COMPILER  = (Get-Command "cl").source -replace '\\','/'
$script:PARALLEL_LEVEL = (Get-WmiObject -class Win32_processor).NumberOfLogicalProcessors

# Extract the CL version for export to build scripts:
$script:CL_VERSION_STRING = & cl.exe /?
if ($script:CL_VERSION_STRING -match "Version (\d+\.\d+)\.\d+") {
    $CL_VERSION = [version]$matches[1]
    Write-Host "Detected cl.exe version: $CL_VERSION"
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
$env:CUDAHOSTCXX = $HOST_COMPILER.FullName
$env:CXX = $HOST_COMPILER.FullName

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
Write-Host "Current commit is:"
Write-Host "$(git log -1)"
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
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$CMAKE_OPTIONS
    )

    $step = "$BUILD_NAME (configure)"

    # CMake must be invoked in the same directory as the presets file:
    pushd ".."

    # Echo and execute command to stdout:
    $configure_command = "cmake --preset $PRESET $CMAKE_OPTIONS --log-level VERBOSE"
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

    sccache_stats('Start')

    cmake --build --preset $PRESET -v
    $test_result = $LastExitCode

    $preset_dir = "${BUILD_DIR}/${PRESET}"
    $sccache_json = "${preset_dir}/sccache_stats.json"
    sccache --show-adv-stats --stats-format=json > "${sccache_json}"

    sccache_stats('Stop')

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

    sccache_stats('Start')

    ctest --preset $PRESET
    $test_result = $LastExitCode

    sccache_stats('Stop')

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
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$CMAKE_OPTIONS
    )

    configure_preset "$BUILD_NAME" "$PRESET" "$CMAKE_OPTIONS"
    build_preset "$BUILD_NAME" "$PRESET"
}

function sccache_stats {
    Param (
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [ValidateSet('Start','Stop')]
        [string]$MODE
    )

    $sccache_stats = sccache -s
    If($MODE -eq 'Start') {
        [int]$script:sccache_compile_requests = ($sccache_stats[0] -replace '[^\d]+')
        [int]$script:sccache_cache_hits_cpp   = ($sccache_stats[2] -replace '[^\d]+')
        [int]$script:sccache_cache_hits_cuda  = ($sccache_stats[3] -replace '[^\d]+')
        [int]$script:sccache_cache_miss_cpp   = ($sccache_stats[5] -replace '[^\d]+')
        [int]$script:sccache_cache_miss_cuda  = ($sccache_stats[6] -replace '[^\d]+')
    } else {
        [int]$final_sccache_compile_requests = ($sccache_stats[0] -replace '[^\d]+')
        [int]$final_sccache_cache_hits_cpp   = ($sccache_stats[2] -replace '[^\d]+')
        [int]$final_sccache_cache_hits_cuda  = ($sccache_stats[3] -replace '[^\d]+')
        [int]$final_sccache_cache_miss_cpp   = ($sccache_stats[5] -replace '[^\d]+')
        [int]$final_sccache_cache_miss_cuda  = ($sccache_stats[6] -replace '[^\d]+')

        [int]$total_requests  = $final_sccache_compile_requests - $script:sccache_compile_requests
        [int]$total_hits_cpp  = $final_sccache_cache_hits_cpp   - $script:sccache_cache_hits_cpp
        [int]$total_hits_cuda = $final_sccache_cache_hits_cuda  - $script:sccache_cache_hits_cuda
        [int]$total_miss_cpp  = $final_sccache_cache_miss_cpp   - $script:sccache_cache_miss_cpp
        [int]$total_miss_cuda = $final_sccache_cache_miss_cuda  - $script:sccache_cache_miss_cuda
        If ( $total_requests -gt 0 ) {
            [int]$hit_rate_cpp  = $total_hits_cpp  / $total_requests * 100;
            [int]$hit_rate_cuda = $total_hits_cuda / $total_requests * 100;
            echo "sccache hits cpp:  $total_hits_cpp  `t| misses: $total_miss_cpp  `t| hit rate: $hit_rate_cpp%"
            echo "sccache hits cuda: $total_hits_cuda `t| misses: $total_miss_cuda `t| hit rate: $hit_rate_cuda%"
        } else {
            echo "sccache stats: N/A No new compilation requests"
        }
    }
}

Export-ModuleMember -Function configure_preset, build_preset, test_preset, configure_and_build_preset, sccache_stats
Export-ModuleMember -Variable BUILD_DIR, CL_VERSION
