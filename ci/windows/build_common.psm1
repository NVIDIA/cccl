
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


# We need the full path to cl because otherwise cmake will replace CMAKE_CXX_COMPILER with the full path
# and keep CMAKE_CUDA_HOST_COMPILER at "cl" which breaks our cmake script
$script:HOST_COMPILER  = (Get-Command "cl").source -replace '\\','/'
$script:PARALLEL_LEVEL = (Get-WmiObject -class Win32_processor).NumberOfLogicalProcessors

If($null -eq $env:DEVCONTAINER_NAME) {
    $script:BUILD_DIR="$PSScriptRoot/../../build/local"
} else {
    $script:BUILD_DIR="$PSScriptRoot/../../build/$DEVCONTAINER_NAME"
}

If(!(test-path -PathType container "../build")) {
    New-Item -ItemType Directory -Path "../build"
}

# The most recent build will always be symlinked to cccl/build/latest
New-Item -ItemType Directory -Path "$BUILD_DIR" -Force

# replace sccache binary to get it working with MSVC
$script:path_to_sccache =(gcm sccache).Source
Remove-Item $path_to_sccache -Force
Invoke-WebRequest -Uri "https://github.com/robertmaynard/sccache/releases/download/nvcc_msvc_v1/sccache.exe" -OutFile $path_to_sccache

$script:COMMON_CMAKE_OPTIONS= @(
    "-S .."
    "-B $BUILD_DIR"
    "-G Ninja"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CXX_STANDARD=$CXX_STANDARD"
    "-DCMAKE_CUDA_STANDARD=$CXX_STANDARD"
    "-DCMAKE_CXX_COMPILER=$HOST_COMPILER"
    "-DCMAKE_CUDA_HOST_COMPILER=$HOST_COMPILER"
    "-DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCHS"
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
)

Write-Host "========================================"
Write-Host "Begin build"
Write-Host "pwd=$pwd"
Write-Host "HOST_COMPILER=$HOST_COMPILER"
Write-Host "CXX_STANDARD=$CXX_STANDARD"
Write-Host "GPU_ARCHS=$GPU_ARCHS"
Write-Host "PARALLEL_LEVEL=$PARALLEL_LEVEL"
Write-Host "BUILD_DIR=$BUILD_DIR"
Write-Host "Current commit is:"
Write-Host "$(git log -1)"
Write-Host "========================================"

function configure {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        $CMAKE_OPTIONS
    )

    $FULL_CMAKE_OPTIONS = $script:COMMON_CMAKE_OPTIONS + $CMAKE_OPTIONS
    cmake $FULL_CMAKE_OPTIONS
    $test_result = $LastExitCode

    If ($test_result -ne 0) {
        throw 'Step Failed'
    }
}

function build {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$BUILD_NAME
    )

    sccache_stats('Start')

    cmake --build $script:BUILD_DIR --parallel $script:PARALLEL_LEVEL
    $test_result = $LastExitCode

    sccache_stats('Stop')
    echo "${BUILD_NAME} build complete"
    If ($test_result -ne 0) {
         throw 'Step Failed'
    }
}

function configure_and_build {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$BUILD_NAME,
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        $CMAKE_OPTIONS
    )

    configure -CMAKE_OPTIONS $CMAKE_OPTIONS
    build -BUILD_NAME $BUILD_NAME
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

Export-ModuleMember -Function configure, build, configure_and_build, sccache_stats
Export-ModuleMember -Variable BUILD_DIR
