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

# Get the number of logical processors
$N_CPUS = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
if (-not $N_CPUS) { $N_CPUS = [Environment]::ProcessorCount } # Fallback for cross-platform
if (-not $N_CPUS) { $N_CPUS = $env:NUMBER_OF_PROCESSORS } # Fallback for cross-platform
$N_CPUS_MINUS_1 = if ($N_CPUS -gt 1) { $N_CPUS - 1 } else { 1 }

# Set variables as read-only constants
Set-Variable -Name N_CPUS -Value $N_CPUS -Option ReadOnly -Force
Set-Variable -Name N_CPUS_MINUS_1 -Value $N_CPUS_MINUS_1 -Option ReadOnly -Force

# Provide a default value for PARALLEL_LEVEL if it does not exist in the environment
if (-not $env:PARALLEL_LEVEL) {
    $PARALLEL_LEVEL = $N_CPUS_MINUS_1
} else {
    $PARALLEL_LEVEL = [int]$env:PARALLEL_LEVEL
}

# Safely check environment variables with defaults
$USE_SCCACHE_DIST = if ($env:USE_SCCACHE_DIST) { [bool]::Parse($env:USE_SCCACHE_DIST) } else { $false }
$SCCACHE_DIST_URL_SET = -not [string]::IsNullOrEmpty($env:SCCACHE_DIST_URL)

# If PARALLEL_LEVEL <= 0, assume build cluster and tune parallelism to as many
# concurrent preprocessor calls we think we can do without OOM'ing the machine
if ($USE_SCCACHE_DIST -and $SCCACHE_DIST_URL_SET -and $PARALLEL_LEVEL -le 0) {
    # Memory (in KB) used by each `sccache <compiler> ...` invocation from ninja
    # * 1.5Mb for the shell launched by ninja
    # * 6MiB for each sccache client process
    # * round up to 10MiB
    $mem_per_job = 10 * 1024
    # Assume preprocessor invocations take ~300Mb or so
    $mem_for_preprocessing = $N_CPUS * 300 * 1024
    # It's usually around 400-600MiB, but be conservative
    # and assume the sccache daemon will use 1GiB of RAM
    $mem_for_sccache_daemon = 1 * 1024 * 1024
    # Available memory (in KB) using CIM/WMI
    $mem_available = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory
    # Stay under 95% for CI
    $mem_available = [math]::Floor($mem_available * 95 / 100)
    # Total job count is available memory after accounting for `nproc` concurrent preprocessor
    # calls divided by the amount of memory required to invoke the sccache thin client process
    $PARALLEL_LEVEL = [math]::Floor(($mem_available - $mem_for_sccache_daemon - $mem_for_preprocessing) / $mem_per_job)
}

if ($PARALLEL_LEVEL -le 0) {
    $PARALLEL_LEVEL = $N_CPUS_MINUS_1
}

# Export to environment block
$env:PARALLEL_LEVEL = $PARALLEL_LEVEL

Write-Host "=== Docker Container Resource Info ==="
Write-Host "Number of Processors: $script:N_CPUS"
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

# limit the number of concurrent link steps to avoid OOM'ing CI
$script:GLOBAL_CMAKE_OPTIONS += ' "-DCMAKE_JOB_POOLS=link_jobs=' + "$N_CPUS_MINUS_1" + '"'
$script:GLOBAL_CMAKE_OPTIONS += ' "-DCMAKE_JOB_POOL_LINK=link_jobs"'

if ($CUDA_ARCH) {
    $script:GLOBAL_CMAKE_OPTIONS += ' "-DCMAKE_CUDA_ARCHITECTURES={0}"' -f $CUDA_ARCH
}

# Default to pedantic mode in CI (GitHub Actions)
if ($env:GITHUB_ACTIONS) {
    $script:GLOBAL_CMAKE_OPTIONS += ' "-DCCCL_ENABLE_WERROR=ON" "-DCCCL_ENABLE_PRAGMA_SYSTEM_HEADER=OFF"'
} else {
    $script:GLOBAL_CMAKE_OPTIONS += ' "-DCCCL_ENABLE_WERROR=OFF" "-DCCCL_ENABLE_PRAGMA_SYSTEM_HEADER=ON"'
}

if (-not $env:CCCL_BUILD_INFIX) {
    $env:CCCL_BUILD_INFIX = ""
}

# Presets will be configured in this directory:
$BUILD_DIR = "$PSScriptRoot/../../build/$env:CCCL_BUILD_INFIX"

# The most recent build will always be symlinked to cccl/build/latest
New-Item -ItemType Directory -Path "$BUILD_DIR" -Force

# Convert to an absolute path:
$BUILD_DIR = (Get-Item -Path "$BUILD_DIR").FullName

# Prepare environment for CMake:
if ($PARALLEL_LEVEL -ge $N_CPUS_MINUS_1) {
    $env:CMAKE_BUILD_PARALLEL_LEVEL = $N_CPUS_MINUS_1
} else {
    $env:CMAKE_BUILD_PARALLEL_LEVEL = $PARALLEL_LEVEL
}
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
Write-Host "PARALLEL_LEVEL=$env:PARALLEL_LEVEL"
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
    $CURRENT_PATH = Split-Path $pwd -leaf
    If($CURRENT_PATH -ne "windows") {
        Write-Host "Moving to repo root"
        pushd "$PSScriptRoot/../.."
    }

    & bash -c ". ./ci/pretty_printing.sh; begin_group '$step'"

    # Echo and execute command to stdout:
    $configure_command = "cmake --preset $PRESET --log-level VERBOSE"
    if ($LOCAL_CMAKE_OPTIONS) {
        $configure_command += " $LOCAL_CMAKE_OPTIONS"
    }
    if ($script:GLOBAL_CMAKE_OPTIONS) {
        $configure_command += " $script:GLOBAL_CMAKE_OPTIONS"
    }

    Write-Host $configure_command

    $env:SCCACHE_NO_DIST_COMPILE="1"

    $duration = [math]::Round((Measure-Command { Invoke-Expression $configure_command | Out-Default }).TotalSeconds)

    $test_result = $LastExitCode

    Remove-Item Env:\SCCACHE_NO_DIST_COMPILE

    & bash -c ". ./ci/pretty_printing.sh; end_group '$step' $test_result $duration"

    If($CURRENT_PATH -ne "windows") {
        popd
    }

    If ($test_result -ne 0) {
        throw "$step Failed"
    }
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
    $CURRENT_PATH = Split-Path $pwd -leaf
    If($CURRENT_PATH -ne "windows") {
        Write-Host "Moving to repo root"
        pushd "$PSScriptRoot/../.."
    }

    & bash -c ". ./ci/pretty_printing.sh; begin_group '$step'"

    sccache -z >$null

    $preset_dir = "${BUILD_DIR}/${PRESET}"
    $sccache_json = "${preset_dir}/sccache_stats.json"

    # Echo and execute command to stdout:
    $build_command = "cmake --build --preset $PRESET --parallel $env:PARALLEL_LEVEL"
    # $build_command = "cmake --build --preset $PRESET -v --parallel $env:PARALLEL_LEVEL"

    Write-Host $build_command

    $duration = [math]::Round((Measure-Command { Invoke-Expression $build_command | Out-Default }).TotalSeconds)

    $test_result = $LastExitCode

    sccache --show-adv-stats --stats-format=json > "${sccache_json}"

    & bash -c ". ./ci/pretty_printing.sh; end_group '$step' $test_result $duration"

    sccache --show-adv-stats

    If($CURRENT_PATH -ne "windows") {
        popd
    }

    If ($test_result -ne 0) {
        throw "$step Failed"
    }
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
    $CURRENT_PATH = Split-Path $pwd -leaf
    If($CURRENT_PATH -ne "windows") {
        Write-Host "Moving to repo root"
        pushd "$PSScriptRoot/../.."
    }

    & bash -c ". ./ci/pretty_printing.sh; begin_group '$step'"

    sccache -z >$null

    # Echo and execute command to stdout:
    $test_command = "ctest --preset $PRESET"

    Write-Host $build_command

    $duration = [math]::Round((Measure-Command { Invoke-Expression $test_command | Out-Default }).TotalSeconds)
    $test_result = $LastExitCode

    & bash -c ". ./ci/pretty_printing.sh; end_group '$step' $test_result $duration"

    sccache --show-adv-stats

    If($CURRENT_PATH -ne "windows") {
        popd
    }

    If ($test_result -ne 0) {
         throw "$step Failed"
    }
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

function Invoke-Checked {
    <#
    .SYNOPSIS
        Runs a script block and throws if the last native command in it exits
        non-zero. $ErrorActionPreference = "Stop" does not make native commands
        (python/pip/pytest/...) throw, so their $LASTEXITCODE must be checked
        explicitly; this wraps that boilerplate into one call.
    .EXAMPLE
        Invoke-Checked { & $python -m pip install pytest } "pip install failed"
    #>
    param(
        [Parameter(Mandatory, Position = 0)][scriptblock]$ScriptBlock,
        [Parameter(Position = 1)][string]$ErrorMessage = "Native command failed"
    )
    & $ScriptBlock
    if ($LASTEXITCODE -ne 0) {
        throw "$ErrorMessage (exit code $LASTEXITCODE)"
    }
}

Export-ModuleMember -Function configure_preset, build_preset, test_preset, configure_and_build_preset, Invoke-Checked
Export-ModuleMember -Variable BUILD_DIR, CL_VERSION
