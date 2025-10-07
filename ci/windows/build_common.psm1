Param(
    [Parameter(Mandatory = $false)]
    [Alias("std")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(17, 20)]
    [int]$CXX_STANDARD = 17,
    [Parameter(Mandatory = $false)]
    [Alias("arch")]
    [string]$CUDA_ARCH = ""
)

$ErrorActionPreference = "Stop"

# We need the full path to cl because otherwise cmake will replace CMAKE_CXX_COMPILER with the full path
# and keep CMAKE_CUDA_HOST_COMPILER at "cl" which breaks our cmake script
$script:HOST_COMPILER = (Get-Command "cl").source -replace '\\', '/'
$script:PARALLEL_LEVEL = (Get-WmiObject -class Win32_processor).NumberOfLogicalProcessors

Write-Host "=== Docker Container Resource Info ==="
Write-Host "Number of Processors: $script:PARALLEL_LEVEL"
Get-WmiObject Win32_OperatingSystem | ForEach-Object {
    Write-Host ("Memory: total={0:N1} GB, free={1:N1} GB" -f ($_.TotalVisibleMemorySize/1MB), ($_.FreePhysicalMemory/1MB))
}

# Extract the CL version for export to build scripts:
$script:CL_VERSION_STRING = & cl.exe /?
if ($script:CL_VERSION_STRING -match "Version (\d+\.\d+)\.\d+") {
    $CL_VERSION = [version]$matches[1]
    Write-Host "Detected cl.exe version: $CL_VERSION"
}

$script:GLOBAL_CMAKE_OPTIONS = ""
if ($CUDA_ARCH -ne "") {
    # Quote the value to ensure it's treated as a single argument, even with semicolons
    $script:GLOBAL_CMAKE_OPTIONS += "`"-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH`" "
}

if (-not $env:CCCL_BUILD_INFIX) {
    $env:CCCL_BUILD_INFIX = ""
}

# Presets will be configured in this directory:
$BUILD_DIR = "../build/$env:CCCL_BUILD_INFIX"

If (!(test-path -PathType container "../build")) {
    New-Item -ItemType Directory -Path "../build"
}

# The most recent build will always be symlinked to cccl/build/latest
New-Item -ItemType Directory -Path "$BUILD_DIR" -Force

# Convert to an absolute path:
$BUILD_DIR = (Get-Item -Path "$BUILD_DIR").FullName

# Prepare environment for CMake:
if (-not $env:CMAKE_BUILD_PARALLEL_LEVEL -or [string]::IsNullOrWhiteSpace($env:CMAKE_BUILD_PARALLEL_LEVEL)) {
    # Only set CMAKE_BUILD_PARALLEL_LEVEL if it's not already defined.
    $env:CMAKE_BUILD_PARALLEL_LEVEL = $script:PARALLEL_LEVEL
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
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$CMAKE_OPTIONS
    )

    $step = "$BUILD_NAME (configure)"

    # CMake must be invoked in the same directory as the presets file:
    pushd ".."

    # Echo and execute command to stdout:
    $configure_command = "cmake --preset $PRESET $script:GLOBAL_CMAKE_OPTIONS $CMAKE_OPTIONS --log-level VERBOSE"
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

    cmake --build --preset $PRESET -v -- -k 0
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
        [ValidateSet('Start', 'Stop')]
        [string]$MODE
    )

    $sccache_stats = sccache -s
    If ($MODE -eq 'Start') {
        [int]$script:sccache_compile_requests = ($sccache_stats[0] -replace '[^\d]+')
        [int]$script:sccache_cache_hits_cpp = ($sccache_stats[2] -replace '[^\d]+')
        [int]$script:sccache_cache_hits_cuda = ($sccache_stats[3] -replace '[^\d]+')
        [int]$script:sccache_cache_miss_cpp = ($sccache_stats[5] -replace '[^\d]+')
        [int]$script:sccache_cache_miss_cuda = ($sccache_stats[6] -replace '[^\d]+')
    }
    else {
        [int]$final_sccache_compile_requests = ($sccache_stats[0] -replace '[^\d]+')
        [int]$final_sccache_cache_hits_cpp = ($sccache_stats[2] -replace '[^\d]+')
        [int]$final_sccache_cache_hits_cuda = ($sccache_stats[3] -replace '[^\d]+')
        [int]$final_sccache_cache_miss_cpp = ($sccache_stats[5] -replace '[^\d]+')
        [int]$final_sccache_cache_miss_cuda = ($sccache_stats[6] -replace '[^\d]+')

        [int]$total_requests = $final_sccache_compile_requests - $script:sccache_compile_requests
        [int]$total_hits_cpp = $final_sccache_cache_hits_cpp - $script:sccache_cache_hits_cpp
        [int]$total_hits_cuda = $final_sccache_cache_hits_cuda - $script:sccache_cache_hits_cuda
        [int]$total_miss_cpp = $final_sccache_cache_miss_cpp - $script:sccache_cache_miss_cpp
        [int]$total_miss_cuda = $final_sccache_cache_miss_cuda - $script:sccache_cache_miss_cuda
        If ( $total_requests -gt 0 ) {
            [int]$hit_rate_cpp = $total_hits_cpp / $total_requests * 100;
            [int]$hit_rate_cuda = $total_hits_cuda / $total_requests * 100;
            echo "sccache hits cpp:  $total_hits_cpp  `t| misses: $total_miss_cpp  `t| hit rate: $hit_rate_cpp%"
            echo "sccache hits cuda: $total_hits_cuda `t| misses: $total_miss_cuda `t| hit rate: $hit_rate_cuda%"
        }
        else {
            echo "sccache stats: N/A No new compilation requests"
        }
    }
}

Export-ModuleMember -Function configure_preset, build_preset, test_preset, configure_and_build_preset, sccache_stats
Export-ModuleMember -Variable BUILD_DIR, CL_VERSION

# Additional shared helpers for Windows Python/CI scripts
function Get-Python {
    Param([Parameter(Mandatory = $true)][string]$Version)
    $exe = $null
    try { $exe = (& py -$Version -c "import sys; print(sys.executable)" 2>$null) } catch {}
    if (-not $exe) {
        $exe = (Get-Command python).Source
        $ver = & $exe -c "import sys; print('%d.%d'%sys.version_info[:2])"
        if ($ver -ne $Version) { throw "Requested Python $Version not found" }
    }
    return $exe
}

function Get-CudaMajor {
    if ($env:CUDA_PATH) {
        $nvcc = Join-Path $env:CUDA_PATH "bin/nvcc.exe"
        if (Test-Path $nvcc) {
            $out = & $nvcc --version 2>&1
            $text = ($out -join "`n")
            if ($text -match 'release\s+(\d+)\.') { return $Matches[1] }
        }
        # Fallback: parse major from CUDA_PATH like ...\v13.0 or ...\CUDA\13
        $pathMatch = [regex]::Match($env:CUDA_PATH, 'v?(\d+)(?:\.\d+)?')
        if ($pathMatch.Success) { return $pathMatch.Groups[1].Value }
    }
    return '13'
}

function Convert-ToUnixPath {
    Param([Parameter(Mandatory = $true)][string]$p)
    return ($p -replace "\\", "/")
}

function Get-RepoRoot {
    return (Resolve-Path "$PSScriptRoot/../..")
}

function Ensure-CudaCcclWheel {
    Param(
        [Parameter(Mandatory = $true)][string]$PyVersion,
        [Parameter(Mandatory = $false)][switch]$UseNinja
    )

    $repoRoot = Get-RepoRoot
    $wheelhouse = Join-Path $repoRoot "wheelhouse"
    New-Item -ItemType Directory -Path $wheelhouse -Force | Out-Null

    $wheel = Get-ChildItem $wheelhouse -Filter "cuda_cccl-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $wheel) {
        $buildScript = Join-Path $PSScriptRoot "build_cuda_cccl_python.ps1"
        $psArgs = ('-File', $buildScript, '-py-version', $PyVersion)
        if ($UseNinja) { $psArgs += '-UseNinja' }
        & powershell @psArgs | Write-Host
        $wheel = Get-ChildItem $wheelhouse -Filter "cuda_cccl-*.whl" | Select-Object -First 1
    }
    if (-not $wheel) { throw "cuda_cccl wheel not found in $wheelhouse" }
    return $wheel.FullName
}

function Get-OnePathMatch {
    [CmdletBinding(DefaultParameterSetName = 'FileSet')]
    param(
        [Parameter(Mandatory)]
        [string] $Path,

        [Parameter(Mandatory)]
        [string] $Pattern,

        [Parameter(Mandatory, ParameterSetName = 'FileSet')]
        [switch] $File,

        [Parameter(Mandatory, ParameterSetName = 'DirSet')]
        [switch] $Directory,

        [switch] $Recurse
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Path not found or not a directory: $Path"
    }

    $gciArgs = @{
        LiteralPath = $Path
        ErrorAction = 'SilentlyContinue'
    }

    if ($Recurse) { $gciArgs['Recurse'] = $true }
    if ($PSCmdlet.ParameterSetName -eq 'FileSet') {
        $gciArgs['File'] = $true
    }
    else {
        $gciArgs['Directory'] = $true
    }

    $pathMatches = @(
        Get-ChildItem @gciArgs |
        Where-Object { $_.Name -match $Pattern } |
        Select-Object -ExpandProperty FullName
    )

    if ($pathMatches.Count -ne 1) {
        $kind = if ($PSCmdlet.ParameterSetName -eq 'FileSet') { 'file' }
        else { 'directory' }
        $indented = ($pathMatches | ForEach-Object { "    $_" }) -join "`n"

        $msg = @"
Expected exactly one $kind name matching regex:
  $Pattern
under:
  $Path
Found:
  $($pathMatches.Count)

$indented
"@
        throw $msg
    }

    return $pathMatches[0]
}

Export-ModuleMember -Function Get-Python, Get-CudaMajor, Convert-ToUnixPath, Get-RepoRoot, Ensure-CudaCcclWheel, Get-OnePathMatch
