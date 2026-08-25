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

# A process killed by a structured exception exits with an NTSTATUS code, which
# PowerShell surfaces as a large negative number. These are the codes a native
# crash produces; naming them is the difference between "exit code -1073740791"
# and knowing the process died of heap corruption.
$script:NtStatusNames = @{
    "C0000005" = "STATUS_ACCESS_VIOLATION"
    "C000001D" = "STATUS_ILLEGAL_INSTRUCTION"
    "C0000094" = "STATUS_INTEGER_DIVIDE_BY_ZERO"
    "C00000FD" = "STATUS_STACK_OVERFLOW"
    "C0000374" = "STATUS_HEAP_CORRUPTION"
    "C0000409" = "STATUS_STACK_BUFFER_OVERRUN"
    "C0000420" = "STATUS_ASSERTION_FAILURE"
}

function Format-ExitCode {
    <#
    .SYNOPSIS
        Renders an exit code, naming it when it is a Windows NTSTATUS crash code.
    .EXAMPLE
        Format-ExitCode -1073740791   # -> "-1073740791 (0xC0000409 STATUS_STACK_BUFFER_OVERRUN)"
    #>
    # [int64] so both the negative and the raw unsigned spelling of a code bind.
    param([Parameter(Mandatory, Position = 0)][int64]$Code)

    $unsigned = $Code -band 0xFFFFFFFFL
    $hex = "{0:X8}" -f $unsigned
    $name = $script:NtStatusNames[$hex]

    if ($name) { return "$Code (0x$hex $name)" }
    # Suffix the literal: PowerShell parses 0xC0000000 as a negative [int], which
    # would make this comparison true for every ordinary exit code.
    if ($unsigned -ge 0xC0000000L) { return "$Code (0x$hex)" }
    return "$Code"
}

function Show-FaultHandlerDumps {
    <#
    .SYNOPSIS
        Prints per-worker faulthandler dumps written by tests/conftest.py.
    .DESCRIPTION
        A crashed pytest-xdist worker takes its stderr with it, so the native
        traceback only survives in these files. Call this after a test run --
        including when it failed, which is when the dumps actually exist.
    #>
    param([Parameter(Position = 0)][string]$Path)

    if (-not $Path -or -not (Test-Path $Path)) {
        Write-Host "faulthandler: no dump directory at '$Path' -- the conftest hook never ran."
        return
    }

    # Report the redirect markers and every dump file, empty ones included. A
    # missing traceback is otherwise ambiguous between "the hook never ran" and
    # "the hook ran but the crash bypassed the handler".
    $markers = @(Get-ChildItem -Path $Path -Filter "fhinit-*.log" -ErrorAction SilentlyContinue)
    $dumps = @(Get-ChildItem -Path $Path -Filter "faulthandler-*.log" -ErrorAction SilentlyContinue)

    Write-Host "faulthandler: $($markers.Count) worker(s) installed the redirect, $($dumps.Count) dump file(s) present."
    foreach ($marker in $markers) {
        Write-Host "  init: $((Get-Content $marker.FullName -Raw).Trim())"
    }
    foreach ($dump in $dumps) {
        Write-Host ("  dump: {0} ({1} bytes)" -f $dump.Name, $dump.Length)
    }

    $captured = @($dumps | Where-Object { $_.Length -gt 0 })
    if ($captured.Count -eq 0) {
        Write-Host "faulthandler: no traceback captured. If a worker died, the crash bypassed the handler (e.g. Windows __fastfail)."
        return
    }

    foreach ($dump in $captured) {
        Write-Host "::group::Native traceback from $($dump.Name)"
        Get-Content $dump.FullName | ForEach-Object { Write-Host $_ }
        Write-Host "::endgroup::"
    }
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
        throw "$ErrorMessage (exit code $(Format-ExitCode $LASTEXITCODE))"
    }
}

Export-ModuleMember -Function configure_preset, build_preset, test_preset, configure_and_build_preset, Invoke-Checked, Format-ExitCode, Show-FaultHandlerDumps
Export-ModuleMember -Variable BUILD_DIR, CL_VERSION
