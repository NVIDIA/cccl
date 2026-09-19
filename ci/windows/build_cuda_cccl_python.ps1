<#
.SYNOPSIS
    Build Python cuda-cccl wheels on Windows.

.DESCRIPTION
    This script is the Windows analog to the Linux ../build_cuda_cccl_python.sh
    script.  It is responsible for building CUDA 12.x and CUDA 13.x wheels that
    are then merged together into a singular cuda-cccl wheel.

    A single CUDA 12.9 builder image (i.e. Docker devcontainer) is used to
    build each distinct Python/MSVC combo.  Much like the Linux approach, this
    script detects when launched via the outer 12.9 instance, builds a `cu12`
    wheel, then dispatches a inner Docker instance (Docker-out-of-Docker) to
    execute this script with `-OnlyCudaMajor 13 -SkipUpload` parameters, which
    yields a `cu13` build.

    Upon completion of the `cu13` build, the outer 12.9 container merges both
    `cu12` and `cu13` wheels into a single cuda-cccl wheel, repairs it with
    delvewheel so it carries its own MSVC C++ runtime (see
    Repair-CudaCcclWheel), and uploads the result via the standard CCCL CI
    artifact upload mechanisms.

.PARAMETER PyVersion
    **Required.** The Python version to use for building the wheel, expressed
    as `<major>.<minor>` (e.g. `3.11`) or a free-threaded version such as
    `3.14t`.

.PARAMETER OnlyCudaMajor
    Optional. Restricts the build to a single CUDA major version (`12` or `13`).
    When set, only that version is built and the *merge* step is skipped.

.PARAMETER Cuda13Image
    Optional. The Docker image name used for a nested build of the CUDA 13
    wheel when the outer container defaults to CUDA 12.9.  The default value
    matches the RAPIDS dev-container image that contains the required
    toolchain: `rapidsai/devcontainers:26.06-cuda13.0-cl14.44-windows2022`.

.PARAMETER SkipUpload
    When set, prevents the final wheel(s) from being uploaded as a GitHub
    Actions artifact even when the script detects it is running inside an
    Action.

.EXAMPLE
    # Build a single cuda-cccl wheel for Python 3.13 (consisting of both CUDA
    # 12 and 13 versions), and, if in CI, upload the resulting wheel as an
    # artifact.
    .\build_cuda_cccl_python.ps1 -PyVersion 3.11
#>

[CmdletBinding()]
Param(
    [Parameter(Mandatory = $true)]
    [Alias("py-version")]
    [ValidatePattern("^\d+\.\d+t?$")]
    [string]$PyVersion,

    [Parameter(Mandatory = $false)]
    [ValidateSet('12', '13')]
    [string]$OnlyCudaMajor,

    [Parameter(Mandatory = $false)]
    [string]$Cuda13Image = "rapidsai/devcontainers:26.06-cuda13.0-cl14.44-windows2022",

    [Parameter(Mandatory = $false)]
    [switch]$SkipUpload
)

$ErrorActionPreference = "Stop"

# Import shared helpers.
Import-Module "$PSScriptRoot/build_common.psm1"
Import-Module "$PSScriptRoot/build_common_python.psm1" -Force

# Resolve repo root from this script's location.
$RepoRoot = Resolve-Path "$PSScriptRoot/../.."
Write-Host "Repo root: $RepoRoot"

# Get the full path to the python.exe for the version we need.
Write-Host "Looking for Python version $PyVersion..."
$PythonExe = Get-Python -Version $PyVersion
Write-Host "Using Python: $PythonExe"
& $PythonExe -m pip --version

# Ensure MSVC is available.
$clPath = (Get-Command cl).Source
if (-not $clPath) {
    throw "cl.exe not found in PATH. Run from a Developer PowerShell prompt."
}
Write-Host "Found cl.exe at: $clPath"

function Resolve-CudaPathForMajor {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('12', '13')]
        [string]$Major
    )
    $candidates = @()
    Get-ChildItem Env: |
    Where-Object { $_.Name -match "^CUDA_PATH_V${Major}_(\d+)$" } |
    ForEach-Object {
        $minor = [int]([regex]::Match(
                $_.Name,
                "^CUDA_PATH_V${Major}_(\d+)$"
            ).Groups[1].Value)
        $candidates += [PSCustomObject]@{
            Minor = $minor;
            Path  = $_.Value
        }
    }

    if ($candidates.Count -gt 0) {
        return ($candidates | Sort-Object -Property Minor -Descending |
            Select-Object -First 1).Path
    }

    if ($env:CUDA_PATH) {
        $maybe = $env:CUDA_PATH
        $nvcc = Join-Path $maybe 'bin/nvcc.exe'
        if (Test-Path $nvcc) {
            $out = & $nvcc --version 2>&1
            $text = ($out -join "`n")
            if ($text -match 'release\s+(\d+)\.') {
                if ($Matches[1] -eq $Major) {
                    return $maybe
                }
            }
        }
    }

    return $null
}

# If $OnlyCudaMajor is present, it means we're being launched from a
# nested Docker container build (12.x launched a 13.x build via DooD).
if ($OnlyCudaMajor) {
    $CudaMajorsToBuild = @($OnlyCudaMajor)
}
else {
    $CudaMajorsToBuild = @('12', '13')
}
$DoMerge = -not [bool]$OnlyCudaMajor

# Base pip/CMake options
$pipBaseConfigArgs = @(
    '-C', 'cmake.define.CMAKE_C_COMPILER=cl.exe',
    '-C', 'cmake.define.CMAKE_CXX_COMPILER=cl.exe'
)

$env:CMAKE_GENERATOR = "Ninja"

# Ensure wheelhouse directories exist.
$Wheelhouse = Join-Path $RepoRoot "wheelhouse"
New-Item -ItemType Directory -Path $Wheelhouse -Force | Out-Null
${null} = New-Item -ItemType Directory -Path (Join-Path $RepoRoot 'wheelhouse_cu12') -Force
${null} = New-Item -ItemType Directory -Path (Join-Path $RepoRoot 'wheelhouse_cu13') -Force

function Invoke-Cuda13NestedBuild {
    <#
    .SYNOPSIS
        Run the nested Docker build for CUDA 13 when we are already inside a
        CUDA 12 builder image.

    .DESCRIPTION
        This routine launches a Docker devcontainer CUDA 13 build for the given
        Python version by way of Docker-out-of-Docker (DooD) facilities.
    #>
    [CmdletBinding()]
    param (
        [Parameter(Mandatory)] [string] $Cuda13Image,
        [Parameter(Mandatory)] [string] $PyVersion,
        [ValidateNotNullOrEmpty()] [string] $HostWorkspace = $env:HOST_WORKSPACE,
        [ValidateNotNullOrEmpty()] [string] $ContainerWorkspace = $env:CONTAINER_WORKSPACE
    )

    # Validate required environment variables.
    if (-not $HostWorkspace) {
        throw "HOST_WORKSPACE env var is not set; required for DooD " +
        "nested docker mounts on Windows."
    }
    if (-not $ContainerWorkspace) {
        throw "CONTAINER_WORKSPACE env var is not set; required for " +
        "DooD nested docker mounts on Windows."
    }

    # Validate Docker CLI availability.
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "docker CLI not found in the devcontainer image (required for DooD)."
    }

    Write-Host "Checking DooD connectivity..."
    $dockerVersionOutput = & docker version 2>&1
    $dockerExitCode = $LASTEXITCODE
    $dockerVersionOutput | Out-Host
    if ($dockerExitCode -ne 0) {
        throw "DooD connectivity check failed (exit code $dockerExitCode). See Docker output above."
    }
    Write-Host "DooD appears to be working, continuing..."

    # Detect outer-container resources so we can set sensible limits.
    $os = Get-WmiObject -Class Win32_OperatingSystem
    $totalGB = [math]::Floor($os.TotalVisibleMemorySize / 1MB) # KB -> GB
    $procCount = [Environment]::ProcessorCount

    # Leave a little head-room so the outer container doesn't starve
    $memLimitGB = [math]::Max(2, [int]([math]::Floor($totalGB * 0.9)))
    $cpuCount = [math]::Max(2, $procCount)

    Write-Host "Launching nested Docker for CUDA 13 build using image: $Cuda13Image"
    $targetFile = Join-Path $ContainerWorkspace 'ci\windows\build_cuda_cccl_python.ps1'
    $dockerArgs = @(
        'run', '--rm', '-i',
        '--cpu-count', "$cpuCount",
        '--memory', "${memLimitGB}g",
        '--workdir', $ContainerWorkspace,
        '--mount', "type=bind,source=$HostWorkspace,target=$ContainerWorkspace",
        '--env', "py_version=$PyVersion",
        '--env', "GITHUB_ACTIONS=$($env:GITHUB_ACTIONS)",
        '--env', "GITHUB_RUN_ID=$($env:GITHUB_RUN_ID)",
        '--env', "JOB_ID=$($env:JOB_ID)",
        $Cuda13Image,
        'PowerShell.exe', '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', $targetFile,
        '-py-version', $PyVersion,
        '-OnlyCudaMajor', '13',
        '-SkipUpload'
    )

    Write-Host ("About to invoke: docker " + ($dockerArgs -join ' '))
    Invoke-Checked { & docker @dockerArgs } 'Nested CUDA 13 wheel build failed'
}

function Build-CudaCcclWheel {
    <#
    .SYNOPSIS
        Perform the regular wheel build for a given CUDA major version.

    .DESCRIPTION
        This routine is used to build both CUDA 12 and CUDA 13 based wheels,
        and is called from normal "outer" Docker containers, as well as the
        "inner" nested ones.
    #>
    [CmdletBinding()]
    param (
        [Parameter(Mandatory)] [ValidateSet('12', '13')] [string] $Major,
        [Parameter(Mandatory)] [string] $RepoRoot,
        [Parameter(Mandatory)] [string] $PythonExe,
        [Parameter(Mandatory)] [string[]] $PipBaseConfigArgs
    )

    # Resolve CUDA toolkit location for the requested major version.
    $CudaPathForMajor = Resolve-CudaPathForMajor -Major $Major
    if (-not $CudaPathForMajor) {
        throw "CUDA Toolkit $Major not found. Ensure CUDA_PATH_V${Major}_* " +
        "is set or matching toolkit is installed."
    }

    $NvccForMajor = Join-Path $CudaPathForMajor 'bin/nvcc.exe'
    if (-not (Test-Path $NvccForMajor)) {
        throw "nvcc not found at $NvccForMajor"
    }

    # Convert Windows paths to Unix-style for CMake
    $NvccUnix = Convert-ToUnixPath $NvccForMajor
    $CudaUnix = Convert-ToUnixPath $CudaPathForMajor

    # Build the pip configuration arguments that inject the CUDA toolchain.
    $pipConfigArgs = $PipBaseConfigArgs + @(
        '-C', "cmake.define.CMAKE_CUDA_COMPILER=$NvccUnix",
        '-C', "cmake.define.CUDAToolkit_ROOT=$CudaUnix"
    )

    $extra = "cu$Major"
    # Use separate output directories for 12 vs 13.
    $outDir = Join-Path $RepoRoot "wheelhouse_$extra"

    Write-Host "Building cuda-cccl wheel for CUDA $Major at $CudaPathForMajor..."

    # Run pip wheel to build the wheel.
    $pythonArgs = @(
        '-m', 'pip', 'wheel',
        '--no-deps',
        '-w', $outDir,
        '.',
        '-v'
    ) + $pipConfigArgs

    Write-Host ("python " + ($pythonArgs -join ' '))
    Invoke-Checked { & $PythonExe @pythonArgs } "Wheel build failed for CUDA $Major"

    # Normalise the wheel filename (append .cu12/.cu13) and prune duplicates.
    $builtWheel = Get-OnePathMatch -Path $outDir `
        -Pattern '^cuda_cccl-.*\.whl' `
        -File
    if (-not $builtWheel) {
        throw "Failed to locate built wheel in $outDir for CUDA $Major"
    }

    $builtName = [System.IO.Path]::GetFileName($builtWheel)
    if ($builtName -notmatch ".cu$Major\.whl$") {
        $newName = ([System.IO.Path]::GetFileNameWithoutExtension($builtName)) `
            + ".cu$Major.whl"
        Write-Host "Renaming wheel to: $newName"
        Rename-Item -Path $builtWheel -NewName $newName -Force
    }

    # Remove any stray wheels that lack the .cuXX suffix.
    Get-ChildItem -Path $outDir -Filter 'cuda_cccl-*.whl' |
    Where-Object { $_.Name -notmatch "\.cu$Major\.whl$" } |
    ForEach-Object {
        Write-Host "Removing duplicate wheel: $($_.FullName)"
        Remove-Item -Force $_.FullName
    }
}

function Repair-CudaCcclWheel {
    <#
    .SYNOPSIS
        Bundle the MSVC C++ runtime into the merged wheel. This is the Windows
        counterpart of the auditwheel repair in ../build_cuda_cccl_python.sh.

    .DESCRIPTION
        cccl.c.parallel.dll links msvcp140.dll dynamically, and Python ships
        only vcruntime140*.dll, so an unrepaired wheel takes whatever
        msvcp140.dll the user's machine has.
    #>
    [CmdletBinding()]
    param (
        [Parameter(Mandatory)] [string] $Wheelhouse,
        [Parameter(Mandatory)] [string] $RepoRoot,
        [Parameter(Mandatory)] [string] $PythonExe
    )

    $wheel = Get-OnePathMatch -Path $Wheelhouse -Pattern '^cuda_cccl-.*\.whl' -File

    Invoke-Checked { & $PythonExe -m pip install 'delvewheel>=1.13.1' | Write-Host } 'Failed to install delvewheel'

    # Vendor the System32 msvcp140.dll, the copy the Visual Studio installer put
    # there alongside this image's toolset. Passed via --add-path so delvewheel
    # takes this copy rather than the first one it meets on PATH.
    $systemMsvcp = Join-Path $env:SystemRoot 'System32\msvcp140.dll'
    Write-Host "System msvcp140.dll is $((Get-Item $systemMsvcp).VersionInfo.FileVersion)"

    $repairedDir = Join-Path $RepoRoot 'wheelhouse_repaired'
    ${null} = New-Item -ItemType Directory -Path $repairedDir -Force

    # cccl.c.parallel*.dll / libnvcc.dll: already in the wheel, loaded via
    # os.add_dll_directory in _bindings.py; unexcluded, delvewheel looks for
    # them on PATH and fails.
    # --analyze-existing: also read the import tables of DLLs already in the
    # wheel. Without it only the .pyd files are analysed, the repair succeeds,
    # and msvcp140.dll is still imported from the system.
    # --namespace-pkg cuda: never create cuda/__init__.py (shared namespace
    # with cuda-bindings and cuda-core); the loader patch lands in
    # cuda/compute/__init__.py instead.
    $delvewheelArgs = @(
        '-m', 'delvewheel', 'repair', $wheel,
        '-w', $repairedDir,
        '--analyze-existing',
        '--namespace-pkg', 'cuda',
        '--exclude', ('cccl.c.parallel*.dll;libnvcc.dll;' +
            'nvrtc64_*.dll;nvrtc-builtins64_*.dll;nvJitLink_*.dll;nvfatbin*.dll;' +
            'cudart64_*.dll;nvcuda.dll;dbghelp.dll'),
        '--add-path', (Split-Path -Parent $systemMsvcp)
    )
    Write-Host ("python " + ($delvewheelArgs -join ' '))
    Invoke-Checked { & $PythonExe @delvewheelArgs } 'delvewheel repair failed'

    $repaired = Get-OnePathMatch -Path $repairedDir -Pattern '^cuda_cccl-.*\.whl' -File
    Remove-Item -Force $wheel
    Move-Item -Force $repaired $Wheelhouse
    Remove-Item $repairedDir -Recurse -Force -ErrorAction SilentlyContinue
}

# Main build entry code.
Push-Location (Join-Path $RepoRoot 'python/cuda_cccl')
try {
    foreach ($major in $CudaMajorsToBuild) {

        # Nested Docker build for CUDA 13 for when we are currently inside a
        # CUDA 12 image.
        if (-not $OnlyCudaMajor -and $major -eq '13' -and $Cuda13Image) {
            Invoke-Cuda13NestedBuild `
                -Cuda13Image $Cuda13Image `
                -PyVersion $PyVersion

            continue
        }

        # Perform a normal build for the current major version.  This may
        # be invoked from either an "outer" or inner "nested" image.
        Build-CudaCcclWheel `
            -Major $major `
            -RepoRoot $RepoRoot `
            -PythonExe $PythonExe `
            -PipBaseConfigArgs $pipBaseConfigArgs
    }
}
finally {
    Pop-Location
}


# Merge the two major-version wheels (if both were built).  This will fail if
# either wheel can't be found.  This only runs on the outer (non-nested)
# container image.
if ($DoMerge) {

    $Cu12Wheel = Get-OnePathMatch `
        -Path (Join-Path $RepoRoot 'wheelhouse_cu12') `
        -Pattern '^cuda_cccl-.*\.cu12\.whl' `
        -File

    $Cu13Wheel = Get-OnePathMatch `
        -Path (Join-Path $RepoRoot 'wheelhouse_cu13') `
        -Pattern '^cuda_cccl-.*\.cu13\.whl' `
        -File

    Write-Host "Found CUDA 12 wheel: $Cu12Wheel"
    Write-Host "Found CUDA 13 wheel: $Cu13Wheel"

    Write-Host 'Merging CUDA wheels...'
    Invoke-Checked { & $PythonExe -m pip install wheel | Write-Host } 'Failed to install wheel for merging'

    $WheelhouseMerged = Join-Path $RepoRoot 'wheelhouse_merged'
    ${null} = New-Item -ItemType Directory -Path $WheelhouseMerged -Force

    $mergePy = Join-Path $RepoRoot 'python/cuda_cccl/merge_cuda_wheels.py'
    Invoke-Checked { & $PythonExe $mergePy $Cu12Wheel $Cu13Wheel --output-dir $WheelhouseMerged } 'Merging wheels failed'

    # Clean up the per-major directories and move the merged wheel into the
    # final location.
    Get-ChildItem $Wheelhouse -Filter '*.whl' |
    ForEach-Object {
        Remove-Item -Force $_.FullName
    }
    $MergedWheel = Get-OnePathMatch `
        -Path $WheelhouseMerged `
        -Pattern '^cuda_cccl-.*\.whl' `
        -File
    Move-Item -Force $MergedWheel $Wheelhouse

    Remove-Item $WheelhouseMerged -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $RepoRoot 'wheelhouse_cu12') `
        -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $RepoRoot 'wheelhouse_cu13') `
        -Recurse -Force -ErrorAction SilentlyContinue

    Repair-CudaCcclWheel `
        -Wheelhouse $Wheelhouse `
        -RepoRoot $RepoRoot `
        -PythonExe $PythonExe

    Write-Host 'Final wheels in wheelhouse:'
    Get-ChildItem $Wheelhouse -Filter '*.whl' |
    ForEach-Object {
        Write-Host " - $($_.Name)"
    }
}

# dbghelp.dll is deliberately not bundled by the repair above and comes from
# C:\Windows\System32, whose copy is often much older than the one Visual
# Studio ships. We use a single symbol from it, UnDecorateSymbolName (via
# nvrtcGetTypeName). If c.parallel JIT compilation ever misbehaves in the wild
# on Windows, an out-of-date dbghelp.dll is a possible culprit.

# Optionally upload the wheel artifact.
if ($env:GITHUB_ACTIONS -and -not $SkipUpload) {
    Push-Location $RepoRoot
    try {
        Write-Host 'GITHUB_ACTIONS detected; uploading wheel artifact'
        $wheelArtifactName = (& bash -lc "ci/util/workflow/get_wheel_artifact_name.sh").Trim()
        if (-not $wheelArtifactName) {
            throw 'Failed to resolve wheel artifact name'
        }
        Write-Host "Wheel artifact name: $wheelArtifactName"

        $uploadCmd = "ci/util/artifacts/upload.sh $wheelArtifactName 'wheelhouse/.*'"
        Invoke-Checked { & bash -lc $uploadCmd } 'Wheel artifact upload failed'
    }
    finally {
        Pop-Location
    }
}
