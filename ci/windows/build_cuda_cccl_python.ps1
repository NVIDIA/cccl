<#
.SYNOPSIS
    Build Python cuda-cccl wheels for the cuda.compute and cuda.coop packages
    on Windows.

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
    `cu12` and `cu13` wheels into a single cuda-cccl wheel, and uploads that
    via the standard CCCL CI artifact upload mechanisms.

.PARAMETER PyVersion
    **Required.** The Python version to use for building the wheel, expressed
    as `<major>.<minor>` (e.g. `3.11`).

.PARAMETER OnlyCudaMajor
    Optional. Restricts the build to a single CUDA major version (`12` or `13`).
    When set, only that version is built and the *merge* step is skipped.

.PARAMETER Cuda13Image
    Optional. The Docker image name used for a nested build of the CUDA 13
    wheel when the outer container defaults to CUDA 12.9.  The default value
    matches the RAPIDS dev‑container image that contains the required
    toolchain: `rapidsai/devcontainers:25.12-cuda13.0-cl14.44-windows2022`.

.PARAMETER SkipUpload
    When set, prevents the final wheel(s) from being uploaded as a GitHub
    Actions artifact even when the script detects it is running inside an
    Action.

.EXAMPLE
    # Build a single cuda-cccl wheel for Python 3.13 (consisting of both CUDA
    # 12 and 13 versions), and, if in CI, upload the resulting wheel as an
    # artifact.
    .\build_cuda_cccl_python.ps1 -PyVersion 3.11
#>

[CmdletBinding()]
Param(
    [Parameter(Mandatory = $true)]
    [Alias("py-version")]
    [ValidatePattern("^\d+\.\d+$")]
    [string]$PyVersion,

    [Parameter(Mandatory = $false)]
    [ValidateSet('12', '13')]
    [string]$OnlyCudaMajor,

    [Parameter(Mandatory = $false)]
    [string]$Cuda13Image = "rapidsai/devcontainers:25.12-cuda13.0-cl14.44-windows2022",

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
        Run the nested Docker build for CUDA 13 when we are already inside a
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

    # Detect outer‑container resources so we can set sensible limits.
    $os = Get-WmiObject -Class Win32_OperatingSystem
    $totalGB = [math]::Floor($os.TotalVisibleMemorySize / 1MB) # KB -> GB
    $procCount = [Environment]::ProcessorCount

    # Leave a little head‑room so the outer container doesn't starve
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
    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) {
        throw 'Nested CUDA 13 wheel build failed'
    }
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

    # Convert Windows paths to Unix‑style for CMake
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
        '-w', $outDir,
        ".[${extra}]",
        '-v'
    ) + $pipConfigArgs

    Write-Host ("python " + ($pythonArgs -join ' '))
    & $PythonExe @pythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Wheel build failed for CUDA $Major"
    }

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


# Merge the two major‑version wheels (if both were built).  This will fail if
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
    & $PythonExe -m pip install wheel | Write-Host
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to install wheel for merging'
    }

    $WheelhouseMerged = Join-Path $RepoRoot 'wheelhouse_merged'
    ${null} = New-Item -ItemType Directory -Path $WheelhouseMerged -Force

    $mergePy = Join-Path $RepoRoot 'python/cuda_cccl/merge_cuda_wheels.py'
    & $PythonExe $mergePy $Cu12Wheel $Cu13Wheel --output-dir $WheelhouseMerged
    if ($LASTEXITCODE -ne 0) {
        throw 'Merging wheels failed'
    }

    # Clean up the per‑major directories and move the merged wheel into the
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

    Write-Host 'Final wheels in wheelhouse:'
    Get-ChildItem $Wheelhouse -Filter '*.whl' |
    ForEach-Object {
        Write-Host " - $($_.Name)"
    }
}

# If it turns out we need delvewheel, we'd handle it here, after the merging
# of wheels.  The two DLLs that seem like they might be problematic are
# msvc140p.dll, and dbghelp.dll.  The former comes from llvmlite, upon which
# we depend.  Dbghelp.dll ships in C:\Windows\System32, but that will often
# be a much older version compared to the one used by Visual Studio.  We only
# use one symbol from Dbghelp.dll: UnDecorateSymbolName, which is used by
# nvrtc.  If we encounter weird issues with c.parallel jit compilation and
# nvrtc in the wild on Windows, an out-of-date Dbghelp.dll could possibly be
# the culprit.
#
# For now, though, it doesn't appear to be necessary.

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
        & bash -lc $uploadCmd
        if ($LASTEXITCODE -ne 0) {
            throw 'Wheel artifact upload failed'
        }
    }
    finally {
        Pop-Location
    }
}
