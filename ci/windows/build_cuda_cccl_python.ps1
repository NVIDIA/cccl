Param(
    [Parameter(Mandatory = $true)]
    [Alias("py-version")]
    [ValidatePattern("^\d+\.\d+$")]
    [string]$PyVersion,
    [Parameter(Mandatory = $false)]
    [ValidateSet('12', '13')]
    [Alias('ctk', 'cuda')]
    [string]$CudaVersion,
    [Parameter(Mandatory = $false)]
    [switch]$UseNinja,
    [Parameter(Mandatory = $false)]
    [ValidateSet('12', '13')]
    [string]$OnlyCudaMajor,
    [Parameter(Mandatory = $false)]
    [string]$Cuda13Image,
    [Parameter(Mandatory = $false)]
    [switch]$SkipUpload
)

$ErrorActionPreference = "Stop"

# Resolve repo root from this script's location
$RepoRoot = Resolve-Path "$PSScriptRoot/../.."
Write-Host "Repo root: $RepoRoot"

# Import shared helpers
Import-Module "$PSScriptRoot/build_common.psm1" -Verbose

$PythonExe = Get-Python -Version $PyVersion
Write-Host "Using Python: $PythonExe"
& $PythonExe -m pip --version

# Ensure MSVC is available
$clPath = (Get-Command cl).Source
if (-not $clPath) {
    throw "cl.exe not found in PATH. Run from a Developer PowerShell prompt."
}
Write-Host "Found cl.exe at: $clPath"

function Resolve-CudaPathForMajor {
    Param([Parameter(Mandatory = $true)][ValidateSet('12', '13')][string]$Major)
    $candidates = @()
    Get-ChildItem Env: | Where-Object { $_.Name -match "^CUDA_PATH_V${Major}_(\d+)$" } | ForEach-Object {
        $minor = [int]([regex]::Match($_.Name, "^CUDA_PATH_V${Major}_(\d+)$").Groups[1].Value)
        $candidates += [PSCustomObject]@{ Minor = $minor; Path = $_.Value }
    }
    if ($candidates.Count -gt 0) {
        return ($candidates | Sort-Object -Property Minor -Descending | Select-Object -First 1).Path
    }
    if ($env:CUDA_PATH) {
        $maybe = $env:CUDA_PATH
        $nvcc = Join-Path $maybe 'bin/nvcc.exe'
        if (Test-Path $nvcc) {
            $out = & $nvcc --version 2>&1
            $text = ($out -join "`n")
            if ($text -match 'release\s+(\d+)\.') { if ($Matches[1] -eq $Major) { return $maybe } }
        }
    }
    return $null
}

# Determine which CTKs to build
if ($OnlyCudaMajor) { $CudaMajorsToBuild = @($OnlyCudaMajor) } else { $CudaMajorsToBuild = @('12', '13') }
$DoMerge = -not [bool]$OnlyCudaMajor

# Fallback to env-provided nested image if parameter not supplied
if (-not $Cuda13Image -and $env:CUDA13_IMAGE) { $Cuda13Image = $env:CUDA13_IMAGE }

# Base pip/CMake options
$pipBaseConfigArgs = @(
    '-C', 'cmake.define.CMAKE_C_COMPILER=cl.exe',
    '-C', 'cmake.define.CMAKE_CXX_COMPILER=cl.exe'
)

# Prefer Ninja if requested and available
if ($UseNinja) {
    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        $env:CMAKE_GENERATOR = "Ninja"
        Write-Host "CMAKE_GENERATOR=Ninja"
    }
    else {
        Write-Host "Ninja not found; proceeding with default generator" -ForegroundColor Yellow
    }
}

# Ensure wheelhouse exists at repo root for CI artifact collection
$Wheelhouse = Join-Path $RepoRoot "wheelhouse"
New-Item -ItemType Directory -Path $Wheelhouse -Force | Out-Null
${null} = New-Item -ItemType Directory -Path (Join-Path $RepoRoot 'wheelhouse_cu12') -Force
${null} = New-Item -ItemType Directory -Path (Join-Path $RepoRoot 'wheelhouse_cu13') -Force

Push-Location (Join-Path $RepoRoot 'python/cuda_cccl')
try {
    foreach ($major in $CudaMajorsToBuild) {
        if (-not $OnlyCudaMajor -and $major -eq '13' -and $Cuda13Image) {
            if (-not $env:HOST_WORKSPACE) { throw 'HOST_WORKSPACE env var is not set; required for DooD nested docker mounts on Windows.' }

            if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
              throw "docker CLI not found in the devcontainer image (required for DooD)."
            }

            Write-Host "Checking DooD connectivity..."
            docker version | Out-Host

            $hostWorkspace = $env:HOST_WORKSPACE
            Write-Host "Launching nested Docker for CUDA 13 build using image: $Cuda13Image"
            $dockerCmd = @(
                'docker', 'run', '--rm', '-i',
                '--workdir', '/workspace/',
                '--mount', "type=bind,source=$hostWorkspace,target=/workspace/",
                '--env', "py_version=$PyVersion",
                '--env', "GITHUB_ACTIONS=$($env:GITHUB_ACTIONS)",
                '--env', "GITHUB_RUN_ID=$($env:GITHUB_RUN_ID)",
                '--env', "JOB_ID=$($env:JOB_ID)",
                $Cuda13Image,
                'PowerShell.exe', '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', '/workspace/ci/windows/build_cuda_cccl_python.ps1',
                '-py-version', $PyVersion,
                '-OnlyCudaMajor', '13',
                '-UseNinja',
                '-SkipUpload'
            )
            Write-Host ($dockerCmd -join ' ')
            & $dockerCmd
            if ($LASTEXITCODE -ne 0) { throw 'Nested CUDA 13 wheel build failed' }
            continue
        }

        $CudaPathForMajor = Resolve-CudaPathForMajor -Major $major
        if (-not $CudaPathForMajor) { throw "CUDA Toolkit $major not found. Ensure CUDA_PATH_V${major}_* is set or matching toolkit is installed." }
        $NvccForMajor = Join-Path $CudaPathForMajor 'bin/nvcc.exe'
        if (-not (Test-Path $NvccForMajor)) { throw "nvcc not found at $NvccForMajor" }
        $NvccUnix = Convert-ToUnixPath $NvccForMajor
        $CudaPathUnix = Convert-ToUnixPath $CudaPathForMajor
        $pipConfigArgs = $pipBaseConfigArgs + @('-C', "cmake.define.CMAKE_CUDA_COMPILER=$NvccUnix", '-C', "cmake.define.CUDAToolkit_ROOT=$CudaPathUnix")

        $extra = "cu$major"
        $outDir = if ($major -eq '12') { (Join-Path $RepoRoot 'wheelhouse_cu12') } else { (Join-Path $RepoRoot 'wheelhouse_cu13') }
        Write-Host "Building cuda-cccl wheel for CUDA $major at $CudaPathForMajor..."
        $PythonArgs = @('-m', 'pip', 'wheel', '-w', $outDir, ".[${extra}]", '-v') + $pipConfigArgs
        Write-Host ("python " + ($PythonArgs -join ' '))
        & $PythonExe @PythonArgs
        if ($LASTEXITCODE -ne 0) { throw "Wheel build failed for CUDA $major" }
    }
}
finally { Pop-Location }

if ($DoMerge) {
    $Cu12Wheel = Get-OnePathMatch -Path (Join-Path $RepoRoot 'wheelhouse_cu12') -Pattern '^cuda_cccl-.*\.whl' -File
    $Cu13Wheel = Get-OnePathMatch -Path (Join-Path $RepoRoot 'wheelhouse_cu13') -Pattern '^cuda_cccl-.*\.whl' -File
    Write-Host "Found CUDA 12 wheel: $Cu12Wheel"
    Write-Host "Found CUDA 13 wheel: $Cu13Wheel"

    Write-Host 'Merging CUDA wheels...'
    & $PythonExe -m pip install wheel | Write-Host
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install wheel for merging' }

    $WheelhouseMerged = Join-Path $RepoRoot 'wheelhouse_merged'
    ${null} = New-Item -ItemType Directory -Path $WheelhouseMerged -Force
    & $PythonExe (Join-Path $RepoRoot 'python/cuda_cccl/merge_cuda_wheels.py') $Cu12Wheel $Cu13Wheel --output-dir $WheelhouseMerged
    if ($LASTEXITCODE -ne 0) { throw 'Merging wheels failed' }

    Get-ChildItem $Wheelhouse -Filter '*.whl' | ForEach-Object { Remove-Item -Force $_.FullName }
    $MergedWheel = Get-OnePathMatch -Path $WheelhouseMerged -Pattern '^cuda_cccl-.*\.whl' -File
    Move-Item -Force $MergedWheel $Wheelhouse
    Remove-Item $WheelhouseMerged -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $RepoRoot 'wheelhouse_cu12') -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $RepoRoot 'wheelhouse_cu13') -Recurse -Force -ErrorAction SilentlyContinue

    Write-Host 'Final wheels in wheelhouse:'
    Get-ChildItem $Wheelhouse -Filter '*.whl' | ForEach-Object { Write-Host " - $($_.Name)" }
}

# I don't think any of this delvewheel stuff is needed.

#$BuildRoot = Join-Path $RepoRoot "python/cuda_cccl/build"
## There should only be one directory in $BuildRoot, verify this now.
#$BuildDir = Get-OnePathMatch -Path $BuildRoot -Pattern '.*' -Directory
#Write-Host "Using build directory: $BuildDir"
#
#$ReleaseDir = Join-Path $BuildDir "Release"
## Raise an error if $ReleaseDir does not exist.
#if (-not (Test-Path $ReleaseDir)) {
#    throw "Release directory $ReleaseDir does not exist"
#}
#Write-Host "Using release directory: $ReleaseDir"
#
## Find exactly one _bindings_impl-*.pyd under $ReleaseDir.
#$BindingsImplPyd = Get-OnePathMatch -Path $ReleaseDir `
#    -Pattern '^_bindings_impl\..*\.pyd$' -File
#Write-Host "Using bindings implementation: $BindingsImplPyd"
#
## The c.parallel DLL will live in a quirkier location, e.g.:
## build/cp313-cp313-win_amd64/_parent_cccl/c/parallel/cuda_cccl/Release/cccl.c.parallel.dll
#$CParallelDir = Join-Path $BuildDir "_parent_cccl/c/parallel/cuda_cccl/Release"
#$CParallelDll = Get-OnePathMatch -Path $CParallelDir `
#    -Pattern '^cccl\.c\.parallel\.dll$' -File
#Write-Host "Using c.parallel DLL: $CPParallelDll"
#
## Install and run delvewheel to repair Windows wheels (embed needed DLLs)
#Write-Host "Installing delvewheel..."
#$DelveInstallArgs = @('-m', 'pip', 'install', '--disable-pip-version-check', '--quiet', 'delvewheel')
#& $PythonExe @DelveInstallArgs
#if ($LASTEXITCODE -ne 0) { throw "Failed to install delvewheel" }
#
#$WheelhouseFinal = Join-Path $RepoRoot "wheelhouse_final"
#New-Item -ItemType Directory -Path $WheelhouseFinal -Force | Out-Null
#
#Write-Host "Repairing wheel with delvewheel: $($wheel.FullName)"
#$CudaBin = Join-Path $CudaPath 'bin'
#$excludeDlls = @('nvcuda.dll')
#$excludeDlls += (Get-ChildItem $CudaBin -Filter 'nvrtc64_*.dll' -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
#$excludeDlls += (Get-ChildItem $CudaBin -Filter 'nvJitLink64_*.dll' -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
#$excludeJoined = ($excludeDlls | Select-Object -Unique) -join ';'
#Write-Host "Excluding DLLs: $excludeJoined"
#
#$includedDlls = @($CParallelDll, $BindingsImplPyd)
#$includedJoined = ($includedDlls | Select-Object -Unique) -join ';'
#
#$DelveArgs = @(
#    '-m', 'delvewheel', 'repair',
#    '--wheel-dir', $WheelhouseFinal,
#    '--include', $includedJoined,
#    '--exclude', $excludeJoined,
#    $BuiltWheel
#)
#Write-Host ("python " + ($DelveArgs -join ' '))
#& $PythonExe @DelveArgs
#if ($LASTEXITCODE -ne 0) { throw "delvewheel repair failed for $($wheel.Name)" }
#
## Replace original wheels with repaired ones
#Remove-Item (Join-Path $Wheelhouse '*.whl') -ErrorAction SilentlyContinue
#Get-ChildItem $WheelhouseFinal -Filter "cuda_cccl-*.whl" | ForEach-Object { Move-Item -Force $_.FullName $Wheelhouse }
#Remove-Item $WheelhouseFinal -Recurse -Force -ErrorAction SilentlyContinue
#
#Write-Host "Final repaired wheel in ${Wheelhouse}:" -ForegroundColor Green
#Get-ChildItem $Wheelhouse -Filter "cuda_cccl-*.whl" | ForEach-Object { Write-Host " - $($_.Name)" }

if ($env:GITHUB_ACTIONS -and -not $SkipUpload) {
    Push-Location $RepoRoot
    try {
        Write-Host 'GITHUB_ACTIONS detected; uploading wheel artifact'
        $wheelArtifactName = (& bash -lc "ci/util/workflow/get_wheel_artifact_name.sh").Trim()
        if (-not $wheelArtifactName) { throw 'Failed to resolve wheel artifact name' }
        Write-Host "Wheel artifact name: $wheelArtifactName"
        $uploadCmd = "ci/util/artifacts/upload.sh $wheelArtifactName 'wheelhouse/.*'"
        & bash -lc $uploadCmd
        if ($LASTEXITCODE -ne 0) { throw 'Wheel artifact upload failed' }
    }
    finally { Pop-Location }
}
