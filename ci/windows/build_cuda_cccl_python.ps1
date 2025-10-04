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
    [switch]$UseNinja
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

# Locate CUDA toolkit
if (-not $env:CUDA_PATH) {
    throw "CUDA_PATH is not set. Please install CUDA Toolkit and ensure CUDA_PATH is in the environment."
}
$CudaPath = $env:CUDA_PATH
$Nvcc = Join-Path $CudaPath "bin/nvcc.exe"
if (-not (Test-Path $Nvcc)) {
    throw "nvcc not found at $Nvcc"
}

# Determine CUDA major version if not provided
if (-not $CudaVersion) {
    $CudaVersion = Get-CudaMajor
}
if ($CudaVersion -NotIn @('12', '13')) {
    # codespell: ignore
    throw "Unsupported/unknown CUDA version '$CudaVersion'. Supported: 12 or 13."
}
Write-Host "Using CUDA Toolkit: $CudaVersion at $CudaPath"

# Prepare build options for scikit-build-core via pip -C settings
# Convert Windows paths to forward slashes for CMake friendliness
$NvccUnix = Convert-ToUnixPath $Nvcc
$CudaPathUnix = Convert-ToUnixPath $CudaPath

$pipConfigArgs = @(
    '-C', "cmake.define.CMAKE_C_COMPILER=cl.exe",
    '-C', "cmake.define.CMAKE_CXX_COMPILER=cl.exe",
    '-C', "cmake.define.CMAKE_CUDA_COMPILER=$NvccUnix",
    '-C', "cmake.define.CUDAToolkit_ROOT=$CudaPathUnix"
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

# Choose the extras specifier to pull the correct dependency set
$extra = "cu$CudaVersion"

# Ensure wheelhouse exists at repo root for CI artifact collection
$Wheelhouse = Join-Path $RepoRoot "wheelhouse"
New-Item -ItemType Directory -Path $Wheelhouse -Force | Out-Null

Push-Location (Join-Path $RepoRoot "python/cuda_cccl")
try {
    Write-Host "Building cuda-cccl wheel for CUDA $CudaVersion..."
    $PythonArgs = @('-m', 'pip', 'wheel', '-w', $Wheelhouse, ".[${extra}]", '-v') + $pipConfigArgs
    Write-Host ("python " + ($PythonArgs -join ' '))
    & $PythonExe @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Wheel build failed" }
}
finally {
    Pop-Location
}

$BuiltWheel = Get-OnePathMatch -Path $Wheelhouse -Pattern '^cuda_cccl-.*\.whl' -File
Write-Host "Built wheel: $BuiltWheel"

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

if ($env:GITHUB_ACTIONS) {
    Write-Host "GITHUB_ACTIONS detected; ensure workflow picks up artifacts from wheelhouse/"
}
