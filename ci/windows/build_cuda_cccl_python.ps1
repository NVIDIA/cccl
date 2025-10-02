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
Import-Module "$PSScriptRoot/build_common.psm1"

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
    $nvccOut = & $Nvcc --version
    if ($nvccOut -Match "release (\d+)\.") { $CudaVersion = $Matches[1] }
}
if ($CudaVersion -NotIn @('12', '13')) { # codespell: ignore
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
    $args = @('-m', 'pip', 'wheel', '-w', $Wheelhouse, ".[${extra}]", '-v') + $pipConfigArgs
    Write-Host ("python " + ($args -join ' '))
    & $PythonExe @args
    if ($LASTEXITCODE -ne 0) { throw "Wheel build failed" }
}
finally {
    Pop-Location
}

Write-Host "Built wheels in ${Wheelhouse}:" -ForegroundColor Green
Get-ChildItem $Wheelhouse -Filter "cuda_cccl-*.whl" | ForEach-Object { Write-Host " - $($_.Name)" }

if ($env:GITHUB_ACTIONS) {
    Write-Host "GITHUB_ACTIONS detected; ensure workflow picks up artifacts from wheelhouse/"
}
