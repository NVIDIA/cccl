Param(
    [Parameter(Mandatory = $true)]
    [Alias("py-version")]
    [ValidatePattern("^\d+\.\d+t?$")]
    [string]$PyVersion
)

$ErrorActionPreference = "Stop"

# Import shared helpers
Import-Module "$PSScriptRoot/build_common.psm1"
Import-Module "$PSScriptRoot/build_common_python.psm1"

$python = Get-Python -Version $PyVersion
$cudaMajor = Get-CudaMajor

$repoRoot = Get-RepoRoot

${wheelPath} = Get-CudaCcclWheel
& $python -m pip install -U pip pytest pytest-xdist
# CuPy is required by the cuda.compute examples and is not part of the test extras
& $python -m pip install "${wheelPath}[test-cu$cudaMajor]" "cupy-cuda${cudaMajor}x"

Push-Location (Join-Path $repoRoot "python/cuda_cccl/tests")
try {
    & $python -m pytest -n 6 test_examples.py
}
finally { Pop-Location }
