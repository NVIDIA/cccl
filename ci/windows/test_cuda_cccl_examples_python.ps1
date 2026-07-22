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

# Native commands (python.exe / pip / pytest) only set $LASTEXITCODE on failure;
# $ErrorActionPreference = "Stop" does not make them throw, so a non-zero exit
# must be checked explicitly or a failed pip/pytest is masked by a later
# successful command and the job passes green.
& $python -m pip install -U pip pytest pytest-xdist
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install pytest / pytest-xdist"
}
# CuPy is required by the cuda.compute examples and is not part of the test extras
& $python -m pip install "${wheelPath}[test-cu$cudaMajor]" "cupy-cuda${cudaMajor}x"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install cuda_cccl test extra / cupy"
}

Push-Location (Join-Path $repoRoot "python/cuda_cccl/tests")
try {
    & $python -m pytest -n 6 test_examples.py
    if ($LASTEXITCODE -ne 0) {
        throw "examples tests failed"
    }
}
finally { Pop-Location }
