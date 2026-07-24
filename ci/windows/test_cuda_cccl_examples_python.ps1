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
# pytest-benchmark is for the host-benchmark smoke test below.
& $python -m pip install -U pip pytest pytest-xdist pytest-benchmark
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install pytest / pytest-xdist / pytest-benchmark"
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

# Smoke-test the host-overhead benchmark harness: run every benchmark case
# exactly once (pass/fail only, no timing) so harness rot fails CI here instead
# of silently surviving until someone runs the perf suite. --benchmark-disable
# makes pytest-benchmark invoke each benchmarked callable a single time. This
# lane already installs cupy + numba (for the examples), which the benchmark
# suite also needs, so only pytest-benchmark is added above.
Push-Location (Join-Path $repoRoot "python/cuda_cccl/benchmarks/compute/host")
try {
    & $python -m pytest -v --benchmark-disable .
    if ($LASTEXITCODE -ne 0) {
        throw "host benchmark smoke test failed"
    }
}
finally { Pop-Location }
