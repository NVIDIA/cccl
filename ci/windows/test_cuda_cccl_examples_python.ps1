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

# Pin cuda-toolkit to the container's CTK minor (CCCL_PYTHON_TEST_LATEST_CTK=1
# opts out). See build_common_python.psm1.
Set-CtkPin

$repoRoot = Get-RepoRoot

${wheelPath} = Get-CudaCcclWheel

# pytest-benchmark is for the host-benchmark smoke test below.
Invoke-Checked { & $python -m pip install -U pip pytest pytest-xdist pytest-benchmark } "Failed to install pytest / pytest-xdist / pytest-benchmark"
# CuPy is required by the cuda.compute examples and is not part of the test extras
Invoke-Checked { & $python -m pip install "${wheelPath}[test-cu$cudaMajor]" "cupy-cuda${cudaMajor}x" } "Failed to install cuda_cccl test extra / cupy"

Push-Location (Join-Path $repoRoot "python/cuda_cccl/tests")
try {
    Invoke-Checked { & $python -m pytest -n 6 test_examples.py } "examples tests failed"
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
    Invoke-Checked { & $python -m pytest -v --benchmark-disable . } "host benchmark smoke test failed"
}
finally { Pop-Location }
