<#
.SYNOPSIS
    Test payload: the cuda.compute examples and benchmark smoke tests.
.DESCRIPTION
    Invoked by ci/windows/test_cuda_cccl_examples_python.ps1, which has already put the
    cuda_cccl wheel in wheelhouse/. This normally runs in the minimal container;
    see "Testing Python in a minimal container" in
    docs/infrastructure/ci/references/ci_scripts.rst.

    build_common.psm1 must NOT be imported here: it resolves cl.exe at import
    time, which by design does not exist in the minimal image.
#>
Param(
    [Parameter(Mandatory = $true)]
    [Alias("py-version")]
    [ValidatePattern("^\d+\.\d+t?$")]
    [string]$PyVersion,

    [Alias("ctk-mode")]
    [string]$CtkMode = ""
)

$ErrorActionPreference = "Stop"

Import-Module "$PSScriptRoot/build_common_python.psm1"

Assert-MinimalEnvironment

Install-MsvcRuntime

$python = Get-Python -Version $PyVersion
$cudaMajor = Get-CudaMajor
$ctkFlavor = Get-CtkExtraFlavor $CtkMode
Set-CtkPin $CtkMode

$repoRoot = Get-RepoRoot
$wheelPath = Get-OnePathMatch -Path (Join-Path $repoRoot 'wheelhouse') -Pattern '^cuda_cccl-.*\.whl' -File

# pytest-benchmark is for the host-benchmark smoke test below.
Invoke-Checked { & $python -m pip install -U pip pytest pytest-xdist pytest-benchmark } "Failed to install pytest / pytest-xdist / pytest-benchmark"
# CuPy is required by the cuda.compute examples and is not part of the test extras.
# It resolves curand/cublas/... through cuda-pathfinder at first use and nothing
# else installs them, so request its own `ctk` extra on the pip-toolkit lanes.
# sysctk lanes stay bare: the system toolkit provides them there.
$cupyReq = "cupy-cuda${cudaMajor}x"
if ($ctkFlavor -eq 'cu') { $cupyReq = "${cupyReq}[ctk]" }
Invoke-Checked { & $python -m pip install "${wheelPath}[test-$ctkFlavor$cudaMajor]" $cupyReq } "Failed to install cuda_cccl test extra / cupy"

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
