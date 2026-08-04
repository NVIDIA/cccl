Param(
    [Parameter(Mandatory = $true)]
    [Alias("py-version")]
    [ValidatePattern("^\d+\.\d+t?$")]
    [string]$PyVersion,

    [Alias("ctk-mode")]
    [string]$CtkMode = ""
)

$ErrorActionPreference = "Stop"

# Import shared helpers
Import-Module "$PSScriptRoot/build_common.psm1"
Import-Module "$PSScriptRoot/build_common_python.psm1"

$python = Get-Python -Version $PyVersion
$cudaMajor = Get-CudaMajor
$ctkFlavor = Get-CtkExtraFlavor $CtkMode

# Pin cuda-toolkit to the container's CTK minor (-ctk-mode latest
# opts out). See build_common_python.psm1.
Set-CtkPin $CtkMode

$repoRoot = Get-RepoRoot

${wheelPath} = Get-CudaCcclWheel
$wheelhouse = Split-Path -Parent $wheelPath
$null = Get-OnePathMatch -Path $wheelhouse -Pattern '^cuda_compute-.*\.whl' -File
$null = Get-OnePathMatch -Path $wheelhouse -Pattern '^cccl_headers-.*\.whl' -File

# pytest-benchmark is for the host-benchmark smoke test below.
Invoke-Checked { & $python -m pip install -U pip pytest pytest-xdist pytest-benchmark } "Failed to install pytest / pytest-xdist / pytest-benchmark"
# CuPy is required by the cuda.compute examples and is not part of the test extras
Invoke-Checked {
    & $python -m pip install --find-links $wheelhouse "${wheelPath}[test-$ctkFlavor$cudaMajor]" "cupy-cuda${cudaMajor}x"
} "Failed to install cuda-cccl test extra / cupy"
Invoke-Checked { & $python -m pip check } "Installed cuda-cccl environment is inconsistent"
Invoke-Checked {
    & $python -c "import importlib.metadata as m; assert len({m.version(name) for name in ('cccl-headers', 'cuda-compute', 'cuda-cccl')}) == 1"
} "cuda-cccl did not install the coordinated compute and headers versions"

Invoke-Checked {
    & $python -m pytest -n 0 -v (Join-Path $repoRoot "python/cuda_cccl/tests")
} "cuda-cccl metadata tests failed"

Push-Location (Join-Path $repoRoot "python/cuda_compute/tests")
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
Push-Location (Join-Path $repoRoot "python/cuda_compute/benchmarks/compute/host")
try {
    Invoke-Checked { & $python -m pytest -v --benchmark-disable . } "host benchmark smoke test failed"
}
finally { Pop-Location }
