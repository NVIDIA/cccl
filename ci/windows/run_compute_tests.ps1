<#
.SYNOPSIS
    Test payload: the cuda.compute pytest suite.
.DESCRIPTION
    Invoked by ci/windows/test_cuda_compute_python.ps1, which has already put the
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

# Pin cuda-toolkit to the lane's CTK minor (-ctk-mode latest opts out).
Set-CtkPin $CtkMode

$repoRoot = Get-RepoRoot
$wheelPath = Get-OnePathMatch -Path (Join-Path $repoRoot 'wheelhouse') -Pattern '^cuda_cccl-.*\.whl' -File

Invoke-Checked { & $python -m pip install -U pip pytest pytest-xdist } "Failed to install pytest / pytest-xdist"
Invoke-Checked { & $python -m pip install "$wheelPath[test-$ctkFlavor$cudaMajor]" } "Failed to install cuda_cccl test extra"

Push-Location (Join-Path $repoRoot "python/cuda_cccl/tests")
try {
    Invoke-Checked { & $python -m pytest -n 6 -v compute/ -m "not large and not free_threading" } "compute tests (not large) failed"
    Invoke-Checked { & $python -m pytest -n 0 -v compute/ -m "large and not free_threading" } "compute tests (large) failed"

    # Mirrors the free-threading section of ci/util/python/run_compute_tests.sh;
    # see there for the full rationale. The suites carry the free_threading marker
    # so the runs above exclude them, and on a GIL interpreter they would only
    # skip themselves -- run them when the interpreter is genuinely free-threaded.
    if (Test-FreeThreadedPython $python) {
        Invoke-Checked { & $python -c "import sys; assert not sys._is_gil_enabled(), 'GIL is enabled; free-threading tests have no signal'" } "interpreter is not GIL-free; free-threading tests have no signal"

        # -n 0: these spawn and barrier-synchronize their own worker threads, so
        # pytest itself must stay in a single process. Selected by marker rather
        # than by filename so a future suite is picked up automatically.
        Invoke-Checked { & $python -m pytest -n 0 -v -m free_threading compute/ } "free-threading tests failed"

        # Broad thread-safety sweep. "not large" avoids re-creating the memory
        # pressure the -n 6 / -n 0 split above exists to prevent;
        # "not free_threading" because those just ran with their own workers.
        Invoke-Checked { & $python -m pip install pytest-run-parallel } "Failed to install pytest-run-parallel"
        Invoke-Checked { & $python -m pytest -n 0 -v --parallel-threads=2 compute/ -m "not large and not free_threading" } "parallel-threads sweep failed"
    }

    # ml_dtypes (the NumPy bfloat16 extension dtype) is deliberately not in the
    # test extras, so the sweeps above match a user's default install, where the
    # bfloat16 tests skip themselves. Install it last and run them explicitly.
    Invoke-Checked { & $python -m pip install ml_dtypes } "Failed to install ml_dtypes"
    Invoke-Checked { & $python -m pytest -n 6 -v compute/test_bfloat16.py } "bfloat16 tests failed"
}
finally { Pop-Location }
