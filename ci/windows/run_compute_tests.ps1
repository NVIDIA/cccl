<#
.SYNOPSIS
    Test payload: the cuda.compute pytest suite.
.DESCRIPTION
    Invoked by ci/windows/test_cuda_compute_python.ps1, which has already put the
    cuda_cccl wheel in wheelhouse/.

    This may run inside the minimal container (see run_in_minimal_container.ps1),
    so everything here must work with nothing but Python and the wheel's declared
    pip dependencies. Note in particular that build_common.psm1 must NOT be
    imported: it resolves cl.exe at import time, which by design does not exist
    in the minimal image.
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

function Invoke-Step {
    param([scriptblock]$Command, [string]$ErrorMessage)
    & $Command
    if ($LASTEXITCODE -ne 0) { throw $ErrorMessage }
}

$python = Get-Python -Version $PyVersion
$cudaMajor = Get-CudaMajor
$ctkFlavor = Get-CtkExtraFlavor $CtkMode

# Pin cuda-toolkit to the lane's CTK minor (-ctk-mode latest opts out).
Set-CtkPin $CtkMode

$repoRoot = Get-RepoRoot
$wheelPath = Get-OnePathMatch -Path (Join-Path $repoRoot 'wheelhouse') -Pattern '^cuda_cccl-.*\.whl' -File

Invoke-Step { & $python -m pip install -U pip pytest pytest-xdist } "Failed to install pytest / pytest-xdist"
Invoke-Step { & $python -m pip install "$wheelPath[test-$ctkFlavor$cudaMajor]" } "Failed to install cuda_cccl test extra"

Push-Location (Join-Path $repoRoot "python/cuda_cccl/tests")
try {
    Invoke-Step { & $python -m pytest -n 6 -v compute/ -m "not large and not free_threading" } "compute tests (not large) failed"
    Invoke-Step { & $python -m pytest -n 0 -v compute/ -m "large and not free_threading" } "compute tests (large) failed"

    # The bfloat16 tests require ml_dtypes (the NumPy bfloat16 extension dtype),
    # which is deliberately not part of the test extras so that the sweeps above
    # run in an environment matching a user's default install (where the bfloat16
    # tests skip themselves). Install it last and run those tests explicitly.
    Invoke-Step { & $python -m pip install ml_dtypes } "Failed to install ml_dtypes"
    Invoke-Step { & $python -m pytest -n 6 -v compute/test_bfloat16.py } "bfloat16 tests failed"
}
finally { Pop-Location }
