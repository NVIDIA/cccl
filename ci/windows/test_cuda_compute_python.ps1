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

$wheelPath = Get-CudaCcclWheel

Invoke-Checked { & $python -m pip install -U pip pytest pytest-xdist } "Failed to install pytest / pytest-xdist"
Invoke-Checked { & $python -m pip install "$wheelPath[test-$ctkFlavor$cudaMajor]" } "Failed to install cuda_cccl test extra"

Push-Location (Join-Path $repoRoot "python/cuda_cccl/tests")
try {
    Invoke-Checked { & $python -m pytest -n 6 -v compute/ -m "not large and not free_threading" } "compute tests (not large) failed"
    Invoke-Checked { & $python -m pytest -n 0 -v compute/ -m "large and not free_threading" } "compute tests (large) failed"

    # The bfloat16 tests require ml_dtypes (the NumPy bfloat16 extension dtype),
    # which is deliberately not part of the test extras so that the sweeps above
    # run in an environment matching a user's default install (where the bfloat16
    # tests skip themselves). Install it last and run those tests explicitly.
    Invoke-Checked { & $python -m pip install ml_dtypes } "Failed to install ml_dtypes"
    Invoke-Checked { & $python -m pytest -n 6 -v compute/test_bfloat16.py } "bfloat16 tests failed"
}
finally { Pop-Location }
