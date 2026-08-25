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

# A crashed xdist worker takes its stderr with it, so faulthandler output has to
# go to a file rather than the console: tests/conftest.py writes one per worker
# when CCCL_FAULTHANDLER_DIR is set, and the finally block below prints them.
$env:PYTHONFAULTHANDLER = "1"
$env:CCCL_FAULTHANDLER_DIR = Join-Path $repoRoot "faulthandler-dumps"

Push-Location (Join-Path $repoRoot "python/cuda_cccl/tests")
try {
    Invoke-Checked { & $python -m pytest -n 6 -v compute/ -m "not large and not free_threading" } "compute tests (not large) failed"
    Invoke-Checked { & $python -m pytest -n 0 -v compute/ -m "large and not free_threading" } "compute tests (large) failed"
}
finally {
    Show-FaultHandlerDumps $env:CCCL_FAULTHANDLER_DIR
    Pop-Location
}
