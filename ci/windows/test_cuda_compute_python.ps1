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

$wheelPath = Get-CudaCcclWheel

& $python -m pip install -U pip pytest pytest-xdist
& $python -m pip install "$wheelPath[test-cu$cudaMajor]"

Push-Location (Join-Path $repoRoot "python/cuda_cccl/tests")
try {
    & $python -m pytest -n 6 -v compute/ -m "not large and not free_threading"
    & $python -m pytest -n 0 -v compute/ -m "large and not free_threading"
    if ($PyVersion -eq "3.14t") {
        # The free-threading stress tests create their own worker threads; keep
        # pytest itself serial so the signal is not diluted by xdist process noise.
        & $python -m pytest -n 0 -v compute/ -m "free_threading"
    }
}
finally { Pop-Location }
