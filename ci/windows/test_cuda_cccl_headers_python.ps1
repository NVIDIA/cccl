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
& $python -m pip install "${wheelPath}[test-cu$cudaMajor]"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install cuda_cccl test extra"
}

Push-Location (Join-Path $repoRoot "python/cuda_cccl/tests")
try {
    & $python -m pytest -n auto -v headers/
    if ($LASTEXITCODE -ne 0) {
        throw "headers tests failed"
    }
}
finally { Pop-Location }
