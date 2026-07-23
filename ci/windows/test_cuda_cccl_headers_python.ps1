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

# Pin cuda-toolkit wheels to the container's CTK minor. A lane can set
# CCCL_PYTHON_TEST_LATEST_CTK=1 to skip the pin and test the latest minor.
if ($env:CCCL_PYTHON_TEST_LATEST_CTK -ne "1") {
    $cudaVersion = Get-CudaVersion
    $env:PIP_CONSTRAINT = Join-Path ([System.IO.Path]::GetTempPath()) "ctk-constraint.txt"
    "cuda-toolkit==$cudaVersion.*" | Out-File -FilePath $env:PIP_CONSTRAINT -Encoding ascii
} else {
    # Clear any inherited constraint so this lane truly tests the latest minor.
    Remove-Item Env:\PIP_CONSTRAINT -ErrorAction SilentlyContinue
}

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
