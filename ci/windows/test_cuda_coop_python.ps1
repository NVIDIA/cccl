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

${wheelPath} = Get-CudaCcclWheel
& $python -m pip install -U pip pytest pytest-xdist
& $python -m pip install "${wheelPath}[test-cu$cudaMajor]"

Push-Location (Join-Path (Get-RepoRoot) "python/cuda_cccl/tests")
try {
    & $python -m pytest -n auto -v coop/_experimental/
}
finally { Pop-Location }
