<#
.SYNOPSIS
    Test payload: the cuda.cccl.headers suite.
.DESCRIPTION
    Invoked by ci/windows/test_cuda_cccl_headers_python.ps1, which has already put the
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

# cuda.cccl is pure Python, so no MSVC runtime is needed here.

$python = Get-Python -Version $PyVersion
$cudaMajor = Get-CudaMajor
$ctkFlavor = Get-CtkExtraFlavor $CtkMode
Set-CtkPin $CtkMode

$repoRoot = Get-RepoRoot
$wheelPath = Get-OnePathMatch -Path (Join-Path $repoRoot 'wheelhouse') -Pattern '^cuda_cccl-.*\.whl' -File

Invoke-Checked { & $python -m pip install -U pip pytest pytest-xdist } "Failed to install pytest / pytest-xdist"
Invoke-Checked { & $python -m pip install "${wheelPath}[test-$ctkFlavor$cudaMajor]" } "Failed to install cuda_cccl test extra"

Push-Location (Join-Path $repoRoot "python/cuda_cccl/tests")
try {
    Invoke-Checked { & $python -m pytest -n auto -v headers/ } "headers tests failed"
}
finally { Pop-Location }
