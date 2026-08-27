<#
.SYNOPSIS
    Test payload: the cuda.compute pytest suite.
.DESCRIPTION
    Invoked by ci/windows/test_cuda_compute_python.ps1, which has already put the
    cuda_cccl wheel in wheelhouse/. This normally runs in the minimal container;
    see "Testing Python in a minimal container" in
    docs/infrastructure/ci/references/ci_overview.rst.

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

# Server Core ships no MSVC runtime, and every C++ Python extension used here
# links against it -- numba's _typeconv and cccl.c.parallel.dll among them. It
# is a Windows prerequisite rather than a packaging gap, so install it instead
# of expecting a wheel to carry it. vcruntime140*.dll ship next to some
# interpreters, which makes msvcp140.dll the reliable probe.
$msvcp = Join-Path $env:SystemRoot 'System32\msvcp140.dll'
if (-not (Test-Path $msvcp)) {
    Write-Host "Installing the MSVC runtime redistributable..."
    $installer = Join-Path $env:TEMP 'vc_redist.x64.exe'
    Invoke-WebRequest -UseBasicParsing `
        -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $installer
    Start-Process -Wait -FilePath $installer -ArgumentList '/quiet', '/norestart'
    if (-not (Test-Path $msvcp)) {
        throw "vc_redist.x64.exe did not install msvcp140.dll."
    }
}

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

    # ml_dtypes (the NumPy bfloat16 extension dtype) is deliberately not in the
    # test extras, so the sweeps above match a user's default install, where the
    # bfloat16 tests skip themselves. Install it last and run them explicitly.
    Invoke-Checked { & $python -m pip install ml_dtypes } "Failed to install ml_dtypes"
    Invoke-Checked { & $python -m pytest -n 6 -v compute/test_bfloat16.py } "bfloat16 tests failed"
}
finally { Pop-Location }
