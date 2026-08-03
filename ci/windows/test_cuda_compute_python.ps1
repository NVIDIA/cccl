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

$wheelPath = Get-CudaComputeWheel
$wheelhouse = Split-Path -Parent $wheelPath
$headersWheelPath = Get-OnePathMatch -Path $wheelhouse -Pattern '^cccl_headers-.*\.whl' -File

Invoke-Checked { & $python -m pip install -U pip pytest pytest-xdist } "Failed to install pytest / pytest-xdist"
Invoke-Checked {
    & $python -m pip install --find-links $wheelhouse $headersWheelPath "$wheelPath[test-$ctkFlavor$cudaMajor]"
} "Failed to install cuda-compute test extra"
Invoke-Checked { & $python -m pip check } "Installed cuda-compute environment is inconsistent"
Invoke-Checked {
    & $python -c "import importlib.util; assert importlib.util.find_spec('cuda.cccl.headers')"
} "cuda-compute did not install its cccl-headers dependency"
Invoke-Checked {
    & $python -c "import importlib.metadata as m; assert all(d.metadata['Name'] != 'cuda-cccl' for d in m.distributions())"
} "Direct cuda-compute install unexpectedly installed cuda-cccl"

Push-Location (Join-Path $repoRoot "python/cuda_compute/tests")
try {
    Invoke-Checked { & $python -m pytest -n 6 -v compute/ -m "not large and not free_threading" } "compute tests (not large) failed"
    Invoke-Checked { & $python -m pytest -n 0 -v compute/ -m "large and not free_threading" } "compute tests (large) failed"
}
finally { Pop-Location }
