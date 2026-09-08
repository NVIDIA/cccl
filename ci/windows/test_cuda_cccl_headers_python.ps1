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

${wheelPath} = Get-CudaCcclWheel

Invoke-Checked { & $python -m pip install -U pip pytest pytest-xdist } "Failed to install pytest / pytest-xdist"
Invoke-Checked { & $python -m pip install "${wheelPath}[test-$ctkFlavor$cudaMajor]" } "Failed to install cuda_cccl test extra"

Push-Location (Join-Path $repoRoot "python/cuda_cccl/tests")
try {
    Invoke-Checked { & $python -m pytest -n auto -v headers/ } "headers tests failed"
}
finally { Pop-Location }
