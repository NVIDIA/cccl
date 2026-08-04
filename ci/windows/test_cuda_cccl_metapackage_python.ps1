Param(
    [Parameter(Mandatory = $true)]
    [Alias("py-version")]
    [ValidatePattern("^\d+\.\d+t?$")]
    [string]$PyVersion,

    [Alias("ctk-mode")]
    [string]$CtkMode = ""
)

$ErrorActionPreference = "Stop"

Import-Module "$PSScriptRoot/build_common.psm1"
Import-Module "$PSScriptRoot/build_common_python.psm1"

$python = Get-Python -Version $PyVersion
$cudaMajor = Get-CudaMajor
$ctkFlavor = Get-CtkExtraFlavor $CtkMode
Set-CtkPin $CtkMode

$wheelPath = Get-CudaCcclWheel
$wheelhouse = Split-Path -Parent $wheelPath

# Install only the metapackage from a path. Pip must resolve its exact
# cuda-compute dependency from the coordinated local wheelhouse.
Invoke-Checked {
    & $python -m pip install --find-links $wheelhouse "${wheelPath}[minimal-$ctkFlavor$cudaMajor]"
} "Failed to install cuda-cccl metapackage"
Invoke-Checked { & $python -m pip check } "Installed cuda-cccl environment is inconsistent"
Invoke-Checked {
    & $python -c "import importlib.metadata as m, importlib.util as u; import cuda.compute; assert m.version('cuda-cccl') == m.version('cuda-compute'); assert u.find_spec('cuda.cccl') is None"
} "cuda-cccl dependency or module-ownership check failed"
