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

$cudaCcclWheel = Get-CudaCcclWheel
$cudaComputeWheel = Get-CudaComputeWheel
$wheelhouse = Split-Path -Parent $cudaCcclWheel

# Pass both coordinated wheels as explicit local requirements so an
# equal-version index candidate cannot replace cuda-compute. The metapackage
# extra still supplies compute's selected dependencies, with index access left
# available for those third-party packages.
Invoke-Checked {
    & $python -m pip install --find-links $wheelhouse `
        $cudaComputeWheel `
        "${cudaCcclWheel}[minimal-$ctkFlavor$cudaMajor]"
} "Failed to install cuda-cccl metapackage"
Invoke-Checked { & $python -m pip check } "Installed cuda-cccl environment is inconsistent"
$verifyMetapackage = @'
import importlib.metadata
import importlib.util

import cuda.compute

metapackage_version = importlib.metadata.version("cuda-cccl")
compute_version = importlib.metadata.version("cuda-compute")
if metapackage_version != compute_version:
    raise RuntimeError(
        f"cuda-cccl {metapackage_version} != cuda-compute {compute_version}"
    )
if importlib.util.find_spec("cuda.cccl") is not None:
    raise RuntimeError("cuda-cccl must not install a cuda.cccl module")
'@
Invoke-Checked {
    $verifyMetapackage | & $python -
} "cuda-cccl dependency or module-ownership check failed"
