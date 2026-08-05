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
$wheelhouse = Split-Path -Parent $cudaCcclWheel
$null = Get-OnePathMatch `
    -Path $wheelhouse -Pattern '^cuda_compute-.*\.whl' -File

# Install only the metapackage from an explicit path. Its exact dependency must
# resolve transitively from the sibling wheelhouse; index access remains
# available for the selected CUDA extra's third-party dependencies.
Invoke-Checked {
    & $python -m pip install --find-links $wheelhouse `
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
