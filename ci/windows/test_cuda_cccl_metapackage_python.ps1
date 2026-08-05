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
$cudaComputeWheel = Get-OnePathMatch `
    -Path $wheelhouse -Pattern '^cuda_compute-.*\.whl' -File

# Install both coordinated wheels from explicit paths. Index access remains
# available for the selected CUDA extra's third-party dependencies.
Invoke-Checked {
    & $python -m pip install --find-links $wheelhouse `
        "${cudaCcclWheel}[minimal-$ctkFlavor$cudaMajor]" `
        $cudaComputeWheel
} "Failed to install cuda-cccl metapackage"
Invoke-Checked { & $python -m pip check } "Installed cuda-cccl environment is inconsistent"
Invoke-Checked {
    & $python -c "import importlib.metadata as m, importlib.util as u; import cuda.compute; assert m.version('cuda-cccl') == m.version('cuda-compute'); assert u.find_spec('cuda.cccl') is None"
} "cuda-cccl dependency or module-ownership check failed"
