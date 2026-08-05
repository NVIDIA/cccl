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

$wheelhouse = Get-CcclPythonWheelhouse
$cudaCcclWheel = Get-OnePathMatch `
    -Path $wheelhouse -Pattern '^cuda_cccl-.*\.whl' -File
$cudaComputeWheel = Get-OnePathMatch `
    -Path $wheelhouse -Pattern '^cuda_compute-.*\.whl' -File

# Constrain cuda-compute to the coordinated local wheel without making it a
# direct install requirement. This proves the metapackage resolves compute
# transitively while preventing an equal-version index candidate from winning;
# index access remains available for third-party dependencies.
$cudaComputeWheelUri = & $python -c `
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve().as_uri())' `
    $cudaComputeWheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create cuda-compute wheel URI (exit code $LASTEXITCODE)"
}
$constraintFile = [System.IO.Path]::GetTempFileName()
try {
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText(
        $constraintFile, "cuda-compute @ $cudaComputeWheelUri`n", $utf8NoBom
    )
    Invoke-Checked {
        & $python -m pip install --constraint $constraintFile `
            --find-links $wheelhouse `
            $cudaCcclWheel
    } "Failed to install cuda-cccl metapackage"
    # Resolve the forwarded extra separately. pip releases before 26.2 can
    # assert when one solve combines a direct-URL constraint for the base
    # project with a second requirement for that project's extra.
    Invoke-Checked {
        & $python -m pip install --find-links $wheelhouse `
            "${cudaCcclWheel}[minimal-$ctkFlavor$cudaMajor]"
    } "Failed to install cuda-cccl metapackage extra"
}
finally {
    Remove-Item -LiteralPath $constraintFile -Force
}
Invoke-Checked { & $python -m pip check } "Installed cuda-cccl environment is inconsistent"
$verifyMetapackage = @'
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

import cuda.compute

compute_distribution = importlib.metadata.distribution("cuda-compute")
direct_url_text = compute_distribution.read_text("direct_url.json")
if direct_url_text is None:
    raise RuntimeError("cuda-compute is missing direct_url.json provenance")
compute_url = json.loads(direct_url_text)["url"]
expected_compute_url = Path(sys.argv[1]).resolve().as_uri()
if compute_url != expected_compute_url:
    raise RuntimeError(
        f"cuda-compute came from {compute_url}, expected {expected_compute_url}"
    )

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
    $verifyMetapackage | & $python - $cudaComputeWheel
} "cuda-cccl dependency or module-ownership check failed"
