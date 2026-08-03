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
$repoRoot = Get-RepoRoot

${wheelPath} = Get-CcclHeadersWheel
$wheelhouse = Split-Path -Parent $wheelPath

Invoke-Checked { & $python -m pip install -U pip pytest pytest-xdist } "Failed to install pytest / pytest-xdist"
Invoke-Checked {
    & $python -m pip install --find-links $wheelhouse $wheelPath
} "Failed to install cccl-headers"
Invoke-Checked { & $python -m pip check } "Installed cccl-headers environment is inconsistent"
Invoke-Checked {
    & $python -c "import importlib.metadata as m; names={d.metadata['Name'] for d in m.distributions()}; assert 'cuda-compute' not in names and 'cuda-cccl' not in names"
} "Standalone cccl-headers install pulled in compute or the metapackage"

Push-Location (Join-Path $repoRoot "python/cccl_headers")
try {
    Invoke-Checked { & $python -m pytest -n auto -v tests/ } "headers tests failed"
}
finally { Pop-Location }
