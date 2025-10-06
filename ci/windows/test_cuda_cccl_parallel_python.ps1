Param(
    [Parameter(Mandatory = $true)]
    [Alias("py-version")]
    [ValidatePattern("^\d+\.\d+$")]
    [string]$PyVersion
)

$ErrorActionPreference = "Stop"

# Import shared helpers
Import-Module "$PSScriptRoot/build_common.psm1"

$python = Get-Python -Version $PyVersion
$cudaMajor = Get-CudaMajor

$repoRoot = Get-RepoRoot

if ($env:GITHUB_ACTIONS) {
    # In CI: download previously built wheel artifact to the repo root
    Push-Location $repoRoot
    try {
        $wheelArtifactName = (& bash -lc "ci/util/workflow/get_wheel_artifact_name.sh").Trim()
        if (-not $wheelArtifactName) { throw "Failed to resolve wheel artifact name" }
        $repoRootPosix = Convert-ToUnixPath $repoRoot
        & bash -lc "ci/util/artifacts/download.sh $wheelArtifactName $repoRootPosix"
        if ($LASTEXITCODE -ne 0) { throw "Failed to download wheel artifact '$wheelArtifactName'" }
    }
    finally { Pop-Location }
    $wheelhouse = Join-Path $repoRoot "wheelhouse"
    $wheelPath = Get-OnePathMatch -Path $wheelhouse -Pattern '^cuda_cccl-.*\.whl' -File
}
else {
    # Local/dev: build the wheel if missing
    $wheelPath = Ensure-CudaCcclWheel -PyVersion $PyVersion -UseNinja
}

& $python -m pip install "$wheelPath[test-cu$cudaMajor]"

Push-Location (Join-Path $repoRoot "python/cuda_cccl/tests")
try {
    & $python -m pytest -n 6 -v parallel/ -m "not large"
    & $python -m pytest -n 0 -v parallel/ -m "large"
}
finally { Pop-Location }
