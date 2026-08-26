<#
.SYNOPSIS
    Run a CI script inside a deliberately minimal sibling container.
.DESCRIPTION
    cuda.compute is supposed to work with nothing installed beyond its declared
    pip dependencies -- no host compiler, no system CUDA toolkit. The CCCL
    devcontainer has both, so a test running there cannot tell the difference
    between "we depend only on our wheels" and "we happened to find cl.exe and
    the CUDA toolkit lying around". This runs the payload in an image that has
    neither, so that difference fails loudly.

    The job itself still runs in the devcontainer; this launches a sibling
    through the host's Docker daemon (Docker-outside-of-Docker), the same
    arrangement Invoke-Cuda13NestedBuild in build_cuda_cccl_python.ps1 uses.
    Everything the CI harness needs -- gh to fetch the wheel artifact, the
    workflow helpers -- stays in the devcontainer. Only the payload runs inside.

    The image must match the host kernel, because GPU access requires process
    isolation. The devcontainer images are LTSC 2022, so the default is too.
.PARAMETER Script
    Repo-relative path to the script to run, e.g. 'ci\windows\run_compute_tests.ps1'.
    Repo-relative so it resolves the same on both sides of the mount.
.PARAMETER ScriptArgs
    Arguments forwarded to that script.
#>
Param(
    [Parameter(Mandatory = $true)]
    [string]$Script,

    [string[]]$ScriptArgs = @()
)

$ErrorActionPreference = "Stop"

Import-Module "$PSScriptRoot/build_common_python.psm1"

$image = if ($env:CCCL_MINIMAL_CONTAINER_IMAGE) {
    $env:CCCL_MINIMAL_CONTAINER_IMAGE
} else {
    'mcr.microsoft.com/windows/servercore:ltsc2022'
}

# The host daemon resolves the bind-mount source, so it must be a host path.
# The CI action sets both of these for exactly this purpose.
$hostWorkspace = $env:HOST_WORKSPACE
$containerWorkspace = $env:CONTAINER_WORKSPACE
if (-not $hostWorkspace) {
    throw "HOST_WORKSPACE is not set; required to bind-mount the workspace into a sibling container."
}
if (-not $containerWorkspace) {
    throw "CONTAINER_WORKSPACE is not set; required to bind-mount the workspace into a sibling container."
}

if ([System.IO.Path]::IsPathRooted($Script)) {
    throw "Script '$Script' must be a path relative to the repo root."
}

# The whole approach rests on reaching the host daemon; say so plainly rather
# than failing somewhere inside `docker run`.
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker CLI not found in the devcontainer image; cannot launch a sibling container."
}
& docker version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Cannot reach the host Docker daemon; cannot launch a sibling container."
}

# Windows exposes GPUs to a container as a whole device class rather than
# individually, and only under process isolation. Ask the driver whether there
# is anything to pass on.
$gpuArgs = @()
& nvidia-smi -L *> $null
if ($LASTEXITCODE -eq 0) {
    $gpuArgs = @('--isolation=process', '--device', 'class/5B45201D-F2F2-4F3B-85BB-30FF1F953599')
} else {
    Write-Host "No GPU visible here; the payload will run without one."
}

# There is no nvcc in the sibling, so hand it the version resolved out here.
$cudaVersion = Get-CudaVersion

$payload = Join-Path $containerWorkspace $Script

$dockerArgs = @(
    'run', '--rm', '-i'
) + $gpuArgs + @(
    '--mount', "type=bind,source=$hostWorkspace,target=$containerWorkspace",
    '--workdir', $containerWorkspace,
    '--env', "CCCL_CUDA_VERSION=$cudaVersion",
    '--env', "CCCL_PYTHON_USE_V2=$($env:CCCL_PYTHON_USE_V2)",
    '--env', "GITHUB_ACTIONS=$($env:GITHUB_ACTIONS)",
    '--env', 'PYTHONDONTWRITEBYTECODE=1',
    $image,
    'powershell.exe', '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass',
    '-File', $payload
) + $ScriptArgs

Write-Host "Running in minimal container: $image"
Write-Host "  docker $($dockerArgs -join ' ')"
& docker @dockerArgs
exit $LASTEXITCODE
