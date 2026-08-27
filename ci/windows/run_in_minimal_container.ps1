<#
.SYNOPSIS
    Run a CI script inside a deliberately minimal sibling container.
.DESCRIPTION
    See "Testing Python in a minimal container" in
    docs/infrastructure/ci/references/ci_overview.rst for what this buys and why
    the lanes are split this way.
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
    # The image must match the host kernel, because GPU access requires process
    # isolation. The devcontainer images are LTSC 2022, so this is too.
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

# Windows exposes GPUs as a whole device class rather than individually, and
# only under process isolation. Every lane that reaches this helper is a GPU
# lane, so request the class unconditionally; docker fails loudly if it is
# absent.
$gpuArgs = @(
    '--isolation=process',
    '--device', 'class/5B45201D-F2F2-4F3B-85BB-30FF1F953599'
)

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
