# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

<#
.SYNOPSIS
    Build the pure-Python cuda-coop wheel on Windows.

.PARAMETER PythonExe
    Python executable used to install build and create the wheel.

.PARAMETER Wheelhouse
    Output directory. Defaults to wheelhouse/cuda_coop under the repository.
#>

[CmdletBinding()]
Param(
    [Parameter(Mandatory = $false)]
    [ValidateNotNullOrEmpty()]
    [string]$PythonExe = "python",

    [Parameter(Mandatory = $false)]
    [ValidateNotNullOrEmpty()]
    [string]$Wheelhouse
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$PackageRoot = Join-Path $RepoRoot "python/cuda_coop"
if (-not $Wheelhouse) {
    $Wheelhouse = Join-Path $RepoRoot "wheelhouse/cuda_coop"
}
$Wheelhouse = [System.IO.Path]::GetFullPath($Wheelhouse)
New-Item -ItemType Directory -Path $Wheelhouse -Force | Out-Null

Get-ChildItem -LiteralPath $Wheelhouse -Filter "cuda_coop-*.whl" -File |
    Remove-Item -Force

if (-not $env:SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUDA_COOP) {
    $SourceRevision = & git -C $RepoRoot rev-parse --short=12 HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to determine the cuda-coop source revision."
    }
    $SourceRevision = $SourceRevision.Trim()
    $env:SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUDA_COOP = `
        "0.1.0.dev0+g$SourceRevision"
}

& $PythonExe -m pip install build
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the Python build frontend."
}

$BuildArgs = @(
    "-m", "build",
    "--wheel",
    "--outdir", $Wheelhouse,
    $PackageRoot
)
& $PythonExe @BuildArgs
if ($LASTEXITCODE -ne 0) {
    throw "cuda-coop wheel build failed."
}

$Wheels = @(Get-ChildItem -LiteralPath $Wheelhouse -Filter "cuda_coop-*.whl" -File)
if ($Wheels.Count -ne 1) {
    throw "Expected one cuda-coop wheel under $Wheelhouse; found $($Wheels.Count)."
}
if ($Wheels[0].Name -notmatch "-py3-none-any\.whl$") {
    throw "Expected a py3-none-any wheel; found $($Wheels[0].Name)."
}

Write-Host "Built cuda-coop wheel: $($Wheels[0].FullName)"
