# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build the standalone sdist and build its universal wheel from that archive.
[CmdletBinding()]
Param(
    [Parameter(Mandatory = $true)]
    [Alias("py-version")]
    [ValidatePattern("^\d+\.\d+$")]
    [string]$PyVersion
)
$ErrorActionPreference = "Stop"
Import-Module "$PSScriptRoot/build_common_python.psm1" -Force
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$PythonExe = Get-Python -Version $PyVersion
$PackageRoot = Join-Path $RepoRoot "python/cuda_coop_cutlass"
$Wheelhouse = Join-Path $RepoRoot "wheelhouse"
& $PythonExe -m pip install build
if ($LASTEXITCODE -ne 0) { throw "Failed to install the build frontend." }
& $PythonExe -m build --outdir $Wheelhouse $PackageRoot
if ($LASTEXITCODE -ne 0) { throw "cuda-coop-cutlass source/wheel build failed." }
$Wheels = @(Get-ChildItem -LiteralPath $Wheelhouse -Filter "cuda_coop_cutlass-*.whl" -File)
$Sources = @(Get-ChildItem -LiteralPath $Wheelhouse -Filter "cuda_coop_cutlass-*.tar.gz" -File)
if ($Wheels.Count -ne 1 -or $Sources.Count -ne 1) {
    throw "Expected exactly one cuda-coop-cutlass wheel and sdist."
}
& $PythonExe (Join-Path $RepoRoot "ci/validate_cuda_coop_cutlass_artifacts.py") $Wheels[0].FullName $Sources[0].FullName
if ($LASTEXITCODE -ne 0) { throw "cuda-coop-cutlass artifact validation failed." }
