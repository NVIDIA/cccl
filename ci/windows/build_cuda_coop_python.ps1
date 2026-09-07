# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

<#
.SYNOPSIS
    Build and validate the pure-Python cuda-coop wheel on Windows.

.PARAMETER PyVersion
    Python version used to build the wheel.
#>

[CmdletBinding()]
Param(
    [Parameter(Mandatory = $true)]
    [Alias("py-version")]
    [ValidatePattern("^\d+\.\d+t?$")]
    [string]$PyVersion
)

$ErrorActionPreference = "Stop"

Import-Module "$PSScriptRoot/build_common_python.psm1" -Force

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$PackageRoot = Join-Path $RepoRoot "python/cuda_coop"
$Wheelhouse = Join-Path $RepoRoot "wheelhouse"
$PythonExe = Get-Python -Version $PyVersion

New-Item -ItemType Directory -Path $Wheelhouse -Force | Out-Null
Get-ChildItem -LiteralPath $Wheelhouse -Filter "cuda_coop-*.whl" -File |
    Remove-Item -Force

& $PythonExe -m pip install build
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the Python build frontend."
}

& $PythonExe -m build --wheel --outdir $Wheelhouse $PackageRoot
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

$Validator = Join-Path $RepoRoot "ci/validate_cuda_coop_wheel.py"
& $PythonExe $Validator $Wheels[0].FullName
if ($LASTEXITCODE -ne 0) {
    throw "cuda-coop wheel validation failed."
}

$SmokeEnv = Join-Path ([System.IO.Path]::GetTempPath()) (
    "cuda-coop-smoke-" + [guid]::NewGuid().ToString("N")
)
try {
    & $PythonExe -m venv $SmokeEnv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the cuda-coop wheel smoke-test environment."
    }

    $SmokePython = Join-Path $SmokeEnv "Scripts/python.exe"
    & $SmokePython -m pip install --disable-pip-version-check $Wheels[0].FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install the cuda-coop wheel."
    }

    $SmokeCommand = @'
import sys
from pathlib import Path

import cuda
import cuda.coop
from cuda.coop._headers import resolve_include_paths

paths = resolve_include_paths(
    start=Path(sys.prefix),
    required_headers=(
        "cub/block/block_load.cuh",
        "cuda/experimental/coop.cuh",
        "cuda/std/cstdint",
    ),
)
assert cuda.coop.__version__ != "0+unknown"
assert getattr(cuda, "__file__", None) is None
assert cuda.coop.this_block().kind == "block"
assert paths.origin == "cuda-coop wheel header bundle"
'@
    $SmokeScript = Join-Path $SmokeEnv "validate_install.py"
    Set-Content -LiteralPath $SmokeScript -Value $SmokeCommand -Encoding UTF8
    & $SmokePython -I $SmokeScript
    if ($LASTEXITCODE -ne 0) {
        throw "Installed cuda-coop wheel smoke test failed."
    }
}
finally {
    if (Test-Path -LiteralPath $SmokeEnv) {
        Remove-Item -LiteralPath $SmokeEnv -Recurse -Force
    }
}

Write-Host "Built cuda-coop wheel: $($Wheels[0].FullName)"
