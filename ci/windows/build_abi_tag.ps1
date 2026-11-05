# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Param(
    [Alias("std")]
    [ValidateSet(17, 20, 23)]
    [int]$CXX_STANDARD = 17,
    [Alias("cxx")]
    [ValidateNotNullOrEmpty()]
    [string]$HOST_COMPILER = "cl"
)

$ErrorActionPreference = "Stop"

& nvcc --version
if ($LASTEXITCODE -ne 0) {
    throw "Failed to query nvcc version (exit code $LASTEXITCODE)"
}
$hostInfo = & $HOST_COMPILER /?
if ($LASTEXITCODE -ne 0) {
    throw "Failed to query host compiler version (exit code $LASTEXITCODE)"
}
$hostInfo | Select-String "Version" | Select-Object -First 1 | ForEach-Object { Write-Host $_ }

$buildDir = Join-Path ([System.IO.Path]::GetTempPath()) ("cccl-abi-tag-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $buildDir | Out-Null
try {
    # Reject warnings about ignored attributes in both the CUDA and host frontends.
    $nvccArgs = @(
        "-ccbin", $HOST_COMPILER,
        "-std=c++$CXX_STANDARD",
        "-arch=sm_75",
        "--Werror", "all-warnings",
        "-Xcompiler=/WX",
        "-c", (Join-Path $PSScriptRoot "../abi_tag.cu"),
        "-o", (Join-Path $buildDir "abi_tag.obj")
    )
    $displayArgs = $nvccArgs | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }
    Write-Host ("nvcc " + ($displayArgs -join " "))
    & nvcc @nvccArgs
    if ($LASTEXITCODE -ne 0) {
        throw "abi_tag probe failed (exit code $LASTEXITCODE)"
    }
} finally {
    Remove-Item -Recurse -Force $buildDir
}
