# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ....support.paths import PACKAGE_ROOT

NATIVE_ROOT = PACKAGE_ROOT.parents[1] / "tile/cpp/tilelib"


def _include_root(variable, candidates, header):
    configured = os.environ.get(variable)
    if configured:
        for relative in candidates:
            include = Path(configured) / relative
            if (include / header).is_file():
                return include
    pytest.skip(f"NVRTC lifetime test requires {variable} with {header}")


def test_nvrtc_program_cleanup_on_returns_and_allocation_errors(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")

    sdk_include = _include_root(
        "CUDA_COOP_TILE_SDK_ROOT",
        ("include",),
        "tile_ir/linkage_provider/LinkageProviderAPI.h",
    )
    nvrtc_include = _include_root(
        "CUDA_COOP_TILE_CUDA_ROOT",
        ("bin/nvrtc/include", "include"),
        "nvrtc.h",
    )
    executable = tmp_path / "nvrtc_program_lifetime"
    fixture = (
        PACKAGE_ROOT.parents[1]
        / "tile/cpp/test/support/nvrtc_program_lifetime.cpp"
    )
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O0",
            "-g0",
            "-ffunction-sections",
            "-fdata-sections",
            f"-I{NATIVE_ROOT}",
            f"-I{sdk_include}",
            f"-I{nvrtc_include}",
            str(fixture),
            str(NATIVE_ROOT / "provider/runtime/nvrtc_compiler.cpp"),
            str(NATIVE_ROOT / "provider/primitive_spec.cpp"),
            "-Wl,--gc-sections",
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    executed = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True
    )
    assert executed.returncode == 0, executed.stdout + executed.stderr
