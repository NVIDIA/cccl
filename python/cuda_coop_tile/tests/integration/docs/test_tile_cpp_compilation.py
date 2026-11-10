# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile documented C++ kernels without a GPU or a native provider."""

import os
import subprocess

import pytest

from ...support.paths import REPO_ROOT
from ._tile_cpp_examples import documented_cases

CASES, _ = documented_cases(REPO_ROOT, REPO_ROOT / "docs/tile/cpp")


@pytest.mark.backend_tile_cpp
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_tile_cpp_documentation_compiles(case, tmp_path):
    source = tmp_path / f"{case['name']}.cu"
    source.write_text(case["source"], encoding="utf-8")
    result = subprocess.run(
        [
            os.environ.get("CUDA_COOP_TILE_CPP_TEST_NVCC")
            or os.environ.get("CUDACXX")
            or "nvcc",
            "--enable-tile",
            "--tile-preview=simt",
            "--tilebc",
            "-std=c++20",
            "-arch=sm_120",
            "-I",
            str(REPO_ROOT / "tile/cpp/include"),
            "-I",
            str(REPO_ROOT / "libcudacxx/include"),
            str(source),
            "-o",
            str(tmp_path / f"{case['name']}.tilebc"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
