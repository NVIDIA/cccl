# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import shutil
import subprocess

import pytest

from ..support.frontend import (
    CPP_API_SHAPES_SOURCE,
    TILE_ROOT,
    make_cpp_frontend_stubs,
)


def test_cpp_api_compile_shapes_with_stub_tile_frontend(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip(
            "a C++20 compiler is required for the C++ Tile API shape test"
        )

    include = tmp_path / "include"
    make_cpp_frontend_stubs(include)
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-fsyntax-only",
            f"-I{include}",
            f"-I{TILE_ROOT / 'include'}",
            str(CPP_API_SHAPES_SOURCE),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_cpp_api_reports_missing_linkage_provider_frontend(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip(
            "a C++20 compiler is required for the C++ Tile API capability test"
        )

    include = tmp_path / "include"
    make_cpp_frontend_stubs(include)
    cuda_tile = include / "cuda_tile.h"
    stub = cuda_tile.read_text(encoding="utf-8")
    stub = stub.replace("#define __experimental_foreign_decl__\n", "")
    stub = stub.replace("#define __experimental_foreign_name__(name)\n", "")
    cuda_tile.write_text(stub, encoding="utf-8")

    source = tmp_path / "missing_linkage_provider_frontend.cpp"
    source.write_text("#include <cuda/coop/tile.cuh>\n", encoding="utf-8")
    result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-fsyntax-only",
            f"-I{include}",
            f"-I{TILE_ROOT / 'include'}",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert (
        "requires a CUDA C++ Tile frontend with experimental "
        "linkage-provider declarations"
    ) in result.stderr
