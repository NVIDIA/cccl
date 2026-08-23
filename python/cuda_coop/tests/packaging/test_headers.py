# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from cuda.coop._headers import (
    CoopIncludePaths,
    HeaderResolutionError,
    resolve_include_paths,
)

_PACKAGE_ROOT = Path(__file__).parents[2]


def test_source_resolution_uses_one_coherent_cccl_header_set() -> None:
    paths = resolve_include_paths(
        start=Path(__file__),
        required_headers=(
            "cub/block/block_load.cuh",
            "cub/block/block_store.cuh",
            "thrust/detail/raw_pointer_cast.h",
            "cuda/experimental/coop.cuh",
            "cuda/std/cstdint",
        ),
    )

    assert paths.origin.startswith("CCCL source checkout ")
    assert tuple(path.relative_to(_PACKAGE_ROOT.parents[1]) for path in paths.cccl) == (
        Path("thrust"),
        Path("cub"),
        Path("cudax/include"),
        Path("libcudacxx/include"),
    )


def test_required_header_diagnostic_never_falls_back_to_toolkit() -> None:
    with pytest.raises(HeaderResolutionError, match="does not fall back"):
        resolve_include_paths(
            start=Path(__file__),
            required_headers=("cub/block/not_a_primitive.cuh",),
        )


def test_cuda_headers_are_required_only_when_compiling() -> None:
    paths = CoopIncludePaths(
        cccl=(Path("/private/cccl/include"),),
        cuda=(),
        origin="test",
    )
    with pytest.raises(HeaderResolutionError, match="cuda_runtime.h"):
        paths.as_tuple()


def test_package_metadata_includes_cudax_header_bundle() -> None:
    cmake = (_PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "CCCLInstallRules" in cmake
    assert "CCCL_ENABLE_CUDAX ON" in cmake
    assert '"../../cudax/LICENSE.TXT"' in pyproject
