# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path

import pytest

from ....support.paths import PACKAGE_ROOT

NATIVE_ROOT = PACKAGE_ROOT.parents[1] / "tile/cpp/tilelib"

CPP_TEST = (
    PACKAGE_ROOT.parents[1] / "tile/cpp/test/support/implementation_runtime.cpp"
)


def _sdk_include() -> Path:
    raw_root = os.environ.get("CUDA_COOP_TILE_SDK_ROOT")
    if not raw_root:
        pytest.skip("host conformance test requires CUDA_COOP_TILE_SDK_ROOT")
    include = Path(raw_root) / "include"
    assert (include / "tile_ir/linkage_provider/LinkageProviderAPI.h").is_file()
    return include


def _toolkit_nvrtc() -> tuple[Path, Path]:
    raw_root = os.environ.get("CUDA_COOP_TILE_CUDA_ROOT")
    if raw_root:
        root = Path(raw_root)
        for include, library in (
            (
                root / "bin" / "nvrtc" / "include",
                root / "bin" / "nvrtc" / "lib64" / "libnvrtc.so",
            ),
            (root / "include", root / "lib64" / "libnvrtc.so"),
        ):
            if include.joinpath("nvrtc.h").is_file() and library.is_file():
                return include, library
    pytest.skip("host conformance test requires CUDA_COOP_TILE_CUDA_ROOT")
