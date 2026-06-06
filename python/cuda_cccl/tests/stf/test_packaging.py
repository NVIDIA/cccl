# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Packaging checks: the wheel must ship the STF C development headers and
shared library so external C/CUDA consumers can build against CUDASTF.

These are pure path/file-existence checks (no compiler, no GPU). They import
the lightweight ``cuda.stf._experimental.paths`` submodule, exercising the
cheap path-discovery route that does not load the STF extension.
"""

import pytest

from cuda.stf._experimental.paths import (
    get_include_paths,
    get_library_dir,
    get_library_path,
)


@pytest.fixture
def include_root():
    # All include_paths fields point at the same shipped CCCL include root.
    return get_include_paths().libcudacxx


def test_stf_c_header_shipped(include_root):
    stf_h = include_root / "cccl" / "c" / "experimental" / "stf" / "stf.h"
    assert stf_h.exists()


def test_cudax_places_header_shipped(include_root):
    places = include_root / "cuda" / "experimental" / "places.cuh"
    assert places.exists()


def test_cudax_stf_header_shipped(include_root):
    stf_cuh = include_root / "cuda" / "experimental" / "stf.cuh"
    assert stf_cuh.exists()


def test_library_dir_resolves():
    lib_dir = get_library_dir()
    assert lib_dir.is_dir()


def test_library_path_resolves():
    lib_path = get_library_path()
    assert lib_path.exists()
    assert lib_path.parent == get_library_dir()
