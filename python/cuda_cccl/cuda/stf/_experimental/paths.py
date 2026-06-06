# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Locate the CUDASTF C development headers and shared library.

These helpers let external C/CUDA projects compile and link against the same
STF C ABI that the Python bindings use. Importing this module is cheap: it does
*not* load the STF extension (``_stf_bindings_impl``) or preload CUDA libraries,
so it is safe to use from build scripts.

The shipped include root (see :func:`get_include_paths`) contains both the C STF
header ``cccl/c/experimental/stf/stf.h`` and the cudax headers
``cuda/experimental/*.cuh``, alongside libcudacxx/CUB/Thrust.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path

from cuda.cccl._cuda_version_utils import detect_cuda_version, get_recommended_extra
from cuda.cccl.headers import get_include_paths as _get_cccl_include_paths

# Shared library produced by the cccl.c.experimental.stf target (Linux-only).
_STF_LIBRARY_NAME = "libcccl.c.experimental.stf.so"


def get_include_paths():
    """Return the CCCL include paths needed to compile against the STF C API.

    The returned :class:`~cuda.cccl.headers.include_paths.IncludePaths` exposes
    an include root that contains the C STF header
    (``cccl/c/experimental/stf/stf.h``) and the cudax headers
    (``cuda/experimental/*.cuh``), in addition to libcudacxx/CUB/Thrust.
    """
    return _get_cccl_include_paths()


@lru_cache()
def get_library_dir() -> Path:
    """Return the directory containing the STF C shared library."""
    rel = Path(get_recommended_extra(detect_cuda_version())) / "cccl"

    with as_file(files("cuda.stf._experimental")) as f:
        lib_dir = Path(f) / rel

    if not (lib_dir / _STF_LIBRARY_NAME).exists():
        # Editable installs serve the .py files from the source tree but place
        # compiled artifacts elsewhere; fall back to scanning sys.path (mirrors
        # cuda.cccl.headers.include_paths.get_include_paths).
        for sp in sys.path:
            candidate = Path(sp).resolve() / "cuda" / "stf" / "_experimental" / rel
            if (candidate / _STF_LIBRARY_NAME).exists():
                lib_dir = candidate
                break
        else:
            raise RuntimeError(
                f"Unable to locate the CUDASTF library '{_STF_LIBRARY_NAME}'. "
                "Reinstall cuda-cccl with the matching extra "
                "(e.g. `pip install cuda-cccl[cu13]`)."
            )
    return lib_dir


@lru_cache()
def get_library_path() -> Path:
    """Return the full path to the STF C shared library."""
    return get_library_dir() / _STF_LIBRARY_NAME
