# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

from ._stf_bindings import _BINDINGS_AVAILABLE  # type: ignore[attr-defined]


@lru_cache()
def get_include_path() -> Path:
    """Return the path to CUDASTF C API and cudax headers installed with this package.

    This directory contains:
    - cccl/c/experimental/stf/stf.h  (the C API header)
    - cuda/experimental/__stf/...     (cudax/STF C++ headers)

    Downstream packages that compile C/C++/CUDA extensions against the STF C API
    should add this path to their include directories to ensure ABI consistency.
    Similar to numpy.get_include().
    """
    pkg_dir = Path(__file__).resolve().parent
    include_dir = pkg_dir / "include"
    if include_dir.is_dir():
        return include_dir
    for sp in sys.path:
        candidate = Path(sp) / "cuda" / "stf" / "include"
        if candidate.is_dir():
            return candidate
    raise RuntimeError(
        "Cannot locate cuda.stf include directory. "
        "The cuda-cccl package may not have been installed correctly."
    )


@lru_cache()
def get_library_path() -> Path:
    """Return the path to the directory containing libcccl.c.experimental.stf.so.

    Downstream packages that link against the STF C API should use this as
    their library search path to ensure ABI consistency with the runtime.
    """
    from cuda.cccl._cuda_version_utils import detect_cuda_version, get_recommended_extra

    cuda_version = detect_cuda_version()
    extra_name = get_recommended_extra(cuda_version)
    lib_name = "libcccl.c.experimental.stf.so"

    # Check relative to this file first (works for non-editable installs)
    pkg_dir = Path(__file__).resolve().parent
    lib_dir = pkg_dir / extra_name / "cccl"
    if (lib_dir / lib_name).exists():
        return lib_dir
    # Editable installs: the .so may be in site-packages while __file__ is in the source tree
    for sp in sys.path:
        candidate = Path(sp) / "cuda" / "stf" / extra_name / "cccl"
        if (candidate / lib_name).exists():
            return candidate
    raise RuntimeError(
        f"Cannot locate {lib_name} for CUDA {cuda_version}. "
        "The cuda-cccl package may not have been installed correctly."
    )


@runtime_checkable
class ExecPlaceLike(Protocol):
    """Protocol for objects that can be used as execution places.

    Any object implementing this protocol can be passed to ctx.task(),
    exec_place_grid.create(), task.set_exec_place(), etc.

    Built-in exec_place and exec_place_grid satisfy this protocol.
    External packages can define custom execution places by implementing
    _as_stf_exec_place() (which should return an exec_place wrapping an
    opaque handle obtained from stf_exec_place_opaque_wrap()).
    """

    @property
    def kind(self) -> str: ...

    def _as_stf_exec_place(self) -> "exec_place": ...


@runtime_checkable
class DataPlaceLike(Protocol):
    """Protocol for objects that can be used as data places.

    Any object implementing this protocol can be passed wherever a data_place
    is expected. External packages can define custom data places by implementing
    _as_stf_data_place().
    """

    @property
    def kind(self) -> str: ...

    def _as_stf_data_place(self) -> "data_place": ...


if not _BINDINGS_AVAILABLE:
    __all__ = [
        "_BINDINGS_AVAILABLE",
        "ExecPlaceLike",
        "DataPlaceLike",
        "get_include_path",
        "get_library_path",
    ]

    def __getattr__(name: str):
        raise AttributeError(
            f"Cannot access 'cuda.stf.{name}' because CUDASTF bindings are not available. "
            "This typically means you're running on a CPU-only machine without CUDA drivers installed, "
            "or that cuda-cccl was not built with STF support."
        )
else:
    from ._stf_bindings import (
        context,
        data_place,
        dep,
        exec_place,
        exec_place_grid,
    )

    __all__ = [
        "_BINDINGS_AVAILABLE",
        "ExecPlaceLike",
        "DataPlaceLike",
        "get_include_path",
        "get_library_path",
        "context",
        "dep",
        "exec_place",
        "exec_place_grid",
        "data_place",
    ]
