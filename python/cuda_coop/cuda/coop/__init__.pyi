# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative primitives shared by supported CUDA Python DSLs."""

from typing import Literal

from ._core.api.load_store import load, store
from ._core.api.temp_storage import TempStorage, TempStorageLike
from ._core.api.thread_data import ThreadData, ThreadDataLike
from ._core.api.thread_group import (
    Hierarchy,
    ThreadGroup,
    ThreadHierarchy,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)

__version__: str

def register(backend: Literal["numba-cuda-mlir", "numba_cuda_mlir"]) -> None: ...

__all__ = [
    "Hierarchy",
    "TempStorage",
    "TempStorageLike",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "__version__",
    "load",
    "register",
    "store",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]
