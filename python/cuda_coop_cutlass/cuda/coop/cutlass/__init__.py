# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative CUDA primitives for the CUTLASS CuTe DSL."""

from ._compiler._runtime import validate_cutlass_runtime
from ._group_load_store import load, store
from ._temp_storage import TempStorage
from ._thread_data import ThreadData
from ._thread_group import (
    Hierarchy,
    ThreadGroup,
    ThreadHierarchy,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)
from ._typing import TempStorageLike, ThreadDataLike

__all__ = [
    "load",
    "store",
    "Hierarchy",
    "ThreadGroup",
    "ThreadHierarchy",
    "ThreadData",
    "TempStorage",
    "TempStorageLike",
    "ThreadDataLike",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]


def __dir__():
    """List the public API for interactive completion."""
    return sorted(__all__)


validate_cutlass_runtime()
del validate_cutlass_runtime
