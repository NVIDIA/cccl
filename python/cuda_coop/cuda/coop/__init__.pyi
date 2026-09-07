# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative primitives shared by supported CUDA Python DSLs."""

from ._core.api.load_store import load, store
from ._core.api.reduce import reduce, sum
from ._core.api.scan import (
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
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

__all__ = [
    "Hierarchy",
    "TempStorage",
    "TempStorageLike",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "__version__",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "reduce",
    "scan",
    "store",
    "sum",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]
