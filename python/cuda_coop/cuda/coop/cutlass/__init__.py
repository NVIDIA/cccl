# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative primitives for the CUTLASS CuTe DSL compiler.

Importing this namespace validates the optional runtime and lets the common
API recognize the active CuTe DSL compiler environment. Calls inside that
environment use these implementations. The namespace provides block and warp
Load/Store, thread groups, and per-thread register payloads.
TempStorage descriptors control scratch for block transpose algorithms.
"""

from .._core.api import TempStorageLike, ThreadDataLike
from ._compiler._activation import register_trace_context
from ._group_load_store import load, store
from ._group_reduce import reduce, sum
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

__all__ = [
    "TempStorage",
    "TempStorageLike",
    "Hierarchy",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    "load",
    "store",
    "reduce",
    "sum",
]


def __dir__():
    """List the qualified public API for interactive completion."""

    return sorted(__all__)


register_trace_context()
del register_trace_context
