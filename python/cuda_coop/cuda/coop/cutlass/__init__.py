# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative primitives for the CUTLASS CuTe DSL compiler.

Importing this namespace validates the optional runtime and lets the common
API recognize the active CuTe DSL compiler environment. Calls inside that
environment use these implementations. The namespace provides Load/Store,
Reduce, Sum, Scan, Exchange, Shuffle, and supported group queries and
synchronization. Reduce and Scan accept only built-in operators.

Qualified calls also accept the CuTe payload forms documented by each
operation. The per-operation docs describe which block calls accept a
TempStorage descriptor to control their temporary shared-memory storage.
"""

from .._core.api import TempStorageLike, ThreadDataLike
from ._compiler._activation import register_trace_context
from ._group_exchange import exchange
from ._group_load_store import load, store
from ._group_merge_sort import merge_sort_keys, merge_sort_pairs
from ._group_reduce import reduce, sum
from ._group_scan import (
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
from ._group_shuffle import shuffle
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
    "Hierarchy",
    "TempStorage",
    "TempStorageLike",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "merge_sort_keys",
    "merge_sort_pairs",
    "reduce",
    "scan",
    "shuffle",
    "store",
    "sum",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]


def __dir__():
    """List the qualified public API for interactive completion."""

    return sorted(__all__)


register_trace_context()
del register_trace_context
