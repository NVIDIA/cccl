# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative primitives for the CUTLASS CuTe DSL compiler.

Importing this namespace checks for a compatible CUTLASS CuTe DSL runtime
and raises an error if none is available. The import also registers a check
for the active CuTe compiler environment, so common ``cuda.coop`` calls
traced there use these implementations. The namespace provides Load/Store,
Reduce, Sum, Scan, Exchange, Shuffle, Merge Sort, Radix Sort, Radix Rank,
TopK, Adjacent Difference, Discontinuity, Histogram, Run Length Decode,
Batched Warp Reduction, and thread-group handles such as ``this_block()``
with rank/count queries and synchronization. Reduction and Scan calls accept
only built-in operators.

Calls through ``cuda.coop.cutlass`` (qualified calls) also accept the CuTe
payload forms documented by each operation. The per-operation docs describe
which block calls accept a TempStorage descriptor to control their temporary
shared-memory storage.
"""

from .._core.api import TempStorageLike, ThreadDataLike
from ._compiler._activation import register_trace_context
from ._group_exchange import exchange
from ._group_histogram import histogram
from ._group_load_store import load, store
from ._group_merge_sort import merge_sort_keys, merge_sort_pairs
from ._group_neighbors import adjacent_difference, discontinuity
from ._group_radix_sort import (
    radix_rank_keys,
    radix_sort_keys,
    radix_sort_pairs,
)
from ._group_reduce import reduce, sum
from ._group_reduce_batched import reduce_batched
from ._group_run_length import run_length_decode, run_length_decode_into
from ._group_scan import (
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
from ._group_shuffle import shuffle
from ._group_topk import (
    topk_max_keys,
    topk_max_pairs,
    topk_min_keys,
    topk_min_pairs,
)
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
    "adjacent_difference",
    "discontinuity",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "histogram",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank_keys",
    "radix_sort_keys",
    "radix_sort_pairs",
    "reduce",
    "reduce_batched",
    "run_length_decode",
    "run_length_decode_into",
    "scan",
    "shuffle",
    "store",
    "sum",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
]


def __dir__():
    """List the qualified public API for interactive completion."""

    return sorted(__all__)


register_trace_context()
del register_trace_context
