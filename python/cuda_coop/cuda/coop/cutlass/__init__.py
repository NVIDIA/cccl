# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative primitives for the CUTLASS CuTe DSL compiler."""

from .._core.api import TempStorageLike, ThreadDataLike
from ._compiler._activation import register_trace_context
from ._group_exchange import exchange
from ._group_load_store import load, store
from ._group_merge_sort import merge_sort_keys, merge_sort_pairs
from ._group_neighbors import adjacent_difference, discontinuity
from ._group_radix import radix_rank, radix_sort_keys, radix_sort_pairs
from ._group_reduce import reduce, sum
from ._group_scan import (
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
from ._group_shuffle import shuffle
from ._group_topk import topk_max_keys, topk_max_pairs, topk_min_keys, topk_min_pairs
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
    "scan",
    "exclusive_scan",
    "inclusive_scan",
    "exclusive_sum",
    "inclusive_sum",
    "shuffle",
    "exchange",
    "adjacent_difference",
    "discontinuity",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_sort_keys",
    "radix_sort_pairs",
    "radix_rank",
    "topk_min_keys",
    "topk_min_pairs",
    "topk_max_keys",
    "topk_max_pairs",
]


def __dir__():
    return sorted(__all__)


register_trace_context()
del register_trace_context
