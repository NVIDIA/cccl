# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative CUDA primitives for the CUTLASS CuTe DSL."""

from ._compiler._runtime import validate_cutlass_runtime
from ._group_exchange import exchange
from ._group_load_store import load, store
from ._group_merge_sort import merge_sort_keys, merge_sort_pairs
from ._group_radix_sort import (
    radix_rank_keys,
    radix_sort_keys,
    radix_sort_pairs,
)
from ._group_reduce import reduce, sum
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
from ._typing import TempStorageLike, ThreadDataLike

__all__ = [
    "topk_min_keys",
    "topk_max_keys",
    "topk_min_pairs",
    "topk_max_pairs",
    "radix_sort_keys",
    "radix_sort_pairs",
    "radix_rank_keys",
    "merge_sort_keys",
    "merge_sort_pairs",
    "exchange",
    "shuffle",
    "scan",
    "exclusive_scan",
    "inclusive_scan",
    "exclusive_sum",
    "inclusive_sum",
    "reduce",
    "sum",
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
