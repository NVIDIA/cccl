# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative primitives shared by supported CUDA Python DSLs."""

from ._core.api.adjacent_difference import (
    adjacent_difference as adjacent_difference,
)
from ._core.api.discontinuity import discontinuity as discontinuity
from ._core.api.exchange import exchange as exchange
from ._core.api.histogram import histogram as histogram
from ._core.api.load_store import load as load
from ._core.api.load_store import store as store
from ._core.api.merge_sort import merge_sort_keys as merge_sort_keys
from ._core.api.merge_sort import merge_sort_pairs as merge_sort_pairs
from ._core.api.radix import radix_rank as radix_rank
from ._core.api.radix import radix_sort_keys as radix_sort_keys
from ._core.api.radix import radix_sort_pairs as radix_sort_pairs
from ._core.api.reduce import reduce as reduce
from ._core.api.reduce import sum as sum
from ._core.api.run_length_decode import run_length_decode as run_length_decode
from ._core.api.scan import exclusive_scan as exclusive_scan
from ._core.api.scan import exclusive_sum as exclusive_sum
from ._core.api.scan import inclusive_scan as inclusive_scan
from ._core.api.scan import inclusive_sum as inclusive_sum
from ._core.api.scan import scan as scan
from ._core.api.shuffle import shuffle as shuffle
from ._core.api.temp_storage import TempStorage as TempStorage
from ._core.api.temp_storage import TempStorageLike as TempStorageLike
from ._core.api.thread_data import ThreadData as ThreadData
from ._core.api.thread_data import ThreadDataLike as ThreadDataLike
from ._core.api.thread_group import Hierarchy as Hierarchy
from ._core.api.thread_group import ThreadGroup as ThreadGroup
from ._core.api.thread_group import ThreadHierarchy as ThreadHierarchy
from ._core.api.thread_group import this_block as this_block
from ._core.api.thread_group import this_cluster as this_cluster
from ._core.api.thread_group import this_grid as this_grid
from ._core.api.thread_group import this_thread as this_thread
from ._core.api.thread_group import this_warp as this_warp
from ._core.api.topk import topk_max_keys as topk_max_keys
from ._core.api.topk import topk_max_pairs as topk_max_pairs
from ._core.api.topk import topk_min_keys as topk_min_keys
from ._core.api.topk import topk_min_pairs as topk_min_pairs

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
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_pairs",
    "reduce",
    "run_length_decode",
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
