# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR-qualified group-first cooperative primitives."""

from .._core.api import TempStorageLike, ThreadDataLike
from ._group_exchange import exchange
from ._group_histogram import histogram as histogram
from ._group_load_store import load, store
from ._group_merge_sort import merge_sort_keys as merge_sort_keys
from ._group_merge_sort import merge_sort_pairs as merge_sort_pairs
from ._group_neighbors import adjacent_difference as adjacent_difference
from ._group_neighbors import discontinuity as discontinuity
from ._group_radix import radix_rank as radix_rank
from ._group_radix import radix_sort_keys as radix_sort_keys
from ._group_radix import radix_sort_pairs as radix_sort_pairs
from ._group_reduce import reduce, sum
from ._group_run_length import run_length_decode as run_length_decode
from ._group_run_length import run_length_decode_into as run_length_decode_into
from ._group_scan import (
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
from ._group_shuffle import shuffle
from ._group_topk import topk_max_keys, topk_max_pairs, topk_min_keys, topk_min_pairs
from ._stateful_function import StatefulFunction
from ._temp_storage import TempStorage
from ._thread_data import ThreadData, local, shared
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
    "topk_min_keys",
    "topk_min_pairs",
    "topk_max_keys",
    "topk_max_pairs",
    "Hierarchy",
    "TempStorage",
    "TempStorageLike",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
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
    "run_length_decode_into",
    "scan",
    "shuffle",
    "store",
    "sum",
    "StatefulFunction",
    "local",
    "shared",
]
