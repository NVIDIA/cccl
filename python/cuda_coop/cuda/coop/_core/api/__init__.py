# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Collect the common kernel API and identify its constructor functions.

The root package exports these names. Compiler adapters use their identities
to recognize cooperative operations without changing the public call syntax.
"""

from .exchange import exchange
from .histogram import histogram as histogram
from .load_store import load, store
from .merge_sort import merge_sort_keys as merge_sort_keys
from .merge_sort import merge_sort_pairs as merge_sort_pairs
from .neighbors import adjacent_difference, discontinuity
from .radix_sort import radix_rank_keys, radix_sort_keys, radix_sort_pairs
from .reduce import reduce, sum
from .reduce_batched import reduce_batched as reduce_batched
from .run_length import run_length_decode as run_length_decode
from .run_length import run_length_decode_into as run_length_decode_into
from .scan import (
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
from .shuffle import shuffle
from .temp_storage import TempStorage, TempStorageLike
from .thread_data import ThreadData, ThreadDataLike
from .thread_group import (
    Hierarchy,
    ThreadGroup,
    ThreadHierarchy,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)
from .topk import (
    topk_max_keys,
    topk_max_pairs,
    topk_min_keys,
    topk_min_pairs,
)

# Constructors and group factories bypass the operation registration decorator.
# The Numba rewrite requires this tag and the exported function identity, so
# an unrelated callable with the same name is not treated as a constructor.
for _member_name in (
    "TempStorage",
    "ThreadData",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
):
    globals()[_member_name].__cuda_coop_backend_member__ = _member_name
del _member_name


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
