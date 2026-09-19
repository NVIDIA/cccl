# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Import-light common API organized by cooperative primitive family.

Leaf modules own public argument capture and validation; this facade preserves
the documented root export order and compiler-backend marker contract. It does
not own semantic lowering, provider rendering, or backend compiler state.
"""

from .exchange import exchange  # noqa: F401
from .histogram import histogram as histogram
from .load_store import load, store  # noqa: F401
from .merge_sort import merge_sort_keys as merge_sort_keys
from .merge_sort import merge_sort_pairs as merge_sort_pairs
from .neighbors import adjacent_difference, discontinuity  # noqa: F401
from .radix import radix_rank, radix_sort_keys, radix_sort_pairs  # noqa: F401
from .reduce import reduce, sum  # noqa: F401
from .reduce_batched import reduce_batched as reduce_batched
from .run_length import run_length_decode as run_length_decode
from .run_length import run_length_decode_into as run_length_decode_into
from .scan import (  # noqa: F401
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
from .shuffle import shuffle  # noqa: F401
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
from .topk import (  # noqa: F401
    topk_max_keys,
    topk_max_pairs,
    topk_min_keys,
    topk_min_pairs,
)

# Descriptor constructors and group factories do not use the family
# registration decorator. The Numba rewrite recognizes their exact exported
# identity plus this tag, which rejects same-named impostor callables.
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
    "adjacent_difference",
    "discontinuity",
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
    "histogram",
    "load",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank",
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
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
]
