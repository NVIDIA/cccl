# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Import-light portable API organized by cooperative primitive family.

Leaf modules own public argument capture and validation; this facade preserves
the documented root export order and compiler-backend marker contract. It does
not own semantic lowering, provider rendering, or backend compiler state.
"""

from ._dispatch import _GROUP_OPERATIONS
from .adjacent_difference import adjacent_difference  # noqa: F401
from .discontinuity import discontinuity  # noqa: F401
from .exchange import exchange  # noqa: F401
from .histogram import histogram  # noqa: F401
from .load_store import load, store  # noqa: F401
from .merge_sort import merge_sort_keys, merge_sort_pairs  # noqa: F401
from .radix import (  # noqa: F401
    radix_rank,
    radix_sort_keys,
    radix_sort_pairs,
)
from .reduce import reduce, sum  # noqa: F401
from .run_length_decode import run_length_decode  # noqa: F401
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

for _member_name in (
    "TempStorage",
    "ThreadData",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    *_GROUP_OPERATIONS,
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
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    *_GROUP_OPERATIONS,
]
del _GROUP_OPERATIONS
