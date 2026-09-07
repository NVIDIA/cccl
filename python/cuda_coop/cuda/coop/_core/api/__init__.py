# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Import-light portable API organized by cooperative primitive family.

Leaf modules own public argument capture and validation; this facade preserves
the documented root export order and compiler-backend marker contract. It does
not own semantic lowering, provider rendering, or backend compiler state.
"""

from ._dispatch import _GROUP_OPERATIONS
from .load_store import load, store  # noqa: F401
from .reduce import reduce, sum  # noqa: F401
from .scan import (  # noqa: F401
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
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
