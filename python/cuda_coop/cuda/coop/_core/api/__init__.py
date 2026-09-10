# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Import-light portable API organized by cooperative primitive family.

Leaf modules own public argument capture and validation; this facade preserves
the documented root export order and compiler-backend marker contract. It does
not own semantic lowering, provider rendering, or backend compiler state.
"""

from .exchange import exchange  # noqa: F401
from .load_store import load, store  # noqa: F401
from .reduce import reduce, sum  # noqa: F401
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
    "reduce",
    "scan",
    "shuffle",
    "store",
    "sum",
]
