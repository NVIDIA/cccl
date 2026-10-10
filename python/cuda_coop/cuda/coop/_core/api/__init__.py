# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Collect the common kernel API and identify its constructor functions.

The root package exports these names. Compiler adapters use their identities
to recognize cooperative operations without changing the public call syntax.
"""

from .load_store import load, store
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

# Constructors and group factories bypass the operation registration decorator.
# Compiler rewrites require this tag and the exported function identity, so
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
    "load",
    "store",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]
