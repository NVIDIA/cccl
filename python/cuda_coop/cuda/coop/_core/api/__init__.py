# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative operations such as loading and storing a block of values.

This module exports the public functions and group descriptions. Compiler
adapters recognize these functions by identity and lower their kernel calls.
"""

from .load_store import load, store  # noqa: F401
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
# registration decorator. Compiler rewrites recognize their exact exported
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
    "load",
    "store",
]
