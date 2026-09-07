# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Merge Sort markers for Numba-CUDA-MLIR.

These markers define fresh-result semantics.  Callable hashing, sentinel
validation, and block/warp provider selection are compiler responsibilities.
"""

from __future__ import annotations

from typing import Any

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation("merge_sort_keys")
def merge_sort_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: Any = None,
    compare_op: Any = None,
) -> Any:
    """Merge-sort keys across a block or warp group."""

    return group_primitive_marker(
        "merge_sort_keys",
        group,
        keys,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
        compare_op=compare_op,
    )


@group_operation("merge_sort_pairs")
def merge_sort_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: Any = None,
    compare_op: Any = None,
) -> tuple[Any, Any]:
    """Merge-sort keys and associated values across a block or warp group."""

    return group_primitive_marker(
        "merge_sort_pairs",
        group,
        keys,
        values,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
        compare_op=compare_op,
    )


__all__ = ["merge_sort_keys", "merge_sort_pairs"]
