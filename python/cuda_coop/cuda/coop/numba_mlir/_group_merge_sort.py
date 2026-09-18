# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Merge Sort entry points."""

from __future__ import annotations

from typing import Any

from .._core.api._payload import (
    TempStorageLike,
    ThreadDataLike,
    _ReadableThreadDataLike,
)
from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation(
    "merge_sort_keys", family_module="cuda.coop.numba_mlir._compiler._group_merge_sort"
)
def merge_sort_keys(
    group: ThreadGroup,
    keys: _ReadableThreadDataLike[Any],
    /,
    *,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: TempStorageLike | None = None,
    compare_op: Any = None,
) -> ThreadDataLike[Any]:
    """Return sorted blocked payloads without modifying the inputs."""

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


@group_operation(
    "merge_sort_pairs", family_module="cuda.coop.numba_mlir._compiler._group_merge_sort"
)
def merge_sort_pairs(
    group: ThreadGroup,
    keys: _ReadableThreadDataLike[Any],
    values: _ReadableThreadDataLike[Any],
    /,
    *,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: TempStorageLike | None = None,
    compare_op: Any = None,
) -> tuple[ThreadDataLike[Any], ThreadDataLike[Any]]:
    """Return sorted blocked payloads without modifying the inputs."""

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
