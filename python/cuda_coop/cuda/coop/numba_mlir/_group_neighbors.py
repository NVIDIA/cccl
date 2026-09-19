# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba block neighbor APIs with optional stateless binary operators."""

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
    "adjacent_difference",
    family_module="cuda.coop.numba_mlir._compiler._group_neighbors",
)
def adjacent_difference(
    group: ThreadGroup,
    values: _ReadableThreadDataLike[Any],
    /,
    *,
    direction: str = "left",
    valid_items: Any = None,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    temp_storage: TempStorageLike | None = None,
    difference_op: Any = None,
) -> ThreadDataLike[Any]:
    """Common Adjacent Difference semantics with a custom binary operator.

    difference_op(current, neighbor) must be a stateless device-compilable
    callable returning the input dtype. None selects subtraction. Fixed-size
    local arrays are also accepted. Inputs are preserved, including the
    invalid suffix. Groups must be complete blocks.
    """
    return group_primitive_marker(
        "adjacent_difference",
        group,
        values,
        direction=direction,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
        difference_op=difference_op,
    )


@group_operation(
    "discontinuity", family_module="cuda.coop.numba_mlir._compiler._group_neighbors"
)
def discontinuity(
    group: ThreadGroup,
    values: _ReadableThreadDataLike[Any],
    /,
    *,
    mode: str = "heads",
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    temp_storage: TempStorageLike | None = None,
    flag_op: Any = None,
) -> ThreadDataLike[Any] | tuple[ThreadDataLike[Any], ThreadDataLike[Any]]:
    """Common full-tile Discontinuity semantics with a binary predicate.

    flag_op(previous, current) determines heads; flag_op(current, next)
    determines tails. It must be a stateless device-compilable predicate.
    None selects inequality. Fixed-size local arrays are also accepted.
    The result flag dtype is int32; heads_and_tails returns (heads, tails).
    """
    return group_primitive_marker(
        "discontinuity",
        group,
        values,
        mode=mode,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
        flag_op=flag_op,
    )


__all__ = ["adjacent_difference", "discontinuity"]
