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
    """Compute neighbor differences with an optional device operator.

    Shared parameters, participation, boundaries, partial tiles, and scratch
    behavior follow :func:`cuda.coop.adjacent_difference`.

    Additional parameters
    ---------------------
    values : ThreadDataLike or local array
        Fixed-size local arrays are accepted in addition to ThreadData.
    difference_op : callable, optional
        Stateless device-compilable ``difference_op(current, neighbor)``
        returning the input dtype. ``None`` selects subtraction. The call
        receives the previous neighbor for left differences and the next
        neighbor for right differences.

    Returns
    -------
    ThreadDataLike
        The fresh payload described by the common operation, including when
        the input is a local array.
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
    """Flag adjacent items with an optional device predicate.

    Shared parameters, participation, boundaries, and scratch behavior follow
    :func:`cuda.coop.discontinuity`.

    Additional parameters
    ---------------------
    values : ThreadDataLike or local array
        Fixed-size local arrays are accepted in addition to ThreadData.
    flag_op : callable, optional
        Stateless device-compilable binary predicate. Heads evaluate
        ``flag_op(previous, current)``; tails evaluate
        ``flag_op(current, next)``. ``None`` selects inequality.

    Returns
    -------
    ThreadDataLike or tuple of ThreadDataLike
        The common operation's int32 flag payload, or ``(heads, tails)``
        for ``mode="heads_and_tails"``, including for local-array inputs.
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
