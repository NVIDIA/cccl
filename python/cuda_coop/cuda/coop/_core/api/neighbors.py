# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common block Adjacent Difference and Discontinuity APIs."""

from __future__ import annotations

from typing import Any

from ..block.neighbors import validate_neighbor_options
from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _common_group_operation,
    _group_primitive_marker,
)
from ._payload import (
    TempStorageLike,
    ThreadDataLike,
    _ReadableThreadDataLike,
    _validate_common_numeric_value,
    _validate_common_temp_storage,
)


def _validate_payload(operation, values, temp_storage):
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            operation,
            "values",
            values,
            require_thread_data=True,
            allow_readonly_thread_data=True,
        )
        if temp_storage is not None:
            _validate_common_temp_storage(operation, temp_storage)


@_common_group_operation("adjacent_difference", group_kinds=("block",))
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
) -> ThreadDataLike[Any]:
    """Return blocked neighbor differences without modifying the input.

    A complete block processes fixed-size numeric ThreadData. Left subtracts
    the previous element from the current element; right subtracts the next
    element. A missing neighbor leaves the boundary input unchanged unless
    the matching tile boundary scalar is supplied. Boundary scalars must be
    uniform across the block and match the input dtype.

    valid_items is a uniform count in [0, block size * items per thread].
    The invalid suffix is copied unchanged. Right partial tiles cannot use
    tile_successor_item. All input slots must be initialized, including the
    suffix. Temporary storage is automatic unless temp_storage is supplied.
    Use the qualified API for a custom binary difference operator.
    """
    validate_neighbor_options(
        "adjacent_difference",
        direction,
        partial=valid_items is not None,
        predecessor=tile_predecessor_item is not None,
        successor=tile_successor_item is not None,
    )
    _validate_payload("adjacent_difference", values, temp_storage)
    return _group_primitive_marker(
        "adjacent_difference",
        group,
        values,
        direction=direction,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
    )


@_common_group_operation("discontinuity", group_kinds=("block",))
def discontinuity(
    group: ThreadGroup,
    values: _ReadableThreadDataLike[Any],
    /,
    *,
    mode: str = "heads",
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[Any] | tuple[ThreadDataLike[Any], ThreadDataLike[Any]]:
    """Flag unequal adjacent items in a full, blocked block tile.

    Return int32 ThreadData for heads or tails, or (heads, tails) for
    heads_and_tails. Results have the input extent; the input is preserved.
    An absent tile predecessor forces the first head to one; an absent
    successor forces the last tail to one. Supplied boundary scalars must
    match the input dtype and be uniform across the block. Only the boundary
    arguments relevant to the selected mode are accepted.

    Partial tiles are unsupported: padding becomes part of the tile and can
    change its flags, including the last valid tail. Storage is automatic
    unless temp_storage is supplied. Use the qualified API for a custom
    binary predicate.
    """
    validate_neighbor_options(
        "discontinuity",
        mode,
        predecessor=tile_predecessor_item is not None,
        successor=tile_successor_item is not None,
    )
    _validate_payload("discontinuity", values, temp_storage)
    return _group_primitive_marker(
        "discontinuity",
        group,
        values,
        mode=mode,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
    )


__all__ = ["adjacent_difference", "discontinuity"]
