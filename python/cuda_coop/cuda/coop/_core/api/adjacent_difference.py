# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable adjacent-difference entry point.

This frontend normalizes the shared direction selector and delegates the call
without owning CUB specialization or compiler lifecycle. Boundary operands
remain explicit in the portable API contract.
"""

from __future__ import annotations

from typing import Any

from ..block import BlockAdjacentDifferenceDirection
from ..thread_group import ThreadGroup
from ._dispatch import (
    _ADJACENT_DIFFERENCE_DIRECTIONS,
    _backend_module_name,
    _group_primitive_marker,
    _portable_selector,
)
from ._payload import _validate_common_numeric_value


def adjacent_difference(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    direction: Any = BlockAdjacentDifferenceDirection.LEFT,
    valid_items: Any = None,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Compute groupwise differences through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    direction = _portable_selector(
        "adjacent_difference",
        "direction",
        direction,
        _ADJACENT_DIFFERENCE_DIRECTIONS,
    )
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "adjacent_difference", "value", value, require_thread_data=True
        )

    return _group_primitive_marker(
        "adjacent_difference",
        group,
        value,
        direction=direction,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
    )


__all__ = ["adjacent_difference"]
