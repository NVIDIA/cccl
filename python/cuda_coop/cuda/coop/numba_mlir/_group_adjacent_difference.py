# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first adjacent-difference marker for Numba-CUDA-MLIR.

The marker retains portable defaults while compiler lowering validates the
physical-block restriction and materializes the comparison callable.
"""

from __future__ import annotations

from typing import Any

from cuda.coop._core.block import BlockAdjacentDifferenceDirection

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation("adjacent_difference")
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
    difference_op: Any = None,
) -> Any:
    """Compute adjacent differences across a complete physical block."""

    return group_primitive_marker(
        "adjacent_difference",
        group,
        value,
        direction=direction,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
        difference_op=difference_op,
    )


__all__ = ["adjacent_difference"]
