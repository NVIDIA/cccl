# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first discontinuity marker for Numba-CUDA-MLIR.

The public signature is kept separate from block provider construction so
planner diagnostics do not depend on private implementation module names.
"""

from __future__ import annotations

from typing import Any

from cuda.coop._core.block import BlockDiscontinuityMode

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation("discontinuity")
def discontinuity(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: Any = BlockDiscontinuityMode.HEADS,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    temp_storage: Any = None,
    flag_op: Any = None,
) -> Any:
    """Flag adjacent-item boundaries across a complete physical block."""

    return group_primitive_marker(
        "discontinuity",
        group,
        value,
        mode=mode,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
        flag_op=flag_op,
    )


__all__ = ["discontinuity"]
