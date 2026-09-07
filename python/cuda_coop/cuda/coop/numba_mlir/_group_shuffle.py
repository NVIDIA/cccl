# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first shuffle marker for Numba-CUDA-MLIR.

This file is the semantic entry point for shuffle navigation.  CUB-specific
mode normalization and provider construction are lower-level responsibilities.
"""

from __future__ import annotations

from typing import Any

from cuda.coop._core.block import BlockShuffleMode

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation("shuffle")
def shuffle(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: Any = BlockShuffleMode.DOWN,
    distance: Any = 1,
    block_prefix: Any = None,
    block_suffix: Any = None,
) -> Any:
    """Shuffle scalar values or fixed-size per-thread tiles within a block."""

    return group_primitive_marker(
        "shuffle",
        group,
        value,
        mode=mode,
        distance=distance,
        block_prefix=block_prefix,
        block_suffix=block_suffix,
    )


__all__ = ["shuffle"]
