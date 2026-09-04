# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Shuffle marker for Numba-CUDA-MLIR."""

from __future__ import annotations

from typing import Any

from .._core.api._payload import ThreadDataLike, _ReadableThreadDataLike
from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation(
    "shuffle",
    family_module="cuda.coop.numba_mlir._compiler._group_shuffle",
)
def shuffle(
    group: ThreadGroup,
    value: _ReadableThreadDataLike[Any],
    /,
    *,
    mode: Any = "down",
    distance: Any = 1,
) -> ThreadDataLike[Any]:
    """Shuffle a scalar or fixed-size per-thread payload within a block."""

    return group_primitive_marker(
        "shuffle",
        group,
        value,
        mode=mode,
        distance=distance,
    )


__all__ = ["shuffle"]
