# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Exchange marker for Numba-CUDA-MLIR."""

from __future__ import annotations

from typing import Any

from .._core.api._payload import ThreadDataLike, _ReadableThreadDataLike
from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation(
    "exchange",
    family_module="cuda.coop.numba_mlir._compiler._group_exchange",
)
def exchange(
    group: ThreadGroup,
    value: _ReadableThreadDataLike[Any],
    /,
    *,
    mode: Any = "striped_to_blocked",
    ranks: _ReadableThreadDataLike[Any] | None = None,
    valid_flags: _ReadableThreadDataLike[Any] | None = None,
    warp_time_slicing: bool = False,
) -> ThreadDataLike[Any]:
    """Rearrange a fixed-size per-thread payload within a group."""

    return group_primitive_marker(
        "exchange",
        group,
        value,
        mode=mode,
        ranks=ranks,
        valid_flags=valid_flags,
        warp_time_slicing=warp_time_slicing,
    )


__all__ = ["exchange"]
