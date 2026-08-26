# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first exchange marker for Numba-CUDA-MLIR.

This module owns the portable exchange signature; block/warp CUB selection is
performed later from the resolved group hierarchy.
"""

from __future__ import annotations

from typing import Any

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation("exchange")
def exchange(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: str = "striped_to_blocked",
    ranks: Any = None,
    valid_flags: Any = None,
    warp_time_slicing: bool = False,
) -> Any:
    """Rearrange a fixed-size per-thread tile within a group.

    ``warp_time_slicing`` is block-only and is unavailable for guarded or
    flagged scatter-to-striped modes.
    """

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
