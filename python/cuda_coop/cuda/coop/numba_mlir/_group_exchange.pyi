# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exchange signatures for block and warp groups."""

from typing import Any

from typing_extensions import TypeVar

from .._typing import ThreadDataLike as _ThreadDataLike
from ._thread_group import ThreadGroup

_ItemT = TypeVar("_ItemT")

def exchange(
    group: ThreadGroup[Any],
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    mode: str = "striped_to_blocked",
    ranks: Any = None,
    valid_flags: Any = None,
    warp_time_slicing: bool = False,
) -> _ThreadDataLike[_ItemT]:
    """Rearrange a fixed-size per-thread tile within a group.

    ``warp_time_slicing`` is block-only and is unavailable for guarded or
    flagged scatter-to-striped modes.
    """
