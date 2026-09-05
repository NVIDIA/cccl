# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable discontinuity entry point.

The frontend validates the shared heads/tails selector and delegates to the
active compiler backend. Result-shape planning and provider rendering remain
outside this module.
"""

from __future__ import annotations

from typing import Any

from ..block import BlockDiscontinuityMode
from ..thread_group import ThreadGroup
from ._dispatch import (
    _DISCONTINUITY_MODES,
    _group_primitive_marker,
    _portable_selector,
)


def discontinuity(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: Any = BlockDiscontinuityMode.HEADS,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Compute groupwise discontinuities through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    mode = _portable_selector("discontinuity", "mode", mode, _DISCONTINUITY_MODES)

    return _group_primitive_marker(
        "discontinuity",
        group,
        value,
        mode=mode,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
    )


__all__ = ["discontinuity"]
