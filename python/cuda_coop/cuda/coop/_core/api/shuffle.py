# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative shuffle entry point.

This module validates the common shuffle modes and static distance contract
before delegation. CUB specialization and compiler-specific payload handling
remain in the semantic planner and backend.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

from ..block import BlockShuffleMode
from ..thread_group import ThreadGroup
from ._dispatch import (
    _SHUFFLE_MODES,
    _backend_module_name,
    _group_primitive_marker,
    _portable_selector,
)
from ._payload import _validate_common_numeric_value


def shuffle(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: Any = BlockShuffleMode.DOWN,
    distance: Any = 1,
) -> Any:
    """Unit-shift a block payload through the compiler-selected backend.

    The portable API accepts only a fixed-size per-thread payload, ``up`` or
    ``down`` mode, and the unit distance ``1``. The vacated first item for
    ``up`` or last item for ``down`` is undefined. Use the qualified
    ``cuda.coop.<backend>`` API for scalar, boundary-output, or other-distance
    behavior.
    """

    mode = _portable_selector("shuffle", "mode", mode, _SHUFFLE_MODES)
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "shuffle", "value", value, require_thread_data=True
        )
        normalized_distance = getattr(distance, "value", distance)
        if (
            isinstance(normalized_distance, bool)
            or not isinstance(normalized_distance, Integral)
            or int(normalized_distance) != 1
        ):
            raise ValueError(
                "cuda.coop.shuffle distance must be exactly 1 in the portable API; "
                "use a backend-qualified import for other distances"
            )
        distance = 1

    return _group_primitive_marker(
        "shuffle",
        group,
        value,
        mode=mode,
        distance=distance,
    )


__all__ = ["shuffle"]
