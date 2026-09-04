# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative Shuffle entry point."""

from __future__ import annotations

from enum import Enum
from numbers import Integral
from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _group_primitive_marker,
    _portable_group_operation,
    _portable_selector,
)
from ._payload import (
    ThreadDataLike,
    _ReadableThreadDataLike,
    _validate_common_numeric_value,
)

_PORTABLE_SHUFFLE_MODES = frozenset({"down", "up"})


@_portable_group_operation(
    "shuffle",
    group_kinds=("block",),
)
def shuffle(
    group: ThreadGroup,
    value: _ReadableThreadDataLike[Any],
    /,
    *,
    mode: Any = "down",
    distance: Any = 1,
) -> ThreadDataLike[Any]:
    """Unit-shift a per-thread payload within a complete block."""

    mode = _portable_selector(
        "shuffle",
        "mode",
        mode,
        _PORTABLE_SHUFFLE_MODES,
    )
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "shuffle",
            "value",
            value,
            allow_readonly_thread_data=True,
            require_thread_data=True,
        )
        if (
            isinstance(distance, (bool, Enum))
            or not isinstance(distance, Integral)
            or int(distance) != 1
        ):
            raise ValueError(
                "cuda.coop.shuffle distance must be exactly 1 in the portable "
                "API; use cuda.coop.numba_mlir for scalar Shuffle"
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
