# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative exchange entry point.

The frontend admits only the shared exchange modes before delegating to an
active backend. Group resolution and CUB block/warp specialization are owned by
the portable planner and backend compiler layers.
"""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _EXCHANGE_MODES,
    _backend_module_name,
    _group_primitive_marker,
    _portable_selector,
)
from ._payload import _validate_common_numeric_value


def exchange(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: str = "striped_to_blocked",
) -> Any:
    """Exchange values across a group through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    mode = _portable_selector("exchange", "mode", mode, _EXCHANGE_MODES)
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "exchange", "value", value, require_thread_data=True
        )

    return _group_primitive_marker(
        "exchange",
        group,
        value,
        mode=mode,
    )


__all__ = ["exchange"]
