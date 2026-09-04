# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative exchange entry point."""

from __future__ import annotations

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

_PORTABLE_EXCHANGE_MODES = frozenset(
    {
        "striped_to_blocked",
        "blocked_to_striped",
    }
)


@_portable_group_operation(
    "exchange",
    group_kinds=("block", "warp", "threads_within_warp"),
)
def exchange(
    group: ThreadGroup,
    value: _ReadableThreadDataLike[Any],
    /,
    *,
    mode: Any = "striped_to_blocked",
) -> ThreadDataLike[Any]:
    """Rearrange a per-thread payload within the selected group."""

    mode = _portable_selector(
        "exchange",
        "mode",
        mode,
        _PORTABLE_EXCHANGE_MODES,
    )
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "exchange",
            "value",
            value,
            allow_readonly_thread_data=True,
            require_thread_data=True,
        )
    return _group_primitive_marker(
        "exchange",
        group,
        value,
        mode=mode,
    )


__all__ = ["exchange"]
