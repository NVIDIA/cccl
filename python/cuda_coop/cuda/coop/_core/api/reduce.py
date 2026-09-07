# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative reduction entry points.

The root functions retain the conservative common reduction controls and
delegate to the active backend. Semantic planning and CUDAX/CUB selection live
in the portable group family and backend lowering layers.
"""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _REDUCE_ALGORITHMS,
    _backend_module_name,
    _group_primitive_marker,
    _portable_operator,
    _portable_selector,
)
from ._payload import (
    _ReadableThreadDataLike,
    _validate_common_integer_value,
    _validate_common_numeric_operator,
    _validate_common_numeric_value,
)

_PARTIAL_REDUCTION_GROUP_KINDS = frozenset({"block", "warp", "threads_within_warp"})


def _validate_portable_reduce_options(
    operation: str,
    group: ThreadGroup,
    value: Any,
    *,
    broadcast: bool,
    valid_items: Any,
    algorithm: Any,
) -> None:
    """Enforce the conservative portable reduction overload matrix."""

    if _backend_module_name() is None:
        return
    if not isinstance(broadcast, bool):
        raise TypeError(f"cuda.coop.{operation} broadcast must be a bool")
    if valid_items is not None:
        static_valid_items = _validate_common_integer_value(
            operation,
            "valid_items",
            valid_items,
        )
        if (
            not isinstance(group, ThreadGroup)
            or group.kind not in _PARTIAL_REDUCTION_GROUP_KINDS
        ):
            raise ValueError(
                f"cuda.coop.{operation} valid_items requires a block or warp group"
            )
        if broadcast is not False:
            raise ValueError(
                f"cuda.coop.{operation} valid_items requires broadcast=False"
            )
        if isinstance(value, _ReadableThreadDataLike):
            raise ValueError(
                f"cuda.coop.{operation} valid_items supports scalar values only"
            )
        if static_valid_items is not None:
            if static_valid_items < 1:
                raise ValueError(
                    f"cuda.coop.{operation} valid_items must be at least 1"
                )
            if group.static_size is not None and static_valid_items > group.static_size:
                raise ValueError(
                    f"cuda.coop.{operation} valid_items {static_valid_items} "
                    f"exceeds group size {group.static_size}"
                )
    if algorithm is not None:
        if not isinstance(group, ThreadGroup) or group.kind != "block":
            raise ValueError(
                f"cuda.coop.{operation} algorithm selection requires a block group"
            )
        if broadcast is not False:
            raise ValueError(
                f"cuda.coop.{operation} algorithm selection requires broadcast=False"
            )


def reduce(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    binary_op: Any = None,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Reduce values across a group through the compiler-selected backend.

    With ``broadcast=False``, only group rank zero has a defined result; other
    members receive an implementation placeholder and must not consume it.
    Every member must still participate in the collective.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "reduce",
        "algorithm",
        algorithm,
        _REDUCE_ALGORITHMS,
        allow_none=True,
    )
    binary_op = _portable_operator("reduce", "binary_op", binary_op)
    if _backend_module_name() is not None:
        _validate_common_numeric_operator(
            "reduce",
            "value",
            value,
            binary_op,
            allow_readonly_thread_data=True,
        )
    _validate_portable_reduce_options(
        "reduce",
        group,
        value,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )

    return _group_primitive_marker(
        "reduce",
        group,
        value,
        binary_op=binary_op,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


def sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Sum values across a group through the compiler-selected backend.

    With ``broadcast=False``, only group rank zero has a defined result; other
    members receive an implementation placeholder and must not consume it.
    Every member must still participate in the collective.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "sum",
        "algorithm",
        algorithm,
        _REDUCE_ALGORITHMS,
        allow_none=True,
    )
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "sum",
            "value",
            value,
            allow_readonly_thread_data=True,
        )
    _validate_portable_reduce_options(
        "sum",
        group,
        value,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )

    return _group_primitive_marker(
        "sum",
        group,
        value,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


__all__ = ["reduce", "sum"]
