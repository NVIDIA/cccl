# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative reduction entry points."""

from __future__ import annotations

from enum import Enum
from typing import Any

from ..dtype_policy import validate_portable_integer_value_dtype_name
from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _group_primitive_marker,
    _portable_group_operation,
    _validate_portable_operation_group,
)
from ._payload import (
    _ReadableThreadDataLike,
    _validate_common_integer_value,
    _validate_common_numeric_value,
)

_PARTIAL_REDUCTION_GROUP_KINDS = frozenset({"block", "warp", "threads_within_warp"})
_PORTABLE_REDUCTION_GROUP_KINDS = (
    "thread",
    "warp",
    "threads_within_warp",
    "block",
    "warps_within_block",
    "cluster",
)
_PORTABLE_REDUCE_ALGORITHMS = frozenset(
    {"raking_commutative_only", "raking", "warp_reductions"}
)
_PORTABLE_OPERATOR_ALIASES = {
    "+": "sum",
    "sum": "sum",
    "add": "sum",
    "plus": "sum",
    "*": "multiplies",
    "mul": "multiplies",
    "multiply": "multiplies",
    "multiplies": "multiplies",
    "min": "min",
    "minimum": "min",
    "max": "max",
    "maximum": "max",
    "&": "bit_and",
    "bit_and": "bit_and",
    "|": "bit_or",
    "bit_or": "bit_or",
    "^": "bit_xor",
    "bit_xor": "bit_xor",
}
_BITWISE_OPERATORS = frozenset({"bit_and", "bit_or", "bit_xor"})


def _is_plain_string(value: Any) -> bool:
    return isinstance(value, str) and not isinstance(value, Enum)


def _portable_reduce_algorithm(operation: str, value: Any) -> Any:
    if _backend_module_name() is None or value is None:
        return value
    if not _is_plain_string(value):
        raise TypeError(f"cuda.coop.{operation} algorithm must be a string")
    token = value.strip().lower().replace("-", "_")
    if token not in _PORTABLE_REDUCE_ALGORITHMS:
        choices = ", ".join(sorted(_PORTABLE_REDUCE_ALGORITHMS))
        raise ValueError(
            f"cuda.coop.{operation} algorithm must be one of: {choices}; "
            "use a backend-qualified import for backend-only controls"
        )
    return token


def _portable_reduce_operator(value: Any) -> Any:
    if _backend_module_name() is None or value is None:
        return value
    if not _is_plain_string(value):
        raise TypeError("cuda.coop.reduce binary_op must be a string")
    token = value.strip().lower().replace("-", "_")
    try:
        return _PORTABLE_OPERATOR_ALIASES[token]
    except KeyError:
        choices = ", ".join(sorted(set(_PORTABLE_OPERATOR_ALIASES.values())))
        raise ValueError(
            "cuda.coop.reduce binary_op must be one of: "
            f"{choices}; use a backend-qualified import for custom operators"
        ) from None


def _validate_portable_reduce_options(
    operation: str,
    group: ThreadGroup,
    value: Any,
    *,
    broadcast: bool,
    valid_items: Any,
    algorithm: Any,
) -> None:
    if _backend_module_name() is None:
        return
    if isinstance(group, ThreadGroup) and group.kind == "grid":
        raise NotImplementedError(
            f"cuda.coop.{operation} does not support grid groups because grid "
            "reduction requires hidden per-launch workspace"
        )
    _validate_portable_operation_group(operation, group)
    if not isinstance(broadcast, bool):
        raise TypeError(f"cuda.coop.{operation} broadcast must be a bool")
    if valid_items is not None:
        static_valid_items = _validate_common_integer_value(
            operation,
            "valid_items",
            valid_items,
        )
        if group.kind not in _PARTIAL_REDUCTION_GROUP_KINDS:
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
        if group.kind != "block":
            raise ValueError(
                f"cuda.coop.{operation} algorithm selection requires a block group"
            )
        if broadcast is not False:
            raise ValueError(
                f"cuda.coop.{operation} algorithm selection requires broadcast=False"
            )


def _validate_portable_reduce_value(
    operation: str,
    value: Any,
    operator: Any,
) -> None:
    dtype_name = _validate_common_numeric_value(
        operation,
        "value",
        value,
        allow_readonly_thread_data=True,
    )
    assert dtype_name is not None
    if operator in _BITWISE_OPERATORS:
        validate_portable_integer_value_dtype_name(
            dtype_name,
            operation=operation,
            parameter="value",
        )


@_portable_group_operation(
    "reduce",
    group_kinds=_PORTABLE_REDUCTION_GROUP_KINDS,
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

    With ``broadcast=False``, only group rank zero has a defined result. Every
    member must still participate in the collective.
    """

    algorithm = _portable_reduce_algorithm("reduce", algorithm)
    binary_op = _portable_reduce_operator(binary_op)
    if _backend_module_name() is not None:
        _validate_portable_reduce_value("reduce", value, binary_op)
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


@_portable_group_operation(
    "sum",
    group_kinds=_PORTABLE_REDUCTION_GROUP_KINDS,
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

    With ``broadcast=False``, only group rank zero has a defined result. Every
    member must still participate in the collective.
    """

    algorithm = _portable_reduce_algorithm("sum", algorithm)
    if _backend_module_name() is not None:
        _validate_portable_reduce_value("sum", value, None)
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
