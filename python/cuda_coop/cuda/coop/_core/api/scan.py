# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative scan entry points."""

from __future__ import annotations

from typing import Any

from ..dtype_policy import validate_portable_integer_value_dtype_name
from ..scan import normalize_scan_operator_alias
from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _group_primitive_marker,
    _portable_group_operation,
    _portable_selector,
    _validate_portable_operation_group,
)
from ._payload import (
    _ReadableThreadDataLike,
    _validate_common_numeric_scalar,
    _validate_common_numeric_value,
    _validate_common_temp_storage,
)

_PORTABLE_SCAN_GROUP_KINDS = ("block", "warp", "threads_within_warp")
_PORTABLE_SCAN_MODES = frozenset({"exclusive", "inclusive"})
_PORTABLE_SCAN_ALGORITHMS = frozenset({"raking", "raking_memoize", "warp_scans"})
_BITWISE_OPERATORS = frozenset({"bit_and", "bit_or", "bit_xor"})
_WARP_GROUP_KINDS = frozenset({"warp", "threads_within_warp"})


def _portable_scan_operator(operation: str, value: Any) -> Any:
    if _backend_module_name() is None or value is None:
        return value
    if not isinstance(value, str):
        raise TypeError(
            f"cuda.coop.{operation} scan_op must be a string; use a "
            "backend-qualified import for custom operators"
        )
    operator = normalize_scan_operator_alias(value)
    if operator is None:
        choices = "bit_and, bit_or, bit_xor, max, min, multiplies, sum"
        raise ValueError(
            f"cuda.coop.{operation} scan_op must be one of: "
            f"{choices}; use a backend-qualified import for custom operators"
        ) from None
    return operator


def _validate_portable_scan_value(
    operation: str,
    value: Any,
    scan_op: Any,
) -> str:
    dtype_name = _validate_common_numeric_value(
        operation,
        "value",
        value,
        allow_readonly_thread_data=True,
    )
    assert dtype_name is not None
    if scan_op in _BITWISE_OPERATORS:
        validate_portable_integer_value_dtype_name(
            dtype_name,
            operation=operation,
            parameter="value",
        )
    return dtype_name


def _validate_portable_scan_options(
    operation: str,
    group: ThreadGroup,
    value: Any,
    *,
    mode: str,
    scan_op: Any,
    initial_value: Any,
    algorithm: Any,
    temp_storage: Any,
) -> None:
    if _backend_module_name() is None:
        return
    _validate_portable_operation_group(operation, group)
    if mode == "inclusive" and initial_value is not None:
        raise ValueError(
            f"cuda.coop.{operation} initial_value is not supported for inclusive scans"
        )
    if mode == "exclusive" and scan_op not in {None, "sum"}:
        if initial_value is None:
            raise ValueError(
                f"cuda.coop.{operation} non-sum exclusive scans require initial_value"
            )
    if initial_value is not None:
        _validate_common_numeric_scalar(operation, "initial_value", initial_value)
    if group.kind in _WARP_GROUP_KINDS:
        if isinstance(value, _ReadableThreadDataLike):
            raise TypeError(
                f"cuda.coop.{operation} value must be a portable numeric scalar "
                "for warp scans"
            )
        if algorithm is not None:
            raise ValueError(
                f"cuda.coop.{operation} algorithm selection is supported only "
                "for blocks"
            )
        if temp_storage is not None:
            raise ValueError(
                f"cuda.coop.{operation} temp_storage is supported only for blocks"
            )
    elif temp_storage is not None:
        _validate_common_temp_storage(operation, temp_storage)


def _scan_call(
    operation: str,
    group: ThreadGroup,
    value: Any,
    *,
    mode: str,
    scan_op: Any,
    initial_value: Any,
    algorithm: Any,
    temp_storage: Any,
) -> Any:
    scan_op = _portable_scan_operator(operation, scan_op)
    if _backend_module_name() is not None:
        _validate_portable_scan_value(operation, value, scan_op)
    _validate_portable_scan_options(
        operation,
        group,
        value,
        mode=mode,
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )
    kwargs = {
        "algorithm": algorithm,
        "temp_storage": temp_storage,
    }
    if operation in {"scan", "exclusive_scan", "inclusive_scan"}:
        kwargs["scan_op"] = scan_op
    if operation in {"scan", "exclusive_scan"}:
        kwargs["initial_value"] = initial_value
    if operation == "scan":
        kwargs["mode"] = mode
    return _group_primitive_marker(operation, group, value, **kwargs)


@_portable_group_operation("scan", group_kinds=_PORTABLE_SCAN_GROUP_KINDS)
def scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Scan values across a block or warp group through the active backend."""

    mode = _portable_selector(
        "scan",
        "mode",
        mode,
        _PORTABLE_SCAN_MODES,
    )
    algorithm = _portable_selector(
        "scan",
        "algorithm",
        algorithm,
        _PORTABLE_SCAN_ALGORITHMS,
        allow_none=True,
    )
    return _scan_call(
        "scan",
        group,
        value,
        mode=mode,
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


@_portable_group_operation(
    "exclusive_sum",
    group_kinds=_PORTABLE_SCAN_GROUP_KINDS,
)
def exclusive_sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return an out-of-place exclusive prefix sum."""

    algorithm = _portable_selector(
        "exclusive_sum",
        "algorithm",
        algorithm,
        _PORTABLE_SCAN_ALGORITHMS,
        allow_none=True,
    )
    return _scan_call(
        "exclusive_sum",
        group,
        value,
        mode="exclusive",
        scan_op=None,
        initial_value=None,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


@_portable_group_operation(
    "inclusive_sum",
    group_kinds=_PORTABLE_SCAN_GROUP_KINDS,
)
def inclusive_sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return an out-of-place inclusive prefix sum."""

    algorithm = _portable_selector(
        "inclusive_sum",
        "algorithm",
        algorithm,
        _PORTABLE_SCAN_ALGORITHMS,
        allow_none=True,
    )
    return _scan_call(
        "inclusive_sum",
        group,
        value,
        mode="inclusive",
        scan_op=None,
        initial_value=None,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


@_portable_group_operation(
    "exclusive_scan",
    group_kinds=_PORTABLE_SCAN_GROUP_KINDS,
)
def exclusive_scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return an out-of-place exclusive scan."""

    algorithm = _portable_selector(
        "exclusive_scan",
        "algorithm",
        algorithm,
        _PORTABLE_SCAN_ALGORITHMS,
        allow_none=True,
    )
    return _scan_call(
        "exclusive_scan",
        group,
        value,
        mode="exclusive",
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


@_portable_group_operation(
    "inclusive_scan",
    group_kinds=_PORTABLE_SCAN_GROUP_KINDS,
)
def inclusive_scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    scan_op: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return an out-of-place inclusive scan."""

    algorithm = _portable_selector(
        "inclusive_scan",
        "algorithm",
        algorithm,
        _PORTABLE_SCAN_ALGORITHMS,
        allow_none=True,
    )
    return _scan_call(
        "inclusive_scan",
        group,
        value,
        mode="inclusive",
        scan_op=scan_op,
        initial_value=None,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


__all__ = [
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "scan",
]
