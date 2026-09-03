# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared payload validation for portable root calls.

Family frontends use these import-light helpers before delegating to a compiler
backend. The validators define the conservative portable contract and do not
infer backend-specific types or construct lowering plans.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Protocol, TypeVar, runtime_checkable

from ..dtype_policy import (
    validate_portable_integer_value_dtype_name,
    validate_portable_numeric_dtype_name,
)

_ItemT = TypeVar("_ItemT")


@runtime_checkable
class _ReadableThreadDataLike(Protocol[_ItemT]):
    """Readable fixed-size per-thread payload understood by supported backends."""

    items_per_thread: int
    dtype: Any | None

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _ItemT: ...


@runtime_checkable
class ThreadDataLike(_ReadableThreadDataLike[_ItemT], Protocol[_ItemT]):
    """Mutable fixed-size per-thread payload understood by supported backends."""

    def __setitem__(self, index: int, value: _ItemT) -> None: ...


@runtime_checkable
class TempStorageLike(Protocol):
    """Explicit cooperative scratch descriptor understood by supported backends."""

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool | None
    sharing: str


def _validate_common_temp_storage(operation: str, value: Any) -> None:
    """Require the portable explicit temporary-storage representation."""

    if isinstance(value, TempStorageLike):
        return
    raise TypeError(
        f"cuda.coop.{operation} temp_storage must satisfy TempStorageLike; "
        "construct it with cuda.coop.TempStorage()"
    )


def _validate_common_thread_data_payload(
    operation: str,
    parameter: str,
    value: Any,
    *,
    allow_readonly: bool = False,
) -> None:
    """Require the portable fixed-size payload representation."""

    protocol = _ReadableThreadDataLike if allow_readonly else ThreadDataLike
    if isinstance(value, protocol):
        return
    raise TypeError(
        f"cuda.coop.{operation} requires a fixed-size ThreadData {parameter} "
        "payload; use a backend-qualified import for backend-specific scalar "
        "or register payloads"
    )


def _common_thread_data_extent(
    operation: str,
    parameter: str,
    value: _ReadableThreadDataLike[Any],
) -> int:
    """Return one positive trace-static portable payload extent."""

    extent = value.items_per_thread
    if isinstance(extent, bool) or not isinstance(extent, Integral):
        raise TypeError(
            f"cuda.coop.{operation} {parameter}.items_per_thread must be a "
            "compile-time positive integer"
        )
    normalized = int(extent)
    if normalized <= 0:
        raise ValueError(
            f"cuda.coop.{operation} {parameter}.items_per_thread must be a "
            "compile-time positive integer"
        )
    if len(value) != normalized:
        raise ValueError(
            f"cuda.coop.{operation} {parameter}.items_per_thread must match "
            "the payload item count"
        )
    return normalized


def _common_payload_dtype(
    operation: str,
    parameter: str,
    value: _ReadableThreadDataLike[Any],
) -> Any:
    """Return the declared or item-inferred dtype for a portable payload."""

    dtype = value.dtype
    if dtype is None and len(value) > 0:
        item = value[0]
        dtype = getattr(item, "dtype", None)
        if dtype is None:
            dtype = type(item)
        dtype_name = _common_numeric_dtype_name(dtype)
        for index in range(1, len(value)):
            item = value[index]
            item_dtype = getattr(item, "dtype", None)
            if item_dtype is None:
                item_dtype = type(item)
            if _common_numeric_dtype_name(item_dtype) != dtype_name:
                raise TypeError(
                    f"cuda.coop.{operation} {parameter} items must have one "
                    "common dtype"
                )
    return dtype


def _common_numeric_dtype_name(dtype: Any) -> str:
    """Normalize a Python, NumPy, or structural compiler numeric dtype."""

    if dtype is int:
        return "int32"
    if dtype is float:
        return "float32"

    dtype_name = getattr(dtype, "name", None)
    if not isinstance(dtype_name, str):
        dtype_name = getattr(dtype, "__name__", None)
    if isinstance(dtype_name, str):
        dtype_name = dtype_name.lower()
        for prefix in ("int", "uint", "float", "complex"):
            suffix = dtype_name[len(prefix) :] if dtype_name.startswith(prefix) else ""
            if suffix.isdigit():
                return dtype_name
        if dtype_name in {"bool", "boolean"}:
            return dtype_name

    width = getattr(dtype, "width", None)
    if width is None:
        width = getattr(dtype, "bitwidth", None)
    signed = getattr(dtype, "signed", None)
    if isinstance(width, Integral) and not isinstance(width, bool):
        if isinstance(signed, bool):
            return f"{'int' if signed else 'uint'}{int(width)}"

    if isinstance(dtype_name, str):
        return dtype_name
    return str(dtype).lower()


def _is_common_numeric_scalar(value: Any) -> bool:
    """Return whether ``value`` has the portable scalar representation."""

    if type(value) in {int, float}:
        return True
    value_type = type(value)
    if (value_type.__module__ or "").split(".", 1)[0] == "numpy":
        return value_type.__name__ in {
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "float32",
            "float64",
        }
    width = getattr(value, "width", None)
    return (
        isinstance(width, Integral)
        and not isinstance(width, bool)
        and width > 0
        and getattr(value, "dtype", None) is not None
        and callable(getattr(value, "ir_value", None))
    )


def _validate_common_numeric_scalar(
    operation: str,
    parameter: str,
    value: Any,
) -> str:
    """Require one portable numeric scalar rather than a thread payload."""

    if not _is_common_numeric_scalar(value):
        raise TypeError(
            f"cuda.coop.{operation} {parameter} must be a portable numeric "
            "scalar; use a backend-qualified import for backend-specific values"
        )
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = type(value)
    return validate_portable_numeric_dtype_name(
        _common_numeric_dtype_name(dtype),
        operation=operation,
        parameter=parameter,
    )


def _validate_common_integer_value(
    operation: str,
    parameter: str,
    value: Any,
) -> int | None:
    """Validate a static or compiler-owned portable integer value."""

    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if not (
        _is_common_numeric_scalar(value)
        and isinstance(getattr(value, "signed", None), bool)
    ):
        raise TypeError(
            f"cuda.coop.{operation} {parameter} must be a portable integer value"
        )
    dtype = getattr(value, "dtype", None)
    assert dtype is not None
    validate_portable_integer_value_dtype_name(
        _common_numeric_dtype_name(dtype),
        operation=operation,
        parameter=parameter,
    )
    return None


def _validate_common_numeric_value(
    operation: str,
    parameter: str,
    value: Any,
    *,
    allow_untyped_thread_data: bool = False,
    allow_readonly_thread_data: bool = False,
    require_thread_data: bool = False,
) -> str | None:
    """Require one portable scalar or fixed-size per-thread payload."""

    protocol = _ReadableThreadDataLike if allow_readonly_thread_data else ThreadDataLike
    if isinstance(value, protocol):
        _common_thread_data_extent(operation, parameter, value)
        if value.dtype is None and allow_untyped_thread_data:
            return
        dtype = _common_payload_dtype(operation, parameter, value)
    else:
        if require_thread_data:
            _validate_common_thread_data_payload(
                operation,
                parameter,
                value,
                allow_readonly=allow_readonly_thread_data,
            )
            raise AssertionError("unreachable")
        if not _is_common_numeric_scalar(value):
            raise TypeError(
                f"cuda.coop.{operation} requires the portable API's numeric "
                f"scalar or fixed-size ThreadData {parameter} payload; use a "
                "backend-qualified import for backend-specific payloads"
            )
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            dtype = type(value)
    return validate_portable_numeric_dtype_name(
        _common_numeric_dtype_name(dtype),
        operation=operation,
        parameter=parameter,
    )


__all__ = ["TempStorageLike", "ThreadDataLike"]
