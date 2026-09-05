# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared payload and dtype validation for portable root calls.

Family frontends use these import-light helpers before delegating to a compiler
backend. The validators define the conservative portable contract and do not
infer backend-specific types or construct lowering plans.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Protocol, TypeVar, runtime_checkable

from ..dtype_policy import (
    validate_portable_integer_key_dtype_name,
    validate_portable_integer_value_dtype_name,
    validate_portable_numeric_dtype_name,
)

_BITWISE_OPERATORS = frozenset({"bit_and", "bit_or", "bit_xor"})

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


def _common_integer_dtype_name(dtype: Any) -> str:
    """Normalize a Python or structural compiler integer dtype to its name."""

    return _common_numeric_dtype_name(dtype)


def _is_common_numeric_scalar(value: Any) -> bool:
    """Return whether ``value`` has the portable scalar representation."""

    if type(value) in {int, float}:
        return True
    value_type = type(value)
    if (value_type.__module__ or "").split(".", 1)[0] == "numpy":
        return value_type.__name__ in {
            "uint8",
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
                f"cuda.coop.{operation} requires a portable numeric scalar or "
                f"fixed-size ThreadData {parameter} payload; use a "
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


def _validate_common_numeric_operator(
    operation: str,
    parameter: str,
    value: Any,
    operator: Any,
    *,
    allow_readonly_thread_data: bool = False,
) -> None:
    """Require an operator supported for the portable payload dtype."""

    dtype_name = _validate_common_numeric_value(
        operation,
        parameter,
        value,
        allow_readonly_thread_data=allow_readonly_thread_data,
    )
    assert dtype_name is not None
    if operator in _BITWISE_OPERATORS:
        validate_portable_integer_value_dtype_name(
            dtype_name,
            operation=operation,
            parameter=parameter,
        )


def _validate_common_run_length_decode_dtype(
    parameter: str,
    value: ThreadDataLike[Any],
    *,
    allow_uint8: bool,
) -> tuple[int, bool]:
    """Require one portable decode dtype and return its width and signedness."""

    dtype_name = _common_integer_dtype_name(
        _common_payload_dtype("run_length_decode", parameter, value)
    )
    validator = (
        validate_portable_integer_value_dtype_name
        if allow_uint8
        else validate_portable_integer_key_dtype_name
    )
    dtype_name = validator(
        dtype_name,
        operation="run_length_decode",
        parameter=parameter,
    )
    return int(
        dtype_name.removeprefix("u").removeprefix("int")
    ), not dtype_name.startswith("u")


def _is_compiler_integer(value: Any) -> bool:
    """Return whether a dynamic value exposes the portable integer protocol."""

    missing = object()
    width = getattr(value, "width", missing)
    signed = getattr(value, "signed", missing)
    dtype = getattr(value, "dtype", missing)
    ir_value = getattr(value, "ir_value", missing)
    return (
        isinstance(width, Integral)
        and not isinstance(width, bool)
        and width > 0
        and isinstance(signed, bool)
        and dtype is not missing
        and callable(ir_value)
    )


def _validate_common_run_length_decode_controls(
    *,
    decoded_items_per_thread: Any,
    decoded_window_offset: Any,
    run_length_width: int,
    run_length_signed: bool,
) -> None:
    """Validate trace-static output shape and the portable window scalar."""

    if isinstance(decoded_items_per_thread, bool) or not isinstance(
        decoded_items_per_thread, Integral
    ):
        raise TypeError(
            "cuda.coop.run_length_decode decoded_items_per_thread must be a "
            "compile-time positive integer"
        )
    if int(decoded_items_per_thread) <= 0:
        raise ValueError(
            "cuda.coop.run_length_decode decoded_items_per_thread must be a "
            "compile-time positive integer"
        )

    if isinstance(decoded_window_offset, bool):
        raise TypeError(
            "cuda.coop.run_length_decode decoded_window_offset must be an "
            "int-like scalar"
        )
    if isinstance(decoded_window_offset, Integral):
        normalized_offset = int(decoded_window_offset)
        if normalized_offset < 0:
            raise ValueError(
                "cuda.coop.run_length_decode decoded_window_offset must be non-negative"
            )
        value_bits = run_length_width - 1 if run_length_signed else run_length_width
        if normalized_offset >= 1 << value_bits:
            raise ValueError(
                "cuda.coop.run_length_decode decoded_window_offset must be "
                "representable in the run_lengths dtype"
            )
        return
    if not _is_compiler_integer(decoded_window_offset):
        raise TypeError(
            "cuda.coop.run_length_decode decoded_window_offset must be an "
            "int-like scalar"
        )


def _validate_common_integer_dtype(
    operation: str,
    parameter: str,
    dtype: Any,
    *,
    allow_uint8: bool,
) -> None:
    """Require one portable integral dtype used by a specialized operation."""

    validator = (
        validate_portable_integer_value_dtype_name
        if allow_uint8
        else validate_portable_integer_key_dtype_name
    )
    validator(
        _common_integer_dtype_name(dtype),
        operation=operation,
        parameter=parameter,
    )


__all__ = ["TempStorageLike", "ThreadDataLike"]
