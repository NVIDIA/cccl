# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check common payload representations before a tracing backend uses them.

A per-thread payload exposes a fixed extent, one dtype and indexed items.
Structural checks keep this module independent of compiler imports.
Read-only inputs, writable outputs and scratch descriptors have different
requirements, so callers choose the appropriate protocol.

Numeric checks accept common Python, NumPy and compiler scalar forms, then
apply the shared dtype policy. A recognized shape alone is not enough: the
common API still accepts only the shared dtypes and not backend-specific
register tensors.
"""

from __future__ import annotations

import operator
from numbers import Integral
from typing import Any, Protocol, SupportsIndex, TypeVar, runtime_checkable

from ..dtype_policy import (
    validate_common_integer_value_dtype_name,
    validate_common_numeric_dtype_name,
)

_ItemT = TypeVar("_ItemT")


@runtime_checkable
class _ReadableThreadDataLike(Protocol[_ItemT]):
    """Describe a fixed number of readable values owned by one thread.

    ``items_per_thread`` is the fixed extent. ``dtype`` can remain unknown
    until a supported producer establishes the element type. The compiler
    backend supplies the storage and indexed access.
    """

    items_per_thread: int
    dtype: object | None

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _ItemT: ...


@runtime_checkable
class ThreadDataLike(_ReadableThreadDataLike[_ItemT], Protocol[_ItemT]):
    """Mutable fixed-size per-thread payload understood by supported backends.

    See :ref:`per-thread payloads <coop-common-payloads>` for construction, item
    access, and dtype requirements.
    """

    def __setitem__(self, index: int, value: _ItemT) -> None: ...


@runtime_checkable
class TempStorageLike(Protocol):
    """Explicit scratch descriptor understood by supported backends.

    See :ref:`temporary storage <coop-common-storage>` for construction,
    allocation sharing, and synchronization.
    """

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool
    sharing: str

    def reserve(
        self, num_elems: int, dtype: object, *, alignment: int | None = None
    ) -> Any:
        """Return a typed array that follows the descriptor's sharing policy."""
        ...


def _normalize_alignment(alignment: SupportsIndex | None) -> int | None:
    """Normalize a payload or scratch alignment request to bytes.

    Preserve ``None`` so the compiler can choose the alignment. Explicit
    requests must be positive powers of two. Accept integer-like values
    through ``__index__``, but reject booleans as accidental requests.
    """

    if alignment is None:
        return None
    if isinstance(alignment, bool):
        raise TypeError("alignment must be an integer or None")
    try:
        alignment = operator.index(alignment)
    except TypeError as exc:
        raise TypeError("alignment must be an integer or None") from exc
    if alignment <= 0:
        raise ValueError("alignment must be a positive integer")
    if alignment & (alignment - 1):
        raise ValueError("alignment must be a power of 2")
    return alignment


def _validate_common_temp_storage(operation: str, value: Any) -> None:
    """Check that explicit scratch exposes the common descriptor fields.

    This structural check does not allocate storage or validate an operation's
    required capacity, alignment or synchronization. The backend checks those
    requirements when it plans the call.
    """

    if isinstance(value, TempStorageLike):
        return
    raise TypeError(
        f"cuda.coop.{operation} temp_storage must satisfy TempStorageLike; "
        "construct it with cuda.coop.TempStorage()"
    )


def _validate_common_thread_data_payload(
    operation: str,
    parameter: str,
    value: object,
    *,
    allow_readonly: bool = False,
) -> None:
    """Check for readable or writable fixed-size payload operations.

    Inputs can opt into the read-only protocol; outputs require indexed
    writes. This check covers the interface only. Separate helpers validate
    the extent and dtype before the backend interprets individual items.
    """

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
    """Return a positive payload extent known during Python tracing.

    Require a host integer rather than a compiler value, and compare it
    with the payload's reported length. Both values must agree before
    callers compute the group tile size from them.
    """

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
    """Read a declared dtype, or infer one from populated payload items.

    Trust an explicit dtype without reading the payload. Otherwise use
    each item's dtype attribute or Python type, rejecting mixed normalized
    names. Call this only when the items already hold values. An untyped
    Load output is skipped instead, so the source array can set its dtype.
    """

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
    """Describe a dtype without importing its compiler or NumPy.

    Map Python int and float aliases to the common 32-bit types. Prefer an
    explicit numeric name; integer-like compiler types can instead expose
    width and signedness. Return other names for the policy validator to
    reject with an operation-specific message. This helper does not establish
    that the resulting dtype is supported.
    """

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
            suffix = (
                dtype_name[len(prefix) :]
                if dtype_name.startswith(prefix)
                else ""
            )
            if suffix.isdigit():
                return dtype_name
        if dtype_name in {"bool", "boolean"}:
            return dtype_name

    width = getattr(dtype, "width", None)
    if width is None:
        width = getattr(dtype, "bitwidth", None)
    signed = getattr(dtype, "signed", None)
    if (
        isinstance(width, Integral)
        and (not isinstance(width, bool))
        and isinstance(signed, bool)
    ):
        return f"{'int' if signed else 'uint'}{int(width)}"

    if isinstance(dtype_name, str):
        return dtype_name
    return str(dtype).lower()


def _is_common_numeric_scalar(value: Any) -> bool:
    """Recognize the scalar forms checked by the common dtype policy.

    Accept Python numeric literals, listed NumPy scalar types, or compiler
    values with a positive width, dtype and callable ir_value accessor.
    Structural recognition avoids importing the compiler. The caller must
    still validate the dtype; this predicate does not read the device value.
    """

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
    value: object,
) -> str:
    """Validate a scalar representation and return its common dtype name.

    Use the value's ``dtype`` attribute when present (NumPy and compiler
    scalars), otherwise its Python type. The shared policy rejects
    unsupported widths and kinds before a backend can interpret the value
    using a wider qualified contract.
    """

    if not _is_common_numeric_scalar(value):
        raise TypeError(
            f"cuda.coop.{operation} {parameter} must be a numeric scalar "
            "supported by the common API; use a backend-qualified import "
            "for backend-specific values"
        )
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = type(value)
    return validate_common_numeric_dtype_name(
        _common_numeric_dtype_name(dtype),
        operation=operation,
        parameter=parameter,
    )


def _validate_common_integer_value(
    operation: str,
    parameter: str,
    value: object,
) -> int | None:
    """Distinguish a static integer from a validated compiler integer.

    Return a Python int for a non-Boolean Integral value so callers can check
    its range immediately. Return None for a supported compiler integer whose
    device value is not available during tracing. That return means range
    checks remain for lowering; it does not mean the argument was omitted.
    """

    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if not (
        _is_common_numeric_scalar(value)
        and isinstance(getattr(value, "signed", None), bool)
    ):
        raise TypeError(
            f"cuda.coop.{operation} {parameter} must be an integer value "
            "supported by the common API"
        )
    dtype = getattr(value, "dtype", None)
    assert dtype is not None
    validate_common_integer_value_dtype_name(
        _common_numeric_dtype_name(dtype),
        operation=operation,
        parameter=parameter,
    )
    return None


def _validate_common_numeric_value(
    operation: str,
    parameter: str,
    value: object,
    *,
    allow_untyped_thread_data: bool = False,
    allow_readonly_thread_data: bool = False,
    require_thread_data: bool = False,
) -> str | None:
    """Validate a common scalar or fixed-size payload and return its dtype.

    Callers can require a payload, accept read-only inputs, or defer dtype
    inference for an untyped output. Every payload needs a positive static
    extent that matches its length. For an untyped output that allows
    deferral, return None without reading items: Load must set the dtype
    before those slots are used. Other payloads need one supported numeric
    dtype. Scalar inputs follow the same dtype policy.
    """

    protocol = (
        _ReadableThreadDataLike
        if allow_readonly_thread_data
        else ThreadDataLike
    )
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
                f"cuda.coop.{operation} requires the common API's numeric "
                f"scalar or fixed-size ThreadData {parameter} payload; use a "
                "backend-qualified import for backend-specific payloads"
            )
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            dtype = type(value)
    return validate_common_numeric_dtype_name(
        _common_numeric_dtype_name(dtype),
        operation=operation,
        parameter=parameter,
    )


__all__ = [
    "TempStorageLike",
    "ThreadDataLike",
    "_ReadableThreadDataLike",
    "_common_payload_dtype",
    "_common_thread_data_extent",
    "_normalize_alignment",
    "_validate_common_integer_value",
    "_validate_common_numeric_scalar",
    "_validate_common_numeric_value",
    "_validate_common_temp_storage",
    "_validate_common_thread_data_payload",
]
