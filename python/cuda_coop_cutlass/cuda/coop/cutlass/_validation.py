# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate static payload and shared-storage descriptions for CuTe tracing."""

from __future__ import annotations

import operator
from numbers import Integral
from typing import Any, Protocol, SupportsIndex, TypeVar, runtime_checkable

_ItemT = TypeVar("_ItemT")


@runtime_checkable
class _ReadableThreadDataLike(Protocol[_ItemT]):
    """Describe a fixed number of readable values owned by one thread.

    ``items_per_thread`` is the fixed extent. ``dtype`` can remain unknown
    until a supported producer establishes the element type. CuTe supplies
    the storage and indexed access.
    """

    items_per_thread: int
    dtype: object | None

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _ItemT: ...


@runtime_checkable
class ThreadDataLike(_ReadableThreadDataLike[_ItemT], Protocol[_ItemT]):
    """Mutable fixed-size per-thread payload understood by CUTLASS.

    See :class:`cuda.coop.cutlass.ThreadData` for construction, item access,
    and dtype requirements.
    """

    def __setitem__(self, index: int, value: _ItemT) -> None: ...


@runtime_checkable
class TempStorageLike(Protocol):
    """Explicit scratch descriptor understood by CUTLASS.

    See :class:`cuda.coop.cutlass.TempStorage` for construction, allocation
    sharing, and synchronization.
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

    def release(self, dtype: object | None = None) -> object:
        """Consume the descriptor and return its complete native shared array.

        An omitted dtype returns bytes. An explicit dtype must fit the
        region's existing capacity and alignment. No allocation or barrier is
        added. Every alias becomes unavailable for subsequent primitive,
        reservation, or release calls on any reachable control-flow path.
        """
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


def _thread_data_extent(
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
            f"cuda.coop.cutlass.{operation} "
            f"{parameter}.items_per_thread must be a "
            "compile-time positive integer"
        )
    normalized = int(extent)
    if normalized <= 0:
        raise ValueError(
            f"cuda.coop.cutlass.{operation} "
            f"{parameter}.items_per_thread must be a "
            "compile-time positive integer"
        )
    if len(value) != normalized:
        raise ValueError(
            f"cuda.coop.cutlass.{operation} {parameter}.items_per_thread "
            "must match the payload item count"
        )
    return normalized


def _payload_dtype(
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
        dtype_name = _numeric_dtype_name(dtype)
        for index in range(1, len(value)):
            item = value[index]
            item_dtype = getattr(item, "dtype", None)
            if item_dtype is None:
                item_dtype = type(item)
            if _numeric_dtype_name(item_dtype) != dtype_name:
                raise TypeError(
                    f"cuda.coop.cutlass.{operation} {parameter} "
                    "items must have one dtype"
                )
    return dtype


def _numeric_dtype_name(dtype: Any) -> str:
    """Describe a dtype without importing its compiler or NumPy.

    Map Python int and float aliases to 32-bit types. Prefer an
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


def _validate_temp_storage(operation: str, value: Any) -> None:
    """Check the descriptor interface before planning its shared region."""
    if not isinstance(value, TempStorageLike):
        raise TypeError(
            f"cuda.coop.cutlass.{operation} temp_storage must satisfy "
            "TempStorageLike; construct it with cuda.coop.cutlass.TempStorage()"
        )


def _validate_numeric_value(
    operation: str,
    parameter: str,
    value: object,
    *,
    allow_untyped_thread_data: bool = False,
    allow_readonly_thread_data: bool = False,
    require_thread_data: bool = False,
) -> None:
    """Validate readable payload shape and homogeneous dtype metadata.

    The consuming CUTLASS operation checks its supported scalar dtypes. This
    check preserves structural readable payloads without imposing an extra
    frontend dtype policy or importing a compiler during payload inspection.
    """
    protocol = (
        _ReadableThreadDataLike
        if allow_readonly_thread_data
        else ThreadDataLike
    )
    if isinstance(value, protocol):
        _thread_data_extent(operation, parameter, value)
        if value.dtype is None and allow_untyped_thread_data:
            return
        _payload_dtype(operation, parameter, value)
    elif require_thread_data:
        raise TypeError(
            f"cuda.coop.cutlass.{operation} requires a fixed-size "
            f"ThreadData {parameter} payload"
        )
