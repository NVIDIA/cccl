# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common block radix ordering operations."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from .._bindings import ArgumentBinding
from ..block.radix import make_radix_bit_range
from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _common_group_operation,
    _group_primitive_marker,
    _validate_common_operation_group,
)
from ._payload import (
    _common_thread_data_extent,
    _validate_common_integer_value,
    _validate_common_numeric_value,
    _validate_common_temp_storage,
)


def _radix_bounds(operation, key_width, begin_bit, end_bit, radix_bits=None):
    for name, value in (
        ("begin_bit", begin_bit),
        ("end_bit", end_bit),
        ("radix_bits", radix_bits),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral)
        ):
            raise TypeError(
                f"cuda.coop.{operation} {name} must be a compile-time integer"
            )
    if radix_bits is not None and radix_bits <= 0:
        raise ValueError("radix_bits must be positive")
    if end_bit is None:
        end_bit = (
            begin_bit + (4 if radix_bits is None else radix_bits)
            if operation == "radix_rank"
            else key_width
        )
    if radix_bits is not None and end_bit - begin_bit != radix_bits:
        raise ValueError("radix_bits must match end_bit - begin_bit")
    make_radix_bit_range(begin_bit=begin_bit, end_bit=end_bit, bit_width=key_width)
    if operation == "radix_rank" and end_bit - begin_bit > 8:
        raise ValueError("radix_rank bit width must be <= 8")
    return int(begin_bit), int(end_bit)


def _validate(
    operation,
    group,
    keys,
    values,
    begin_bit,
    end_bit,
    descending,
    temp_storage,
    radix_bits=None,
):
    if _backend_module_name() is None:
        return
    _validate_common_operation_group(operation, group)
    name = _validate_common_numeric_value(
        operation,
        "keys",
        keys,
        require_thread_data=True,
        allow_readonly_thread_data=True,
    )
    if name not in {"int32", "uint32", "int64", "uint64"}:
        raise TypeError(
            f"cuda.coop.{operation} keys require int32, uint32, int64, or uint64"
        )
    if operation == "radix_sort_pairs":
        _validate_common_numeric_value(
            operation,
            "values",
            values,
            require_thread_data=True,
            allow_readonly_thread_data=True,
        )
        if _common_thread_data_extent(
            operation, "keys", keys
        ) != _common_thread_data_extent(operation, "values", values):
            raise ValueError("keys and values must have the same items_per_thread")
    if not isinstance(descending, bool):
        raise TypeError(f"cuda.coop.{operation} descending must be a compile-time bool")
    width = int(name[-2:])
    if operation == "radix_rank":
        _radix_bounds(operation, width, begin_bit, end_bit, radix_bits)
    else:
        begin = _validate_common_integer_value(operation, "begin_bit", begin_bit)
        end = (
            width
            if end_bit is None
            else _validate_common_integer_value(operation, "end_bit", end_bit)
        )
        make_radix_bit_range(
            begin_bit=ArgumentBinding.runtime() if begin is None else begin,
            end_bit=ArgumentBinding.runtime() if end is None else end,
            bit_width=width,
        )
    if temp_storage is not None:
        _validate_common_temp_storage(operation, temp_storage)


@_common_group_operation("radix_sort_keys", group_kinds=("block",))
def radix_sort_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    descending: bool = False,
    temp_storage: Any = None,
) -> Any:
    """Return stable, blocked radix-sorted integral keys without mutation.

    Parameters
    ----------
    group : ThreadGroup
        The complete physical block returned by ``this_block()``. All block
        threads must participate with identical options and payload extents.
    keys : ThreadDataLike
        Fixed-size per-thread keys with int32, uint32, int64, or uint64 dtype.
        The input sequence is the flattened blocked arrangement.
    begin_bit, end_bit : int or compiler integer
        Half-open interval in CUB's ordered key representation. The default
        begin is zero; omitted end selects the full key width, even when begin
        is nonzero. Bounds may be runtime values but must be block-uniform and
        satisfy ``0 <= begin_bit < end_bit <= key_width``. Known bounds are
        checked during compilation; invalid runtime bounds trap before narrowing
        to CUB's integer arguments.
    descending : bool
        Compile-time selector for descending instead of ascending digit order.
    temp_storage : TempStorageLike, optional
        Caller-owned block scratch. Omit to allocate scratch automatically.
        An explicit descriptor must satisfy the specialization's size and
        alignment. With ``auto_sync=False``, the caller synchronizes before
        reusing it.

    Returns
    -------
    ThreadDataLike
        Sorted keys in blocked arrangement with the input dtype and extent.
        The input payload is preserved. Equal selected digits retain their
        original blocked order, including for descending sorts.

    Notes
    -----
    Wraps CUB ``BlockRadixSort::Sort`` or ``SortDescending``. For signed
    integers, the sign bit is inverted before selecting the bit interval,
    then restored in the returned keys. Use ``cuda.coop.numba_mlir`` for
    floating-point keys, scalar or local-array payloads, and striped output.
    """
    _validate(
        "radix_sort_keys",
        group,
        keys,
        None,
        begin_bit,
        end_bit,
        descending,
        temp_storage,
    )
    return _group_primitive_marker(
        "radix_sort_keys",
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
    )


@_common_group_operation("radix_sort_pairs", group_kinds=("block",))
def radix_sort_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    descending: bool = False,
    temp_storage: Any = None,
) -> tuple[Any, Any]:
    """Return stable sorted keys and associated numeric values without mutation.

    Parameters
    ----------
    group : ThreadGroup
        A complete physical block; every thread participates.
    keys, values : ThreadDataLike
        Fixed-size per-thread payloads with matching extents. Keys use int32,
        uint32, int64, or uint64. Values use the common API's numeric dtypes:
        signed or unsigned 8-, 16-, 32-, or 64-bit integers, float32, or
        float64.
    begin_bit, end_bit : int or compiler integer
        Block-uniform half-open interval in CUB's ordered key representation.
        Omitted end selects the key width. Require
        ``0 <= begin_bit < end_bit <= key_width``. Invalid static bounds fail
        compilation; invalid runtime bounds trap before narrowing. Signed keys
        invert their sign bit before digit
        extraction; returned keys retain their original representation.
    descending : bool
        Compile-time order selector. Equal digits retain their input order
        for both ascending and descending sorts.
    temp_storage : TempStorageLike, optional
        Explicit block scratch; omitted storage is allocated automatically.
        Its requested size and alignment must cover the specialization. The
        caller supplies reuse synchronization when ``auto_sync=False``.

    Returns
    -------
    tuple[ThreadDataLike, ThreadDataLike]
        Keys and associated values in blocked arrangement, preserving both
        input dtypes, their matching extent, and key/value association. Neither
        input payload is modified.

    Notes
    -----
    Wraps the key/value overload of CUB ``BlockRadixSort::Sort`` or
    ``SortDescending``. Qualified Numba-CUDA-MLIR calls additionally support
    floating-point keys, scalar or local-array payloads, and striped output.
    """
    _validate(
        "radix_sort_pairs",
        group,
        keys,
        values,
        begin_bit,
        end_bit,
        descending,
        temp_storage,
    )
    return _group_primitive_marker(
        "radix_sort_pairs",
        group,
        keys,
        values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
    )


@_common_group_operation("radix_rank", group_kinds=("block",))
def radix_rank(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    radix_bits: int | None = None,
    descending: bool = False,
) -> Any:
    """Return stable int32 digit ranks without mutating integral keys.

    Parameters
    ----------
    group : ThreadGroup
        A complete physical block. All threads participate with identical
        compile-time controls and per-thread payload extents.
    keys : ThreadDataLike
        Fixed-size int32, uint32, int64, or uint64 per-thread keys in blocked
        arrangement.
    begin_bit, end_bit : int, optional
        Compile-time half-open interval in CUB's ordered representation.
        Begin defaults to zero; omitted end is begin plus ``radix_bits`` or
        four when that option is omitted. The interval must remain within
        the key width and contain one through eight bits.
    radix_bits : int, optional
        Compile-time digit width. When end is also supplied, it must equal
        ``end_bit - begin_bit``.
    descending : bool
        Compile-time selector that places greater digits before smaller ones.

    Returns
    -------
    ThreadDataLike
        Signed int32 ranks with the same per-thread extent as keys. Ranks
        form a permutation of the block tile's indices. Equal digits retain
        flattened blocked input order. The keys are not modified.

    Notes
    -----
    Uses CUB ``BlockRadixRank::RankKeys`` with a digit extractor. Signed keys
    invert their sign bit before digit extraction, matching radix sort's
    ordered representation. Scratch allocation and its reuse barrier are
    automatic. The qualified API also accepts scalars and local arrays and
    can write exclusive digit prefixes into a caller-provided output array.
    """
    _validate(
        "radix_rank",
        group,
        keys,
        None,
        begin_bit,
        end_bit,
        descending,
        None,
        radix_bits,
    )
    return _group_primitive_marker(
        "radix_rank",
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        radix_bits=radix_bits,
        descending=descending,
    )


__all__ = ["radix_rank", "radix_sort_keys", "radix_sort_pairs"]
