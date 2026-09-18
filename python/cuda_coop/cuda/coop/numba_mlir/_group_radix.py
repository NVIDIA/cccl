# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block radix operations for Numba-CUDA-MLIR."""

from __future__ import annotations

from typing import Any

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation(
    "radix_sort_keys", family_module="cuda.coop.numba_mlir._compiler._group_radix"
)
def radix_sort_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    temp_storage: Any = None,
    blocked_to_striped: bool = False,
) -> Any:
    """Return stable block radix-sorted keys without modifying the input.

    Parameters
    ----------
    group : ThreadGroup
        The complete physical block from ``this_block()``; all threads
        participate with identical controls and per-thread extents.
    keys : scalar, ThreadDataLike, or local array
        One scalar or a fixed-size payload of int32, uint32, int64, uint64,
        float32, or float64 keys in blocked arrangement.
    begin_bit, end_bit : int or compiler integer
        Block-uniform half-open interval with
        ``0 <= begin_bit < end_bit <= key_width``. End defaults to the key
        width. Invalid runtime bounds trap before narrowing to CUB's integer
        arguments.
    descending : bool
        Compile-time selector for descending ordering.
    temp_storage : TempStorageLike, optional
        Explicit block scratch, otherwise allocated automatically. Requested
        size and alignment must satisfy the specialization. Synchronize before
        reuse when its descriptor uses ``auto_sync=False``.
    blocked_to_striped : bool
        Compile-time selector for striped output. False returns blocked
        output; true maps output item ``i`` at thread ``t`` to sorted index
        ``i * block_threads + t``.

    Returns
    -------
    scalar or ThreadDataLike
        Sorted keys, retaining the input dtype and scalar or array shape.
        Equal selected digits preserve blocked input order in both directions.

    Notes
    -----
    Wraps CUB ``BlockRadixSort::Sort``, ``SortDescending``, or their
    ``BlockedToStriped`` variants. CUB inverts the sign bit for signed integer
    keys; for floating-point keys it inverts all bits of negative values and
    only the sign bit of nonnegative values before digit selection. Returned
    keys keep their original representations. Negative and positive zero
    compare equivalently; NaNs follow CUB's transformed-bit ordering rather
    than a numeric total order. The input payload is preserved.
    """
    return group_primitive_marker(
        "radix_sort_keys",
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
        blocked_to_striped=blocked_to_striped,
    )


@group_operation(
    "radix_sort_pairs", family_module="cuda.coop.numba_mlir._compiler._group_radix"
)
def radix_sort_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    temp_storage: Any = None,
    blocked_to_striped: bool = False,
) -> tuple[Any, Any]:
    """Return stable radix-sorted keys and their associated values.

    Parameters
    ----------
    group : ThreadGroup
        The complete physical block; all block threads participate.
    keys, values : scalar, ThreadDataLike, or local array
        Both operands must have matching scalar or array shape and extent.
        Keys support 32- and 64-bit signed integers, unsigned integers, and
        floating-point values. Associated values support the portable numeric
        dtypes, including 8- and 16-bit integers.
    begin_bit, end_bit : int or compiler integer
        Block-uniform half-open interval in CUB's transformed key bits, with
        ``0 <= begin_bit < end_bit <= key_width``. Omitted end selects the key
        width. Invalid runtime bounds trap before narrowing to CUB's integer
        arguments.
    descending : bool
        Compile-time order selector. Equal selected digits retain flattened
        blocked input order in either direction.
    temp_storage : TempStorageLike, optional
        Explicit block scratch. Omit for automatic allocation and reuse
        synchronization. Explicit size and alignment must cover the selected
        specialization; ``auto_sync=False`` requires caller synchronization.
    blocked_to_striped : bool
        Compile-time selector for striped instead of blocked output, applied
        to both keys and associated values.

    Returns
    -------
    tuple
        Sorted keys and associated values, preserving their dtypes, matching
        shape, and association. Neither input payload is modified.

    Notes
    -----
    Wraps the paired CUB ``BlockRadixSort`` overloads. Signed integer and
    floating-point bit transformations, signed-zero behavior, NaN ordering,
    bit intervals, and striped indexing are the same as ``radix_sort_keys``.
    """
    return group_primitive_marker(
        "radix_sort_pairs",
        group,
        keys,
        values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
        blocked_to_striped=blocked_to_striped,
    )


@group_operation(
    "radix_rank", family_module="cuda.coop.numba_mlir._compiler._group_radix"
)
def radix_rank(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    radix_bits: Any | None = None,
    descending: bool = False,
    exclusive_digit_prefix: Any = None,
) -> Any:
    """Return stable block-wide int32 digit ranks and optional bin prefixes.

    Parameters
    ----------
    group : ThreadGroup
        The complete physical block. Every thread must participate with
        identical compile-time controls and payload extents.
    keys : scalar, ThreadDataLike, or local array
        int32, uint32, int64, or uint64 keys in blocked arrangement. Keys
        remain unchanged; signed keys invert the sign bit before extracting
        the selected digit.
    begin_bit, end_bit : int, optional
        Compile-time half-open interval within the key width. The digit has
        one through eight bits. Begin defaults to zero; omitted end is begin
        plus ``radix_bits``, or begin plus four if both are omitted.
    radix_bits : int, optional
        Compile-time digit width; when end is explicit, must equal
        ``end_bit - begin_bit``.
    descending : bool
        Compile-time selector for descending digit ordering.
    exclusive_digit_prefix : ThreadDataLike or local array, optional
        Writable int32 side output, distinct from the keys. Each thread must
        provide ``max(1, ceil(2**radix_bits / block_threads))`` items. Bin
        indices are distributed in blocked order, including descending mode.
        Each prefix counts keys with smaller digits for ascending order or
        greater digits for descending order. Slots beyond the number of bins
        have no defined value and must not be consumed.

    Returns
    -------
    int32 scalar or ThreadDataLike
        Ranks with the input's scalar or array shape and a fixed signed int32
        dtype. Equal digits retain flattened blocked input order.

    Notes
    -----
    Wraps CUB ``BlockRadixRank::RankKeys`` with an optional exclusive-prefix
    output. Scratch allocation and its reuse barrier are automatic.
    """
    return group_primitive_marker(
        "radix_rank",
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        radix_bits=radix_bits,
        descending=descending,
        exclusive_digit_prefix=exclusive_digit_prefix,
    )


__all__ = ["radix_rank", "radix_sort_keys", "radix_sort_pairs"]
