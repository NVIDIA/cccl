# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Merge Sort entry points."""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _group_primitive_marker,
    _portable_group_operation,
)
from ._payload import (
    TempStorageLike,
    ThreadDataLike,
    _ReadableThreadDataLike,
    _validate_common_numeric_value,
    _validate_common_temp_storage,
)


@_portable_group_operation(
    "merge_sort_keys", group_kinds=("block", "warp", "threads_within_warp")
)
def merge_sort_keys(
    group: ThreadGroup,
    keys: _ReadableThreadDataLike[Any],
    /,
    *,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[Any]:
    """Return keys sorted across a block or warp in blocked order.

    Each thread contributes a fixed-size payload in blocked order: its items
    occupy consecutive positions in the group tile. Every group member must
    call this operation, with the same controls. The inputs remain unchanged.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, ``this_warp()``, or a logical warp produced by
        ``this_warp().group_by(width)``. Blocks require a power-of-two total
        thread count; multidimensional blocks are supported. Logical warp
        widths are 1, 2, 4, 8, 16, and 32. Each group must be complete.
    keys : ThreadDataLike
        Per-thread keys with a positive, compile-time item count. Supported
        dtypes are signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        ``float32``, and ``float64``. The compiler may infer the dtype from
        writes to ``ThreadData`` or a preceding ``load``.
    descending : bool, optional
        Compile-time sort direction. ``False`` sorts in ascending order;
        ``True`` sorts in descending order.
    valid_items : integer, optional
        Number of valid items in the entire group tile, from zero through
        ``group_size * items_per_thread``. Valid items form the initial blocked
        prefix. Supply this together with ``oob_default`` for a partial tile.
        Runtime counts must have a signed integer dtype up to 64 bits or an
        unsigned integer dtype up to 32 bits. Invalid static counts fail
        compilation; invalid runtime counts trap before CUB narrows them.
    oob_default : scalar, optional
        Key sentinel for a partial tile. Choose a value that sorts after the
        valid keys: an upper bound for ascending order or a lower bound for
        descending order. Runtime scalars must match the key dtype exactly;
        representable ordinary Python numeric literals are converted to it.
        ``valid_items`` and ``oob_default`` must be uniform within the group.
    temp_storage : TempStorageLike, optional
        Caller-provided scratch for a block group. Omit it to let the compiler
        manage scratch. Warp groups always use compiler-managed storage with
        a separate slice for each physical or logical warp. When sharing
        scratch between block calls, retain automatic synchronization or
        synchronize explicitly before reuse.

    Returns
    -------
    ThreadDataLike
        Sorted keys in blocked order, with the input dtype and per-thread
        item count. Equal keys have no stability guarantee.
        For a partial tile, only the first ``valid_items`` positions of the
        group result are defined; the remaining positions are unspecified.

    Notes
    -----
    The Numba backend uses ``cub::BlockMergeSort::Sort`` or
    ``cub::WarpMergeSort::Sort`` on copies of the input payloads. Floating-point
    keys must obey the comparison's ordering requirements.
    Use ``cuda.coop.numba_mlir`` for fixed-size Numba local-array inputs or
    a custom comparison predicate.

    See Also
    --------
    merge_sort_pairs
    """

    if not isinstance(descending, bool):
        raise TypeError("descending must be a compile-time bool")
    if (valid_items is None) != (oob_default is None):
        raise ValueError("valid_items and oob_default must be provided together")
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "merge_sort_keys",
            "keys",
            keys,
            require_thread_data=True,
            allow_readonly_thread_data=True,
        )
        if temp_storage is not None:
            _validate_common_temp_storage("merge_sort_keys", temp_storage)
    return _group_primitive_marker(
        "merge_sort_keys",
        group,
        keys,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
    )


@_portable_group_operation(
    "merge_sort_pairs", group_kinds=("block", "warp", "threads_within_warp")
)
def merge_sort_pairs(
    group: ThreadGroup,
    keys: _ReadableThreadDataLike[Any],
    values: _ReadableThreadDataLike[Any],
    /,
    *,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadDataLike[Any], ThreadDataLike[Any]]:
    """Return key/value pairs sorted across a block or warp in blocked order.

    Each thread contributes a fixed-size payload in blocked order: its items
    occupy consecutive positions in the group tile. Every group member must
    call this operation, with the same controls. The inputs remain unchanged.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, ``this_warp()``, or a logical warp produced by
        ``this_warp().group_by(width)``. Blocks require a power-of-two total
        thread count; multidimensional blocks are supported. Logical warp
        widths are 1, 2, 4, 8, 16, and 32. Each group must be complete.
    keys : ThreadDataLike
        Per-thread keys with a positive, compile-time item count. Supported
        dtypes are signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        ``float32``, and ``float64``. The compiler may infer the dtype from
        writes to ``ThreadData`` or a preceding ``load``.
    values : ThreadDataLike
        Values associated with the keys, with the same per-thread item count.
        Values use the same numeric dtype set, independently of the key dtype.
    descending : bool, optional
        Compile-time sort direction. ``False`` sorts in ascending order;
        ``True`` sorts in descending order.
    valid_items : integer, optional
        Number of valid items in the entire group tile, from zero through
        ``group_size * items_per_thread``. Valid items form the initial blocked
        prefix. Supply this together with ``oob_default`` for a partial tile.
        Runtime counts must have a signed integer dtype up to 64 bits or an
        unsigned integer dtype up to 32 bits. Invalid static counts fail
        compilation; invalid runtime counts trap before CUB narrows them.
    oob_default : scalar, optional
        Key sentinel for a partial tile. Choose a value that sorts after the
        valid keys: an upper bound for ascending order or a lower bound for
        descending order. Runtime scalars must match the key dtype exactly;
        representable ordinary Python numeric literals are converted to it.
        ``valid_items`` and ``oob_default`` must be uniform within the group.
    temp_storage : TempStorageLike, optional
        Caller-provided scratch for a block group. Omit it to let the compiler
        manage scratch. Warp groups always use compiler-managed storage with
        a separate slice for each physical or logical warp. When sharing
        scratch between block calls, retain automatic synchronization or
        synchronize explicitly before reuse.

    Returns
    -------
    tuple[ThreadDataLike, ThreadDataLike]
        Sorted keys and corresponding values, in blocked order. Each result
        retains its input's dtype and per-thread item count. Key/value
        associations are preserved; equal keys have no stability guarantee.
        For a partial tile, only the first ``valid_items`` positions of the
        group result are defined; the remaining positions are unspecified.

    Notes
    -----
    The Numba backend uses ``cub::BlockMergeSort::Sort`` or
    ``cub::WarpMergeSort::Sort`` on copies of the input payloads. Floating-point
    keys must obey the comparison's ordering requirements.
    Use ``cuda.coop.numba_mlir`` for fixed-size Numba local-array inputs or
    a custom comparison predicate.

    See Also
    --------
    merge_sort_keys
    """

    if not isinstance(descending, bool):
        raise TypeError("descending must be a compile-time bool")
    if (valid_items is None) != (oob_default is None):
        raise ValueError("valid_items and oob_default must be provided together")
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "merge_sort_pairs",
            "keys",
            keys,
            require_thread_data=True,
            allow_readonly_thread_data=True,
        )
        _validate_common_numeric_value(
            "merge_sort_pairs",
            "values",
            values,
            require_thread_data=True,
            allow_readonly_thread_data=True,
        )
        if temp_storage is not None:
            _validate_common_temp_storage("merge_sort_pairs", temp_storage)
    return _group_primitive_marker(
        "merge_sort_pairs",
        group,
        keys,
        values,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
    )


__all__ = ["merge_sort_keys", "merge_sort_pairs"]
