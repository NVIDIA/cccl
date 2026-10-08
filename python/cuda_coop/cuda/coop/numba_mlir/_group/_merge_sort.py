# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose block and warp MergeSort calls to Numba device kernels.

Each call returns newly allocated per-thread payloads in blocked order.
Keys and their optional values stay associated, and the input arrays stay
unchanged. The qualified API also accepts fixed local arrays and a custom
stateless comparison callback; group planning resolves these marker calls
before ordinary compiler typing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy
import numpy as np

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    IntegerValue,
)

from ..._core.api._payload import (
    TempStorageLike,
    ThreadDataLike,
)
from .._compiler._operations import group_operation
from .._thread_group import BlockGroup, WarpGroup
from ._marker import group_primitive_marker

_ValueT = TypeVar("_ValueT", bound=CommonNumericScalar)

_KeyT = TypeVar("_KeyT", bound=CommonNumericScalar)


@group_operation(
    "merge_sort_keys",
    family_module="cuda.coop.numba_mlir._compiler._group_merge_sort",
)
def merge_sort_keys(
    group: BlockGroup | WarpGroup,
    keys: CommonThreadDataLike[_KeyT] | numpy.ndarray,
    /,
    *,
    descending: bool = False,
    valid_items: IntegerValue | None = None,
    oob_default: CommonNumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
    compare_op: Callable[[_KeyT, _KeyT], bool | np.bool_] | None = None,
) -> ThreadDataLike[_KeyT]:
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
        Fixed-size one-dimensional Numba local arrays are also accepted.
    descending : bool, optional
        Compile-time sort direction. ``False`` sorts in ascending order;
        ``True`` sorts in descending order.
    valid_items : integer, optional
        Number of valid items in the entire group tile, from zero through
        ``group_size * items_per_thread``. Valid items occupy the first
        ``valid_items`` positions in blocked order. Supply this together with
        ``oob_default`` for a partial tile.
        Runtime counts must have a signed integer dtype up to 64 bits or an
        unsigned integer dtype up to 32 bits. Invalid static counts fail
        compilation; invalid runtime counts trap before CUB narrows them.
    oob_default : scalar, optional
        Key sentinel for a partial tile. Choose a value that sorts after the
        valid keys: an upper bound for ascending order or a lower bound for
        descending order. Typed runtime values and NumPy scalar constants
        must match the key dtype exactly. An ordinary Python int constant
        can convert to any key dtype within range. A Python float constant
        requires a floating key dtype and must be within its finite range;
        float-to-integer conversion is rejected. Floating keys also accept
        positive or negative infinity as a sentinel.
        ``valid_items`` and ``oob_default`` must be uniform within the group.
    temp_storage : TempStorageLike, optional
        Caller-provided scratch for a block group. Omit it to let the compiler
        manage scratch. Warp groups always use compiler-managed storage with
        a separate slice for each physical or logical warp. When sharing
        scratch between block calls, set ``auto_sync=True`` or
        synchronize explicitly before reuse.
    compare_op : callable, optional
        Compile-time stateless predicate ``compare_op(left, right)`` that
        defines a strict weak ordering of keys. It must be device-compatible.
        The predicate sets the ordering and cannot be combined with
        ``descending=True``. For partial tiles, choose a sentinel that follows
        all valid keys under this predicate.

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
    ``cub::WarpMergeSort::Sort`` on copies of the input payloads.
    Floating-point keys must obey the comparison's ordering requirements.

    See Also
    --------
    merge_sort_pairs

    Examples
    --------
    Sort keys and key/index pairs with a custom descending predicate.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_merge_sort_examples.py
        :language: python
        :start-after: # qualified-sort-example-begin
        :end-before: # qualified-sort-example-end
        :dedent: 4
    """

    return group_primitive_marker(
        "merge_sort_keys",
        group,
        keys,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
        compare_op=compare_op,
    )


@group_operation(
    "merge_sort_pairs",
    family_module="cuda.coop.numba_mlir._compiler._group_merge_sort",
)
def merge_sort_pairs(
    group: BlockGroup | WarpGroup,
    keys: CommonThreadDataLike[_KeyT] | numpy.ndarray,
    values: CommonThreadDataLike[_ValueT] | numpy.ndarray,
    /,
    *,
    descending: bool = False,
    valid_items: IntegerValue | None = None,
    oob_default: CommonNumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
    compare_op: Callable[[_KeyT, _KeyT], bool | np.bool_] | None = None,
) -> tuple[ThreadDataLike[_KeyT], ThreadDataLike[_ValueT]]:
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
        Fixed-size one-dimensional Numba local arrays are also accepted.
    values : ThreadDataLike
        Values associated with the keys, with the same per-thread item count.
        Values use the same numeric dtype set, independently of the key dtype.
        Fixed-size one-dimensional Numba local arrays are also accepted.
    descending : bool, optional
        Compile-time sort direction. ``False`` sorts in ascending order;
        ``True`` sorts in descending order.
    valid_items : integer, optional
        Number of valid items in the entire group tile, from zero through
        ``group_size * items_per_thread``. Valid items occupy the first
        ``valid_items`` positions in blocked order. Supply this together with
        ``oob_default`` for a partial tile.
        Runtime counts must have a signed integer dtype up to 64 bits or an
        unsigned integer dtype up to 32 bits. Invalid static counts fail
        compilation; invalid runtime counts trap before CUB narrows them.
    oob_default : scalar, optional
        Key sentinel for a partial tile. Choose a value that sorts after the
        valid keys: an upper bound for ascending order or a lower bound for
        descending order. Typed runtime values and NumPy scalar constants
        must match the key dtype exactly. An ordinary Python int constant
        can convert to any key dtype within range. A Python float constant
        requires a floating key dtype and must be within its finite range;
        float-to-integer conversion is rejected. Floating keys also accept
        positive or negative infinity as a sentinel.
        ``valid_items`` and ``oob_default`` must be uniform within the group.
    temp_storage : TempStorageLike, optional
        Caller-provided scratch for a block group. Omit it to let the compiler
        manage scratch. Warp groups always use compiler-managed storage with
        a separate slice for each physical or logical warp. When sharing
        scratch between block calls, set ``auto_sync=True`` or
        synchronize explicitly before reuse.
    compare_op : callable, optional
        Compile-time stateless predicate ``compare_op(left, right)`` that
        defines a strict weak ordering of keys. It must be device-compatible.
        The predicate sets the ordering and cannot be combined with
        ``descending=True``. For partial tiles, choose a sentinel that follows
        all valid keys under this predicate.

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
    ``cub::WarpMergeSort::Sort`` on copies of the input payloads.
    Floating-point keys must obey the comparison's ordering requirements.

    See Also
    --------
    merge_sort_keys

    Examples
    --------
    Sort keys and key/index pairs with a custom descending predicate.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_merge_sort_examples.py
        :language: python
        :start-after: # qualified-sort-example-begin
        :end-before: # qualified-sort-example-end
        :dedent: 4
    """

    return group_primitive_marker(
        "merge_sort_pairs",
        group,
        keys,
        values,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
        compare_op=compare_op,
    )


__all__ = ["merge_sort_keys", "merge_sort_pairs"]
