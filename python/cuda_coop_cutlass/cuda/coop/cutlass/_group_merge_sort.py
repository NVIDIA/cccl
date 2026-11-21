# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A merge sort puts the keys held by a group's threads in order.

``merge_sort_keys`` sorts the keys. ``merge_sort_pairs`` also carries a
value with each key, such as the position the key came from. Both return
new payloads and leave their inputs unchanged. The sorted keys go back to
the threads in :term:`blocked` order: with two items per thread, the thread
with rank 0 gets the first two keys, the thread with rank 1 the next two,
and so on.

.. figure:: visualizations/merge-sort.svg
   :alt: Eight threads hold two keys each, from 7 and 12 in thread 0 to 13
         and 2 in thread 7, and a letter from A to P marks each key. One
         row sorts each thread's two keys, and two rows merge the runs of
         two and of four threads. In the last row the keys run from 0 and
         1 in thread 0 to 14 and 15 in thread 7, each with its letter.
   :width: 100%

   ``merge_sort_pairs`` over one block of eight threads with two items
   each. The value of each key is its position in the tile, drawn as a
   letter from A for 0 to P for 15. Key 0 starts in thread 2 at position 5
   and ends in thread 0, still with F.

Each group sorts its own :term:`tile`. ``this_block()`` sorts the items of
the whole block, ``this_warp()`` those of one warp, and
``this_warp().group_by(n)`` those of ``n`` neighboring lanes. The package
has no primitive that merges sorted tiles, so an array that spans several
tiles is left unsorted as a whole.

Both functions take these controls:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Control
     - Groups
     - What it does
   * - ``descending``
     - All
     - Sorts from the largest key to the smallest.
   * - ``valid_items`` with ``oob_default``
     - All
     - Sorts only the first ``valid_items`` items of the tile.
   * - ``temp_storage``
     - Blocks
     - Puts the call's scratch in a :ref:`TempStorage <coop-temp-storage>`.
"""

# Implementation notes.
#
# Expose CUTLASS Merge Sort entry points for keys and key/value pairs.
#
# ``cuda.coop.cutlass.merge_sort_keys`` and ``merge_sort_pairs`` accept
# readable ThreadData payloads and CuTe register tensors or TensorSSA values.
# Snapshot readable ThreadData or adapt register inputs before lowering. Both
# sorts return fresh ThreadData without changing the inputs, so read-only
# inputs work. Key/value pairs keep independent dtypes and matching extents.

from __future__ import annotations

from typing import TypeVar

from cuda.coop.cutlass._planning.thread_group import ThreadGroup
from cuda.coop.cutlass._typing import IntegerValue
from cuda.coop.cutlass._validation import (
    _validate_temp_storage,
)

from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
    _snapshot_readable_payload,
)
from ._thread_group import (
    _require_complete_warp_partition,
    _resolve_primitive_group_from_launch,
)
from ._typing import (
    BlockGroup,
    NumericScalar,
    ReadableThreadDataLike,
    TempStorageLike,
    WarpGroup,
)

_ValueT = TypeVar("_ValueT", bound=NumericScalar)

_KeyT = TypeVar("_KeyT", bound=NumericScalar)


_SCOPE = "cuda.coop.cutlass"


def _merge_sort(
    group, keys, values, *, descending, valid_items, oob_default, temp_storage
):
    """Validate the public sort form and resolve its complete group.

    Require count and sentinel together for partial tiles, and allow explicit
    scratch only for blocks. Snapshot each readable operand before checking
    pair extents. Launch facts then determine the exact block shape and
    complete warp partition needed by planning.
    """

    primitive = "merge_sort_keys" if values is None else "merge_sort_pairs"
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.{primitive} group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.{primitive} requires a block or warp group"
        )
    if not isinstance(descending, bool):
        raise TypeError(
            f"{_SCOPE}.{primitive} descending must be a compile-time bool"
        )
    if (valid_items is None) != (oob_default is None):
        raise ValueError(
            "Merge Sort valid_items and oob_default must be provided together"
        )
    if temp_storage is not None:
        if group.kind != "block":
            raise ValueError(
                "Merge Sort temp_storage applies only to block groups"
            )
        _validate_temp_storage(primitive, temp_storage)
    keys = _snapshot_readable_payload(keys, name="keys", primitive=primitive)
    values = (
        None
        if values is None
        else _snapshot_readable_payload(
            values, name="values", primitive=primitive
        )
    )
    if values is not None and keys.items_per_thread != values.items_per_thread:
        raise ValueError(
            "Merge Sort keys and values must have matching "
            "items_per_thread extents"
        )

    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._merge_sort import provider_merge_sort

    launch = current_kernel_launch_facts()
    group = _resolve_primitive_group_from_launch(
        group, launch, feature=primitive
    )
    _require_complete_warp_partition(
        group, feature=primitive, exact_block_dim=launch.exact_block_dim
    )
    return provider_merge_sort(
        group=group,
        launch=launch,
        keys=keys,
        values=values,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
    )


def merge_sort_keys(
    group: BlockGroup | WarpGroup,
    keys: ReadableThreadDataLike[_KeyT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    descending: bool = False,
    valid_items: IntegerValue | None = None,
    oob_default: NumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData:
    """Sort a group's keys into ascending or descending order.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, ``this_warp()``, or ``this_warp().group_by(n)``
        with ``n`` of 1, 2, 4, 8, 16 or 32. ``group_by(1)`` sorts each
        thread's own items. ``this_block()`` needs a block size that is a
        power of two, such as ``block=64`` or ``block=(8, 4, 2)``. A warp
        or a group of lanes needs a block size that is a multiple of 32.

        Every thread of the group makes the call, including a thread whose
        items are all at or past ``valid_items``. A whole warp or group of
        lanes can skip the call while the others make it.
    keys : ThreadData, CuTe register tensor or TensorSSA
        This thread's keys, with the same item count in every thread. A key
        has one of the ten element types listed under
        :ref:`ThreadData <coop-thread-data>`.

        A register tensor or ``TensorSSA`` is read as a payload, and the
        result is a ``ThreadData`` that ``to_register_tensor`` or
        ``to_tensor_ssa`` converts back. An object of your own that has
        ``items_per_thread``, ``dtype``, ``len()`` and integer indexing is
        read as a payload too. With one key per thread, pass a payload of
        one item, such as the one ``load`` returns with
        ``items_per_thread=1``. A scalar such as ``keys[0]`` is rejected
        when the kernel is compiled, and so is a tensor in global or shared
        memory.
    descending : bool, optional
        A Python ``bool``. ``False``, the default, sorts from the smallest
        key to the largest. ``True`` sorts from the largest to the smallest.
    valid_items : int or CuTe integer, optional
        The number of items that take part in the sort: those at the first
        ``valid_items`` positions of the group's tile. Positions are in
        :term:`blocked` order, so with four items per thread
        ``valid_items=6`` covers the thread with rank 0 and the first two
        items of the thread with rank 1. The count ranges from 0 to the
        size of the tile, which is the group's size times the item count.
        The default, ``None``, sorts the whole tile. Pass ``valid_items``
        and ``oob_default`` together, or leave both out.

        The items at or past the count are left out of the sort. They still
        need a value, as every item of a payload does.

        A Python or NumPy integer is fixed when the kernel is compiled, and
        one out of range is rejected then. A CuTe integer, such as a kernel
        argument declared ``cutlass.Int64``, is read while the kernel runs,
        and one out of range traps on the device. Its type is ``Int8`` to
        ``Int64`` or ``Uint8`` to ``Uint32``. Every thread of the group
        passes the same count.
    oob_default : scalar, optional
        A bound on the keys that take part: an upper bound for an ascending
        sort, a lower bound for a descending sort. A key that takes part
        can be equal to it. Every thread of the group passes the same
        value.

        A Python integer or float must be within the range of the key
        type. A Python float needs a ``Float32`` or ``Float64`` key type,
        and it can also be an infinity. A NumPy or CuTe scalar must have
        the key type, and a CuTe scalar can be computed while the kernel
        runs. NaN is rejected when it is a Python or NumPy value. A CuTe
        value is used as it is, so keep NaN out of it.
    temp_storage : TempStorage, optional
        For blocks. Every call reserves scratch in shared memory. Leave
        this out and the compiler provides the scratch and a barrier after
        the call. Pass a :ref:`TempStorage <coop-temp-storage>` to share
        the bytes with other calls. A barrier then follows the call only if
        the ``TempStorage`` was created with ``auto_sync=True``.

        Warps and groups of lanes take ``None``. Each of those groups has
        scratch of its own, and a warp barrier for its lanes follows the
        call.

    Returns
    -------
    cuda.coop.cutlass.ThreadData
        This thread's part of the sorted tile, as a new payload with the
        element type, item count and alignment of ``keys``. ``keys`` keeps
        its values. With ``K`` items per thread, the thread whose
        ``group.rank()`` is ``t`` holds sorted positions ``t * K`` to
        ``t * K + K - 1``, so rank 0 holds the first keys.

        With ``valid_items``, the sorted keys fill the first
        ``valid_items`` positions. The positions after them hold
        unspecified values, which can differ from ``oob_default``. Pass
        the same count as ``valid_items`` to
        :func:`~cuda.coop.cutlass.store` on the same group, so that it
        writes the sorted keys only.

    Notes
    -----
    Keys are compared with the element type's own less-than or
    greater-than. Keys that compare equal come out in an unspecified order.
    With ``Float32`` or ``Float64`` keys, a NaN among the keys that take
    part leaves the whole result unspecified, the other keys included.

    CUB's comparison functor and ``StableSort`` have no counterpart here.
    ``descending`` is the only choice of order.

    See Also
    --------
    :cpp:struct:`cub::BlockMergeSort`, :cpp:struct:`cub::WarpMergeSort`
        The CUB classes behind the call.

    Examples
    --------
    ``sort_groups`` sorts each group of eight lanes by itself. A group
    holds 16 keys, two in each lane. Each run of 16 keys in ``destination``
    is then in ascending order, and the next run starts again from its own
    smallest key.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_merge_sort_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_merge_sort_documentation.py
        :language: python
        :start-after: # cutlass-merge-sort-groups-example-begin
        :end-before: # cutlass-merge-sort-groups-example-end
        :dedent: 4

    ``sort_prefix`` leaves the last three positions of a block's tile out
    of the sort. ``ordered`` holds the other keys in ascending order in its
    first ``valid_items`` positions, and ``store`` writes those positions
    only. The test's keys lie between -15 and 15, so ``oob_default=1000``
    sorts after all of them.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_merge_sort_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_merge_sort_documentation.py
        :language: python
        :start-after: # cutlass-merge-sort-example-begin
        :end-before: # cutlass-merge-sort-example-end
        :dedent: 4
    """
    return _merge_sort(
        group,
        keys,
        None,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
    )


def merge_sort_pairs(
    group: BlockGroup | WarpGroup,
    keys: ReadableThreadDataLike[_KeyT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    values: ReadableThreadDataLike[_ValueT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    descending: bool = False,
    valid_items: IntegerValue | None = None,
    oob_default: NumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData, ThreadData]:
    """Sort a group's keys and move each key's value with it.

    The same as :func:`~cuda.coop.cutlass.merge_sort_keys` with a second
    payload, ``values``. The other parameters are described there.

    Parameters
    ----------
    values : ThreadData, CuTe register tensor or TensorSSA
        This thread's values, one for each key: item ``i`` of ``values``
        belongs to item ``i`` of ``keys``, and the two have the same item
        count. A value has one of the ten element types listed under
        :ref:`ThreadData <coop-thread-data>`, and its type can differ from
        the key type. ``values`` takes every form that ``keys`` takes, and
        the two can have different forms, such as a ``ThreadData`` and a
        register tensor.

    Returns
    -------
    tuple of cuda.coop.cutlass.ThreadData
        ``(sorted_keys, sorted_values)``, two new payloads. The keys are
        placed as ``merge_sort_keys`` places them, and each value is at the
        position of its key. Each payload has the element type, item count
        and alignment of its input. ``keys`` and ``values`` stay unchanged.

        When keys compare equal, their values can come out in any order.
        With ``valid_items``, the positions after the first ``valid_items``
        hold unspecified values in both payloads.

    See Also
    --------
    :cpp:struct:`cub::BlockMergeSort`, :cpp:struct:`cub::WarpMergeSort`
        The CUB classes behind the call.

    Examples
    --------
    ``sort_prefix`` gives each key its position in the tile as its value.
    ``pair_keys`` holds the first ``valid_items`` keys of the tile in
    descending order, and ``pair_indices`` holds the position each of them
    came from. The sort is descending, so ``oob_default=-1000`` lies below
    the test's keys, which run from -15 to 15. Both sorts share
    ``scratch``.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_merge_sort_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_merge_sort_documentation.py
        :language: python
        :start-after: # cutlass-merge-sort-example-begin
        :end-before: # cutlass-merge-sort-example-end
        :dedent: 4
    """
    return _merge_sort(
        group,
        keys,
        values,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
    )


__all__ = ["merge_sort_keys", "merge_sort_pairs"]
