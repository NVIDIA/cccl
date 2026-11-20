# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A top-k call selects the smallest or largest keys of a block's tile.

``k`` is the number of keys to select from the :term:`tile`. The call
returns them in the first positions of the result, starting with thread 0,
in no defined order. The inputs keep their values.

.. figure:: visualizations/topk.svg
   :alt: Eight threads hold two keys each, with a tile position from 0 to
         15 under every key. Threads 6 and 7 are grey and marked ignored.
         In the result, thread 0 holds keys 11 and 14 with positions 8 and
         6, thread 1 holds keys 9 and 12 with positions 4 and 2, and every
         item of threads 2 to 7 is a question mark.
   :width: 100%

   ``topk_max_pairs`` with ``k=4`` and ``valid_items=12`` on eight threads
   with two items each. The value under each key is its position in the
   tile. The four largest of the first twelve keys are 14, 12, 11 and 9.
   They fill the first four positions of the result, here in the order 11,
   14, 9, 12, each with its value. The last four keys, 15 and 13 among
   them, are at or past ``valid_items`` and take no part. ``?`` marks a
   position that holds no selected key.

The four functions differ in which keys they select and in what they
return:

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 30 30 40

   * - Selects
     - Returns the keys
     - Returns the keys and their values
   * - The smallest keys
     - ``topk_min_keys``
     - ``topk_min_pairs``
   * - The largest keys
     - ``topk_max_keys``
     - ``topk_max_pairs``

``topk_min_pairs`` and ``topk_max_pairs`` take a second payload with one
value for each key. All four run on ``this_block()``, and each block of a
launch selects from its own tile. They take these controls:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Control
     - What it does
   * - ``k``
     - Sets the number of keys to select, from zero to the whole tile.
   * - ``valid_items``
     - Selects from the first ``valid_items`` positions of the tile only.
   * - ``temp_storage``
     - Puts the call's scratch in a :ref:`TempStorage <coop-temp-storage>`.

The number of keys selected is ``k``, or ``valid_items`` when that is
smaller. Sorting the selected keys with block merge sort requires a
power-of-two thread count, such as ``block=64``. Pass the result of
``topk_min_keys`` or ``topk_max_keys`` to
:func:`~cuda.coop.cutlass.merge_sort_keys` with ``valid_items`` set to that
number and an ``oob_default`` that sorts after them. After ``topk_min_pairs`` or
``topk_max_pairs``, pass both payloads to
:func:`~cuda.coop.cutlass.merge_sort_pairs` in the same way, and each value
moves with its key. This kernel stores the seven smallest keys of a block
in ascending order:

`Full example and test on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_topk_documentation.py>`__

.. literalinclude::
    ../../../python/cuda_coop_cutlass/tests/runtime/test_topk_documentation.py
    :language: python
    :start-after: # cutlass-topk-sorted-example-begin
    :end-before: # cutlass-topk-sorted-example-end
    :dedent: 4
"""

# Implementation notes.
#
# Adapt readable payloads for block TopK selection.
#
# Turn each input into a fixed-size ThreadData snapshot. Readable payloads,
# including read-only ones, are copied item by item; CuTe register tensors and
# TensorSSA values are converted. Selection reads only these copies and returns
# new ThreadData, so inputs remain unchanged. Keys and values keep independent
# element types and need matching extents. The selected prefix is unsorted.
#
# A runtime k or valid_items is widened to 64 bits and compared with the tile
# size before it is narrowed to CUB's int count, so a count such as 2**32
# traps and does not wrap to 0. A zero count skips the CUB call.

from __future__ import annotations

from typing import TypeVar

from cuda.coop.cutlass._planning.thread_group import ThreadGroup
from cuda.coop.cutlass._typing import IntegralScalar

from ._temp_storage import TempStorage
from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
    _snapshot_readable_payload,
)
from ._typing import (
    BlockGroup,
    NumericScalar,
    ReadableThreadDataLike,
    TempStorageLike,
)

_ValueT = TypeVar("_ValueT", bound=NumericScalar)

_KeyT = TypeVar("_KeyT", bound=NumericScalar)


def _topk(group, keys, values, *, selection, k, valid_items, temp_storage):
    """Validate the block and snapshot each input before lowering.

    Check explicit scratch type and matching pair extents here. The shared
    planner checks one-dimensional launch shape and static counts; the
    provider resolves numeric dtypes and emits the checked C++ call.
    """

    primitive = f"topk_{selection}_{'keys' if values is None else 'pairs'}"
    if not isinstance(group, ThreadGroup):
        raise TypeError(
            f"cuda.coop.cutlass.{primitive} group must be a ThreadGroup"
        )
    if group.kind != "block":
        raise NotImplementedError(
            "TopK supports only complete this_block() groups"
        )
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError("TopK temp_storage must be CUTLASS TempStorage")
    keys = _snapshot_readable_payload(keys, name="keys", primitive=primitive)
    if values is not None:
        values = _snapshot_readable_payload(
            values, name="values", primitive=primitive
        )
        if keys.items_per_thread != values.items_per_thread:
            raise ValueError(
                "TopK keys and values must have matching items_per_thread"
            )

    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._topk import provider_topk

    return provider_topk(
        group=group,
        launch=current_kernel_launch_facts(),
        keys=keys,
        values=values,
        selection=selection,
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def topk_min_keys(
    group: BlockGroup,
    keys: ReadableThreadDataLike[_KeyT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    k: IntegralScalar,
    valid_items: IntegralScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData:
    """Select the smallest keys of a block's tile.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``. The block is one-dimensional, with y and z sizes
        of 1, as in ``launch(grid=1, block=64)``. Every thread of the block
        makes the call, including a thread whose keys are all at or past
        ``valid_items``.
    keys : ThreadData, CuTe register tensor or TensorSSA
        This thread's keys. The tile is in :term:`blocked` order: item ``i``
        of the thread with rank ``t`` is at position
        ``t * items_per_thread + i``. A key has one of the ten element types
        listed under :ref:`ThreadData <coop-thread-data>`.

        A register tensor or ``TensorSSA`` is read as a payload, and the
        result is a ``ThreadData`` that ``to_register_tensor`` or
        ``to_tensor_ssa`` converts back. An object of your own that has
        ``items_per_thread``, ``dtype``, ``len()`` and integer indexing is
        read as a payload too. With one key per thread, pass a payload of
        one item, such as the one ``load`` returns with
        ``items_per_thread=1``. A scalar such as ``keys[0]`` is rejected
        when the kernel is compiled.
    k : int or CuTe integer
        The number of keys to select across the block, from 0 to ``N``,
        where ``N`` is the block size times the item count. Zero selects
        nothing. Every thread of the block passes the same count, and the
        package checks only that each thread's count is in range.

        A Python or NumPy integer is fixed when the kernel is compiled, and
        one out of range is rejected then. A CuTe integer, such as a kernel
        argument declared ``cutlass.Int32``, is read while the kernel runs,
        and one out of range traps on the device. Its type is ``Int8`` to
        ``Int64`` or ``Uint8`` to ``Uint32``.
    valid_items : int or CuTe integer, optional
        The number of keys that take part: those at the first
        ``valid_items`` positions of the tile. The count runs across the
        block, so with four items per thread ``valid_items=6`` covers
        thread 0 and the first two items of thread 1. When ``k`` is at
        least ``valid_items``, every one of those keys is selected. The
        default, ``None``, is the whole tile.

        The count takes the same forms and range as ``k``, and a count out
        of range is handled the same way. Every thread of the block passes
        the same ``valid_items``. One of the two counts can be a Python
        value while the other is a CuTe value.
    temp_storage : TempStorage, optional
        Every call needs scratch. Leave this out and the compiler provides
        the scratch and a barrier after the call. Pass a
        :ref:`TempStorage <coop-temp-storage>` to share the bytes with
        other calls. A barrier then follows the call only if the
        ``TempStorage`` was created with ``auto_sync=True``.

    Returns
    -------
    cuda.coop.cutlass.ThreadData
        A new payload with the element type, item count and alignment of
        ``keys``, which keeps its values. Across the block, the selected
        keys fill the first positions of the result in no defined order:
        ``k`` positions, or ``valid_items`` when that is smaller. With
        ``k=7``, four items per thread and ``valid_items`` left out, thread
        0 holds four selected keys, thread 1 holds three and the other
        threads hold none.

        The positions after the selected keys hold unspecified values, so
        read the selected positions only. Pass the number of keys selected
        as ``valid_items`` to the call that reads the result, such as
        :func:`~cuda.coop.cutlass.store`,
        :func:`~cuda.coop.cutlass.merge_sort_keys` or another top-k call.
        A kernel that reads the items itself reads item ``i`` of the
        thread with rank ``t`` only when ``t * items_per_thread + i`` is
        below that number.

    Notes
    -----
    Several equal keys can compete for the last selected positions. The
    call selects as many of them as there are positions left, and which
    ones it selects is unspecified.

    With ``Float32`` or ``Float64`` keys, ``-0.0`` and ``+0.0`` rank as
    equal, and a selected zero keeps its sign. NaN keys have no defined
    rank.

    The call runs CUB's internal block top-k, which has no public class.

    Examples
    --------
    ``select_extremes`` selects the seven smallest and the seven largest
    keys of a block of 64 threads from one payload, ``keys``. Each ``store``
    writes the seven selected positions.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_topk_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_topk_documentation.py
        :language: python
        :start-after: # cutlass-topk-keys-example-begin
        :end-before: # cutlass-topk-keys-example-end
        :dedent: 4
    """
    return _topk(
        group,
        keys,
        None,
        selection="min",
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def topk_max_keys(
    group: BlockGroup,
    keys: ReadableThreadDataLike[_KeyT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    k: IntegralScalar,
    valid_items: IntegralScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData:
    """Select the largest keys of a block's tile.

    The same as :func:`~cuda.coop.cutlass.topk_min_keys` with the largest
    keys selected. It takes the same parameters.

    Returns
    -------
    cuda.coop.cutlass.ThreadData
        A new payload. Across the block, the selected keys fill the first
        positions of the result in no defined order: ``k`` positions, or
        ``valid_items`` when that is smaller. The positions after them hold
        unspecified values. ``keys`` keeps its values.

    Notes
    -----
    The call runs CUB's internal block top-k, which has no public class.

    Examples
    --------
    ``high`` below holds the seven largest keys. The kernel in
    :doc:`tensor_memory_topk` makes this call with a ``k`` that it reads
    while it runs.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_topk_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_topk_documentation.py
        :language: python
        :start-after: # cutlass-topk-keys-example-begin
        :end-before: # cutlass-topk-keys-example-end
        :dedent: 4
    """
    return _topk(
        group,
        keys,
        None,
        selection="max",
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def topk_min_pairs(
    group: BlockGroup,
    keys: ReadableThreadDataLike[_KeyT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    values: ReadableThreadDataLike[_ValueT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    k: IntegralScalar,
    valid_items: IntegralScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData, ThreadData]:
    """Select the smallest keys of a block's tile, each with its value.

    The same as :func:`~cuda.coop.cutlass.topk_min_keys` with a second
    payload, ``values``. The other parameters are described there.

    Parameters
    ----------
    values : ThreadData, CuTe register tensor or TensorSSA
        This thread's values, one for each key: item ``i`` of ``values``
        belongs to item ``i`` of ``keys``. The two payloads have the same
        item count, and a call with different counts is rejected when the
        kernel is compiled. A value has one of the ten element types listed
        under :ref:`ThreadData <coop-thread-data>`. The value type can
        differ from the key type.

        ``values`` takes the same forms as ``keys``, and a scalar is
        rejected when the kernel is compiled. The two can have different
        forms, such as a register tensor and a ``TensorSSA``.

    Returns
    -------
    tuple of cuda.coop.cutlass.ThreadData
        The selected keys and their values, as two new payloads. The keys
        are placed as ``topk_min_keys`` places them, and each value is at
        the position of its key. Each payload has the element type, item
        count and alignment of its input. The positions after the selected
        keys hold unspecified values in both payloads. ``keys`` and
        ``values`` keep their values.

    Notes
    -----
    The call runs CUB's internal block top-k, which has no public class.

    Examples
    --------
    ``select_pairs`` gives each key its position in the tile as its value
    and leaves the last three positions out with ``valid_items``.
    ``low_indices`` then says where each of the seven smallest of the other
    keys came from, and ``high_indices`` does the same for the seven
    largest. Each of the four ``store`` calls writes the seven selected
    positions.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_topk_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_topk_documentation.py
        :language: python
        :start-after: # cutlass-topk-pairs-example-begin
        :end-before: # cutlass-topk-pairs-example-end
        :dedent: 4
    """
    if values is None:
        raise TypeError("TopK values must be a numeric ThreadData payload")
    return _topk(
        group,
        keys,
        values,
        selection="min",
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def topk_max_pairs(
    group: BlockGroup,
    keys: ReadableThreadDataLike[_KeyT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    values: ReadableThreadDataLike[_ValueT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    k: IntegralScalar,
    valid_items: IntegralScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData, ThreadData]:
    """Select the largest keys of a block's tile, each with its value.

    The same as :func:`~cuda.coop.cutlass.topk_min_pairs` with the largest
    keys selected. It takes the same parameters.

    Returns
    -------
    tuple of cuda.coop.cutlass.ThreadData
        The selected keys and their values, as two new payloads. Each value
        is at the position of its key. The positions after the selected
        keys hold unspecified values in both payloads. ``keys`` and
        ``values`` keep their values.

    Notes
    -----
    The call runs CUB's internal block top-k, which has no public class.

    Examples
    --------
    ``select_pairs`` below leaves the last three positions of the tile out
    with ``valid_items``. ``high`` and ``high_indices`` hold the seven
    largest of the other keys and the position each one came from.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_topk_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_topk_documentation.py
        :language: python
        :start-after: # cutlass-topk-pairs-example-begin
        :end-before: # cutlass-topk-pairs-example-end
        :dedent: 4
    """
    if values is None:
        raise TypeError("TopK values must be a numeric ThreadData payload")
    return _topk(
        group,
        keys,
        values,
        selection="max",
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


__all__ = ["topk_max_keys", "topk_max_pairs", "topk_min_keys", "topk_min_pairs"]
