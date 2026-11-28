# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Two primitives compare each item of a block's tile with its neighbor.

``adjacent_difference`` subtracts the neighbor from the item.
``discontinuity`` flags each item that differs from its neighbor: a head
flag marks the first item of a run of equal values, and a tail flag marks
the last. Both read the payload in :term:`blocked` order: all of thread 0's
items, then thread 1's, and so on. A neighbor can therefore belong to
another thread. Here eight threads hold one item each:

.. code-block:: text

   values                                         2  2  5  5  9  9  9  3
   adjacent_difference(block, values)             2  0  3  0  4  0  0 -6
   adjacent_difference(..., direction="right")    0 -3  0 -4  0  0  6  3
   discontinuity(block, values)                   1  0  1  0  1  0  0  1
   discontinuity(..., mode="tails")               0  1  0  1  0  0  1  1

The two primitives side by side:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - ``adjacent_difference``
     - ``discontinuity``
   * - Each result
     - The item minus its neighbor
     - 1 if the item differs from its neighbor, 0 if the two are equal
   * - Element type of the result
     - That of the input
     - ``Int32``
   * - Neighbor
     - ``direction``: ``"left"`` (the default) for the item before, or
       ``"right"`` for the item after
     - ``mode``: ``"heads"`` (the default) for the item before, ``"tails"``
       for the item after, or ``"heads_and_tails"`` for both
   * - Item at the tile's edge, with no neighbor
     - Copied to the result
     - Flagged with 1

Each block compares its own :term:`tile`. The tile's first item has no item
before it, and its last item has none after it. When the data continues in
another tile, pass the item on the other side of the edge, and the edge item
is compared with it. The two primitives take these controls:

.. list-table::
   :header-rows: 1
   :widths: 28 34 38

   * - Control
     - Taken with
     - What it does
   * - ``tile_predecessor_item``
     - ``"left"``, ``"heads"``, ``"heads_and_tails"``
     - Gives the tile's first item a neighbor before it.
   * - ``tile_successor_item``
     - ``"right"`` without ``valid_items``, ``"tails"``,
       ``"heads_and_tails"``
     - Gives the tile's last item a neighbor after it.
   * - ``valid_items``
     - ``adjacent_difference``
     - Limits the call to the first ``valid_items`` items of the tile and
       copies the others.
   * - ``temp_storage``
     - Every call
     - Puts the call's scratch in a :ref:`TempStorage <coop-temp-storage>`.
"""

# Implementation notes.
#
# Adapt inputs for block adjacent differences and head/tail flags. Copy
# readable payloads or convert CuTe register inputs before planning.
# Adjacent differences retain the value dtype; discontinuity returns Int32
# flags. Both preserve the input and use separate output storage.

from __future__ import annotations

from typing import Literal, TypeVar

from cuda.coop.cutlass._planning.block.neighbors import (
    validate_neighbor_options,
)
from cuda.coop.cutlass._planning.thread_group import ThreadGroup

from ._temp_storage import TempStorage
from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
    _snapshot_readable_payload,
)
from ._typing import (
    BlockGroup,
    IntegerValue,
    NumericScalar,
    ReadableThreadDataLike,
    TempStorageLike,
)

_T = TypeVar("_T", bound=NumericScalar)


def _neighbors(
    group,
    values,
    *,
    operation,
    mode,
    valid_items=None,
    tile_predecessor_item=None,
    tile_successor_item=None,
    temp_storage=None,
):
    """Check group, mode, tile-neighbor options, and scratch type.

    The shared validator rejects outside-tile neighbors that the mode cannot
    use and a successor with a partial right difference. Copy the input before
    the provider resolves the dtype, the static or runtime count, scratch,
    and result arrays.
    """

    if not isinstance(group, ThreadGroup):
        raise TypeError(
            f"cuda.coop.cutlass.{operation} group must be a ThreadGroup"
        )
    if group.kind != "block":
        raise NotImplementedError(
            f"cuda.coop.cutlass.{operation} requires a block group"
        )
    validate_neighbor_options(
        operation,
        mode,
        partial=valid_items is not None,
        predecessor=tile_predecessor_item is not None,
        successor=tile_successor_item is not None,
    )
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError(
            f"cuda.coop.cutlass.{operation} temp_storage must be "
            "CUTLASS TempStorage"
        )
    values = _snapshot_readable_payload(
        values, name="values", primitive=operation
    )
    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._neighbors import provider_neighbors

    return provider_neighbors(
        group=group,
        launch=current_kernel_launch_facts(),
        values=values,
        operation=operation,
        mode=mode,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
    )


def adjacent_difference(
    group: BlockGroup,
    values: ReadableThreadDataLike[_T]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    direction: Literal["left", "right"] = "left",
    valid_items: IntegerValue | None = None,
    tile_predecessor_item: NumericScalar | None = None,
    tile_successor_item: NumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData:
    """Subtract the neighboring item from each item of a block's tile.

    With ``direction="left"`` each difference is an item minus the item
    before it. With ``"right"`` it is an item minus the item after it. An
    item with no neighbor is copied to the result.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, with one, two or three dimensions. Every thread
        of the block makes the call, with or without ``valid_items``.
    values : ThreadData, CuTe register tensor or TensorSSA
        The block's items: a payload with the same element type and item
        count in every thread. Every item needs a value, including the
        items at or past ``valid_items``. An item has one of the ten
        element types listed under :ref:`ThreadData <coop-thread-data>`.
        A register tensor or ``TensorSSA`` is read as a payload.
    direction : {"left", "right"}, optional
        A Python string. The default is ``"left"``.
    valid_items : int or CuTe integer, optional
        The number of items at the start of the tile that take part, from
        zero to the block size times the item count. Every thread passes
        the same count. The call treats the tile as ending at the count
        and copies the items at or past the count to the result. A CuTe
        count is a signed integer of up to 64 bits or an unsigned integer
        of up to 32 bits. A Python count out of range is rejected when the
        kernel is compiled, and a runtime count out of range traps on the
        device. Leave it out to take the whole tile.
    tile_predecessor_item : scalar, optional
        The item before the tile's first item, for ``direction="left"``.
        The first result is the first item minus this value, unless
        ``valid_items`` is zero, which copies the whole tile. Without this
        value, a left difference copies the first item to the first
        result. Only the thread of rank 0 uses the value. A right
        difference takes ``None``.

        The value is a Python number or a scalar computed while the kernel
        runs. A CuTe or NumPy scalar must have the element type of
        ``values``. A Python integer must fit that type. A Python float
        needs a floating type. A finite float must fit that type, and
        infinity and NaN are accepted.
    tile_successor_item : scalar, optional
        The item after the tile's last item, for ``direction="right"``
        when ``valid_items`` is left out. The last result is the last item
        minus this value. Without this value, a right difference copies
        the last item to the last result. Only the last thread of the
        block uses the value. It follows the type rules of
        ``tile_predecessor_item``. A left difference takes ``None``, and so
        does a right difference with ``valid_items``.
    temp_storage : TempStorage, optional
        Both directions use scratch. Leave this out and the compiler
        provides the scratch and a barrier after the call. Pass a
        :ref:`TempStorage <coop-temp-storage>` to share the bytes with
        other calls. A barrier then follows the call only if the
        ``TempStorage`` was created with ``auto_sync=True``.

    Returns
    -------
    cuda.coop.cutlass.ThreadData
        A new payload with the item count and element type of ``values``
        and, for a ``ThreadData`` input, its alignment. Each item holds the
        difference for the input item at the same index in the same
        thread, or a copy of that item. The input keeps its values.

    Notes
    -----
    The call always subtracts, and differences are computed in the element
    type. CUB's ``difference_op`` argument has no counterpart here.

    See Also
    --------
    :cpp:struct:`cub::BlockAdjacentDifference`
        The CUB class behind the call.

    Examples
    --------
    ``differences`` below holds each item minus the item before it, and
    ``tile_predecessor_item=0`` makes the first difference the first item
    minus zero. The ``adjacent_difference`` and ``discontinuity`` calls
    share ``scratch``, a ``TempStorage`` created with ``auto_sync=True``,
    so a barrier follows each of them.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_neighbors_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_neighbors_documentation.py
        :language: python
        :start-after: # cutlass-neighbors-example-begin
        :end-before: # cutlass-neighbors-example-end
        :dedent: 4

    A sequence that spans several tiles, one tile of 256 items on each trip
    of a loop. ``previous`` is zero on the first trip. On each later trip
    it is the last item of the previous tile, read through ``inputs``.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_neighbors_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_neighbors_documentation.py
        :language: python
        :start-after: # cutlass-neighbors-tiles-example-begin
        :end-before: # cutlass-neighbors-tiles-example-end
        :dedent: 4
    """
    return _neighbors(
        group,
        values,
        operation="adjacent_difference",
        mode=direction,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
    )


def discontinuity(
    group: BlockGroup,
    values: ReadableThreadDataLike[_T]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    mode: Literal["heads", "tails", "heads_and_tails"] = "heads",
    tile_predecessor_item: NumericScalar | None = None,
    tile_successor_item: NumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData | tuple[ThreadData, ThreadData]:
    """Flag the items of a block's tile that differ from their neighbor.

    A head flag is 1 where an item differs from the item before it and 0
    where the two are equal, so heads mark the first item of each run of
    equal values. A tail flag compares an item with the item after it in
    the same way, so tails mark the last item of each run.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, with one, two or three dimensions. Every thread
        of the block makes the call.
    values : ThreadData, CuTe register tensor or TensorSSA
        The block's items: a payload with the same element type and item
        count in every thread. Every item needs a value, and every item is
        compared. If the tile is padded, the tail flag of the last real
        item depends on the first padding value. An item has one of the
        ten element types listed under :ref:`ThreadData <coop-thread-data>`.
        A register tensor or ``TensorSSA`` is read as a payload.
    mode : {"heads", "tails", "heads_and_tails"}, optional
        A Python string. ``"heads"``, the default, returns the head flags.
        ``"tails"`` returns the tail flags. ``"heads_and_tails"`` returns
        both from one call.
    tile_predecessor_item : scalar, optional
        The item before the tile's first item, for ``"heads"`` and
        ``"heads_and_tails"``. The first head flag compares the first item
        with this value. Without this value, the first head flag is 1.
        Only the thread of rank 0 uses the value. It follows the type
        rules of ``tile_predecessor_item`` in
        :func:`~cuda.coop.cutlass.adjacent_difference`. ``"tails"`` takes
        ``None``.
    tile_successor_item : scalar, optional
        The item after the tile's last item, for ``"tails"`` and
        ``"heads_and_tails"``. The last tail flag compares the last item
        with this value. Without this value, the last tail flag is 1. Only
        the last thread of the block uses the value. It follows the same
        type rules. ``"heads"`` takes ``None``.
    temp_storage : TempStorage, optional
        Every mode uses scratch. Leave this out and the compiler provides
        the scratch and a barrier after the call. Pass a
        :ref:`TempStorage <coop-temp-storage>` to share the bytes with
        other calls. A barrier then follows the call only if the
        ``TempStorage`` was created with ``auto_sync=True``.

    Returns
    -------
    cuda.coop.cutlass.ThreadData or tuple of ThreadData
        ``Int32`` flags in a new payload with the item count of ``values``
        and, for a ``ThreadData`` input, its alignment.
        ``"heads_and_tails"`` returns two such payloads as the tuple
        ``(heads, tails)``. Each flag has the thread and index of the item
        it describes. The input keeps its values.

    Notes
    -----
    The comparison is always ``!=`` in the element type. CUB's ``flag_op``
    argument has no counterpart here.

    See Also
    --------
    :cpp:struct:`cub::BlockDiscontinuity`
        The CUB class behind the call.

    Examples
    --------
    ``heads`` and ``tails`` below come from one call with
    ``mode="heads_and_tails"``. It reads ``values``, the same payload as the
    ``adjacent_difference`` call.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_neighbors_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_neighbors_documentation.py
        :language: python
        :start-after: # cutlass-neighbors-example-begin
        :end-before: # cutlass-neighbors-example-end
        :dedent: 4
    """
    return _neighbors(
        group,
        values,
        operation="discontinuity",
        mode=mode,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
    )


__all__ = ["adjacent_difference", "discontinuity"]
