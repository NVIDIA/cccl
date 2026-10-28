# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Adapt inputs for block adjacent differences and head/tail flags.

Copy readable payloads or convert CuTe register inputs before planning.
Adjacent differences retain the value dtype; discontinuity returns Int32
flags. Both preserve the input and use separate output storage.
"""

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
    CommonNumericScalar,
    CommonThreadDataLike,
    IntegerValue,
    TempStorageLike,
)

_T = TypeVar("_T", bound=CommonNumericScalar)


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
    values: CommonThreadDataLike[_T]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    direction: Literal["left", "right"] = "left",
    valid_items: IntegerValue | None = None,
    tile_predecessor_item: CommonNumericScalar | None = None,
    tile_successor_item: CommonNumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData:
    """Return blocked neighbor differences, preserving the input payload.

    Parameters
    ----------
    group : ThreadGroup
        A complete ``this_block()`` group. Every member participates in the
        same call; multidimensional blocks use linear thread-rank order.
    values : ThreadData or CuTe register payload
        Readable fixed-size payload in blocked order. Signed and unsigned
        8-, 16-, 32-, and 64-bit integers, float32, and float64 are supported.
        Initialize every input slot, including any invalid suffix. CuTe
        register tensors and TensorSSA values are adapted through
        ``ThreadData.from_payload``.
    direction : {"left", "right"}, optional
        Compile-time choice, default ``"left"``. Subtract the previous item
        from the current item for left differences, or the next item from
        the current item for right differences.
    valid_items : integer, optional
        Block-uniform count in ``[0, block_size * items_per_thread]``.
        Omit it to process the full tile. The suffix beyond this count is
        copied unchanged. For right differences, this argument cannot be
        combined with ``tile_successor_item``.
    tile_predecessor_item, tile_successor_item : scalar, optional
        Block-uniform neighbor outside the tile. A runtime or NumPy scalar
        must match the input dtype. Python literal conversion follows the
        rules below. Left differences accept only a
        predecessor; right differences accept only a successor. Without that
        neighbor, the boundary input is copied unchanged.
    temp_storage : TempStorage, optional
        Explicit block scratch descriptor. Omit it for automatic storage.
        With ``auto_sync=False``, synchronize the block before reusing the
        descriptor in another primitive.

    Returns
    -------
    cuda.coop.cutlass.ThreadData
        Fresh values with the input dtype, extent, and minimum alignment.
        Invalid suffix items and a boundary without an external neighbor
        retain their input values. Only built-in subtraction is supported.

    Notes
    -----
    ``tile_predecessor_item`` and ``tile_successor_item`` given as NumPy or
    runtime scalars must match the input dtype. Plain Python integers must
    fit the input dtype; out-of-range values raise ValueError. Python floats
    require a floating-point input dtype. Finite values must fit its range;
    infinities and NaNs are accepted.

    ``valid_items`` may be a Python integer, a runtime signed integer up to
    64 bits, or a runtime unsigned integer up to 32 bits. An out-of-range
    static count raises ValueError when the kernel compiles. An out-of-range
    runtime count stops the kernel with a trap before conversion to CUB's
    int count.

    See the :doc:`Adjacent Difference visualization
    </python/coop_cutlass/visualizations/adjacent-difference>` for tile
    boundaries.

    Examples
    --------
    Compute left differences using zero as the tile predecessor, then
    mark run heads and tails. Scratch is synchronized automatically before
    reuse by the second collective.

    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_qualified_neighbors_examples.py
        :language: python
        :start-after: # qualified-neighbors-example-begin
        :end-before: # qualified-neighbors-example-end
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
    values: CommonThreadDataLike[_T]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    mode: Literal["heads", "tails", "heads_and_tails"] = "heads",
    tile_predecessor_item: CommonNumericScalar | None = None,
    tile_successor_item: CommonNumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData | tuple[ThreadData, ThreadData]:
    """Flag unequal neighbors in a full blocked tile.

    Parameters
    ----------
    group : ThreadGroup
        A complete ``this_block()`` group. Every member participates in the
        same call; multidimensional blocks use linear thread-rank order.
    values : ThreadData or CuTe register payload
        Readable fixed-size payload in blocked order. Signed and unsigned
        8-, 16-, 32-, and 64-bit integers, float32, and float64 are supported.
        CuTe register tensors and TensorSSA values are adapted through
        ``ThreadData.from_payload``.
    mode : {"heads", "tails", "heads_and_tails"}, optional
        Compile-time result selection, default ``"heads"``. Heads compare
        the previous item with the current item; tails compare the current
        item with the next item. Unequal neighbors produce one, otherwise
        zero.
    tile_predecessor_item, tile_successor_item : scalar, optional
        Block-uniform neighbor outside the tile. A runtime or NumPy scalar
        must match the input dtype. Python literal conversion follows the
        rules below. Heads accept a predecessor, tails
        accept a successor, and ``"heads_and_tails"`` accepts both. Without a
        predecessor the first head is one; without a successor the last tail
        is one.
    temp_storage : TempStorage, optional
        Explicit block scratch descriptor. Omit it for automatic storage.
        With ``auto_sync=False``, synchronize the block before reusing the
        descriptor in another primitive.

    Returns
    -------
    cuda.coop.cutlass.ThreadData or tuple of ThreadData
        Fresh int32 heads or tails, or ``(heads, tails)``. Each result has the
        input extent and minimum alignment. The input remains unchanged.

    Notes
    -----
    Partial tiles are unsupported. Padding participates in comparisons and
    can change the last valid tail. Only built-in inequality is supported.

    ``tile_predecessor_item`` and ``tile_successor_item`` given as NumPy or
    runtime scalars must match the input dtype. Plain Python integers must
    fit the input dtype; out-of-range values raise ValueError. Python floats
    require a floating-point input dtype. Finite values must fit its range;
    infinities and NaNs are accepted.

    See the :doc:`Discontinuity visualization
    </python/coop_cutlass/visualizations/discontinuity>` for head and tail
    flags.

    Examples
    --------
    Mark both ends of each equal-value run and compute adjacent
    differences over the same tile.

    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_qualified_neighbors_examples.py
        :language: python
        :start-after: # qualified-neighbors-example-begin
        :end-before: # qualified-neighbors-example-end
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
