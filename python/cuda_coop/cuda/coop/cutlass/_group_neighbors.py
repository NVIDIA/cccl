# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Input-preserving block neighbor operations for CuTe kernels."""

from cuda.coop._core.block.neighbors import validate_neighbor_options
from cuda.coop._core.thread_group import ThreadGroup

from ._temp_storage import TempStorage
from ._thread_data import _snapshot_readable_payload


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
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"cuda.coop.cutlass.{operation} group must be a ThreadGroup")
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
            f"cuda.coop.cutlass.{operation} temp_storage must be CUTLASS TempStorage"
        )
    values = _snapshot_readable_payload(values, name="values", primitive=operation)
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
    group,
    values,
    /,
    *,
    direction="left",
    valid_items=None,
    tile_predecessor_item=None,
    tile_successor_item=None,
    temp_storage=None,
):
    """Return blocked neighbor differences, preserving the input payload.

    Parameters, partial tiles, boundary values, and scratch reuse follow
    :func:`cuda.coop.adjacent_difference`. The qualified API additionally
    accepts CuTe register tensors and immutable register vectors through
    :meth:`cuda.coop.cutlass.ThreadData.from_payload`.

    Returns
    -------
    cuda.coop.cutlass.ThreadData
        Fresh values with the input dtype, extent, and minimum alignment.
        Invalid suffix items and a boundary without an external neighbor
        retain their input values. Only built-in subtraction is supported.
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
    group,
    values,
    /,
    *,
    mode="heads",
    tile_predecessor_item=None,
    tile_successor_item=None,
    temp_storage=None,
):
    """Flag unequal neighbors in a full blocked tile.

    Parameters, participation, boundary flags, and scratch reuse follow
    :func:`cuda.coop.discontinuity`. CuTe register tensors and immutable
    register vectors are also accepted. Every block member participates,
    including for multidimensional blocks. Custom predicates are unsupported.

    Returns
    -------
    cuda.coop.cutlass.ThreadData or tuple of ThreadData
        Fresh int32 heads or tails, or ``(heads, tails)``. Each result has the
        input extent and minimum alignment. The input remains unchanged.
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
