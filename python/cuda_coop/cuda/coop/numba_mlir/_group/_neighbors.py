# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualified block neighbor markers with optional scalar device callbacks.

These operations accept ThreadData or fixed-size local arrays and return
fresh per-thread arrays with the input extent. A result counts as ThreadData
for later common-API checks only when its input was ThreadData. The common
API defines participation, boundaries, and scratch use; this module adds
stateless callbacks to those contracts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeVar

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
from .._thread_group import BlockGroup
from ._marker import group_primitive_marker

_T = TypeVar("_T", bound=CommonNumericScalar)


@group_operation(
    "adjacent_difference",
    family_module="cuda.coop.numba_mlir._compiler._group_neighbors",
)
def adjacent_difference(
    group: BlockGroup,
    values: CommonThreadDataLike[_T] | numpy.ndarray,
    /,
    *,
    direction: Literal["left", "right"] = "left",
    valid_items: IntegerValue | None = None,
    tile_predecessor_item: CommonNumericScalar | None = None,
    tile_successor_item: CommonNumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
    difference_op: Callable[[_T, _T], _T] | None = None,
) -> ThreadDataLike[_T]:
    """Compute neighbor differences with an optional device operator.

    Shared parameters, participation, boundaries, partial tiles, and scratch
    behavior follow :func:`cuda.coop.adjacent_difference`.

    Additional parameters
    ---------------------
    values : ThreadDataLike or local array
        Fixed-size local arrays are accepted in addition to ThreadData.
    difference_op : callable, optional
        Stateless device-compilable ``difference_op(current, neighbor)``
        returning the input dtype. ``None`` selects subtraction. The call
        receives the previous neighbor for left differences and the next
        neighbor for right differences. Both arguments are scalar elements
        of the input dtype.

    Returns
    -------
    ThreadDataLike
        Fresh blocked values with the input dtype and extent, including the
        unchanged invalid suffix. Common API calls recognize this result as
        ThreadData only when the input was ThreadData.

    Examples
    --------
    Use device callbacks to measure absolute gaps to right neighbors and
    flag transitions whose magnitude exceeds two. The final gap uses zero
    as the tile successor.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_neighbor_examples.py
        :language: python
        :start-after: # qualified-neighbor-example-begin
        :end-before: # qualified-neighbor-example-end
        :dedent: 4
    """
    return group_primitive_marker(
        "adjacent_difference",
        group,
        values,
        direction=direction,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
        difference_op=difference_op,
    )


@group_operation(
    "discontinuity",
    family_module="cuda.coop.numba_mlir._compiler._group_neighbors",
)
def discontinuity(
    group: BlockGroup,
    values: CommonThreadDataLike[_T] | numpy.ndarray,
    /,
    *,
    mode: Literal["heads", "tails", "heads_and_tails"] = "heads",
    tile_predecessor_item: CommonNumericScalar | None = None,
    tile_successor_item: CommonNumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
    flag_op: Callable[[_T, _T], bool | np.bool_] | None = None,
) -> (
    ThreadDataLike[np.int32]
    | tuple[ThreadDataLike[np.int32], ThreadDataLike[np.int32]]
):
    """Flag adjacent items with an optional device predicate.

    Shared parameters, participation, boundaries, and scratch behavior follow
    :func:`cuda.coop.discontinuity`. Every item in the block tile must be
    initialized; this operation has no partial-tile count.

    Additional parameters
    ---------------------
    values : ThreadDataLike or local array
        Fixed-size local arrays are accepted in addition to ThreadData.
    flag_op : callable, optional
        Stateless device-compilable binary predicate. Heads evaluate
        ``flag_op(previous, current)``; tails evaluate
        ``flag_op(current, next)``. Both arguments are scalar elements of
        the input dtype, and the predicate returns a Boolean result. ``None``
        selects inequality.

    Returns
    -------
    ThreadDataLike or tuple of ThreadDataLike
        Fresh int32 flags, or ``(heads, tails)`` for
        ``mode="heads_and_tails"``. Each array has the input extent. Common
        API calls recognize these results as ThreadData only when the input
        was ThreadData.

    Examples
    --------
    Use device callbacks to measure absolute gaps to right neighbors and
    flag transitions whose magnitude exceeds two. The final gap uses zero
    as the tile successor.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_neighbor_examples.py
        :language: python
        :start-after: # qualified-neighbor-example-begin
        :end-before: # qualified-neighbor-example-end
        :dedent: 4
    """
    return group_primitive_marker(
        "discontinuity",
        group,
        values,
        mode=mode,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
        flag_op=flag_op,
    )


__all__ = ["adjacent_difference", "discontinuity"]
