# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS adjacent difference."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, overload

from .._typing import PortableNumericScalar, ValidItems
from ._temp_storage import TempStorage
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData
from ._thread_group import BlockGroup
from ._typing import CutlassNumericT, ScalarValueT

_DifferenceOperator: TypeAlias = Literal["-", "sub", "subtract"]

@overload
def adjacent_difference(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    direction: Literal["left"] = "left",
    valid_items: ValidItems | None = None,
    tile_predecessor_item: CutlassNumericT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    difference_op: _DifferenceOperator | None = None,
) -> ThreadData[CutlassNumericT]:
    """Return left differences for CUTLASS ``value`` across ``group``.

    ``direction`` is left, ``valid_items`` may limit the tile,
    ``tile_predecessor_item`` supplies its boundary,
    ``tile_successor_item`` stays ``None``, ``temp_storage`` supplies scratch,
    ``difference_op`` selects subtraction.
    """

@overload
def adjacent_difference(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    direction: Literal["right"],
    valid_items: ValidItems | None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    difference_op: _DifferenceOperator | None = None,
) -> ThreadData[CutlassNumericT]:
    """Return right differences for CUTLASS ``value`` across ``group``.

    ``direction`` is right, ``valid_items`` may limit the tile, both
    ``tile_predecessor_item`` and ``tile_successor_item`` stay ``None``,
    ``temp_storage`` supplies scratch, and ``difference_op`` selects
    subtraction.
    """

@overload
def adjacent_difference(
    group: BlockGroup,
    value: ThreadData[CutlassNumericT],
    /,
    *,
    direction: Literal["right"],
    valid_items: None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: CutlassNumericT,
    temp_storage: TempStorage | None = None,
    difference_op: _DifferenceOperator | None = None,
) -> ThreadData[CutlassNumericT]:
    """Return full-tile right differences for CUTLASS ``value`` across ``group``.

    ``direction`` is right, ``valid_items`` and ``tile_predecessor_item`` stay
    ``None``, ``tile_successor_item`` supplies the boundary, ``temp_storage``
    supplies scratch, ``difference_op`` selects subtraction.
    """

@overload
def adjacent_difference(
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    direction: Literal["left"] = "left",
    valid_items: ValidItems | None = None,
    tile_predecessor_item: PortableNumericScalar | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    difference_op: _DifferenceOperator | None = None,
) -> ThreadData[Any]:
    """Return left differences for CUTLASS register ``value`` across ``group``.

    ``direction`` is left, ``valid_items`` may limit the tile,
    ``tile_predecessor_item`` supplies its boundary,
    ``tile_successor_item`` stays ``None``, ``temp_storage`` supplies scratch,
    ``difference_op`` selects subtraction.
    """

@overload
def adjacent_difference(
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    direction: Literal["right"],
    valid_items: ValidItems | None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    difference_op: _DifferenceOperator | None = None,
) -> ThreadData[Any]:
    """Return right differences for CUTLASS register ``value`` across ``group``.

    ``direction`` is right, ``valid_items`` may limit the tile, both
    ``tile_predecessor_item`` and ``tile_successor_item`` stay ``None``,
    ``temp_storage`` supplies scratch, and ``difference_op`` selects
    subtraction.
    """

@overload
def adjacent_difference(
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    direction: Literal["right"],
    valid_items: None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: PortableNumericScalar,
    temp_storage: TempStorage | None = None,
    difference_op: _DifferenceOperator | None = None,
) -> ThreadData[Any]:
    """Return full-tile right differences for register ``value`` across ``group``.

    ``direction`` is right, ``valid_items`` and ``tile_predecessor_item`` stay
    ``None``, ``tile_successor_item`` supplies the boundary, ``temp_storage``
    supplies scratch, ``difference_op`` selects subtraction.
    """

@overload
def adjacent_difference(
    group: BlockGroup,
    value: ScalarValueT,
    /,
    *,
    direction: Literal["left"] = "left",
    valid_items: ValidItems | None = None,
    tile_predecessor_item: ScalarValueT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    difference_op: _DifferenceOperator | None = None,
) -> ScalarValueT:
    """Return one left scalar difference for ``value`` across ``group``.

    ``direction`` is left, ``valid_items`` may limit the tile,
    ``tile_predecessor_item`` supplies its boundary,
    ``tile_successor_item`` stays ``None``, ``temp_storage`` supplies scratch,
    ``difference_op`` selects subtraction.
    """

@overload
def adjacent_difference(
    group: BlockGroup,
    value: ScalarValueT,
    /,
    *,
    direction: Literal["right"],
    valid_items: ValidItems | None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    difference_op: _DifferenceOperator | None = None,
) -> ScalarValueT:
    """Return one right scalar difference for ``value`` across ``group``.

    ``direction`` is right, ``valid_items`` may limit the tile, both
    ``tile_predecessor_item`` and ``tile_successor_item`` stay ``None``,
    ``temp_storage`` supplies scratch, and ``difference_op`` selects
    subtraction.
    """

@overload
def adjacent_difference(
    group: BlockGroup,
    value: ScalarValueT,
    /,
    *,
    direction: Literal["right"],
    valid_items: None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: ScalarValueT,
    temp_storage: TempStorage | None = None,
    difference_op: _DifferenceOperator | None = None,
) -> ScalarValueT:
    """Return a full-tile right scalar difference for ``value`` across ``group``.

    ``direction`` is right, ``valid_items`` and ``tile_predecessor_item`` stay
    ``None``, ``tile_successor_item`` supplies the boundary, ``temp_storage``
    supplies scratch, ``difference_op`` selects subtraction.
    """

__all__ = [
    "adjacent_difference",
]
