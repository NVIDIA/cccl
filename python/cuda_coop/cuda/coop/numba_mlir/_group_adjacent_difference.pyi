# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Adjacent-difference signatures for block groups."""

from collections.abc import Callable
from typing import Literal, overload

from typing_extensions import TypeVar

from .._typing import ThreadDataLike as _ThreadDataLike
from .._typing import _ScalarValue as _ScalarValue
from .._typing import _ValidItems as _ValidItems
from ._temp_storage import TempStorage
from ._thread_group import _BlockGroup

_ItemT = TypeVar("_ItemT")

_ScalarT = TypeVar("_ScalarT", bound=_ScalarValue)

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    direction: Literal["left"] = "left",
    valid_items: _ValidItems | None = None,
    tile_predecessor_item: _ItemT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    difference_op: Callable[[_ItemT, _ItemT], _ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return left differences in a fresh per-thread payload."""

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    direction: Literal["right"],
    valid_items: _ValidItems | None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    difference_op: Callable[[_ItemT, _ItemT], _ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return right differences for a full or partial tile."""

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    direction: Literal["right"],
    valid_items: None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: _ItemT,
    temp_storage: TempStorage | None = None,
    difference_op: Callable[[_ItemT, _ItemT], _ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return right differences with a full-tile successor boundary."""

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    direction: Literal["left"] = "left",
    valid_items: _ValidItems | None = None,
    tile_predecessor_item: _ScalarT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    difference_op: Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return one left scalar difference per thread."""

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    direction: Literal["right"],
    valid_items: _ValidItems | None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    difference_op: Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return one right scalar difference per thread."""

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    direction: Literal["right"],
    valid_items: None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: _ScalarT,
    temp_storage: TempStorage | None = None,
    difference_op: Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return one right scalar difference with a successor boundary."""
