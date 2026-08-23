# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable adjacent-difference family."""

from typing import Literal, overload

from typing_extensions import TypeVar

from cuda.coop._typing import TempStorageLike as TempStorageLike
from cuda.coop._typing import ThreadDataLike as ThreadDataLike
from cuda.coop._typing import _PortableNumericScalar as _PortableNumericScalar
from cuda.coop._typing import _ValidItems as _ValidItems

from .thread_group import _BlockGroup

_PortableNumericT = TypeVar("_PortableNumericT", bound=_PortableNumericScalar)

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    direction: Literal["left"] = "left",
    valid_items: _ValidItems | None = None,
    tile_predecessor_item: _PortableNumericT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return left differences for ``value`` across ``group``.

    ``direction`` is left, ``valid_items`` may limit the tile,
    ``tile_predecessor_item`` supplies its boundary,
    ``tile_successor_item`` stays ``None``, and ``temp_storage`` supplies
    optional scratch.
    """

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    direction: Literal["right"],
    valid_items: _ValidItems | None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return right differences for ``value`` across ``group``.

    ``direction`` is right, ``valid_items`` may limit the tile, both
    ``tile_predecessor_item`` and ``tile_successor_item`` stay ``None``, and
    ``temp_storage`` supplies optional scratch.
    """

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    direction: Literal["right"],
    valid_items: None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: _PortableNumericT,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return full-tile right differences for ``value`` across ``group``.

    ``direction`` is right, ``valid_items`` and ``tile_predecessor_item`` stay
    ``None``, ``tile_successor_item`` supplies the boundary, and
    ``temp_storage`` supplies optional scratch.
    """
