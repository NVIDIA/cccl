# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Discontinuity signatures for block groups."""

from collections.abc import Callable
from typing import Literal, overload

from typing_extensions import TypeVar

from .._typing import ScalarValue, ThreadDataLike
from ._temp_storage import TempStorage
from ._thread_group import BlockGroup

_ItemT = TypeVar("_ItemT")

_ScalarT = TypeVar("_ScalarT", bound=ScalarValue)

@overload
def discontinuity(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["heads"] = "heads",
    tile_predecessor_item: _ItemT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    flag_op: Callable[[_ItemT, _ItemT], object] | None = None,
) -> ThreadDataLike[int]:
    """Return fresh signed 32-bit head flags."""

@overload
def discontinuity(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["tails"],
    tile_predecessor_item: None = None,
    tile_successor_item: _ItemT | None = None,
    temp_storage: TempStorage | None = None,
    flag_op: Callable[[_ItemT, _ItemT], object] | None = None,
) -> ThreadDataLike[int]:
    """Return fresh signed 32-bit tail flags."""

@overload
def discontinuity(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["heads_and_tails"],
    tile_predecessor_item: _ItemT | None = None,
    tile_successor_item: _ItemT | None = None,
    temp_storage: TempStorage | None = None,
    flag_op: Callable[[_ItemT, _ItemT], object] | None = None,
) -> tuple[ThreadDataLike[int], ThreadDataLike[int]]:
    """Return fresh signed 32-bit head and tail flag payloads."""

@overload
def discontinuity(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["heads"] = "heads",
    tile_predecessor_item: _ScalarT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    flag_op: Callable[[_ScalarT, _ScalarT], object] | None = None,
) -> int:
    """Return one signed 32-bit scalar head flag per thread."""

@overload
def discontinuity(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["tails"],
    tile_predecessor_item: None = None,
    tile_successor_item: _ScalarT | None = None,
    temp_storage: TempStorage | None = None,
    flag_op: Callable[[_ScalarT, _ScalarT], object] | None = None,
) -> int:
    """Return one signed 32-bit scalar tail flag per thread."""

@overload
def discontinuity(
    group: BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["heads_and_tails"],
    tile_predecessor_item: _ScalarT | None = None,
    tile_successor_item: _ScalarT | None = None,
    temp_storage: TempStorage | None = None,
    flag_op: Callable[[_ScalarT, _ScalarT], object] | None = None,
) -> tuple[int, int]:
    """Return signed 32-bit scalar head and tail flags per thread."""
