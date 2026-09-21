# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Literal, TypeAlias, overload

from cutlass import Int32
from typing_extensions import TypeVar

from .._core.api.thread_group import BlockGroup
from .._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ContextualInitialValue,
    IntegerValue,
    TempStorageLike,
)
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData

_T = TypeVar("_T", bound=CommonNumericScalar)
_RegisterPayload: TypeAlias = CutlassTensorSample | CutlassTensorSSASample

@overload
def adjacent_difference(
    group: BlockGroup,
    values: CommonThreadDataLike[_T],
    /,
    *,
    direction: Literal["left", "right"] = "left",
    valid_items: IntegerValue | None = None,
    tile_predecessor_item: ContextualInitialValue[_T] | None = None,
    tile_successor_item: ContextualInitialValue[_T] | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[_T]: ...
@overload
def adjacent_difference(
    group: BlockGroup,
    values: _RegisterPayload,
    /,
    *,
    direction: Literal["left", "right"] = "left",
    valid_items: IntegerValue | None = None,
    tile_predecessor_item: CommonNumericScalar | None = None,
    tile_successor_item: CommonNumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[Any]: ...
@overload
def discontinuity(
    group: BlockGroup,
    values: CommonThreadDataLike[_T],
    /,
    *,
    mode: Literal["heads", "tails"] = "heads",
    tile_predecessor_item: ContextualInitialValue[_T] | None = None,
    tile_successor_item: ContextualInitialValue[_T] | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[Int32]: ...
@overload
def discontinuity(
    group: BlockGroup,
    values: _RegisterPayload,
    /,
    *,
    mode: Literal["heads", "tails"] = "heads",
    tile_predecessor_item: CommonNumericScalar | None = None,
    tile_successor_item: CommonNumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[Int32]: ...
@overload
def discontinuity(
    group: BlockGroup,
    values: CommonThreadDataLike[_T],
    /,
    *,
    mode: Literal["heads_and_tails"],
    tile_predecessor_item: ContextualInitialValue[_T] | None = None,
    tile_successor_item: ContextualInitialValue[_T] | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[Int32], ThreadData[Int32]]: ...
@overload
def discontinuity(
    group: BlockGroup,
    values: _RegisterPayload,
    /,
    *,
    mode: Literal["heads_and_tails"],
    tile_predecessor_item: CommonNumericScalar | None = None,
    tile_successor_item: CommonNumericScalar | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[Int32], ThreadData[Int32]]: ...
