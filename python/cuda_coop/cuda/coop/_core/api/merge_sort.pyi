# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed Merge Sort payload and group contracts."""

from typing import overload

from typing_extensions import TypeVar

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ContextualInitialValue,
    IntegerValue,
    TempStorageLike,
    ThreadDataLike,
)

from .thread_group import BlockGroup, WarpGroup

_KeyT = TypeVar("_KeyT", bound=CommonNumericScalar)
_ValueT = TypeVar("_ValueT", bound=CommonNumericScalar)

@overload
def merge_sort_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_KeyT]: ...
@overload
def merge_sort_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    descending: bool = False,
    valid_items: IntegerValue,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_KeyT]: ...
@overload
def merge_sort_keys(
    group: WarpGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: None = None,
) -> ThreadDataLike[_KeyT]: ...
@overload
def merge_sort_keys(
    group: WarpGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    descending: bool = False,
    valid_items: IntegerValue,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: None = None,
) -> ThreadDataLike[_KeyT]: ...
@overload
def merge_sort_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadDataLike[_KeyT], ThreadDataLike[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    descending: bool = False,
    valid_items: IntegerValue,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadDataLike[_KeyT], ThreadDataLike[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: WarpGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: None = None,
) -> tuple[ThreadDataLike[_KeyT], ThreadDataLike[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: WarpGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    descending: bool = False,
    valid_items: IntegerValue,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: None = None,
) -> tuple[ThreadDataLike[_KeyT], ThreadDataLike[_ValueT]]: ...
