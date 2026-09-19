# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Built-in Merge Sort signatures for qualified block and warp payloads."""

from typing import Any, TypeAlias, overload

import numpy as np
from cutlass import Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32
from typing_extensions import TypeVar

from .._core.api.thread_group import BlockGroup, WarpGroup
from .._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ContextualInitialValue,
    TempStorageLike,
)
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData

_KeyT = TypeVar("_KeyT", bound=CommonNumericScalar)
_ValueT = TypeVar("_ValueT", bound=CommonNumericScalar)
_RegisterPayload: TypeAlias = CutlassTensorSample | CutlassTensorSSASample
_ValidItems: TypeAlias = (
    int
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | Int8
    | Int16
    | Int32
    | Int64
    | Uint8
    | Uint16
    | Uint32
)

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
) -> ThreadData[_KeyT]: ...
@overload
def merge_sort_keys(
    group: BlockGroup,
    keys: _RegisterPayload,
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[Any]: ...
@overload
def merge_sort_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[_KeyT]: ...
@overload
def merge_sort_keys(
    group: BlockGroup,
    keys: _RegisterPayload,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: CommonNumericScalar,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[Any]: ...
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
) -> ThreadData[_KeyT]: ...
@overload
def merge_sort_keys(
    group: WarpGroup,
    keys: _RegisterPayload,
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: None = None,
) -> ThreadData[Any]: ...
@overload
def merge_sort_keys(
    group: WarpGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: None = None,
) -> ThreadData[_KeyT]: ...
@overload
def merge_sort_keys(
    group: WarpGroup,
    keys: _RegisterPayload,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: CommonNumericScalar,
    temp_storage: None = None,
) -> ThreadData[Any]: ...
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
) -> tuple[ThreadData[_KeyT], ThreadData[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: BlockGroup,
    keys: _RegisterPayload,
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[Any], ThreadData[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: _RegisterPayload,
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[_KeyT], ThreadData[Any]]: ...
@overload
def merge_sort_pairs(
    group: BlockGroup,
    keys: _RegisterPayload,
    values: _RegisterPayload,
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[Any], ThreadData[Any]]: ...
@overload
def merge_sort_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[_KeyT], ThreadData[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: BlockGroup,
    keys: _RegisterPayload,
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: CommonNumericScalar,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[Any], ThreadData[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: _RegisterPayload,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[_KeyT], ThreadData[Any]]: ...
@overload
def merge_sort_pairs(
    group: BlockGroup,
    keys: _RegisterPayload,
    values: _RegisterPayload,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: CommonNumericScalar,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[Any], ThreadData[Any]]: ...
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
) -> tuple[ThreadData[_KeyT], ThreadData[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: WarpGroup,
    keys: _RegisterPayload,
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: None = None,
) -> tuple[ThreadData[Any], ThreadData[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: WarpGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: _RegisterPayload,
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: None = None,
) -> tuple[ThreadData[_KeyT], ThreadData[Any]]: ...
@overload
def merge_sort_pairs(
    group: WarpGroup,
    keys: _RegisterPayload,
    values: _RegisterPayload,
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: None = None,
) -> tuple[ThreadData[Any], ThreadData[Any]]: ...
@overload
def merge_sort_pairs(
    group: WarpGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: None = None,
) -> tuple[ThreadData[_KeyT], ThreadData[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: WarpGroup,
    keys: _RegisterPayload,
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: CommonNumericScalar,
    temp_storage: None = None,
) -> tuple[ThreadData[Any], ThreadData[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: WarpGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: _RegisterPayload,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: None = None,
) -> tuple[ThreadData[_KeyT], ThreadData[Any]]: ...
@overload
def merge_sort_pairs(
    group: WarpGroup,
    keys: _RegisterPayload,
    values: _RegisterPayload,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: CommonNumericScalar,
    temp_storage: None = None,
) -> tuple[ThreadData[Any], ThreadData[Any]]: ...
