# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type MergeSort results and the supported optional-argument combinations.

Keys and values retain independent element types. Overloads pair a partial
count with its key-typed sentinel, restrict caller storage to block groups,
and require ``descending=False`` when a custom comparator is supplied.
The implementation docstrings define ordering and participation contracts.
"""

from collections.abc import Callable
from typing import Literal, overload

import numpy as np
from typing_extensions import TypeVar

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ContextualInitialValue,
    IntegerValue,
    TempStorageLike,
    ThreadDataLike,
)

from .._thread_group import BlockGroup, WarpGroup

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
    compare_op: None = None,
) -> ThreadDataLike[_KeyT]: ...
@overload
def merge_sort_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    descending: Literal[False] = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorageLike | None = None,
    compare_op: Callable[[_KeyT, _KeyT], bool | np.bool_],
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
    compare_op: None = None,
) -> ThreadDataLike[_KeyT]: ...
@overload
def merge_sort_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    descending: Literal[False] = False,
    valid_items: IntegerValue,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: TempStorageLike | None = None,
    compare_op: Callable[[_KeyT, _KeyT], bool | np.bool_],
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
    compare_op: None = None,
) -> ThreadDataLike[_KeyT]: ...
@overload
def merge_sort_keys(
    group: WarpGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    descending: Literal[False] = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: None = None,
    compare_op: Callable[[_KeyT, _KeyT], bool | np.bool_],
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
    compare_op: None = None,
) -> ThreadDataLike[_KeyT]: ...
@overload
def merge_sort_keys(
    group: WarpGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    descending: Literal[False] = False,
    valid_items: IntegerValue,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: None = None,
    compare_op: Callable[[_KeyT, _KeyT], bool | np.bool_],
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
    compare_op: None = None,
) -> tuple[ThreadDataLike[_KeyT], ThreadDataLike[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    descending: Literal[False] = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorageLike | None = None,
    compare_op: Callable[[_KeyT, _KeyT], bool | np.bool_],
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
    compare_op: None = None,
) -> tuple[ThreadDataLike[_KeyT], ThreadDataLike[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    descending: Literal[False] = False,
    valid_items: IntegerValue,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: TempStorageLike | None = None,
    compare_op: Callable[[_KeyT, _KeyT], bool | np.bool_],
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
    compare_op: None = None,
) -> tuple[ThreadDataLike[_KeyT], ThreadDataLike[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: WarpGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    descending: Literal[False] = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: None = None,
    compare_op: Callable[[_KeyT, _KeyT], bool | np.bool_],
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
    compare_op: None = None,
) -> tuple[ThreadDataLike[_KeyT], ThreadDataLike[_ValueT]]: ...
@overload
def merge_sort_pairs(
    group: WarpGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    descending: Literal[False] = False,
    valid_items: IntegerValue,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: None = None,
    compare_op: Callable[[_KeyT, _KeyT], bool | np.bool_],
) -> tuple[ThreadDataLike[_KeyT], ThreadDataLike[_ValueT]]: ...
