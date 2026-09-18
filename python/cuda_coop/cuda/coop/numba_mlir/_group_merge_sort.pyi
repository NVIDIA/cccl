# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed Merge Sort payload and group contracts."""

from typing import Callable, Literal, overload

import numpy as np
from typing_extensions import TypeVar

from cuda.coop._typing import (
    ContextualInitialValue,
    IntegerValue,
    PortableNumericScalar,
    PortableThreadDataLike,
    TempStorageLike,
    ThreadDataLike,
)

from ._thread_group import BlockGroup, WarpGroup

_KeyT = TypeVar("_KeyT", bound=PortableNumericScalar)
_ValueT = TypeVar("_ValueT", bound=PortableNumericScalar)

@overload
def merge_sort_keys(
    group: BlockGroup,
    keys: PortableThreadDataLike[_KeyT],
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
    keys: PortableThreadDataLike[_KeyT],
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
    keys: PortableThreadDataLike[_KeyT],
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
    keys: PortableThreadDataLike[_KeyT],
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
    keys: PortableThreadDataLike[_KeyT],
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
    keys: PortableThreadDataLike[_KeyT],
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
    keys: PortableThreadDataLike[_KeyT],
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
    keys: PortableThreadDataLike[_KeyT],
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
    keys: PortableThreadDataLike[_KeyT],
    values: PortableThreadDataLike[_ValueT],
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
    keys: PortableThreadDataLike[_KeyT],
    values: PortableThreadDataLike[_ValueT],
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
    keys: PortableThreadDataLike[_KeyT],
    values: PortableThreadDataLike[_ValueT],
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
    keys: PortableThreadDataLike[_KeyT],
    values: PortableThreadDataLike[_ValueT],
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
    keys: PortableThreadDataLike[_KeyT],
    values: PortableThreadDataLike[_ValueT],
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
    keys: PortableThreadDataLike[_KeyT],
    values: PortableThreadDataLike[_ValueT],
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
    keys: PortableThreadDataLike[_KeyT],
    values: PortableThreadDataLike[_ValueT],
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
    keys: PortableThreadDataLike[_KeyT],
    values: PortableThreadDataLike[_ValueT],
    /,
    *,
    descending: Literal[False] = False,
    valid_items: IntegerValue,
    oob_default: ContextualInitialValue[_KeyT],
    temp_storage: None = None,
    compare_op: Callable[[_KeyT, _KeyT], bool | np.bool_],
) -> tuple[ThreadDataLike[_KeyT], ThreadDataLike[_ValueT]]: ...
