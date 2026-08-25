# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Merge-sort signatures for block and warp groups."""

from collections.abc import Callable
from typing import TypeAlias, overload

import numpy as np
from typing_extensions import TypeVar

from .._typing import (
    CompilerScalarLike,
    PortableIntegerKey,
    ThreadDataLike,
    ValidItems,
)
from ._temp_storage import TempStorage
from ._thread_group import BlockGroup, WarpGroup

_NumbaOrderedItem: TypeAlias = (
    PortableIntegerKey
    | bool
    | float
    | np.bool_
    | np.int8
    | np.uint8
    | np.int16
    | np.uint16
    | np.float16
    | np.float32
    | np.float64
    | CompilerScalarLike
)

_NumbaMergeSortKeyT = TypeVar("_NumbaMergeSortKeyT", bound=_NumbaOrderedItem)

_NumbaPairValue: TypeAlias = (
    bool
    | int
    | float
    | np.bool_
    | np.int8
    | np.uint8
    | np.int16
    | np.uint16
    | np.int32
    | np.uint32
    | np.int64
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
)

_NumbaPairValueT = TypeVar("_NumbaPairValueT", bound=_NumbaPairValue)

@overload
def merge_sort_keys(
    group: BlockGroup,
    keys: ThreadDataLike[_NumbaMergeSortKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> ThreadDataLike[_NumbaMergeSortKeyT]:
    """Return fresh block-wide merge-sorted keys."""

@overload
def merge_sort_keys(
    group: BlockGroup,
    keys: _NumbaMergeSortKeyT,
    /,
    *,
    descending: bool = False,
    valid_items: ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> _NumbaMergeSortKeyT:
    """Return one fresh merge-sorted key per block member."""

@overload
def merge_sort_keys(
    group: WarpGroup,
    keys: ThreadDataLike[_NumbaMergeSortKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> ThreadDataLike[_NumbaMergeSortKeyT]:
    """Return fresh physical- or logical-warp merge-sorted keys."""

@overload
def merge_sort_keys(
    group: WarpGroup,
    keys: _NumbaMergeSortKeyT,
    /,
    *,
    descending: bool = False,
    valid_items: ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> _NumbaMergeSortKeyT:
    """Return one fresh merge-sorted key per warp member."""

@overload
def merge_sort_pairs(
    group: BlockGroup,
    keys: ThreadDataLike[_NumbaMergeSortKeyT],
    values: ThreadDataLike[_NumbaPairValueT],
    /,
    *,
    descending: bool = False,
    valid_items: ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> tuple[
    ThreadDataLike[_NumbaMergeSortKeyT],
    ThreadDataLike[_NumbaPairValueT],
]:
    """Return fresh block-wide merge-sorted key/value payloads."""

@overload
def merge_sort_pairs(
    group: BlockGroup,
    keys: _NumbaMergeSortKeyT,
    values: _NumbaPairValueT,
    /,
    *,
    descending: bool = False,
    valid_items: ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> tuple[_NumbaMergeSortKeyT, _NumbaPairValueT]:
    """Return one fresh merge-sorted key/value pair per block member."""

@overload
def merge_sort_pairs(
    group: WarpGroup,
    keys: ThreadDataLike[_NumbaMergeSortKeyT],
    values: ThreadDataLike[_NumbaPairValueT],
    /,
    *,
    descending: bool = False,
    valid_items: ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> tuple[
    ThreadDataLike[_NumbaMergeSortKeyT],
    ThreadDataLike[_NumbaPairValueT],
]:
    """Return fresh physical- or logical-warp sorted key/value payloads."""

@overload
def merge_sort_pairs(
    group: WarpGroup,
    keys: _NumbaMergeSortKeyT,
    values: _NumbaPairValueT,
    /,
    *,
    descending: bool = False,
    valid_items: ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> tuple[_NumbaMergeSortKeyT, _NumbaPairValueT]:
    """Return one fresh merge-sorted key/value pair per warp member."""
