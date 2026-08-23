# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Merge-sort signatures for block and warp groups."""

from collections.abc import Callable
from typing import TypeAlias, overload

from numpy import bool_ as _NumpyBool
from numpy import float16 as _NumpyFloat16
from numpy import float32 as _NumpyFloat32
from numpy import float64 as _NumpyFloat64
from numpy import int8 as _NumpyInt8
from numpy import int16 as _NumpyInt16
from numpy import int32 as _NumpyInt32
from numpy import int64 as _NumpyInt64
from numpy import uint8 as _NumpyUint8
from numpy import uint16 as _NumpyUint16
from numpy import uint32 as _NumpyUint32
from numpy import uint64 as _NumpyUint64
from typing_extensions import TypeVar

from .._typing import ThreadDataLike as _ThreadDataLike
from .._typing import _CompilerScalarLike as _CompilerScalarLike
from .._typing import _PortableIntegerKey as _PortableIntegerKey
from .._typing import _ValidItems as _ValidItems
from ._temp_storage import TempStorage
from ._thread_group import _BlockGroup, _WarpGroup

_NumbaOrderedItem: TypeAlias = (
    _PortableIntegerKey
    | bool
    | float
    | _NumpyBool
    | _NumpyInt8
    | _NumpyUint8
    | _NumpyInt16
    | _NumpyUint16
    | _NumpyFloat16
    | _NumpyFloat32
    | _NumpyFloat64
    | _CompilerScalarLike
)

_NumbaMergeSortKeyT = TypeVar("_NumbaMergeSortKeyT", bound=_NumbaOrderedItem)

_NumbaPairValue: TypeAlias = (
    bool
    | int
    | float
    | _NumpyBool
    | _NumpyInt8
    | _NumpyUint8
    | _NumpyInt16
    | _NumpyUint16
    | _NumpyInt32
    | _NumpyUint32
    | _NumpyInt64
    | _NumpyUint64
    | _NumpyFloat16
    | _NumpyFloat32
    | _NumpyFloat64
)

_NumbaPairValueT = TypeVar("_NumbaPairValueT", bound=_NumbaPairValue)

@overload
def merge_sort_keys(
    group: _BlockGroup,
    keys: _ThreadDataLike[_NumbaMergeSortKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> _ThreadDataLike[_NumbaMergeSortKeyT]:
    """Return fresh block-wide merge-sorted keys."""

@overload
def merge_sort_keys(
    group: _BlockGroup,
    keys: _NumbaMergeSortKeyT,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
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
    group: _WarpGroup,
    keys: _ThreadDataLike[_NumbaMergeSortKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> _ThreadDataLike[_NumbaMergeSortKeyT]:
    """Return fresh physical- or logical-warp merge-sorted keys."""

@overload
def merge_sort_keys(
    group: _WarpGroup,
    keys: _NumbaMergeSortKeyT,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
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
    group: _BlockGroup,
    keys: _ThreadDataLike[_NumbaMergeSortKeyT],
    values: _ThreadDataLike[_NumbaPairValueT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> tuple[
    _ThreadDataLike[_NumbaMergeSortKeyT],
    _ThreadDataLike[_NumbaPairValueT],
]:
    """Return fresh block-wide merge-sorted key/value payloads."""

@overload
def merge_sort_pairs(
    group: _BlockGroup,
    keys: _NumbaMergeSortKeyT,
    values: _NumbaPairValueT,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
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
    group: _WarpGroup,
    keys: _ThreadDataLike[_NumbaMergeSortKeyT],
    values: _ThreadDataLike[_NumbaPairValueT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> tuple[
    _ThreadDataLike[_NumbaMergeSortKeyT],
    _ThreadDataLike[_NumbaPairValueT],
]:
    """Return fresh physical- or logical-warp sorted key/value payloads."""

@overload
def merge_sort_pairs(
    group: _WarpGroup,
    keys: _NumbaMergeSortKeyT,
    values: _NumbaPairValueT,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> tuple[_NumbaMergeSortKeyT, _NumbaPairValueT]:
    """Return one fresh merge-sorted key/value pair per warp member."""
