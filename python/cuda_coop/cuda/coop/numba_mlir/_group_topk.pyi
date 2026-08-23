# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Top-k signatures for block groups."""

from typing import TypeAlias

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
from .._typing import _IntegerValue as _IntegerValue
from .._typing import _PortableIntegerKey as _PortableIntegerKey
from .._typing import _ValidItems as _ValidItems
from ._temp_storage import TempStorage
from ._thread_group import _BlockGroup

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

_TopKKeyT = TypeVar("_TopKKeyT", bound=_NumbaOrderedItem)

_TopKValueT = TypeVar("_TopKValueT", bound=_NumbaPairValue)

def topk_max_keys(
    group: _BlockGroup,
    keys: _ThreadDataLike[_TopKKeyT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> _ThreadDataLike[_TopKKeyT]:
    """Select the largest keys into a fresh fixed-size block payload."""

def topk_min_keys(
    group: _BlockGroup,
    keys: _ThreadDataLike[_TopKKeyT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> _ThreadDataLike[_TopKKeyT]:
    """Select the smallest keys into a fresh fixed-size block payload."""

def topk_max_pairs(
    group: _BlockGroup,
    keys: _ThreadDataLike[_TopKKeyT],
    values: _ThreadDataLike[_TopKValueT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[_ThreadDataLike[_TopKKeyT], _ThreadDataLike[_TopKValueT]]:
    """Select largest-key pairs into fresh matching block payloads."""

def topk_min_pairs(
    group: _BlockGroup,
    keys: _ThreadDataLike[_TopKKeyT],
    values: _ThreadDataLike[_TopKValueT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[_ThreadDataLike[_TopKKeyT], _ThreadDataLike[_TopKValueT]]:
    """Select smallest-key pairs into fresh matching block payloads."""
