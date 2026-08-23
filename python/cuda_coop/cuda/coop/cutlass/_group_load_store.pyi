# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS load and store."""

from __future__ import annotations

from typing import overload

from .._typing import _BlockLoadStoreAlgorithm as _BlockLoadStoreAlgorithm
from .._typing import _IntegerValue as _IntegerValue
from .._typing import _PortableNumericScalar as _PortableNumericScalar
from .._typing import _PortableThreadDataLike as _PortableThreadDataLike
from .._typing import _ValidItems as _ValidItems
from .._typing import _WarpLoadStoreAlgorithm as _WarpLoadStoreAlgorithm
from ._temp_storage import TempStorage
from ._thread_data import ThreadData
from ._thread_data import _CutlassTensorSample as _CutlassTensorSample
from ._thread_data import _CutlassTensorSSASample as _CutlassTensorSSASample
from ._thread_group import _BlockGroup as _BlockGroup
from ._thread_group import _WarpGroup as _WarpGroup
from ._typing import _CutlassNumericT as _CutlassNumericT

@overload
def load(
    group: _BlockGroup,
    source: object,
    output: ThreadData[_CutlassNumericT],
    /,
    *,
    algorithm: _BlockLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    oob_default: None = None,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadData[_CutlassNumericT]:
    """Populate and return ``output`` with a block tile."""

@overload
def load(
    group: _BlockGroup,
    source: object,
    output: ThreadData[_CutlassNumericT],
    /,
    *,
    algorithm: _BlockLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems,
    oob_default: _CutlassNumericT,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadData[_CutlassNumericT]:
    """Populate a partial block tile and fill invalid items."""

@overload
def load(
    group: _WarpGroup,
    source: object,
    output: ThreadData[_CutlassNumericT],
    /,
    *,
    algorithm: _WarpLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    oob_default: None = None,
    offset: _IntegerValue | None = None,
    temp_storage: None = None,
) -> ThreadData[_CutlassNumericT]:
    """Populate and return ``output`` with a physical- or logical-warp tile."""

@overload
def load(
    group: _WarpGroup,
    source: object,
    output: ThreadData[_CutlassNumericT],
    /,
    *,
    algorithm: _WarpLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems,
    oob_default: _CutlassNumericT,
    offset: _IntegerValue | None = None,
    temp_storage: None = None,
) -> ThreadData[_CutlassNumericT]:
    """Populate a partial physical- or logical-warp tile and fill invalid items."""

@overload
def store(
    group: _BlockGroup,
    destination: object,
    value: (
        _PortableNumericScalar
        | _PortableThreadDataLike
        | _CutlassTensorSample
        | _CutlassTensorSSASample
    ),
    /,
    *,
    algorithm: _BlockLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Store a scalar or register payload across a block through CUB."""

@overload
def store(
    group: _WarpGroup,
    destination: object,
    value: (
        _PortableNumericScalar
        | _PortableThreadDataLike
        | _CutlassTensorSample
        | _CutlassTensorSSASample
    ),
    /,
    *,
    algorithm: _WarpLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    offset: _IntegerValue | None = None,
    temp_storage: None = None,
) -> None:
    """Store a scalar or register payload across a physical or logical warp."""

__all__ = [
    "load",
    "store",
]
