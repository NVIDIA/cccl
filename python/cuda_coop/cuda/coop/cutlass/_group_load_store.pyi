# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS load and store."""

from __future__ import annotations

from typing import overload

from .._typing import (
    BlockLoadStoreAlgorithm,
    IntegerValue,
    PortableNumericScalar,
    PortableThreadDataLike,
    ValidItems,
    WarpLoadStoreAlgorithm,
)
from ._temp_storage import TempStorage
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData
from ._thread_group import BlockGroup, WarpGroup
from ._typing import CutlassNumericT

@overload
def load(
    group: BlockGroup,
    source: object,
    output: ThreadData[CutlassNumericT],
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadData[CutlassNumericT]:
    """Populate and return ``output`` with a block tile."""

@overload
def load(
    group: BlockGroup,
    source: object,
    output: ThreadData[CutlassNumericT],
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems,
    oob_default: CutlassNumericT,
    offset: IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadData[CutlassNumericT]:
    """Populate a partial block tile and fill invalid items."""

@overload
def load(
    group: WarpGroup,
    source: object,
    output: ThreadData[CutlassNumericT],
    /,
    *,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: None = None,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> ThreadData[CutlassNumericT]:
    """Populate and return ``output`` with a physical- or logical-warp tile."""

@overload
def load(
    group: WarpGroup,
    source: object,
    output: ThreadData[CutlassNumericT],
    /,
    *,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems,
    oob_default: CutlassNumericT,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> ThreadData[CutlassNumericT]:
    """Populate a partial physical- or logical-warp tile and fill invalid items."""

@overload
def store(
    group: BlockGroup,
    destination: object,
    value: (
        PortableNumericScalar
        | PortableThreadDataLike
        | CutlassTensorSample
        | CutlassTensorSSASample
    ),
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Store a scalar or register payload across a block through CUB."""

@overload
def store(
    group: WarpGroup,
    destination: object,
    value: (
        PortableNumericScalar
        | PortableThreadDataLike
        | CutlassTensorSample
        | CutlassTensorSSASample
    ),
    /,
    *,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> None:
    """Store a scalar or register payload across a physical or logical warp."""

__all__ = [
    "load",
    "store",
]
