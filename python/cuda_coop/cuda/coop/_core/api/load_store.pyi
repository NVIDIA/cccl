# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for common cooperative load and store."""

from typing import overload

from typing_extensions import TypeVar

from cuda.coop._typing import (
    BlockLoadStoreAlgorithm,
    CommonNumericScalar,
    CommonThreadDataLike,
    IntegerValue,
    TempStorageLike,
    ThreadDataLike,
    ValidItems,
    WarpLoadStoreAlgorithm,
)

from .thread_group import BlockGroup, WarpGroup

_CommonNumericT = TypeVar("_CommonNumericT", bound=CommonNumericScalar)

@overload
def load(
    group: BlockGroup,
    source: object,
    output: ThreadDataLike[_CommonNumericT],
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Populate ``output`` in place with one cooperative block tile."""

@overload
def load(
    group: BlockGroup,
    source: object,
    output: ThreadDataLike[_CommonNumericT],
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems,
    oob_default: _CommonNumericT | int | float,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Populate a partial block tile and fill invalid items."""

@overload
def load(
    group: WarpGroup,
    source: object,
    output: ThreadDataLike[_CommonNumericT],
    /,
    *,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: None = None,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> None:
    """Populate ``output`` in place with one physical or logical warp tile."""

@overload
def load(
    group: WarpGroup,
    source: object,
    output: ThreadDataLike[_CommonNumericT],
    /,
    *,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems,
    oob_default: _CommonNumericT | int | float,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> None:
    """Populate a partial physical or logical warp tile and fill invalid items."""

@overload
def store(
    group: BlockGroup,
    destination: object,
    value: _CommonNumericT | CommonThreadDataLike[_CommonNumericT],
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Store one scalar or per-thread payload cooperatively across a block."""

@overload
def store(
    group: WarpGroup,
    destination: object,
    value: _CommonNumericT | CommonThreadDataLike[_CommonNumericT],
    /,
    *,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> None:
    """Store one scalar or per-thread payload across a Warp group."""
