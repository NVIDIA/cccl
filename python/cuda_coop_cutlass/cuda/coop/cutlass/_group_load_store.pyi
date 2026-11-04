# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, overload

from typing_extensions import TypeVar

from cuda.coop.cutlass._typing import (
    BlockLoadStoreAlgorithm,
    IntegerValue,
    NumericScalar,
    ReadableThreadDataLike,
    TempStorageLike,
    ValidItems,
    WarpLoadStoreAlgorithm,
)

from ._thread_data import ThreadData
from ._typing import BlockGroup, WarpGroup

_NumericT = TypeVar("_NumericT", bound=NumericScalar)

@overload
def load(
    group: BlockGroup,
    source: object,
    /,
    *,
    items_per_thread: int,
    alignment: int | None = None,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[Any]:
    """Return a fresh payload containing one cooperative block tile."""

@overload
def load(
    group: BlockGroup,
    source: object,
    /,
    *,
    items_per_thread: int,
    alignment: int | None = None,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems,
    oob_default: NumericScalar | float,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[Any]:
    """Return a partial block tile with invalid items filled."""

@overload
def load(
    group: WarpGroup,
    source: object,
    /,
    *,
    items_per_thread: int,
    alignment: int | None = None,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: None = None,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> ThreadData[Any]:
    """Return a fresh payload containing one physical or logical warp tile."""

@overload
def load(
    group: WarpGroup,
    source: object,
    /,
    *,
    items_per_thread: int,
    alignment: int | None = None,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems,
    oob_default: NumericScalar | float,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> ThreadData[Any]:
    """Return a partial warp tile with invalid items filled."""

@overload
def store(
    group: BlockGroup,
    destination: object,
    value: _NumericT | ReadableThreadDataLike[_NumericT],
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
    value: _NumericT | ReadableThreadDataLike[_NumericT],
    /,
    *,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> None:
    """Store one scalar or per-thread payload across a Warp group."""
