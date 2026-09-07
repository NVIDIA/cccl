# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for portable cooperative load and store."""

from typing import overload

from typing_extensions import TypeVar

from cuda.coop._typing import (
    BlockLoadStoreAlgorithm,
    IntegerValue,
    PortableNumericScalar,
    PortableThreadDataLike,
    TempStorageLike,
    ThreadDataLike,
    ValidItems,
    WarpLoadStoreAlgorithm,
)

from .thread_group import BlockGroup, WarpGroup

_PortableNumericT = TypeVar("_PortableNumericT", bound=PortableNumericScalar)

@overload
def load(
    group: BlockGroup,
    source: object,
    output: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Populate and return ``output`` with one cooperative block tile."""

@overload
def load(
    group: BlockGroup,
    source: object,
    output: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems,
    oob_default: _PortableNumericT | int | float,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Populate a partial block tile and fill invalid items."""

@overload
def load(
    group: WarpGroup,
    source: object,
    output: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: None = None,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Populate and return ``output`` with a physical- or logical-warp tile."""

@overload
def load(
    group: WarpGroup,
    source: object,
    output: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems,
    oob_default: _PortableNumericT | int | float,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Populate a partial physical- or logical-warp tile and fill invalid items."""

@overload
def store(
    group: BlockGroup,
    destination: object,
    value: PortableNumericScalar | PortableThreadDataLike,
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
    value: PortableNumericScalar | PortableThreadDataLike,
    /,
    *,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> None:
    """Store one scalar or per-thread payload across a physical or logical warp."""
