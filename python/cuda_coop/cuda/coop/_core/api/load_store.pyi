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
)

from .thread_group import BlockGroup

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

def store(
    group: BlockGroup,
    destination: object,
    value: _PortableNumericT | PortableThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Store one scalar or per-thread payload cooperatively across a block."""
