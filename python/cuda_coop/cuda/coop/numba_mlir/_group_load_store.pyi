# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block and Warp Load/Store signatures for Numba-CUDA-MLIR."""

from typing import overload

from typing_extensions import TypeVar

from .._typing import (
    BlockLoadStoreAlgorithm,
    IntegerValue,
    PortableNumericScalar,
    PortableThreadDataLike,
    TempStorageLike,
    ThreadDataLike,
    ValidItems,
    WarpLoadStoreAlgorithm,
)
from ._thread_group import BlockGroup, WarpGroup

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
) -> ThreadDataLike[_PortableNumericT]: ...
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
) -> ThreadDataLike[_PortableNumericT]: ...
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
) -> ThreadDataLike[_PortableNumericT]: ...
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
) -> ThreadDataLike[_PortableNumericT]: ...
@overload
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
) -> None: ...
@overload
def store(
    group: WarpGroup,
    destination: object,
    value: _PortableNumericT | PortableThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> None: ...
