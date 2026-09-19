# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block and Warp Load/Store signatures for Numba-CUDA-MLIR."""

from typing import overload

from typing_extensions import TypeVar

from .._typing import (
    BlockLoadStoreAlgorithm,
    CommonNumericScalar,
    CommonThreadDataLike,
    IntegerValue,
    TempStorageLike,
    ThreadDataLike,
    ValidItems,
    WarpLoadStoreAlgorithm,
)
from ._thread_group import BlockGroup, WarpGroup

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
) -> None: ...
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
) -> None: ...
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
) -> None: ...
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
) -> None: ...
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
) -> None: ...
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
) -> None: ...
