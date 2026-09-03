# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block Load/Store signatures for the Numba-CUDA-MLIR backend."""

from typing import Literal

from typing_extensions import TypeVar

from .._typing import (
    BlockLoadStoreAlgorithm,
    IntegerValue,
    PortableNumericScalar,
    ThreadDataLike,
    ValidItems,
)
from ._enums import BlockLoadAlgorithm, BlockStoreAlgorithm
from ._temp_storage import TempStorage
from ._thread_group import ThreadGroup

_ItemT = TypeVar("_ItemT", bound=PortableNumericScalar)

def load(
    group: ThreadGroup[Literal["block"]],
    source: object,
    output: ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm | BlockLoadAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: _ItemT | int | float | None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadDataLike[_ItemT]: ...
def store(
    group: ThreadGroup[Literal["block"]],
    destination: object,
    value: _ItemT | ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm | BlockStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> None: ...
