# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shuffle signatures for qualified Block arrays and scalars."""

from typing import Any, Literal, TypeAlias, overload

import numpy as np
from cutlass import Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32
from typing_extensions import TypeVar

from .._core.api.thread_group import BlockGroup
from .._typing import (
    CommonNumericScalar,
    CommonShuffleMode,
    CommonThreadDataLike,
    ScalarShuffleMode,
)
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)
_Distance: TypeAlias = (
    int
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | Int8
    | Int16
    | Int32
    | Int64
    | Uint8
    | Uint16
    | Uint32
)

@overload
def shuffle(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    mode: CommonShuffleMode = "down",
    distance: Literal[1] = 1,
) -> ThreadData[_ItemT]: ...
@overload
def shuffle(
    group: BlockGroup,
    value: CutlassTensorSample | CutlassTensorSSASample,
    /,
    *,
    mode: CommonShuffleMode = "down",
    distance: Literal[1] = 1,
) -> ThreadData[Any]: ...
@overload
def shuffle(
    group: BlockGroup,
    value: _ItemT,
    /,
    *,
    mode: ScalarShuffleMode,
    distance: _Distance = 1,
) -> _ItemT: ...
