# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shuffle signatures for complete block groups."""

from typing import Literal, overload

from typing_extensions import TypeVar

from .._typing import (
    IntegerValue,
    PortableNumericScalar,
    PortableShuffleMode,
    PortableThreadDataLike,
    ScalarShuffleMode,
    ThreadDataLike,
)
from ._thread_group import BlockGroup

_ItemT = TypeVar("_ItemT", bound=PortableNumericScalar)

@overload
def shuffle(
    group: BlockGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    mode: PortableShuffleMode = "down",
    distance: Literal[1] = 1,
) -> ThreadDataLike[_ItemT]: ...
@overload
def shuffle(
    group: BlockGroup,
    value: _ItemT,
    /,
    *,
    mode: ScalarShuffleMode,
    distance: IntegerValue = 1,
) -> _ItemT: ...
