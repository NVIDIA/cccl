# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Literal, overload

import numpy as np
from typing_extensions import TypeVar

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ContextualInitialValue,
    IntegerValue,
    TempStorageLike,
    ThreadDataLike,
)

from .thread_group import BlockGroup

_T = TypeVar("_T", bound=CommonNumericScalar)

def adjacent_difference(
    group: BlockGroup,
    values: CommonThreadDataLike[_T],
    /,
    *,
    direction: Literal["left", "right"] = "left",
    valid_items: IntegerValue | None = None,
    tile_predecessor_item: ContextualInitialValue[_T] | None = None,
    tile_successor_item: ContextualInitialValue[_T] | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_T]: ...
@overload
def discontinuity(
    group: BlockGroup,
    values: CommonThreadDataLike[_T],
    /,
    *,
    mode: Literal["heads", "tails"] = "heads",
    tile_predecessor_item: ContextualInitialValue[_T] | None = None,
    tile_successor_item: ContextualInitialValue[_T] | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[np.int32]: ...
@overload
def discontinuity(
    group: BlockGroup,
    values: CommonThreadDataLike[_T],
    /,
    *,
    mode: Literal["heads_and_tails"],
    tile_predecessor_item: ContextualInitialValue[_T] | None = None,
    tile_successor_item: ContextualInitialValue[_T] | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadDataLike[np.int32], ThreadDataLike[np.int32]]: ...
