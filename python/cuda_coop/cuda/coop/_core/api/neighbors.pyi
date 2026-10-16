# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type common neighbor calls and their per-thread result payloads.

Differences retain the input scalar type. Discontinuity uses int32 flags and
returns either one payload or a pair, selected by the literal mode. Boundary
items follow the input's scalar type, including contextual Python literals.
The implementation documents primitive participation and boundary rules.
"""

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
