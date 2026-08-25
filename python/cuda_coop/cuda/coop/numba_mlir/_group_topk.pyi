# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Top-k signatures for block groups."""

from typing import TypeAlias

import numpy as np
from typing_extensions import TypeVar

from .._typing import (
    CompilerScalarLike,
    IntegerValue,
    PortableIntegerKey,
    ThreadDataLike,
    ValidItems,
)
from ._temp_storage import TempStorage
from ._thread_group import BlockGroup

_NumbaOrderedItem: TypeAlias = (
    PortableIntegerKey
    | bool
    | float
    | np.bool_
    | np.int8
    | np.uint8
    | np.int16
    | np.uint16
    | np.float16
    | np.float32
    | np.float64
    | CompilerScalarLike
)

_NumbaPairValue: TypeAlias = (
    bool
    | int
    | float
    | np.bool_
    | np.int8
    | np.uint8
    | np.int16
    | np.uint16
    | np.int32
    | np.uint32
    | np.int64
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
)

_TopKKeyT = TypeVar("_TopKKeyT", bound=_NumbaOrderedItem)

_TopKValueT = TypeVar("_TopKValueT", bound=_NumbaPairValue)

def topk_max_keys(
    group: BlockGroup,
    keys: ThreadDataLike[_TopKKeyT],
    k: IntegerValue,
    /,
    *,
    valid_items: ValidItems | None = None,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadDataLike[_TopKKeyT]:
    """Select the largest keys into a fresh fixed-size block payload."""

def topk_min_keys(
    group: BlockGroup,
    keys: ThreadDataLike[_TopKKeyT],
    k: IntegerValue,
    /,
    *,
    valid_items: ValidItems | None = None,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadDataLike[_TopKKeyT]:
    """Select the smallest keys into a fresh fixed-size block payload."""

def topk_max_pairs(
    group: BlockGroup,
    keys: ThreadDataLike[_TopKKeyT],
    values: ThreadDataLike[_TopKValueT],
    k: IntegerValue,
    /,
    *,
    valid_items: ValidItems | None = None,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[ThreadDataLike[_TopKKeyT], ThreadDataLike[_TopKValueT]]:
    """Select largest-key pairs into fresh matching block payloads."""

def topk_min_pairs(
    group: BlockGroup,
    keys: ThreadDataLike[_TopKKeyT],
    values: ThreadDataLike[_TopKValueT],
    k: IntegerValue,
    /,
    *,
    valid_items: ValidItems | None = None,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[ThreadDataLike[_TopKKeyT], ThreadDataLike[_TopKValueT]]:
    """Select smallest-key pairs into fresh matching block payloads."""
