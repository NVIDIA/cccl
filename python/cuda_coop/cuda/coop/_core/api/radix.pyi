# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block radix ordering signatures."""

from typing import TypeAlias

import numpy
from typing_extensions import TypeVar

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    CompilerIntegerLike,
    IntegerValue,
    TempStorageLike,
    ThreadDataLike,
)

from .thread_group import BlockGroup

_IntegerKey: TypeAlias = (
    int | numpy.int32 | numpy.uint32 | numpy.int64 | numpy.uint64 | CompilerIntegerLike
)
_KeyT = TypeVar("_KeyT", bound=_IntegerKey)
_RankKeyT = TypeVar("_RankKeyT", bound=_IntegerKey)
_ValueT = TypeVar("_ValueT", bound=CommonNumericScalar)

def radix_sort_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_KeyT]: ...
def radix_sort_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadDataLike[_KeyT], ThreadDataLike[_ValueT]]: ...
def radix_rank(
    group: BlockGroup,
    keys: CommonThreadDataLike[_RankKeyT],
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    radix_bits: int | None = None,
    descending: bool = False,
) -> ThreadDataLike[numpy.int32]: ...
