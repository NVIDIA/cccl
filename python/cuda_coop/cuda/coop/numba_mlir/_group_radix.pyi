# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block radix ordering signatures."""

from typing import TypeAlias, overload

import numpy
from typing_extensions import TypeVar

from cuda.coop._typing import (
    CompilerIntegerLike,
    CompilerScalarLike,
    IntegerValue,
    PortableNumericScalar,
    PortableThreadDataLike,
    TempStorageLike,
    ThreadDataLike,
)

from ._thread_group import BlockGroup

_IntegerKey: TypeAlias = (
    int | numpy.int32 | numpy.uint32 | numpy.int64 | numpy.uint64 | CompilerIntegerLike
)
_KeyT = TypeVar(
    "_KeyT",
    bound=_IntegerKey | float | numpy.float32 | numpy.float64 | CompilerScalarLike,
)
_RankKeyT = TypeVar("_RankKeyT", bound=_IntegerKey)
_ValueT = TypeVar("_ValueT", bound=PortableNumericScalar)

@overload
def radix_sort_keys(
    group: BlockGroup,
    keys: PortableThreadDataLike[_KeyT],
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
    blocked_to_striped: bool = False,
) -> ThreadDataLike[_KeyT]: ...
@overload
def radix_sort_keys(
    group: BlockGroup,
    keys: _KeyT,
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
    blocked_to_striped: bool = False,
) -> _KeyT: ...
@overload
def radix_sort_pairs(
    group: BlockGroup,
    keys: PortableThreadDataLike[_KeyT],
    values: PortableThreadDataLike[_ValueT],
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
    blocked_to_striped: bool = False,
) -> tuple[ThreadDataLike[_KeyT], ThreadDataLike[_ValueT]]: ...
@overload
def radix_sort_pairs(
    group: BlockGroup,
    keys: _KeyT,
    values: _ValueT,
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
    blocked_to_striped: bool = False,
) -> tuple[_KeyT, _ValueT]: ...
@overload
def radix_rank(
    group: BlockGroup,
    keys: PortableThreadDataLike[_RankKeyT],
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    radix_bits: int | None = None,
    descending: bool = False,
    exclusive_digit_prefix: ThreadDataLike[numpy.int32] | None = None,
) -> ThreadDataLike[numpy.int32]: ...
@overload
def radix_rank(
    group: BlockGroup,
    keys: _RankKeyT,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    radix_bits: int | None = None,
    descending: bool = False,
    exclusive_digit_prefix: ThreadDataLike[numpy.int32] | None = None,
) -> numpy.int32: ...
