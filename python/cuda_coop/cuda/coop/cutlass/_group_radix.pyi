# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block radix ordering signatures for qualified CuTe payloads and scalars."""

from typing import Any, TypeAlias, overload

import numpy as np
from cutlass import (
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
)
from typing_extensions import TypeVar

from .._core.api.thread_group import BlockGroup
from .._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    TempStorageLike,
    ThreadDataLike,
)
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData

_IntegerKey: TypeAlias = (
    int | np.int32 | np.uint32 | np.int64 | np.uint64 | Int32 | Uint32 | Int64 | Uint64
)
_SortKey: TypeAlias = _IntegerKey | float | np.float32 | np.float64 | Float32 | Float64
_KeyT = TypeVar("_KeyT", bound=_SortKey)
_RankKeyT = TypeVar("_RankKeyT", bound=_IntegerKey)
_ValueT = TypeVar("_ValueT", bound=CommonNumericScalar)
_RegisterPayload: TypeAlias = CutlassTensorSample | CutlassTensorSSASample
_BitBound: TypeAlias = (
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
_PrefixOutput: TypeAlias = ThreadDataLike[np.int32] | ThreadDataLike[Int32]

@overload
def radix_sort_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    begin_bit: _BitBound = 0,
    end_bit: _BitBound | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
    blocked_to_striped: bool = False,
) -> ThreadData[_KeyT]: ...
@overload
def radix_sort_keys(
    group: BlockGroup,
    keys: _RegisterPayload,
    /,
    *,
    begin_bit: _BitBound = 0,
    end_bit: _BitBound | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
    blocked_to_striped: bool = False,
) -> ThreadData[Any]: ...
@overload
def radix_sort_keys(
    group: BlockGroup,
    keys: _KeyT,
    /,
    *,
    begin_bit: _BitBound = 0,
    end_bit: _BitBound | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
    blocked_to_striped: bool = False,
) -> _KeyT: ...
@overload
def radix_sort_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    begin_bit: _BitBound = 0,
    end_bit: _BitBound | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
    blocked_to_striped: bool = False,
) -> tuple[ThreadData[_KeyT], ThreadData[_ValueT]]: ...
@overload
def radix_sort_pairs(
    group: BlockGroup,
    keys: _RegisterPayload,
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    begin_bit: _BitBound = 0,
    end_bit: _BitBound | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
    blocked_to_striped: bool = False,
) -> tuple[ThreadData[Any], ThreadData[_ValueT]]: ...
@overload
def radix_sort_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: _RegisterPayload,
    /,
    *,
    begin_bit: _BitBound = 0,
    end_bit: _BitBound | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
    blocked_to_striped: bool = False,
) -> tuple[ThreadData[_KeyT], ThreadData[Any]]: ...
@overload
def radix_sort_pairs(
    group: BlockGroup,
    keys: _RegisterPayload,
    values: _RegisterPayload,
    /,
    *,
    begin_bit: _BitBound = 0,
    end_bit: _BitBound | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
    blocked_to_striped: bool = False,
) -> tuple[ThreadData[Any], ThreadData[Any]]: ...
@overload
def radix_sort_pairs(
    group: BlockGroup,
    keys: _KeyT,
    values: _ValueT,
    /,
    *,
    begin_bit: _BitBound = 0,
    end_bit: _BitBound | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
    blocked_to_striped: bool = False,
) -> tuple[_KeyT, _ValueT]: ...
@overload
def radix_rank(
    group: BlockGroup,
    keys: CommonThreadDataLike[_RankKeyT],
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    radix_bits: int | None = None,
    descending: bool = False,
    exclusive_digit_prefix: _PrefixOutput | None = None,
) -> ThreadData[Int32]: ...
@overload
def radix_rank(
    group: BlockGroup,
    keys: _RegisterPayload,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    radix_bits: int | None = None,
    descending: bool = False,
    exclusive_digit_prefix: _PrefixOutput | None = None,
) -> ThreadData[Int32]: ...
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
    exclusive_digit_prefix: _PrefixOutput | None = None,
) -> Int32: ...
