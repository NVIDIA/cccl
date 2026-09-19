# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Input-preserving block TopK signatures for qualified CuTe payloads."""

from typing import Any, TypeAlias, overload

import numpy as np
from cutlass import Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32
from typing_extensions import TypeVar

from .._core.api.thread_group import BlockGroup
from .._typing import CommonNumericScalar, CommonThreadDataLike, TempStorageLike
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData

_KeyT = TypeVar("_KeyT", bound=CommonNumericScalar)
_ValueT = TypeVar("_ValueT", bound=CommonNumericScalar)
_RegisterPayload: TypeAlias = CutlassTensorSample | CutlassTensorSSASample
_Count: TypeAlias = (
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
def topk_min_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    k: _Count,
    valid_items: _Count | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[_KeyT]: ...
@overload
def topk_min_keys(
    group: BlockGroup,
    keys: _RegisterPayload,
    /,
    *,
    k: _Count,
    valid_items: _Count | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[Any]: ...
@overload
def topk_min_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    k: _Count,
    valid_items: _Count | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[_KeyT], ThreadData[_ValueT]]: ...
@overload
def topk_min_pairs(
    group: BlockGroup,
    keys: _RegisterPayload,
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    k: _Count,
    valid_items: _Count | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[Any], ThreadData[_ValueT]]: ...
@overload
def topk_min_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: _RegisterPayload,
    /,
    *,
    k: _Count,
    valid_items: _Count | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[_KeyT], ThreadData[Any]]: ...
@overload
def topk_min_pairs(
    group: BlockGroup,
    keys: _RegisterPayload,
    values: _RegisterPayload,
    /,
    *,
    k: _Count,
    valid_items: _Count | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[Any], ThreadData[Any]]: ...
@overload
def topk_max_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    /,
    *,
    k: _Count,
    valid_items: _Count | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[_KeyT]: ...
@overload
def topk_max_keys(
    group: BlockGroup,
    keys: _RegisterPayload,
    /,
    *,
    k: _Count,
    valid_items: _Count | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[Any]: ...
@overload
def topk_max_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    k: _Count,
    valid_items: _Count | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[_KeyT], ThreadData[_ValueT]]: ...
@overload
def topk_max_pairs(
    group: BlockGroup,
    keys: _RegisterPayload,
    values: CommonThreadDataLike[_ValueT],
    /,
    *,
    k: _Count,
    valid_items: _Count | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[Any], ThreadData[_ValueT]]: ...
@overload
def topk_max_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT],
    values: _RegisterPayload,
    /,
    *,
    k: _Count,
    valid_items: _Count | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[_KeyT], ThreadData[Any]]: ...
@overload
def topk_max_pairs(
    group: BlockGroup,
    keys: _RegisterPayload,
    values: _RegisterPayload,
    /,
    *,
    k: _Count,
    valid_items: _Count | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[ThreadData[Any], ThreadData[Any]]: ...
