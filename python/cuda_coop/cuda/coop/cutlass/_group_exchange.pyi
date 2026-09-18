# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exchange signatures for qualified Block and Warp payloads."""

from typing import Any, Literal, TypeAlias, overload

import numpy as np
from cutlass import Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64
from typing_extensions import TypeVar

from .._core.api.thread_group import BlockGroup, WarpGroup
from .._typing import CommonNumericScalar, CommonThreadDataLike, ExchangeMode
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)
_RegisterPayload: TypeAlias = CutlassTensorSample | CutlassTensorSSASample
_RankScalar: TypeAlias = (
    int | np.int8 | np.int16 | np.int32 | np.int64 | Int8 | Int16 | Int32 | Int64
)
_FlagScalar: TypeAlias = (
    _RankScalar
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | Uint8
    | Uint16
    | Uint32
    | Uint64
)
_Ranks: TypeAlias = CommonThreadDataLike[_RankScalar] | _RegisterPayload
_Flags: TypeAlias = CommonThreadDataLike[_FlagScalar] | _RegisterPayload
_BlockLayoutMode: TypeAlias = (
    ExchangeMode | Literal["warp_striped_to_blocked", "blocked_to_warp_striped"]
)
_BlockScatterMode: TypeAlias = Literal["scatter_to_blocked", "scatter_to_striped"]

@overload
def exchange(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    mode: _BlockLayoutMode = "striped_to_blocked",
    ranks: None = None,
    valid_flags: None = None,
    warp_time_slicing: bool = False,
) -> ThreadData[_ItemT]: ...
@overload
def exchange(
    group: BlockGroup,
    value: _RegisterPayload,
    /,
    *,
    mode: _BlockLayoutMode = "striped_to_blocked",
    ranks: None = None,
    valid_flags: None = None,
    warp_time_slicing: bool = False,
) -> ThreadData[Any]: ...
@overload
def exchange(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    mode: _BlockScatterMode,
    ranks: _Ranks,
    valid_flags: None = None,
    warp_time_slicing: bool = False,
) -> ThreadData[_ItemT]: ...
@overload
def exchange(
    group: BlockGroup,
    value: _RegisterPayload,
    /,
    *,
    mode: _BlockScatterMode,
    ranks: _Ranks,
    valid_flags: None = None,
    warp_time_slicing: bool = False,
) -> ThreadData[Any]: ...
@overload
def exchange(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["scatter_to_striped_guarded"],
    ranks: _Ranks,
    valid_flags: None = None,
    warp_time_slicing: Literal[False] = False,
) -> ThreadData[_ItemT]: ...
@overload
def exchange(
    group: BlockGroup,
    value: _RegisterPayload,
    /,
    *,
    mode: Literal["scatter_to_striped_guarded"],
    ranks: _Ranks,
    valid_flags: None = None,
    warp_time_slicing: Literal[False] = False,
) -> ThreadData[Any]: ...
@overload
def exchange(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["scatter_to_striped_flagged"],
    ranks: _Ranks,
    valid_flags: _Flags,
    warp_time_slicing: Literal[False] = False,
) -> ThreadData[_ItemT]: ...
@overload
def exchange(
    group: BlockGroup,
    value: _RegisterPayload,
    /,
    *,
    mode: Literal["scatter_to_striped_flagged"],
    ranks: _Ranks,
    valid_flags: _Flags,
    warp_time_slicing: Literal[False] = False,
) -> ThreadData[Any]: ...
@overload
def exchange(
    group: WarpGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    mode: ExchangeMode = "striped_to_blocked",
    ranks: None = None,
    valid_flags: None = None,
    warp_time_slicing: Literal[False] = False,
) -> ThreadData[_ItemT]: ...
@overload
def exchange(
    group: WarpGroup,
    value: _RegisterPayload,
    /,
    *,
    mode: ExchangeMode = "striped_to_blocked",
    ranks: None = None,
    valid_flags: None = None,
    warp_time_slicing: Literal[False] = False,
) -> ThreadData[Any]: ...
