# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exchange signatures for block, physical-Warp, and logical-Warp groups."""

from typing import Literal, TypeAlias, overload

from typing_extensions import TypeVar

from .._typing import (
    ExchangeMode,
    IntegralScalar,
    PortableNumericScalar,
    PortableThreadDataLike,
    SignedIntegerScalar,
    ThreadDataLike,
)
from ._thread_group import BlockGroup, WarpGroup

_ItemT = TypeVar("_ItemT", bound=PortableNumericScalar)
_RankT = TypeVar("_RankT", bound=SignedIntegerScalar)
_FlagT = TypeVar("_FlagT", bound=IntegralScalar)
_BlockLayoutExtension: TypeAlias = Literal[
    "warp_striped_to_blocked",
    "blocked_to_warp_striped",
]
_BlockLayoutMode: TypeAlias = ExchangeMode | _BlockLayoutExtension
_BlockScatterMode: TypeAlias = Literal[
    "scatter_to_blocked",
    "scatter_to_striped",
]

@overload
def exchange(
    group: BlockGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    mode: _BlockLayoutMode = "striped_to_blocked",
    ranks: None = None,
    valid_flags: None = None,
    warp_time_slicing: bool = False,
) -> ThreadDataLike[_ItemT]: ...
@overload
def exchange(
    group: BlockGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    mode: _BlockScatterMode,
    ranks: PortableThreadDataLike[_RankT],
    valid_flags: None = None,
    warp_time_slicing: bool = False,
) -> ThreadDataLike[_ItemT]: ...
@overload
def exchange(
    group: BlockGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["scatter_to_striped_guarded"],
    ranks: PortableThreadDataLike[_RankT],
    valid_flags: None = None,
    warp_time_slicing: Literal[False] = False,
) -> ThreadDataLike[_ItemT]: ...
@overload
def exchange(
    group: BlockGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["scatter_to_striped_flagged"],
    ranks: PortableThreadDataLike[_RankT],
    valid_flags: PortableThreadDataLike[_FlagT],
    warp_time_slicing: Literal[False] = False,
) -> ThreadDataLike[_ItemT]: ...
@overload
def exchange(
    group: WarpGroup,
    value: PortableThreadDataLike[_ItemT],
    /,
    *,
    mode: ExchangeMode = "striped_to_blocked",
    ranks: None = None,
    valid_flags: None = None,
    warp_time_slicing: Literal[False] = False,
) -> ThreadDataLike[_ItemT]: ...
