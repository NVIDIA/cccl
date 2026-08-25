# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shuffle signatures for block groups."""

from typing import Literal, overload

from typing_extensions import TypeVar

from .._typing import ThreadDataLike
from ._thread_group import ThreadGroup

_ItemT = TypeVar("_ItemT")

@overload
def shuffle(
    group: ThreadGroup[Literal["block"]],
    value: ThreadDataLike[_ItemT],
    /,
    *,
    mode: str = "down",
    distance: int = 1,
    block_prefix: None = None,
    block_suffix: None = None,
) -> ThreadDataLike[_ItemT]:
    """Shuffle a tile without exposing private boundary outputs."""

@overload
def shuffle(
    group: ThreadGroup[Literal["block"]],
    value: _ItemT,
    /,
    *,
    mode: str = "down",
    distance: int = 1,
    block_prefix: None = None,
    block_suffix: None = None,
) -> _ItemT:
    """Shuffle a scalar value without boundary outputs."""
