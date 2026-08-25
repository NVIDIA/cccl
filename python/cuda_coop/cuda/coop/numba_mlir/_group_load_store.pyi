# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load and store signatures for block and warp groups."""

from typing import Any, Literal, overload

from typing_extensions import TypeVar

from .._typing import ThreadDataLike
from ._enums import (
    BlockLoadAlgorithm,
    BlockStoreAlgorithm,
    WarpLoadAlgorithm,
    WarpStoreAlgorithm,
)
from ._temp_storage import TempStorage
from ._thread_group import ThreadGroup

_ItemT = TypeVar("_ItemT")

@overload
def load(
    group: ThreadGroup[Literal["block"]],
    source: Any,
    output: ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: str | int | BlockLoadAlgorithm = "direct",
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    temp_storage: TempStorage | None = None,
) -> ThreadDataLike[_ItemT]:
    """Load a per-thread tile through a block group."""

@overload
def load(
    group: ThreadGroup[Literal["warp", "threads_within_warp"]],
    source: Any,
    output: ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: str | int | WarpLoadAlgorithm = "direct",
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    temp_storage: None = None,
) -> ThreadDataLike[_ItemT]:
    """Load a per-thread tile through a physical or logical warp."""

@overload
def store(
    group: ThreadGroup[Literal["block"]],
    destination: Any,
    value: Any,
    /,
    *,
    algorithm: str | int | BlockStoreAlgorithm = "direct",
    valid_items: Any = None,
    offset: Any = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Store a per-thread tile through a block group."""

@overload
def store(
    group: ThreadGroup[Literal["warp", "threads_within_warp"]],
    destination: Any,
    value: Any,
    /,
    *,
    algorithm: str | int | WarpStoreAlgorithm = "direct",
    valid_items: Any = None,
    offset: Any = None,
    temp_storage: None = None,
) -> None:
    """Store a per-thread tile through a physical or logical warp."""
