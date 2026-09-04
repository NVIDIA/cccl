# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for portable cooperative Shuffle."""

from typing import Literal

from typing_extensions import TypeVar

from cuda.coop._typing import (
    PortableNumericScalar,
    PortableShuffleMode,
    ThreadDataLike,
)

from .thread_group import BlockGroup

_ItemT = TypeVar("_ItemT", bound=PortableNumericScalar)

def shuffle(
    group: BlockGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    mode: PortableShuffleMode = "down",
    distance: Literal[1] = 1,
) -> ThreadDataLike[_ItemT]:
    """Return a unit-shifted payload without mutating ``value``."""
