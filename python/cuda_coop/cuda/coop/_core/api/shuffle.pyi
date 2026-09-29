# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for common cooperative Shuffle."""

from typing import Literal

from typing_extensions import TypeVar

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonShuffleMode,
    CommonThreadDataLike,
    ThreadDataLike,
)

from .thread_group import BlockGroup

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)

def shuffle(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    mode: CommonShuffleMode = "down",
    distance: Literal[1] = 1,
) -> ThreadDataLike[_ItemT]:
    """Return a unit-shifted payload without mutating ``value``."""
