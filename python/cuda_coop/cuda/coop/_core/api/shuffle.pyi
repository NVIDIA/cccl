# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable shuffle family."""

from typing import Literal

from typing_extensions import TypeVar

from cuda.coop._typing import (
    PortableNumericScalar,
    PortableShuffleMode,
    ThreadDataLike,
)

from .thread_group import BlockGroup

_PortableNumericT = TypeVar("_PortableNumericT", bound=PortableNumericScalar)

def shuffle(
    group: BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: PortableShuffleMode = "down",
    distance: Literal[1] = 1,
) -> ThreadDataLike[_PortableNumericT]:
    """Return a unit-shifted block payload without mutating ``value``.

    ``group`` must be a complete physical block and ``value`` must be a
    fixed-size per-thread payload. ``mode="up"`` shifts the flattened blocked
    tile toward higher item ranks and leaves its first item undefined;
    ``mode="down"`` shifts toward lower ranks and leaves its last item
    undefined. ``distance`` is fixed at the portable value ``1``. Use a
    backend-qualified import for scalar Offset/Rotate or boundary outputs.
    """
