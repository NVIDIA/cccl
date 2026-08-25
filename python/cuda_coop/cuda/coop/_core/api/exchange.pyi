# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable exchange family."""

from typing_extensions import TypeVar

from cuda.coop._typing import ExchangeMode, PortableNumericScalar, ThreadDataLike

from .thread_group import MemoryGroup

_PortableNumericT = TypeVar("_PortableNumericT", bound=PortableNumericScalar)

def exchange(
    group: MemoryGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: ExchangeMode = "striped_to_blocked",
) -> ThreadDataLike[_PortableNumericT]:
    """Return a layout-rearranged ``ThreadData`` payload without mutation.

    ``group`` may be a complete block, physical warp, or logical warp. The
    portable modes are ``"striped_to_blocked"`` and ``"blocked_to_striped"``.
    ``value`` must own one or more items per participant; scalar inputs are
    not supported. The result preserves the input payload's shape and item type.
    """
