# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable exchange family."""

from typing_extensions import TypeVar

from cuda.coop._typing import ExchangeMode as _ExchangeMode
from cuda.coop._typing import ThreadDataLike as ThreadDataLike
from cuda.coop._typing import _PortableNumericScalar as _PortableNumericScalar

from .thread_group import _MemoryGroup

_PortableNumericT = TypeVar("_PortableNumericT", bound=_PortableNumericScalar)

def exchange(
    group: _MemoryGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: _ExchangeMode = "striped_to_blocked",
) -> ThreadDataLike[_PortableNumericT]:
    """Return a layout-rearranged ``ThreadData`` payload without mutation.

    ``group`` may be a complete block, physical warp, or logical warp. The
    portable modes are ``"striped_to_blocked"`` and ``"blocked_to_striped"``.
    ``value`` must own one or more items per participant; scalar inputs are
    not supported. The result preserves the input payload's shape and item type.
    """
