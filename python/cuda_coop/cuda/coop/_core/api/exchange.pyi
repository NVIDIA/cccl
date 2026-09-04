# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for portable cooperative Exchange."""

from typing_extensions import TypeVar

from cuda.coop._typing import ExchangeMode, PortableNumericScalar, ThreadDataLike

from .thread_group import MemoryGroup

_ItemT = TypeVar("_ItemT", bound=PortableNumericScalar)

def exchange(
    group: MemoryGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    mode: ExchangeMode = "striped_to_blocked",
) -> ThreadDataLike[_ItemT]:
    """Return a layout-rearranged payload without mutating ``value``."""
