# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR marker for the current CUDA thread block."""

from __future__ import annotations

from cuda.coop._core.thread_group import ThreadGroup
from cuda.coop._core.thread_group import this_block as _core_this_block

from ._compiler._operations import group_operation


@group_operation("this_block")
def this_block() -> ThreadGroup:
    """Return the current block's compile-time group descriptor."""

    return _core_this_block()


__all__ = ["ThreadGroup", "this_block"]
