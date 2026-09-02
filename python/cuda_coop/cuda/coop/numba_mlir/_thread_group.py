# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR markers for current CUDA thread groups."""

from __future__ import annotations

from cuda.coop._core.thread_group import ThreadGroup
from cuda.coop._core.thread_group import this_block as _core_this_block
from cuda.coop._core.thread_group import this_warp as _core_this_warp

from ._compiler._operations import group_operation


@group_operation("this_block")
def this_block() -> ThreadGroup:
    """Return the current block's compile-time group descriptor."""

    return _core_this_block()


@group_operation("this_warp")
def this_warp() -> ThreadGroup:
    """Return the current physical warp's compile-time descriptor."""

    return _core_this_warp()


__all__ = ["ThreadGroup", "this_block", "this_warp"]
