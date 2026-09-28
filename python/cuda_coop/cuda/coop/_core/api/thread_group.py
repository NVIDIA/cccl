# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Construct symbolic descriptions of the current CUDA thread groups.

The compiler resolves these descriptions against the kernel launch.
"""

from __future__ import annotations

from ..thread_group import (
    Hierarchy,
    ThreadGroup,
    ThreadHierarchy,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)

_core_this_block = this_block
_core_this_cluster = this_cluster
_core_this_grid = this_grid
_core_this_thread = this_thread
_core_this_warp = this_warp

# These names support explicit imports used by the adjacent typing stubs. They
# are overload helpers, not additional root-package exports.
MemoryGroup = ThreadGroup
ReductionGroup = ThreadGroup
BlockGroup = ThreadGroup
WarpGroup = ThreadGroup


def this_thread() -> ThreadGroup:
    """Describe the current thread."""

    return _core_this_thread()


def this_warp() -> ThreadGroup:
    """Describe the current physical warp."""

    return _core_this_warp()


def this_block() -> ThreadGroup:
    """Describe the current CTA."""

    return _core_this_block()


def this_cluster() -> ThreadGroup:
    """Describe the current cluster."""

    return _core_this_cluster()


def this_grid() -> ThreadGroup:
    """Describe the current grid."""

    return _core_this_grid()


__all__ = [
    "Hierarchy",
    "ThreadGroup",
    "ThreadHierarchy",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]
