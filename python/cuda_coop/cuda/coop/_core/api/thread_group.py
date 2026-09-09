# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    """Describe the calling thread as a one-thread group.

    Returns
    -------
    cuda.coop.ThreadGroup
        A symbolic descriptor. Thread groups are not Load or Store targets.
        See :ref:`thread groups <coop-thread-groups>`.
    """

    return _core_this_thread()


def this_warp() -> ThreadGroup:
    """Describe the calling thread's physical warp.

    Returns
    -------
    cuda.coop.ThreadGroup
        A group of 32 consecutive threads in the block's linear thread
        order. Use :meth:`~cuda.coop.ThreadGroup.group_by` to form smaller
        logical warps.

    Notes
    -----
    Warp Load and Store require a block size divisible by 32. Constructing
    this descriptor does not turn a partial final warp into a complete
    group. See :ref:`participation requirements <coop-participation>`.
    """

    return _core_this_warp()


def this_block() -> ThreadGroup:
    """Describe all threads in the calling thread's CUDA block.

    Returns
    -------
    cuda.coop.ThreadGroup
        A descriptor whose size comes from the kernel launch's block
        dimensions. For multidimensional blocks, thread ranks are linearized
        with the x coordinate varying fastest.

    Notes
    -----
    The factory takes no size argument and does not synchronize the block.
    See :ref:`thread groups <coop-thread-groups>` and the primitive's
    :ref:`participation requirements <coop-participation>`.
    """

    return _core_this_block()


def this_cluster() -> ThreadGroup:
    """Describe the thread-block cluster containing the calling thread.

    Returns
    -------
    cuda.coop.ThreadGroup
        A symbolic descriptor. Cluster groups are not Load or Store targets.

    Notes
    -----
    Construction does not create a cluster or enable cluster scheduling.
    See :ref:`thread groups <coop-thread-groups>`.
    """

    return _core_this_cluster()


def this_grid() -> ThreadGroup:
    """Describe all threads in the current kernel grid.

    Returns
    -------
    cuda.coop.ThreadGroup
        A symbolic descriptor. Grid groups are not Load or Store targets.

    Notes
    -----
    Construction does not request a cooperative launch or synchronize
    the grid. See :ref:`thread groups <coop-thread-groups>`.
    """

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
