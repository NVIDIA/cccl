# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common constructors for the current CUDA thread groups.

The constructors either delegate to the active compiler backend or return the
backend-neutral symbolic group used during characterization and planning. This
module does not resolve launch facts or select primitive implementations.
"""

from __future__ import annotations

from typing import Any

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
from ._dispatch import _backend_member, _backend_module_name

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


def _group_constructor(
    name: str,
    fallback: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if _backend_module_name() is None:
        return fallback(*args, **kwargs)
    return _backend_member(name)(*args, **kwargs)


def this_thread() -> ThreadGroup:
    """Describe the calling thread as a one-thread group.

    Returns
    -------
    cuda.coop.ThreadGroup
        A descriptor for the calling thread. Its default ``rank()`` is zero
        and its default ``count()`` is one. Querying an outer level, such as
        ``thread.rank("block")``, gives the thread's rank within that level.

    See Also
    --------
    cuda.coop.ThreadGroup : Group queries and a complete executable example.

    Notes
    -----
    The descriptor uses the current kernel launch; see
    :ref:`thread groups <coop-thread-groups>` and
    :ref:`ranks and sizes <coop-group-queries>`. Constructing a descriptor
    does not synchronize threads or launch a kernel.
    """

    return _group_constructor("this_thread", _core_this_thread)


def this_warp() -> ThreadGroup:
    """Describe the calling thread's physical warp.

    Returns
    -------
    cuda.coop.ThreadGroup
        A group of 32 consecutive threads in the block's linear thread
        order. Use ``group_by(width)`` on this descriptor to form smaller
        logical warps.

    See Also
    --------
    cuda.coop.ThreadGroup.group_by : Logical-warp and mapped-group example.

    Notes
    -----
    Warp primitives require a block size divisible by 32; the descriptor
    does not turn a partial final warp into a complete group. The primitive
    documents its supported logical widths and
    :ref:`participation requirements <coop-participation>`.
    See :ref:`thread groups <coop-thread-groups>` for the group hierarchy.
    """

    return _group_constructor("this_warp", _core_this_warp)


def this_block() -> ThreadGroup:
    """Describe all threads in the calling thread's CUDA block.

    Returns
    -------
    cuda.coop.ThreadGroup
        A descriptor whose size comes from the kernel launch's block
        dimensions. For multidimensional blocks, thread ranks are linearized
        with the x coordinate varying fastest.

    See Also
    --------
    cuda.coop.ThreadGroup : Group queries and a complete executable example.

    Notes
    -----
    The factory takes no size argument and does not synchronize the block.
    See :ref:`thread groups <coop-thread-groups>` and the primitive's
    :ref:`participation requirements <coop-participation>`.
    """

    return _group_constructor("this_block", _core_this_block)


def this_cluster() -> ThreadGroup:
    """Describe the thread-block cluster containing the calling thread.

    Returns
    -------
    cuda.coop.ThreadGroup
        A descriptor spanning the blocks in the current launch's cluster.
        Cluster operations require supported hardware and cluster dimensions
        supplied through the compiler's launch interface.

    Notes
    -----
    The descriptor obtains its dimensions from the launch; it does not
    create a cluster or enable cluster scheduling. See
    :ref:`thread groups <coop-thread-groups>` and each primitive's supported
    scopes. Grid primitives are a separate, unsupported scope.

    Examples
    --------
    Launch a two-block cluster with Numba-CUDA-MLIR and query each block's
    rank within it. This example requires compute capability 9.0 or newer.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_group_examples.py
        :language: python
        :start-after: # cluster-example-begin
        :end-before: # cluster-example-end
        :dedent: 4
    """

    return _group_constructor("this_cluster", _core_this_cluster)


def this_grid() -> ThreadGroup:
    """Describe all threads in the current kernel grid.

    Returns
    -------
    cuda.coop.ThreadGroup
        A launch-wide descriptor for hierarchy queries. ``grid.rank()``
        gives the calling thread's linear rank and ``grid.count()`` gives
        the launch's thread count.

    See Also
    --------
    cuda.coop.ThreadGroup : Group queries and a complete executable example.

    Notes
    -----
    Grid primitives and grid synchronization are unavailable. Constructing
    this descriptor does not request a cooperative launch. See
    :ref:`thread groups <coop-thread-groups>` and
    :ref:`ranks and sizes <coop-group-queries>`.
    """

    group = _group_constructor("this_grid", _core_this_grid)
    if _backend_module_name() is not None and isinstance(group, ThreadGroup):
        assert group.hierarchy is not None
        # Backends distinguish common grid policy from qualified grid access.
        return group.with_hierarchy(group.hierarchy, source="common_root")
    return group


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
