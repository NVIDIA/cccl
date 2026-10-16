# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Attach the Numba-CUDA-MLIR scope to shared thread-group descriptors.

The shared descriptor supplies hierarchy and partition rules. This subclass
keeps the qualified API's compiler scope when a group is resolved or split.
Creating a descriptor does not query a running kernel or synchronize threads.
"""

from __future__ import annotations

from typing import Any

from cuda.coop._core import (
    ThreadHierarchy,
    make_thread_group,
    normalize_thread_level,
)
from cuda.coop._core.thread_group import ThreadGroup as CoreThreadGroup

_ROOT_SCOPE = __name__.rsplit(".", 1)[0]

Hierarchy = ThreadHierarchy


def _thread_group_method_marker(
    group: ThreadGroup,
    operation: str,
    *args: Any,
) -> Any:
    """Reject a group method that has not been consumed by the planner.

    Device compilation replaces recognized method calls with native helpers.
    This host marker has no runtime group implementation and must not reach
    ordinary device typing or execution.
    """

    del group, operation, args
    raise RuntimeError(
        "cuda.coop.numba_mlir ThreadGroup methods are compile-time kernel "
        "constructs and must be lowered by the whole-function planner"
    )


class ThreadGroup(CoreThreadGroup):
    """Describe a CUDA group that the Numba compiler resolves before typing.

    The descriptor identifies participating threads and their hierarchy.
    Queries and synchronization methods become generated device helpers; the
    descriptor itself is erased from kernel IR. Use these methods inside
    ``numba_cuda_mlir.cuda.jit`` code. Direct host calls to the method markers
    raise because no device thread or launch is available there.
    """

    def rank(self, level: str = "thread") -> Any:
        """Return the zero-based rank relative to a hierarchy level.

        An inner level selects the caller's constituent rank within this
        group; an outer level selects this group's rank in the outer group.
        For example, ``this_block().rank()`` gives thread rank within the
        block. The level must be constant. Return uint32, or uint64 if the
        group or queried level is grid.

        Mapped-group rank is meaningful only for members. See :ref:`ranks and
        sizes <coop-group-queries>` for supported levels and use ``rank_as``
        to request another integer dtype.
        """

        return self.rank_as(None, level)

    def count(self, level: str = "thread") -> Any:
        """Count units between this group and a hierarchy level.

        An inner level counts constituents in this group; an outer level
        counts groups of this kind in the outer group. Default ``count()``
        counts threads. The level must be constant. Return uint32, or uint64
        when the group or queried level is grid. Use ``count_as`` for
        another integer dtype.

        Block warp counts include a partial final warp. See :ref:`ranks and
        sizes <coop-group-queries>` for mapped-group limits.
        """

        return self.count_as(None, level)

    def rank_as(self, dtype: Any = None, level: str = "thread") -> Any:
        """Return the hierarchy rank in a selected integer dtype.

        ``dtype`` and ``level`` must be compile-time choices. ``None`` uses
        the same default dtype as ``rank``. Signed and unsigned 8-, 16-, 32-,
        and 64-bit integer types are supported; choose enough width for the
        possible ranks. Level and membership rules follow ``rank(level)``.
        """

        level = normalize_thread_level(
            level,
            scope=_ROOT_SCOPE,
            feature="ThreadGroup.rank",
        )
        return _thread_group_method_marker(self, "rank", dtype, level)

    def count_as(self, dtype: Any = None, level: str = "thread") -> Any:
        """Return the hierarchy count in a selected integer dtype.

        ``dtype`` and ``level`` must be compile-time choices. ``None`` uses
        the same default dtype as ``count``. Signed and unsigned 8-, 16-, 32-,
        and 64-bit integer types are supported; choose enough width for the
        possible counts. Level restrictions follow ``count(level)``.
        """

        level = normalize_thread_level(
            level,
            scope=_ROOT_SCOPE,
            feature="ThreadGroup.count",
        )
        return _thread_group_method_marker(self, "count", dtype, level)

    def sync(self) -> None:
        """Synchronize the participating members of this group.

        All participants must execute the call in converged control flow.
        The Numba backend supports thread, warp, logical-warp, block, and
        supported cluster scopes. Grid synchronization and synchronization of
        mapped physical-warp groups are unavailable. See :ref:`participation
        requirements <coop-participation>` for the primitive-call contract.
        """

        _thread_group_method_marker(self, "sync")

    def sync_aligned(self) -> None:
        """Synchronize an aligned group in converged control flow.

        This has the same supported scopes as ``sync``. For block and cluster
        groups, all threads in each participating block must execute the same
        synchronization instruction in converged control flow. Warp groups use
        their participating lane mask. See
        :ref:`participation requirements <coop-participation>`.
        """

        _thread_group_method_marker(self, "sync_aligned")

    def group_by(
        self,
        count: int,
        *,
        exhaustive: bool = True,
    ) -> ThreadGroup:
        """Partition the parent and retain this backend's descriptor type.

        ``count`` measures threads within a warp or physical warps within a
        block. The validation rules follow ``cuda.coop.ThreadGroup.group_by``.
        The returned descriptor does not synchronize or rearrange threads.
        """
        return super().group_by(count, exhaustive=exhaustive)

    def is_member(self) -> Any:
        """Return whether the calling thread belongs to this group.

        The device helper returns uint8, suitable for an ``if`` condition.
        It is zero for trailing threads excluded by a non-exhaustive mapped
        partition. Use membership to guard rank-dependent work. It does not
        ensure a primitive's convergence or complete participation.
        """

        return _thread_group_method_marker(self, "is_member")


# These names support explicit imports used by the adjacent typing stubs. They
# remain private to this module rather than expanding the qualified facade.
BlockGroup = ThreadGroup
WarpGroup = ThreadGroup


def _make_group(kind: str) -> ThreadGroup:
    """Create a current-group descriptor with this backend's scope."""
    return make_thread_group(
        kind,
        group_type=ThreadGroup,
        scope=_ROOT_SCOPE,
    )


def this_thread() -> ThreadGroup:
    """Describe the current thread."""

    return _make_group("thread")


def this_warp() -> ThreadGroup:
    """Describe the current physical warp."""

    return _make_group("warp")


def this_block() -> ThreadGroup:
    """Describe the current CUDA thread block."""

    return _make_group("block")


def this_cluster() -> ThreadGroup:
    """Describe the current cluster where the launch can represent it."""

    return _make_group("cluster")


def this_grid() -> ThreadGroup:
    """Describe the current grid where the launch can represent it."""

    return _make_group("grid")


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
