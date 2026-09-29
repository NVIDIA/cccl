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
    """Mark a group operation that the whole-function planner must erase."""

    del group, operation, args
    raise RuntimeError(
        "cuda.coop.numba_mlir ThreadGroup methods are compile-time kernel "
        "constructs and must be lowered by the whole-function planner"
    )


class ThreadGroup(CoreThreadGroup):
    """Describe a CUDA thread group for the Numba-CUDA-MLIR planner.

    Membership and launch resolution follow :class:`cuda.coop.ThreadGroup`.
    A descriptor can exist even when a particular operation does not support
    that group; operation planning checks support separately.
    """

    def rank(self, level: str = "thread") -> Any:
        """Return this group's rank relative to another hierarchy level."""

        return self.rank_as(None, level)

    def count(self, level: str = "thread") -> Any:
        """Return this group's count relative to another hierarchy level."""

        return self.count_as(None, level)

    def rank_as(self, dtype: Any = None, level: str = "thread") -> Any:
        level = normalize_thread_level(
            level,
            scope=_ROOT_SCOPE,
            feature="ThreadGroup.rank",
        )
        return _thread_group_method_marker(self, "rank", dtype, level)

    def count_as(self, dtype: Any = None, level: str = "thread") -> Any:
        level = normalize_thread_level(
            level,
            scope=_ROOT_SCOPE,
            feature="ThreadGroup.count",
        )
        return _thread_group_method_marker(self, "count", dtype, level)

    def sync(self) -> None:
        _thread_group_method_marker(self, "sync")

    def sync_aligned(self) -> None:
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
        """Return whether the current thread belongs to this group."""

        return _thread_group_method_marker(self, "is_member")


# These names support explicit imports used by the adjacent typing stubs. They
# remain private to this module rather than expanding the qualified facade.
ReductionGroup = ThreadGroup
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
