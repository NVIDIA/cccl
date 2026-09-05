# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR bindings for shared CUDA thread-group descriptors."""

from __future__ import annotations

from typing import Any

from cuda.coop._core import ThreadHierarchy, make_thread_group, normalize_thread_level
from cuda.coop._core.thread_group import ThreadGroup as PortableThreadGroup

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


class ThreadGroup(PortableThreadGroup):
    """Compile-time CUDA group descriptor for the Numba-CUDA-MLIR frontend."""

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
        return super().group_by(count, exhaustive=exhaustive)

    def is_member(self) -> Any:
        """Return whether the current thread belongs to this group."""

        return _thread_group_method_marker(self, "is_member")


def _make_group(kind: str) -> ThreadGroup:
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
    """Describe the current CTA."""

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
