# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR bindings for shared CUDA thread-group descriptors."""

from __future__ import annotations

from cuda.coop._core import ThreadHierarchy, make_thread_group
from cuda.coop._core.thread_group import ThreadGroup as PortableThreadGroup

_ROOT_SCOPE = __name__.rsplit(".", 1)[0]

Hierarchy = ThreadHierarchy


class ThreadGroup(PortableThreadGroup):
    """Compile-time CUDA group descriptor for the Numba-CUDA-MLIR frontend."""

    def group_by(
        self,
        count: int,
        *,
        exhaustive: bool = True,
    ) -> ThreadGroup:
        return super().group_by(count, exhaustive=exhaustive)


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
