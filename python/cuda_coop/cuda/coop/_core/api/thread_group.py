# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable constructors for the current CUDA thread groups.

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
    """Describe the current thread."""

    return _group_constructor("this_thread", _core_this_thread)


def this_warp() -> ThreadGroup:
    """Describe the current physical warp."""

    return _group_constructor("this_warp", _core_this_warp)


def this_block() -> ThreadGroup:
    """Describe the current CTA."""

    return _group_constructor("this_block", _core_this_block)


def this_cluster() -> ThreadGroup:
    """Describe the current cluster."""

    return _group_constructor("this_cluster", _core_this_cluster)


def this_grid() -> ThreadGroup:
    """Describe the current grid."""

    group = _group_constructor("this_grid", _core_this_grid)
    if _backend_module_name() is not None and isinstance(group, ThreadGroup):
        assert group.hierarchy is not None
        # Backends distinguish portable grid policy from qualified grid access.
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
