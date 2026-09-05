# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compatibility aliases for the canonical CUTLASS thread-group surface."""

from .._thread_group import (
    Hierarchy,
    ThreadGroup,
    ThreadHierarchy,
    cpp_level_expr,
    render_group_decl,
    render_group_decl_lines,
    render_hierarchy_decl,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)

__all__ = [
    "Hierarchy",
    "ThreadGroup",
    "ThreadHierarchy",
    "cpp_level_expr",
    "render_group_decl",
    "render_group_decl_lines",
    "render_hierarchy_decl",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]
