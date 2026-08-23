# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# The family planners import this module's private support names explicitly.
# ruff: noqa: F401

"""Whole-function planning for Numba-CUDA-MLIR group descriptors.

This module owns the imports and registries shared by hierarchy construction
and group-method lowering.
"""

from __future__ import annotations

import inspect
from itertools import count
from typing import Any

from numba_cuda_mlir import types
from numba_cuda_mlir.errors import ForceLiteralArg
from numba_cuda_mlir.extending import (
    WholeFunctionPlanner,
    register_planner,
    require_launch_config,
)
from numba_cuda_mlir.numbair_transforms import ir

from cuda.coop._core import (
    LaunchFactOrigin,
    LaunchFacts,
    ThreadGroup,
    ThreadHierarchy,
    normalize_thread_level,
    resolve_thread_group,
)
from cuda.coop._core import api as _portable_api

from .. import _thread_group as _thread_groups

_NAME_COUNTER = count()
_GROUP_CONSTRUCTORS = {
    _thread_groups.this_thread: _thread_groups.this_thread,
    _thread_groups.this_warp: _thread_groups.this_warp,
    _thread_groups.this_block: _thread_groups.this_block,
    _thread_groups.this_cluster: _thread_groups.this_cluster,
    _thread_groups.this_grid: _thread_groups.this_grid,
    _portable_api.this_thread: _thread_groups.this_thread,
    _portable_api.this_warp: _thread_groups.this_warp,
    _portable_api.this_block: _thread_groups.this_block,
    _portable_api.this_cluster: _thread_groups.this_cluster,
    _portable_api.this_grid: _thread_groups.this_grid,
}
_PORTABLE_GROUP_CONSTRUCTORS = frozenset(
    {
        _portable_api.this_thread,
        _portable_api.this_warp,
        _portable_api.this_block,
        _portable_api.this_cluster,
        _portable_api.this_grid,
    }
)
_GROUP_METHODS = frozenset(
    {
        "rank",
        "count",
        "rank_as",
        "count_as",
        "sync",
        "sync_aligned",
        "group_by",
        "is_member",
    }
)


class GroupRewriteError(Exception):
    """A group descriptor or method could not be lowered safely."""


# Support consumers import the private names they use explicitly.
