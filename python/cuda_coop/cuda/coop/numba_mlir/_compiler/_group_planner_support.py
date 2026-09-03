# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# The family planners import this module's private support names explicitly.
# ruff: noqa: F401

"""Whole-function planning for Numba-CUDA-MLIR group-first primitives.

This module owns IR provenance, hierarchy caching, and call rewriting shared by
all primitive families.  Public signatures and provider construction live in
semantic ``_group_*`` and ``_lowering`` modules; callable recognition is exact
identity based.
"""

from __future__ import annotations

import inspect
from itertools import count
from numbers import Integral
from typing import Any

import numpy as np
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.errors import ForceLiteralArg
from numba_cuda_mlir.extending import (
    WholeFunctionPlanner,
    register_planner,
    require_launch_config,
)
from numba_cuda_mlir.numbair_transforms import ir

import cuda.coop._core.api as _portable_api
import cuda.coop._core.api._dispatch as _portable_dispatch
from cuda.coop._core import (
    LaunchFactOrigin,
    LaunchFacts,
    ThreadGroup,
    ThreadHierarchy,
    normalize_thread_level,
    resolve_thread_group,
)

from .. import _thread_group as _thread_groups
from ._operations import group_operation_name

_cuda_module = cuda

_NAME_COUNTER = count()
_PAYLOAD_DTYPE_LIKE = "like"
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
    """A group-first call was recognized but could not be lowered safely."""


def _group_operation_name(function: Any) -> str | None:
    """Return the group-first operation represented by one marker callable."""

    operation = group_operation_name(function)
    if operation is None:
        operation = _portable_dispatch._portable_group_operation_name(function)
    return operation


def _is_common_root_operation(function: Any, operation: str) -> bool:
    return _portable_dispatch._portable_group_operation_name(function) == operation


def _typed_group_payload_like(
    _prototype: Any,
    _is_array: bool,
    _dtype_policy: str,
    _items_per_thread: int | None = None,
) -> Any:
    raise GroupRewriteError(
        "typed group payload markers must be lowered before device compilation"
    )


# Support consumers import the private names they use explicitly.
