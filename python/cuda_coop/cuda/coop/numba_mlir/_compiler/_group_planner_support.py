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
import operator
from itertools import count
from numbers import Integral
from typing import Any

import numpy as np
from numba_cuda_mlir import cuda as _cuda_module
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
from .._group_adjacent_difference import adjacent_difference
from .._group_discontinuity import discontinuity
from .._group_exchange import exchange
from .._group_histogram import histogram
from .._group_load_store import load, store
from .._group_merge_sort import merge_sort_keys, merge_sort_pairs
from .._group_radix import radix_rank, radix_sort_keys, radix_sort_pairs
from .._group_reduce import reduce, sum
from .._group_run_length_decode import run_length_decode
from .._group_scan import (
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
from .._group_shuffle import shuffle
from .._group_topk import (
    topk_max_keys,
    topk_max_pairs,
    topk_min_keys,
    topk_min_pairs,
)
from .._scan_op import ScanOp
from ._operations import group_operation_name

_NAME_COUNTER = count()
_PAYLOAD_DTYPE_INT32 = "int32"
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
_QUALIFIED_OPERATIONS = (
    "load",
    "store",
    "reduce",
    "sum",
    "scan",
    "exclusive_sum",
    "inclusive_sum",
    "exclusive_scan",
    "inclusive_scan",
    "exchange",
    "adjacent_difference",
    "discontinuity",
    "shuffle",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_pairs",
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
    "histogram",
    "run_length_decode",
)
_ROOT_OPERATIONS = {
    function: group_operation_name(function)
    for function in (
        adjacent_difference,
        discontinuity,
        exchange,
        exclusive_scan,
        exclusive_sum,
        histogram,
        inclusive_scan,
        inclusive_sum,
        load,
        merge_sort_keys,
        merge_sort_pairs,
        radix_rank,
        radix_sort_keys,
        radix_sort_pairs,
        reduce,
        run_length_decode,
        scan,
        shuffle,
        store,
        sum,
        topk_max_keys,
        topk_max_pairs,
        topk_min_keys,
        topk_min_pairs,
    )
}
_ROOT_OPERATIONS.update(
    {
        getattr(_portable_api, name): name
        for name in (
            "load",
            "radix_rank",
            "radix_sort_keys",
            "radix_sort_pairs",
            "store",
            "reduce",
            "sum",
            "scan",
            "exclusive_sum",
            "inclusive_sum",
            "exclusive_scan",
            "inclusive_scan",
            "exchange",
            "adjacent_difference",
            "discontinuity",
            "shuffle",
            "merge_sort_keys",
            "merge_sort_pairs",
            "topk_max_keys",
            "topk_max_pairs",
            "topk_min_keys",
            "topk_min_pairs",
            "histogram",
            "run_length_decode",
        )
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


def _builtin_less(lhs: Any, rhs: Any) -> bool:
    return lhs < rhs


def _builtin_greater(lhs: Any, rhs: Any) -> bool:
    return lhs > rhs


def _static_index(scope_name: str, operation: str, name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{scope_name}.{operation} {name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{scope_name}.{operation} {name} must be an integer") from exc


def _static_bool(scope_name: str, operation: str, name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{scope_name}.{operation} {name} must be a compile-time bool")
    return value


def _builtin_subtract(lhs: Any, rhs: Any) -> Any:
    return lhs - rhs


def _builtin_not_equal(lhs: Any, rhs: Any) -> bool:
    return lhs != rhs


def _histogram_provider_counter_dtype(counter_dtype: Any) -> Any:
    """Use the unsigned CUB accumulator matching the public counter width."""

    if counter_dtype in (types.int32, types.uint32):
        return types.uint32
    if counter_dtype in (types.int64, types.uint64):
        return types.uint64
    return counter_dtype


def _group_operation_name(function: Any) -> str | None:
    """Return the group-first operation represented by one marker callable."""

    operation = _ROOT_OPERATIONS.get(function)
    return operation if operation in _QUALIFIED_OPERATIONS else None


def _is_common_root_operation(function: Any, operation: str) -> bool:
    member = getattr(_portable_api, operation, None)
    return (
        function is member
        and getattr(member, "__cuda_coop_backend_member__", None) == operation
    )


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
