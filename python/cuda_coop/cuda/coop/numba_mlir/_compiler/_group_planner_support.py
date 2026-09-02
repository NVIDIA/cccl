# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# The family planner imports these private support names explicitly.
# ruff: noqa: F401

"""Shared exact-identity support for block group hierarchy planning."""

from __future__ import annotations

import inspect
from itertools import count
from typing import Any

from numba_cuda_mlir import types
from numba_cuda_mlir.extending import (
    WholeFunctionPlanner,
    require_launch_config,
)
from numba_cuda_mlir.numbair_transforms import ir

from cuda.coop._core.api.reduce import reduce as root_reduce
from cuda.coop._core.api.reduce import sum as root_sum
from cuda.coop._core.api.thread_group import this_block as root_this_block
from cuda.coop._core.thread_group import ThreadGroup, normalize_thread_dim

from .._group_reduce import reduce, sum
from .._thread_group import this_block
from ._operations import group_operation_name

_NAME_COUNTER = count()
_GROUP_CONSTRUCTORS = frozenset({root_this_block, this_block})
_ROOT_OPERATIONS = {
    root_reduce: "reduce",
    root_sum: "sum",
    reduce: group_operation_name(reduce),
    sum: group_operation_name(sum),
}


class GroupRewriteError(RuntimeError):
    """A recognized group reduction could not be lowered safely."""


def _callable_from_ir(func_ir: Any, value: Any) -> Any:
    """Resolve one IR callable without accepting module/name lookalikes."""

    if not isinstance(value, ir.Var):
        return None
    try:
        current = func_ir.get_definition(value)
    except KeyError:
        return None
    attrs: list[str] = []
    while isinstance(current, ir.Expr) and current.op == "getattr":
        attrs.append(current.attr)
        try:
            current = func_ir.get_definition(current.value)
        except KeyError:
            return None
    if isinstance(current, (ir.Global, ir.FreeVar, ir.Const)):
        result = current.value
    elif callable(current):
        result = current
    else:
        return None
    try:
        for attr in reversed(attrs):
            result = getattr(result, attr)
    except (AttributeError, ImportError):
        return None
    return result


__all__ = ["GroupRewriteError"]
