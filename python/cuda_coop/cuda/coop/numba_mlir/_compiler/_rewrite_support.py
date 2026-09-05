# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# The registered rewrite imports these private support names explicitly.
# ruff: noqa: F401

"""Small post-inlining provider-rewrite helpers for scalar BlockReduce."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any

from numba_cuda_mlir import types
from numba_cuda_mlir.extending import WholeFunctionPlanner
from numba_cuda_mlir.numbair_transforms import ir

from ._group_planner_support import _callable_from_ir
from ._operations import FactoryOperation, factory_operation

_NAME_COUNTER = count()


class CoopSinglePhaseRewriteError(RuntimeError):
    """A recognized lowering factory could not be materialized safely."""


@dataclass(frozen=True)
class _RewriteMatch:
    inst: ir.Assign
    factory: Any
    metadata: FactoryOperation
    value: ir.Var
    valid_items: ir.Var | None
    factory_kwargs: dict[str, Any]
    factory_func_name: str


def _definition(func_ir: Any, value: Any) -> Any:
    if not isinstance(value, ir.Var):
        return value
    try:
        return func_ir.get_definition(value)
    except KeyError:
        return None


def _constant(state: Any, value: Any, *, name: str) -> Any:
    if not isinstance(value, ir.Var):
        return value
    definition = _definition(state.func_ir, value)
    if isinstance(definition, ir.Arg):
        argtype = state.args[definition.index]
        if isinstance(argtype, types.Literal):
            return argtype.literal_value
        if isinstance(argtype, types.NoneType) or (
            isinstance(argtype, types.Omitted) and argtype.value is None
        ):
            return None
        raise CoopSinglePhaseRewriteError(
            f"cuda.coop lowering factory {name} must be a compile-time constant"
        )
    if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
        return definition.value
    try:
        return state.func_ir.infer_constant(value)
    except Exception as error:
        raise CoopSinglePhaseRewriteError(
            f"cuda.coop lowering factory {name} must be a compile-time constant"
        ) from error


def _factory_from_call(
    func_ir: Any, call: ir.Expr
) -> tuple[Any, FactoryOperation] | None:
    factory = _callable_from_ir(func_ir, call.func)
    metadata = factory_operation(factory)
    if metadata is None:
        return None
    return factory, metadata


__all__ = ["CoopSinglePhaseRewriteError"]
