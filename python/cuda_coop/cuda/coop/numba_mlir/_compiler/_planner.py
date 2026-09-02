# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rewrite exact ``cuda.coop`` block calls after device-function inlining."""

from __future__ import annotations

import inspect
from itertools import count
from numbers import Integral
from typing import Any

from numba_cuda_mlir import types
from numba_cuda_mlir.extending import (
    WholeFunctionPlanner,
    register_planner,
    require_launch_config,
)
from numba_cuda_mlir.numbair_transforms import ir

from cuda.coop._core.block.reduce import (
    normalize_block_reduce_algorithm,
    normalize_block_reduce_operator,
)
from cuda.coop._core.root_api import reduce as root_reduce
from cuda.coop._core.root_api import sum as root_sum
from cuda.coop._core.root_api import this_block as root_this_block
from cuda.coop._core.thread_group import normalize_thread_dim

from .. import reduce, sum, this_block
from ._provider import ReductionMarkerSpec, marker_for, resolve_provider_context

_GROUP_CONSTRUCTORS = frozenset({root_this_block, this_block})
_REDUCTIONS = {
    root_reduce: "reduce",
    root_sum: "sum",
    reduce: "reduce",
    sum: "sum",
}
_NAME_COUNTER = count()


class BlockReducePlanningError(RuntimeError):
    """A recognized block reduction could not be lowered safely."""


class _Planner:
    def __init__(self, state: Any, launch: dict[str, Any]) -> None:
        self.state = state
        self.func_ir = state.func_ir
        self.block_dim = normalize_thread_dim(
            launch["block"], scope="Numba-CUDA-MLIR launch", label="block"
        )
        self.provider_context = resolve_provider_context(state)
        self.descriptors: set[ir.Assign] = set()
        self.replacements: dict[ir.Assign, list[Any]] = {}
        self.dead_callable_names: set[str] = set()

    def _definition(self, value: Any) -> Any:
        if not isinstance(value, ir.Var):
            return value
        try:
            return self.func_ir.get_definition(value)
        except KeyError:
            return None

    def _callable(self, value: Any) -> Any:
        current = self._definition(value)
        attrs: list[str] = []
        while isinstance(current, ir.Expr) and current.op == "getattr":
            attrs.append(current.attr)
            current = self._definition(current.value)
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

    def _constant(self, value: Any, *, name: str) -> Any:
        if not isinstance(value, ir.Var):
            return value
        definition = self._definition(value)
        if isinstance(definition, ir.Arg):
            argtype = self.state.args[definition.index]
            if isinstance(argtype, types.Literal):
                return argtype.literal_value
            raise BlockReducePlanningError(
                f"cuda.coop {name} must be a compile-time constant"
            )
        if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
            return definition.value
        try:
            return self.func_ir.infer_constant(value)
        except Exception as error:
            raise BlockReducePlanningError(
                f"cuda.coop {name} must be a compile-time constant"
            ) from error

    def _is_none(self, value: Any) -> bool:
        try:
            return self._constant(value, name="valid_items") is None
        except BlockReducePlanningError:
            return False

    def _is_descriptor(self, value: Any) -> bool:
        if not isinstance(value, ir.Var):
            return False
        definition = self._definition(value)
        if isinstance(definition, ir.Var):
            return self._is_descriptor(definition)
        if isinstance(definition, ir.Expr) and definition.op == "cast":
            return self._is_descriptor(definition.value)
        return (
            isinstance(definition, ir.Expr)
            and definition.op == "call"
            and self._callable(definition.func) in _GROUP_CONSTRUCTORS
        )

    @staticmethod
    def _bind(function: Any, call: ir.Expr) -> inspect.BoundArguments:
        if call.vararg is not None or call.varkwarg is not None:
            raise BlockReducePlanningError(
                "cuda.coop block reduction does not support *args or **kwargs"
            )
        try:
            bound = inspect.signature(function).bind(*call.args, **dict(call.kws))
        except TypeError as error:
            raise BlockReducePlanningError(str(error)) from error
        bound.apply_defaults()
        return bound

    def _new_var(self, target: ir.Var, stem: str) -> ir.Var:
        return ir.Var(
            target.scope,
            f"__cuda_coop_{stem}_{next(_NAME_COUNTER)}__",
            target.loc,
        )

    def _lower_reduce(
        self,
        statement: ir.Assign,
        call: ir.Expr,
        function: Any,
        operation: str,
    ) -> None:
        bound = self._bind(function, call)
        group = bound.arguments["group"]
        if not self._is_descriptor(group):
            raise BlockReducePlanningError(
                "cuda.coop block reduction group must come from this_block()"
            )
        algorithm = normalize_block_reduce_algorithm(
            self._constant(bound.arguments["algorithm"], name="algorithm")
        ).value
        binary_op = "sum"
        if operation == "reduce":
            binary_op = normalize_block_reduce_operator(
                self._constant(bound.arguments["binary_op"], name="binary_op")
            ).value
        valid_items = bound.arguments["valid_items"]
        has_valid_items = not self._is_none(valid_items)
        if has_valid_items:
            try:
                static_valid_items = self._constant(valid_items, name="valid_items")
            except BlockReducePlanningError:
                static_valid_items = None
            if static_valid_items is not None:
                if isinstance(static_valid_items, bool) or not isinstance(
                    static_valid_items, Integral
                ):
                    raise TypeError("cuda.coop valid_items must be an integer")
                block_threads = (
                    self.block_dim[0] * self.block_dim[1] * self.block_dim[2]
                )
                if not 1 <= int(static_valid_items) <= block_threads:
                    raise ValueError(
                        "cuda.coop static valid_items must be between 1 and "
                        f"the block size ({block_threads})"
                    )
        marker = marker_for(
            ReductionMarkerSpec(
                block_dim=self.block_dim,
                operation=operation,
                binary_op=binary_op,
                algorithm=algorithm,
                has_valid_items=has_valid_items,
            ),
            self.provider_context,
        )
        function_var = self._new_var(statement.target, "block_reduce")
        rewritten = [
            ir.Assign(
                ir.Global(function_var.name, marker, statement.loc),
                function_var,
                statement.loc,
            )
        ]
        args = [bound.arguments["value"]]
        if has_valid_items:
            args.append(valid_items)
        rewritten.append(
            ir.Assign(
                ir.Expr.call(function_var, args, (), statement.loc),
                statement.target,
                statement.loc,
            )
        )
        self.replacements[statement] = rewritten
        self.dead_callable_names.add(call.func.name)

    def _mark_descriptors(self) -> None:
        for block in self.func_ir.blocks.values():
            for statement in block.body:
                if not isinstance(statement, ir.Assign):
                    continue
                call = statement.value
                if (
                    isinstance(call, ir.Expr)
                    and call.op == "call"
                    and self._callable(call.func) in _GROUP_CONSTRUCTORS
                ):
                    self._bind(self._callable(call.func), call)
                    self.descriptors.add(statement)
                    self.dead_callable_names.add(call.func.name)
        descriptor_names = {statement.target.name for statement in self.descriptors}
        changed = True
        while changed:
            changed = False
            for block in self.func_ir.blocks.values():
                for statement in block.body:
                    if not isinstance(statement, ir.Assign):
                        continue
                    source = statement.value
                    if isinstance(source, ir.Expr) and source.op == "cast":
                        source = source.value
                    if (
                        isinstance(source, ir.Var)
                        and source.name in descriptor_names
                        and statement.target.name not in descriptor_names
                    ):
                        self.descriptors.add(statement)
                        descriptor_names.add(statement.target.name)
                        changed = True

    def _validate_descriptor_uses(self) -> None:
        names = {statement.target.name for statement in self.descriptors}
        for block in self.func_ir.blocks.values():
            for statement in block.body:
                used = {value.name for value in statement.list_vars()} & names
                if isinstance(statement, ir.Assign):
                    used.discard(statement.target.name)
                if not used:
                    continue
                if statement in self.descriptors:
                    continue
                if statement in self.replacements:
                    continue
                raise BlockReducePlanningError(
                    "cuda.coop ThreadGroup values are compile-time descriptors "
                    "and may only be passed to reduce or sum"
                )

    def run(self) -> bool:
        self._mark_descriptors()
        for block in self.func_ir.blocks.values():
            for statement in block.body:
                if not isinstance(statement, ir.Assign):
                    continue
                call = statement.value
                if not isinstance(call, ir.Expr) or call.op != "call":
                    continue
                function = self._callable(call.func)
                operation = _REDUCTIONS.get(function)
                if operation is not None:
                    self._lower_reduce(statement, call, function, operation)
        self._validate_descriptor_uses()
        if not (self.descriptors or self.replacements):
            return False
        for block in self.func_ir.blocks.values():
            rewritten: list[Any] = []
            for statement in block.body:
                replacement = self.replacements.get(statement)
                if replacement is not None:
                    rewritten.extend(replacement)
                elif isinstance(statement, ir.Assign) and (
                    statement in self.descriptors
                    or statement.target.name in self.dead_callable_names
                ):
                    rewritten.append(
                        ir.Assign(
                            ir.Const(None, statement.loc),
                            statement.target,
                            statement.loc,
                        )
                    )
                else:
                    rewritten.append(statement)
            block.body = rewritten
        return True


def _has_markers(func_ir: Any) -> bool:
    analyzer = object.__new__(_Planner)
    analyzer.func_ir = func_ir
    for block in func_ir.blocks.values():
        for statement in block.body:
            if not isinstance(statement, ir.Assign):
                continue
            call = statement.value
            if not isinstance(call, ir.Expr) or call.op != "call":
                continue
            function = analyzer._callable(call.func)
            if function in _GROUP_CONSTRUCTORS or function in _REDUCTIONS:
                return True
    return False


@register_planner
class CoopBlockReducePlanner(WholeFunctionPlanner):
    """Lower exact root or qualified block reductions for configured kernels."""

    def run(self) -> bool:
        if not _has_markers(self.state.func_ir):
            return False
        return _Planner(self.state, require_launch_config(self.state)).run()


__all__ = ["BlockReducePlanningError", "CoopBlockReducePlanner"]
