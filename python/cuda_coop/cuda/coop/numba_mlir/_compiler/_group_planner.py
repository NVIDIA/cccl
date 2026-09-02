# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolve group calls against one exact configured launch."""

from __future__ import annotations

from cuda.coop._core.thread_group import ThreadGroup

from ._group_planner_support import (
    _GROUP_CONSTRUCTORS,
    _NAME_COUNTER,
    _ROOT_OPERATIONS,
    Any,
    GroupRewriteError,
    WholeFunctionPlanner,
    _callable_from_ir,
    inspect,
    ir,
    normalize_thread_dim,
    require_launch_config,
    types,
)
from ._group_reduce import _ReducePlanning


class _GroupCallPlanner(_ReducePlanning):
    """Select exact group-reduction factories without payload dtype."""

    error_type = GroupRewriteError

    def __init__(self, state: Any, launch: dict[str, Any]) -> None:
        self.state = state
        self.func_ir = state.func_ir
        self.block_dim = normalize_thread_dim(
            launch["block"], scope="Numba-CUDA-MLIR launch", label="block"
        )
        self.dead_func_names: set[str] = set()
        self.descriptor_assigns: set[ir.Assign] = set()
        self.replacements: dict[ir.Assign, list[Any]] = {}

    def _definition(self, value: Any) -> Any:
        if not isinstance(value, ir.Var):
            return value
        try:
            return self.func_ir.get_definition(value)
        except KeyError:
            return None

    def _callable(self, value: Any) -> Any:
        return _callable_from_ir(self.func_ir, value)

    def _constant(self, value: Any, *, name: str) -> Any:
        if not isinstance(value, ir.Var):
            return value
        definition = self._definition(value)
        if isinstance(definition, ir.Arg):
            argtype = self.state.args[definition.index]
            if isinstance(argtype, types.Literal):
                return argtype.literal_value
            if isinstance(argtype, types.NoneType) or (
                isinstance(argtype, types.Omitted) and argtype.value is None
            ):
                return None
            raise GroupRewriteError(f"cuda.coop {name} must be a compile-time constant")
        if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
            return definition.value
        try:
            return self.func_ir.infer_constant(value)
        except Exception as error:
            raise GroupRewriteError(
                f"cuda.coop {name} must be a compile-time constant"
            ) from error

    def _try_constant(self, value: Any, *, name: str) -> tuple[bool, Any]:
        try:
            return True, self._constant(value, name=name)
        except GroupRewriteError:
            return False, None

    def _is_none(self, value: Any) -> bool:
        is_static, result = self._try_constant(value, name="valid_items")
        return is_static and result is None

    def _descriptor(self, value: Any) -> ThreadGroup | None:
        if not isinstance(value, ir.Var):
            return value if isinstance(value, ThreadGroup) else None
        definition = self._definition(value)
        if isinstance(definition, ir.Var):
            return self._descriptor(definition)
        if isinstance(definition, ir.Expr) and definition.op == "cast":
            return self._descriptor(definition.value)
        if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
            return (
                definition.value if isinstance(definition.value, ThreadGroup) else None
            )
        if isinstance(definition, ir.Expr) and definition.op == "call":
            return _GROUP_CONSTRUCTORS.get(self._callable(definition.func))
        return None

    @staticmethod
    def _bind(function: Any, call: ir.Expr) -> inspect.BoundArguments:
        if call.vararg is not None or call.varkwarg is not None:
            raise GroupRewriteError(
                "cuda.coop group reduction does not support *args or **kwargs"
            )
        try:
            bound = inspect.signature(function).bind(*call.args, **dict(call.kws))
        except TypeError as error:
            raise GroupRewriteError(str(error)) from error
        bound.apply_defaults()
        return bound

    @staticmethod
    def _new_var(scope: Any, loc: ir.Loc, stem: str) -> ir.Var:
        return ir.Var(scope, f"__cuda_coop_group_{stem}_{next(_NAME_COUNTER)}__", loc)

    def _value_var(
        self,
        statements: list[Any],
        *,
        scope: Any,
        loc: ir.Loc,
        stem: str,
        value: Any,
    ) -> ir.Var:
        if isinstance(value, ir.Var):
            return value
        result = self._new_var(scope, loc, stem)
        if value is None or isinstance(value, (bool, int, float, str, tuple)):
            rhs = ir.Const(value, loc)
        else:
            rhs = ir.Global(result.name, value, loc)
        statements.append(ir.Assign(rhs, result, loc))
        return result

    def _rewritten_call(
        self,
        inst: ir.Assign,
        *,
        factory: Any,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> list[Any]:
        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        function_var = self._new_var(scope, loc, "factory")
        statements.append(
            ir.Assign(ir.Global(function_var.name, factory, loc), function_var, loc)
        )
        rewritten_args = [
            self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"arg{index}",
                value=value,
            )
            for index, value in enumerate(args)
        ]
        rewritten_kwargs = tuple(
            (
                name,
                self._value_var(
                    statements,
                    scope=scope,
                    loc=loc,
                    stem=name,
                    value=value,
                ),
            )
            for name, value in kwargs.items()
        )
        statements.append(
            ir.Assign(
                ir.Expr.call(function_var, rewritten_args, rewritten_kwargs, loc),
                inst.target,
                loc,
            )
        )
        return statements

    def _mark_descriptors(self) -> None:
        for block in self.func_ir.blocks.values():
            for inst in block.body:
                if not isinstance(inst, ir.Assign):
                    continue
                call = inst.value
                if isinstance(call, (ir.Global, ir.FreeVar, ir.Const)) and isinstance(
                    call.value, ThreadGroup
                ):
                    self.descriptor_assigns.add(inst)
                    continue
                if (
                    isinstance(call, ir.Expr)
                    and call.op == "call"
                    and self._callable(call.func) in _GROUP_CONSTRUCTORS
                ):
                    self._bind(self._callable(call.func), call)
                    self.descriptor_assigns.add(inst)
                    self.dead_func_names.add(call.func.name)
        descriptor_names = {inst.target.name for inst in self.descriptor_assigns}
        changed = True
        while changed:
            changed = False
            for block in self.func_ir.blocks.values():
                for inst in block.body:
                    if not isinstance(inst, ir.Assign):
                        continue
                    source = inst.value
                    if isinstance(source, ir.Expr) and source.op == "cast":
                        source = source.value
                    if (
                        isinstance(source, ir.Var)
                        and source.name in descriptor_names
                        and inst.target.name not in descriptor_names
                    ):
                        self.descriptor_assigns.add(inst)
                        descriptor_names.add(inst.target.name)
                        changed = True

    def _validate_descriptor_uses(self) -> None:
        descriptor_names = {inst.target.name for inst in self.descriptor_assigns}
        for block in self.func_ir.blocks.values():
            for inst in block.body:
                used_names = {
                    value.name for value in inst.list_vars()
                } & descriptor_names
                if isinstance(inst, ir.Assign):
                    used_names.discard(inst.target.name)
                if not used_names:
                    continue
                if inst in self.descriptor_assigns or inst in self.replacements:
                    continue
                raise GroupRewriteError(
                    "cuda.coop ThreadGroup values are compile-time descriptors "
                    "and may only be passed to reduce or sum"
                )

    def run(self) -> bool:
        self._mark_descriptors()
        for block in self.func_ir.blocks.values():
            for inst in block.body:
                if not isinstance(inst, ir.Assign):
                    continue
                call = inst.value
                if not isinstance(call, ir.Expr) or call.op != "call":
                    continue
                function = self._callable(call.func)
                operation = _ROOT_OPERATIONS.get(function)
                if operation is not None:
                    self._lower_reduce(inst, call, function, operation)
        self._validate_descriptor_uses()
        if not (self.descriptor_assigns or self.replacements or self.dead_func_names):
            return False
        for block in self.func_ir.blocks.values():
            rewritten: list[Any] = []
            for inst in block.body:
                replacement = self.replacements.get(inst)
                if replacement is not None:
                    rewritten.extend(replacement)
                elif isinstance(inst, ir.Assign) and (
                    inst in self.descriptor_assigns
                    or inst.target.name in self.dead_func_names
                ):
                    rewritten.append(
                        ir.Assign(ir.Const(None, inst.loc), inst.target, inst.loc)
                    )
                else:
                    rewritten.append(inst)
            block.body = rewritten
        return True


def has_group_markers(func_ir: Any) -> bool:
    """Return whether exact public group markers remain in the IR."""

    for block in func_ir.blocks.values():
        for inst in block.body:
            if not isinstance(inst, ir.Assign):
                continue
            call = inst.value
            if not isinstance(call, ir.Expr) or call.op != "call":
                continue
            function = _callable_from_ir(func_ir, call.func)
            if function in _GROUP_CONSTRUCTORS or function in _ROOT_OPERATIONS:
                return True
    return False


class CoopGroupHierarchyPlanner(WholeFunctionPlanner):
    """Resolve exact group hierarchy and select a lowering factory first."""

    def run(self) -> bool:
        if not has_group_markers(self.state.func_ir):
            return False
        return _GroupCallPlanner(self.state, require_launch_config(self.state)).run()


__all__ = ["CoopGroupHierarchyPlanner", "GroupRewriteError", "has_group_markers"]
