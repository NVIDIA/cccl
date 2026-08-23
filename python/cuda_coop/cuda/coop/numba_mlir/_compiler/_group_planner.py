# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Whole-function group hierarchy planner for Numba-CUDA-MLIR.

This module resolves compile-time hierarchy descriptors and group methods
against exact launch metadata.
"""

from ._group_planner_support import (
    _GROUP_CONSTRUCTORS,
    _GROUP_METHODS,
    _NAME_COUNTER,
    _PORTABLE_GROUP_CONSTRUCTORS,
    Any,
    ForceLiteralArg,
    GroupRewriteError,
    LaunchFactOrigin,
    LaunchFacts,
    ThreadGroup,
    ThreadHierarchy,
    WholeFunctionPlanner,
    inspect,
    ir,
    normalize_thread_level,
    register_planner,
    require_launch_config,
    resolve_thread_group,
    types,
)


class _GroupCallPlanner:
    """Lower compile-time group descriptors and group methods."""

    def __init__(self, state, launch_config: dict[str, Any]) -> None:
        self.state = state
        self.func_ir = state.func_ir
        self.launch_config = launch_config
        self.launch = self._make_launch_facts(launch_config)
        self.dead_func_names: set[str] = set()
        self.descriptor_assigns: set[ir.Assign] = set()
        self.replacements: dict[ir.Assign, list[Any]] = {}
        self._group_cache: dict[str, ThreadGroup] = {}
        self._hierarchy_cache: dict[str, ThreadHierarchy] = {}
        self._compile_context = None

    def _provider_compile_context(self):
        if self._compile_context is None:
            from ._nvrtc import resolve_compile_context

            self._compile_context = resolve_compile_context()
        return self._compile_context

    @staticmethod
    def _make_launch_facts(config: dict[str, Any]) -> LaunchFacts:
        block = config.get("block")
        grid = config.get("grid")
        cluster = config.get("cluster")
        cluster_launch = cluster is not None
        origins = [
            LaunchFactOrigin(
                fact="exact_block_dim",
                source="numba_cuda_mlir_launch_config",
                verified=True,
            ),
            LaunchFactOrigin(
                fact="exact_grid_dim",
                source="numba_cuda_mlir_launch_config",
                verified=True,
            ),
            LaunchFactOrigin(
                fact="cluster_launch",
                source="numba_cuda_mlir_launch_config",
                verified=True,
            ),
        ]
        if cluster is not None:
            origins.append(
                LaunchFactOrigin(
                    fact="exact_cluster_dim",
                    source="numba_cuda_mlir_launch_config",
                    verified=True,
                )
            )
        return LaunchFacts(
            exact_block_dim=block,
            exact_grid_dim=grid,
            exact_cluster_dim=cluster,
            cluster_launch=cluster_launch,
            cooperative_launch=False,
            provenance=tuple(origins),
        )

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
            obj = current.value
        elif callable(current):
            obj = current
        else:
            return None
        try:
            for attr in reversed(attrs):
                obj = getattr(obj, attr)
        except (AttributeError, ImportError):
            return None
        return obj

    def _constant(self, value: Any) -> Any:
        if not isinstance(value, ir.Var):
            return value
        definition = self._definition(value)
        if isinstance(definition, ir.Arg):
            position = definition.index
            argtype = self.state.args[position]
            if not isinstance(argtype, types.Literal):
                raise ForceLiteralArg({position})
            return argtype.literal_value
        if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
            return definition.value
        hierarchy = self._hierarchy(value)
        if hierarchy is not None:
            return hierarchy
        group = self._group(value)
        if group is not None:
            return group
        try:
            return self.func_ir.infer_constant(value)
        except Exception as exc:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir group arguments that shape provider specialization must be compile-time constants; got {value.name!r}"
            ) from exc

    def _bind(self, function: Any, call: ir.Expr) -> inspect.BoundArguments:
        if call.vararg is not None or call.varkwarg is not None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir group calls do not support *args/**kwargs"
            )
        try:
            bound = inspect.signature(function).bind(*call.args, **dict(call.kws))
        except TypeError as exc:
            raise GroupRewriteError(str(exc)) from exc
        bound.apply_defaults()
        return bound

    def _hierarchy(self, value: Any) -> ThreadHierarchy | None:
        if isinstance(value, ThreadHierarchy):
            return value
        if not isinstance(value, ir.Var):
            return None
        cached = self._hierarchy_cache.get(value.name)
        if cached is not None:
            return cached
        definition = self._definition(value)
        if isinstance(definition, ir.Var):
            return self._hierarchy(definition)
        if isinstance(definition, ir.Expr) and definition.op == "cast":
            return self._hierarchy(definition.value)
        if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
            if isinstance(definition.value, ThreadHierarchy):
                return definition.value
            return None
        if not isinstance(definition, ir.Expr) or definition.op != "call":
            return None
        function = self._callable(definition.func)
        if function is not ThreadHierarchy:
            return None
        self._bind(ThreadHierarchy, definition)
        hierarchy = ThreadHierarchy()
        self._hierarchy_cache[value.name] = hierarchy
        return hierarchy

    def _group(self, value: Any) -> ThreadGroup | None:
        if isinstance(value, ThreadGroup):
            return value
        if not isinstance(value, ir.Var):
            return None
        cached = self._group_cache.get(value.name)
        if cached is not None:
            return cached
        definition = self._definition(value)
        if isinstance(definition, ir.Var):
            return self._group(definition)
        if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
            if isinstance(definition.value, ThreadGroup):
                return definition.value
            return None
        if isinstance(definition, ir.Expr) and definition.op == "cast":
            return self._group(definition.value)
        if not isinstance(definition, ir.Expr) or definition.op != "call":
            return None
        function = self._callable(definition.func)
        if function in _GROUP_CONSTRUCTORS:
            bound = self._bind(function, definition)
            args = []
            kwargs = {}
            parameters = inspect.signature(function).parameters
            for name, parameter in parameters.items():
                argument = self._constant(bound.arguments[name])
                if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    args.append(argument)
                elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                    kwargs[name] = argument
            group = _GROUP_CONSTRUCTORS[function](*args, **kwargs)
            if function in _PORTABLE_GROUP_CONSTRUCTORS:
                assert group.hierarchy is not None
                group = group.with_hierarchy(group.hierarchy, source="common_root")
            self._group_cache[value.name] = group
            return group
        function_definition = self._definition(definition.func)
        if (
            isinstance(function_definition, ir.Expr)
            and function_definition.op == "getattr"
            and (function_definition.attr == "group_by")
        ):
            parent = self._group(function_definition.value)
            if parent is None:
                return None
            if definition.vararg is not None or definition.varkwarg is not None:
                raise GroupRewriteError(
                    "ThreadGroup.group_by does not support *args/**kwargs"
                )
            raw_args = tuple(definition.args)
            raw_kwargs = dict(definition.kws)
            unknown_kwargs = set(raw_kwargs) - {"count", "exhaustive"}
            if unknown_kwargs:
                names = ", ".join(sorted(unknown_kwargs))
                raise GroupRewriteError(
                    f"ThreadGroup.group_by got unexpected keyword(s): {names}"
                )
            if len(raw_args) > 1:
                raise GroupRewriteError(
                    "ThreadGroup.group_by accepts one positional count argument"
                )
            if raw_args and "count" in raw_kwargs:
                raise GroupRewriteError(
                    "ThreadGroup.group_by received count more than once"
                )
            if raw_args:
                count_arg = raw_args[0]
            elif "count" in raw_kwargs:
                count_arg = raw_kwargs["count"]
            else:
                raise GroupRewriteError("ThreadGroup.group_by requires count")
            count_value = self._constant(count_arg)
            exhaustive = self._constant(raw_kwargs.get("exhaustive", True))
            group = parent.group_by(count_value, exhaustive=exhaustive)
            self._group_cache[value.name] = group
            return group
        return None

    def _resolve_group(
        self, group: ThreadGroup, *, feature: str, through_level: str | None = None
    ) -> ThreadGroup:
        resolution = resolve_thread_group(
            group, self.launch, through_level=through_level
        )
        try:
            resolved = resolution.require_supported()
        except NotImplementedError as exc:
            raise NotImplementedError(f"cuda.coop.numba_mlir.{feature} {exc}") from exc
        if group.source == "common_root":
            assert resolved.hierarchy is not None
            resolved = resolved.with_hierarchy(resolved.hierarchy, source="common_root")
        return resolved

    @staticmethod
    def _new_var(scope: Any, loc: ir.Loc, stem: str) -> ir.Var:
        return ir.Var(scope, f"__cuda_coop_group_{stem}_{next(_NAME_COUNTER)}__", loc)

    def _rewritten_call(
        self,
        inst: ir.Assign,
        *,
        factory: Any,
    ) -> list[Any]:
        statements: list[Any] = []
        loc = inst.loc
        function_var = self._new_var(inst.target.scope, loc, "factory")
        statements.append(
            ir.Assign(ir.Global(function_var.name, factory, loc), function_var, loc)
        )
        statements.append(
            ir.Assign(ir.Expr.call(function_var, [], (), loc), inst.target, loc)
        )
        return statements

    def _group_method(self, call: ir.Expr) -> tuple[str, ThreadGroup] | None:
        definition = self._definition(call.func)
        if (
            not isinstance(definition, ir.Expr)
            or definition.op != "getattr"
            or definition.attr not in _GROUP_METHODS
            or (definition.attr == "group_by")
        ):
            return None
        group = self._group(definition.value)
        if group is None:
            return None
        return (definition.attr, group)

    def _lower_group_method(
        self, inst: ir.Assign, call: ir.Expr, *, method: str, group: ThreadGroup
    ) -> None:
        if call.vararg is not None or call.varkwarg is not None:
            raise GroupRewriteError(f"ThreadGroup.{method} does not support splats")
        kwargs = dict(call.kws)
        dtype = None
        level = "thread"
        if method in {"rank", "count"}:
            if len(call.args) > 1 or any((name != "level" for name in kwargs)):
                raise GroupRewriteError(f"invalid ThreadGroup.{method} arguments")
            if call.args and "level" in kwargs:
                raise GroupRewriteError(
                    f"ThreadGroup.{method} received level more than once"
                )
            if call.args:
                level = self._constant(call.args[0])
            elif "level" in kwargs:
                level = self._constant(kwargs["level"])
            operation = method
        elif method in {"rank_as", "count_as"}:
            if len(call.args) > 2 or any(
                (name not in {"dtype", "level"} for name in kwargs)
            ):
                raise GroupRewriteError(f"invalid ThreadGroup.{method} arguments")
            if call.args and "dtype" in kwargs:
                raise GroupRewriteError(
                    f"ThreadGroup.{method} received dtype more than once"
                )
            if len(call.args) > 1 and "level" in kwargs:
                raise GroupRewriteError(
                    f"ThreadGroup.{method} received level more than once"
                )
            if call.args:
                dtype = self._constant(call.args[0])
            elif "dtype" in kwargs:
                dtype = self._constant(kwargs["dtype"])
            if len(call.args) > 1:
                level = self._constant(call.args[1])
            elif "level" in kwargs:
                level = self._constant(kwargs["level"])
            operation = method.removesuffix("_as")
        else:
            if call.args or kwargs:
                raise GroupRewriteError(f"ThreadGroup.{method} accepts no arguments")
            operation = method
        if operation in {"rank", "count"}:
            level = normalize_thread_level(
                level, scope="cuda.coop.numba_mlir", feature=f"ThreadGroup.{operation}"
            )
            group = self._resolve_group(
                group, feature=f"ThreadGroup.{operation}", through_level=level
            )
        else:
            group = self._resolve_group(group, feature=f"ThreadGroup.{operation}")
        if group.kind == "grid" and operation in {"sync", "sync_aligned"}:
            if group.source == "common_root":
                raise NotImplementedError(
                    f"cuda.coop.ThreadGroup.{operation} does not support grid groups in the portable API; use a backend-qualified import for backend-specific grid support"
                )
            raise NotImplementedError(
                "cuda.coop.numba_mlir grid synchronization requires a verified cooperative launch, which the current launch descriptor cannot request"
            )
        from .._lowering._thread_group import make_group_method_invocable

        invocable = make_group_method_invocable(
            group=group,
            operation=operation,
            dtype=dtype,
            level=level,
            compile_context=self._provider_compile_context(),
        )
        self.dead_func_names.add(call.func.name)
        self.replacements[inst] = self._rewritten_call(inst, factory=invocable)

    def _mark_descriptor_calls(self) -> None:
        for block in self.func_ir.blocks.values():
            for inst in block.body:
                if not isinstance(inst, ir.Assign):
                    continue
                call = inst.value
                if not isinstance(call, ir.Expr) or call.op != "call":
                    continue
                function = self._callable(call.func)
                if function is ThreadHierarchy or function in _GROUP_CONSTRUCTORS:
                    self.descriptor_assigns.add(inst)
                    self.dead_func_names.add(call.func.name)
                    continue
                definition = self._definition(call.func)
                if (
                    isinstance(definition, ir.Expr)
                    and definition.op == "getattr"
                    and (definition.attr == "group_by")
                    and (self._group(definition.value) is not None)
                ):
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
                        and (inst.target.name not in descriptor_names)
                    ):
                        self.descriptor_assigns.add(inst)
                        descriptor_names.add(inst.target.name)
                        changed = True

    def _validate_descriptor_uses(self) -> None:
        descriptor_names = {inst.target.name for inst in self.descriptor_assigns}
        if not descriptor_names:
            return
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
                if isinstance(inst, ir.Assign):
                    value = inst.value
                    if (
                        isinstance(value, ir.Expr)
                        and value.op == "getattr"
                        and isinstance(value.value, ir.Var)
                        and (value.value.name in used_names)
                        and (inst.target.name in self.dead_func_names)
                    ):
                        continue
                names = ", ".join(sorted(used_names))
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir ThreadGroup/ThreadHierarchy values are "
                    "compile-time descriptors and may only feed this_*(), "
                    "group_by(), or group methods; "
                    f"descriptor use involving {names!r} would escape to runtime"
                )

    def run(self) -> bool:
        self._mark_descriptor_calls()
        for block in self.func_ir.blocks.values():
            for inst in block.body:
                if not isinstance(inst, ir.Assign):
                    continue
                call = inst.value
                if not isinstance(call, ir.Expr) or call.op != "call":
                    continue
                method = self._group_method(call)
                if method is not None:
                    method_name, group = method
                    self._lower_group_method(
                        inst, call, method=method_name, group=group
                    )
        self._validate_descriptor_uses()
        if not (self.descriptor_assigns or self.replacements or self.dead_func_names):
            return False
        for block in self.func_ir.blocks.values():
            rewritten: list[Any] = []
            for inst in block.body:
                replacement = self.replacements.get(inst)
                if replacement is not None:
                    rewritten.extend(replacement)
                    continue
                if isinstance(inst, ir.Assign) and (
                    inst in self.descriptor_assigns
                    or inst.target.name in self.dead_func_names
                ):
                    rewritten.append(
                        ir.Assign(ir.Const(None, inst.loc), inst.target, inst.loc)
                    )
                    continue
                rewritten.append(inst)
            block.body = rewritten
        return True


def has_group_markers(func_ir) -> bool:
    """Cheaply detect whether the function needs launch metadata."""
    analyzer = object.__new__(_GroupCallPlanner)
    analyzer.func_ir = func_ir
    analyzer._group_cache = {}
    analyzer._hierarchy_cache = {}
    for block in func_ir.blocks.values():
        for inst in block.body:
            if not isinstance(inst, ir.Assign):
                continue
            value = inst.value
            if not isinstance(value, ir.Expr) or value.op != "call":
                continue
            function = analyzer._callable(value.func)
            if function is ThreadHierarchy or function in _GROUP_CONSTRUCTORS:
                return True
            function_definition = analyzer._definition(value.func)
            if (
                isinstance(function_definition, ir.Expr)
                and function_definition.op == "getattr"
                and (function_definition.attr in _GROUP_METHODS)
                and (analyzer._group(function_definition.value) is not None)
            ):
                return True
    return False


@register_planner
class CoopGroupHierarchyPlanner(WholeFunctionPlanner):
    """Resolve compile-time groups against one exact configured launch."""

    def run(self) -> bool:
        if not has_group_markers(self.state.func_ir):
            return False
        launch_config = require_launch_config(self.state)
        return _GroupCallPlanner(self.state, launch_config).run()


__all__ = [
    "CoopGroupHierarchyPlanner",
    "GroupRewriteError",
    "has_group_markers",
]
