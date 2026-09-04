# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared whole-function group planner for Numba-CUDA-MLIR.

This module owns cross-family IR provenance, hierarchy and payload caches,
result construction, and orchestration. Primitive-specific lowering methods
live in the adjacent semantic group mixins.
"""

import cuda.coop._core.api._dispatch as _portable_dispatch

from .._thread_data import ThreadData
from ._group_planner_support import (
    _GROUP_CONSTRUCTORS,
    _GROUP_METHODS,
    _NAME_COUNTER,
    _PAYLOAD_DTYPE_LIKE,
    _PORTABLE_GROUP_CONSTRUCTORS,
    Any,
    ForceLiteralArg,
    GroupRewriteError,
    Integral,
    LaunchFactOrigin,
    LaunchFacts,
    ThreadGroup,
    ThreadHierarchy,
    WholeFunctionPlanner,
    _cuda_module,
    _group_operation_name,
    _is_common_root_operation,
    _portable_api,
    _typed_group_payload_like,
    inspect,
    ir,
    register_planner,
    require_launch_config,
    resolve_thread_group,
    types,
)
from ._group_planning import GroupPlanningContext
from ._operations import group_primitive


class _GroupCallPlanner:
    """Coordinate semantic family lowering against one function IR."""

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
        self.context = GroupPlanningContext(self)

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

    def _all_definitions(self, value: ir.Var) -> tuple[Any, ...]:
        definitions = getattr(self.func_ir, "_definitions", {}).get(value.name, ())
        if definitions:
            return tuple(definitions)
        definition = self._definition(value)
        return () if definition is None else (definition,)

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

    def _try_constant(self, value: Any) -> tuple[bool, Any]:
        """Resolve a constant without requesting dispatcher specialization."""
        if isinstance(value, ir.Var):
            definition = self._definition(value)
            if isinstance(definition, ir.Arg):
                argtype = self.state.args[definition.index]
                if isinstance(argtype, types.Literal):
                    return (True, argtype.literal_value)
                if isinstance(argtype, types.NoneType) or (
                    isinstance(argtype, types.Omitted) and argtype.value is None
                ):
                    return (True, None)
                return (False, None)
        try:
            return (True, self._constant(value))
        except (ForceLiteralArg, GroupRewriteError):
            return (False, None)

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

    def _validate_common_selector(
        self,
        operation: str,
        parameter: str,
        value: Any,
        allowed: frozenset[str],
        *,
        allow_none: bool = False,
    ) -> Any:
        """Validate one common-root selector bypassed by identity rewriting."""
        token = self._constant(value)
        if token is None and allow_none:
            return None
        token = getattr(token, "value", token)
        if isinstance(token, str):
            token = token.strip().lower().replace("-", "_")
        if token not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(
                f"cuda.coop.{operation} {parameter} must be one of: {choices}; use a backend-qualified import for backend-only controls"
            )
        return token

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

    def _is_none(self, value: Any) -> bool:
        resolved, constant = self._try_constant(value)
        return resolved and constant is None

    @staticmethod
    def _merge_array_states(states: tuple[bool | None, ...]) -> bool | None:
        if not states or any((state is False for state in states)):
            return False
        if any((state is True for state in states)):
            return True
        return None

    def _result_source(self, definition: Any, index: int | None = None):
        if not isinstance(definition, ir.Expr) or definition.op != "call":
            return None
        operation = _group_operation_name(self._callable(definition.func))
        registration = None if operation is None else group_primitive(operation)
        if registration is None:
            return None
        results = registration.results
        if index is None:
            if len(results) != 1:
                return None
            result = results[0]
        else:
            if not -len(results) <= index < len(results):
                return None
            result = results[index]
        return result, self._bind(self._callable(definition.func), definition)

    def _is_array_tuple_item(
        self, value: Any, index: int, *, seen: set[str], thread_data_only: bool = False
    ) -> bool | None:
        if not isinstance(value, ir.Var):
            return False
        seen_key = f"{value.name}[{index}]"
        if seen_key in seen:
            return None
        seen.add(seen_key)
        return self._merge_array_states(
            tuple(
                (
                    self._is_array_tuple_item_definition(
                        definition,
                        index,
                        seen=set(seen),
                        thread_data_only=thread_data_only,
                    )
                    for definition in self._all_definitions(value)
                )
            )
        )

    def _is_array_tuple_item_definition(
        self, definition: Any, index: int, *, seen: set[str], thread_data_only: bool
    ) -> bool | None:
        if isinstance(definition, ir.Var):
            return self._is_array_tuple_item(
                definition, index, seen=seen, thread_data_only=thread_data_only
            )
        if not isinstance(definition, ir.Expr):
            return False
        if definition.op in {"cast", "exhaust_iter"}:
            return self._is_array_tuple_item(
                definition.value, index, seen=seen, thread_data_only=thread_data_only
            )
        if definition.op == "phi":
            incoming_values = getattr(definition, "incoming_values", ())
            return self._merge_array_states(
                tuple(
                    (
                        self._is_array_tuple_item(
                            incoming,
                            index,
                            seen=set(seen),
                            thread_data_only=thread_data_only,
                        )
                        for incoming in incoming_values
                    )
                )
            )
        if definition.op == "build_tuple":
            items = tuple(getattr(definition, "items", ()))
            if not -len(items) <= index < len(items):
                return False
            return self._is_array_value(
                items[index], seen=set(seen), thread_data_only=thread_data_only
            )
        if definition.op != "call":
            return False
        resolved = self._result_source(definition, index)
        if resolved is None:
            return False
        result, bound = resolved
        if result.array_parameter is None:
            return False
        return self._is_array_value(
            bound.arguments[result.array_parameter],
            seen=seen,
            thread_data_only=thread_data_only,
        )

    def _is_array_value(
        self,
        value: Any,
        *,
        seen: set[str] | None = None,
        thread_data_only: bool = False,
    ) -> bool | None:
        if not isinstance(value, ir.Var):
            return False
        if seen is None:
            seen = set()
        if value.name in seen:
            return None
        seen.add(value.name)
        return self._merge_array_states(
            tuple(
                (
                    self._is_array_definition(
                        definition, seen=set(seen), thread_data_only=thread_data_only
                    )
                    for definition in self._all_definitions(value)
                )
            )
        )

    def _is_array_definition(
        self, definition: Any, *, seen: set[str], thread_data_only: bool
    ) -> bool | None:
        if isinstance(definition, ir.Var):
            return self._is_array_value(
                definition, seen=seen, thread_data_only=thread_data_only
            )
        if not isinstance(definition, ir.Expr):
            return False
        if definition.op == "cast":
            return self._is_array_value(
                definition.value, seen=seen, thread_data_only=thread_data_only
            )
        if definition.op == "phi":
            incoming_values = getattr(definition, "incoming_values", ())
            return self._merge_array_states(
                tuple(
                    (
                        self._is_array_value(
                            incoming, seen=set(seen), thread_data_only=thread_data_only
                        )
                        for incoming in incoming_values
                    )
                )
            )
        if definition.op in {"getitem", "static_getitem"}:
            index = getattr(definition, "index", None)
            if isinstance(index, ir.Var):
                try:
                    index = self._constant(index)
                except GroupRewriteError:
                    return False
            if isinstance(index, Integral) and (not isinstance(index, bool)):
                return self._is_array_tuple_item(
                    definition.value,
                    int(index),
                    seen=set(seen),
                    thread_data_only=thread_data_only,
                )
            return False
        if definition.op != "call":
            return False
        function = self._callable(definition.func)
        if function in {ThreadData, _portable_api.ThreadData}:
            return True
        if function is _typed_group_payload_like:
            return self._is_array_value(
                definition.args[0], seen=seen, thread_data_only=thread_data_only
            )
        if function is _cuda_module.local.array:
            return not thread_data_only
        resolved = self._result_source(definition)
        if resolved is None:
            return False
        result, bound = resolved
        if result.array_parameter is None:
            return False
        return self._is_array_value(
            bound.arguments[result.array_parameter],
            seen=seen,
            thread_data_only=thread_data_only,
        )

    @staticmethod
    def _new_var(scope: Any, loc: ir.Loc, stem: str) -> ir.Var:
        return ir.Var(scope, f"__cuda_coop_group_{stem}_{next(_NAME_COUNTER)}__", loc)

    def _value_var(
        self, statements: list[Any], *, scope: Any, loc: ir.Loc, stem: str, value: Any
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
        return_alias: ir.Var | tuple[ir.Var, ...] | None = None,
        common_root_operation: str | None = None,
    ) -> list[Any]:
        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        if common_root_operation is not None:
            kwargs = dict(kwargs)
            kwargs.setdefault("_common_root_operation", common_root_operation)
        function_var = self._new_var(scope, loc, "factory")
        statements.append(
            ir.Assign(ir.Global(function_var.name, factory, loc), function_var, loc)
        )
        rewritten_args = [
            self._value_var(
                statements, scope=scope, loc=loc, stem=f"arg{idx}", value=value
            )
            for idx, value in enumerate(args)
        ]
        rewritten_kwargs = tuple(
            (
                (
                    name,
                    self._value_var(
                        statements, scope=scope, loc=loc, stem=name, value=value
                    ),
                )
                for name, value in kwargs.items()
            )
        )
        call_target = (
            inst.target
            if return_alias is None
            else self._new_var(scope, loc, "ignored_result")
        )
        statements.append(
            ir.Assign(
                ir.Expr.call(function_var, rewritten_args, rewritten_kwargs, loc),
                call_target,
                loc,
            )
        )
        if isinstance(return_alias, tuple):
            statements.append(
                ir.Assign(
                    ir.Expr.build_tuple(list(return_alias), loc), inst.target, loc
                )
            )
        elif return_alias is not None:
            statements.append(ir.Assign(return_alias, inst.target, loc))
        return statements

    def _array_operand_state(self, operation: str, value: Any) -> bool:
        state = self._is_array_value(value)
        if state is None:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} could not resolve cyclic array provenance to a concrete scalar or array value"
            )
        return state

    def _thread_data_operand_state(
        self, operation: str, parameter: str, value: Any
    ) -> bool:
        state = self._is_array_value(value, thread_data_only=True)
        if state is None:
            raise GroupRewriteError(
                f"cuda.coop.{operation} could not resolve {parameter} payload provenance"
            )
        return state

    def _array_extent(self, value: Any, *, seen: set[str] | None = None) -> int | None:
        if not isinstance(value, ir.Var):
            return None
        if seen is None:
            seen = set()
        if value.name in seen:
            return None
        seen.add(value.name)
        extents: set[int] = set()
        for definition in self._all_definitions(value):
            extent = self._array_extent_definition(definition, seen=set(seen))
            if extent is not None:
                extents.add(extent)
        if len(extents) > 1:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir array aliases have inconsistent items_per_thread extents"
            )
        return next(iter(extents), None)

    def _array_extent_tuple_item(
        self, value: Any, index: int, *, seen: set[str]
    ) -> int | None:
        if not isinstance(value, ir.Var):
            return None
        seen_key = f"{value.name}[{index}]"
        if seen_key in seen:
            return None
        seen.add(seen_key)
        extents = {
            extent
            for definition in self._all_definitions(value)
            if (
                extent := self._array_extent_tuple_item_definition(
                    definition, index, seen=set(seen)
                )
            )
            is not None
        }
        if len(extents) > 1:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir tuple projections have inconsistent items_per_thread extents"
            )
        return next(iter(extents), None)

    def _array_extent_tuple_item_definition(
        self, definition: Any, index: int, *, seen: set[str]
    ) -> int | None:
        if isinstance(definition, ir.Var):
            return self._array_extent_tuple_item(definition, index, seen=seen)
        if not isinstance(definition, ir.Expr):
            return None
        if definition.op in {"cast", "exhaust_iter"}:
            return self._array_extent_tuple_item(definition.value, index, seen=seen)
        if definition.op == "phi":
            extents = {
                extent
                for incoming in getattr(definition, "incoming_values", ())
                if (
                    extent := self._array_extent_tuple_item(
                        incoming, index, seen=set(seen)
                    )
                )
                is not None
            }
            if len(extents) > 1:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir loop-carried tuple payloads have inconsistent items_per_thread extents"
                )
            return next(iter(extents), None)
        if definition.op == "build_tuple":
            items = tuple(getattr(definition, "items", ()))
            if not -len(items) <= index < len(items):
                return None
            return self._array_extent(items[index], seen=set(seen))
        if definition.op != "call":
            return None
        resolved = self._result_source(definition, index)
        if resolved is None:
            return None
        result, bound = resolved
        if result.array_parameter is None:
            return 1
        return self._array_extent(
            bound.arguments[result.array_parameter],
            seen=seen,
        )

    def _array_extent_definition(
        self, definition: Any, *, seen: set[str]
    ) -> int | None:
        if isinstance(definition, ir.Var):
            return self._array_extent(definition, seen=seen)
        if not isinstance(definition, ir.Expr):
            return None
        if definition.op == "cast":
            return self._array_extent(definition.value, seen=seen)
        if definition.op == "phi":
            extents = {
                extent
                for incoming in getattr(definition, "incoming_values", ())
                if (extent := self._array_extent(incoming, seen=set(seen))) is not None
            }
            if len(extents) > 1:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir loop-carried payloads have inconsistent items_per_thread extents"
                )
            return next(iter(extents), None)
        if definition.op in {"getitem", "static_getitem"}:
            index = getattr(definition, "index", None)
            if isinstance(index, ir.Var):
                try:
                    index = self._constant(index)
                except GroupRewriteError:
                    return None
            if isinstance(index, Integral) and (not isinstance(index, bool)):
                return self._array_extent_tuple_item(
                    definition.value, int(index), seen=set(seen)
                )
            return None
        if definition.op != "call":
            return None
        function = self._callable(definition.func)
        if function is _typed_group_payload_like:
            try:
                is_array = self._constant(definition.args[1])
            except (GroupRewriteError, IndexError):
                return None
            if len(definition.args) >= 4:
                try:
                    extent = self._constant(definition.args[3])
                except GroupRewriteError:
                    return None
                if isinstance(extent, Integral) and (not isinstance(extent, bool)):
                    return int(extent)
                return None
            if is_array is False:
                return 1
            return self._array_extent(definition.args[0], seen=seen)
        if function in {ThreadData, _portable_api.ThreadData}:
            bound = self._bind(function, definition)
            extent_argument = bound.arguments["items_per_thread"]
            try:
                extent = self._constant(extent_argument)
            except GroupRewriteError:
                return None
            if isinstance(extent, Integral) and (not isinstance(extent, bool)):
                return int(extent)
            return None
        if function is _cuda_module.local.array:
            if not definition.args:
                return None
            try:
                extent = self._constant(definition.args[0])
            except GroupRewriteError:
                return None
            if isinstance(extent, Integral) and (not isinstance(extent, bool)):
                return int(extent)
            return None
        resolved = self._result_source(definition)
        if resolved is None:
            return None
        result, bound = resolved
        if result.array_parameter is None:
            return 1
        return self._array_extent(bound.arguments[result.array_parameter], seen=seen)

    def _copy_array_payload(
        self,
        statements: list[Any],
        *,
        operation: str,
        source: ir.Var,
        destination: ir.Var,
        scope: Any,
        loc: ir.Loc,
        known_items_per_thread: int | None = None,
    ) -> None:
        """Copy a static local payload into a fresh result payload."""

        extent = (
            known_items_per_thread
            if known_items_per_thread is not None
            else self._array_extent(source)
        )
        if extent is None:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} could not infer a static "
                "items_per_thread extent for its non-mutating result"
            )
        for item_index in range(extent):
            index = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_copy_index_{item_index}",
                value=item_index,
            )
            item = self._new_var(scope, loc, f"{operation}_copy_item_{item_index}")
            statements.append(ir.Assign(ir.Expr.getitem(source, index, loc), item, loc))
            statements.append(ir.SetItem(destination, index, item, loc))

    def _typed_payload_like(
        self,
        statements: list[Any],
        *,
        scope: Any,
        loc: ir.Loc,
        stem: str,
        prototype: ir.Var,
        is_array: bool,
        dtype_policy: str,
        items_per_thread: Any = None,
    ) -> ir.Var:
        function_var = self._new_var(scope, loc, f"{stem}_payload_factory")
        statements.append(
            ir.Assign(
                ir.Global(function_var.name, _typed_group_payload_like, loc),
                function_var,
                loc,
            )
        )
        is_array_var = self._value_var(
            statements, scope=scope, loc=loc, stem=f"{stem}_is_array", value=is_array
        )
        dtype_policy_var = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{stem}_dtype_policy",
            value=dtype_policy,
        )
        args = [prototype, is_array_var, dtype_policy_var]
        if items_per_thread is not None:
            args.append(
                self._value_var(
                    statements,
                    scope=scope,
                    loc=loc,
                    stem=f"{stem}_items_per_thread",
                    value=items_per_thread,
                )
            )
        payload = self._new_var(scope, loc, f"{stem}_payload")
        statements.append(
            ir.Assign(ir.Expr.call(function_var, args, (), loc), payload, loc)
        )
        return payload

    def _boxed_group_operand(
        self,
        statements: list[Any],
        *,
        operation: str,
        value: ir.Var,
        scope: Any,
        loc: ir.Loc,
    ) -> tuple[ir.Var, bool]:
        """Represent a scalar as a one-item payload for array-only providers."""

        is_array = self._array_operand_state(operation, value)
        if is_array:
            return value, True
        payload = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_input",
            prototype=value,
            is_array=False,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
        )
        index = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_input_index",
            value=0,
        )
        statements.append(ir.SetItem(payload, index, value, loc))
        return payload, False

    def _result_value(
        self,
        statements: list[Any],
        *,
        payload: ir.Var,
        is_array: bool,
        scope: Any,
        loc: ir.Loc,
        stem: str,
    ) -> ir.Var:
        """Return an array payload or unbox its sole scalar item."""

        if is_array:
            return payload
        index = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{stem}_index",
            value=0,
        )
        result = self._new_var(scope, loc, f"{stem}_scalar")
        statements.append(ir.Assign(ir.Expr.getitem(payload, index, loc), result, loc))
        return result

    def _lower_root_operation(
        self, inst: ir.Assign, call: ir.Expr, function: Any, operation: str
    ) -> None:
        bound = self._bind(function, call)
        if bound.arguments.get("kwargs"):
            names = ", ".join(sorted(bound.arguments["kwargs"]))
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} got unexpected keyword(s): {names}"
            )
        group = self._group(bound.arguments["group"])
        if group is None:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} requires a compile-time ThreadGroup from this_*()"
            )
        is_common_root = _is_common_root_operation(function, operation)
        if is_common_root:
            _portable_dispatch._validate_portable_operation_group(operation, group)
        group = self._resolve_group(group, feature=operation)
        registration = group_primitive(operation)
        if registration is None:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir operation {operation!r} has no planner"
            )
        if is_common_root and registration.validate_common_arguments is not None:
            registration.validate_common_arguments(self.context, operation, bound)
        replacement = registration.lower(
            self.context,
            inst,
            operation=operation,
            group=group,
            bound=bound,
            is_common_root=is_common_root,
        )
        self.dead_func_names.add(call.func.name)
        self.replacements[inst] = replacement

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
        del inst, call, group
        raise NotImplementedError(
            f"cuda.coop.numba_mlir ThreadGroup.{method} execution is not part "
            "of the Block Load/Store capability"
        )

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
                    f"cuda.coop.numba_mlir ThreadGroup/ThreadHierarchy values are compile-time descriptors and may only feed this_*(), group_by(), group methods, or group-first primitives; descriptor use involving {names!r} would escape to runtime"
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
                function = self._callable(call.func)
                operation = _group_operation_name(function)
                if operation is not None:
                    self._lower_root_operation(inst, call, function, operation)
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
            if (
                function is ThreadHierarchy
                or function in _GROUP_CONSTRUCTORS
                or _group_operation_name(function) is not None
            ):
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
    """Resolve movement groups against one exact configured launch."""

    def run(self) -> bool:
        if not has_group_markers(self.state.func_ir):
            return False
        if self.is_device_function:
            return False
        launch_config = require_launch_config(self.state)
        return _GroupCallPlanner(self.state, launch_config).run()


__all__ = [
    "CoopGroupHierarchyPlanner",
    "GroupRewriteError",
    "has_group_markers",
]
