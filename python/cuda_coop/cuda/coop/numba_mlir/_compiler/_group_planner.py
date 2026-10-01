# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from enum import Enum
from numbers import Integral
from typing import Any

import numba_cuda_mlir.numba_cuda.types as _numba_types
from numba_cuda_mlir.cuda.local import array as _cuda_local_array
from numba_cuda_mlir.extending import (
    WholeFunctionPlanner,
    register_planner,
    require_launch_config,
)
from numba_cuda_mlir.numba_cuda.core.errors import ForceLiteralArg

import cuda.coop._core.api as _portable_api
import cuda.coop._core.api._dispatch as _portable_dispatch
from cuda.coop._core import (
    LaunchFactOrigin,
    LaunchFacts,
    ThreadGroup,
    ThreadHierarchy,
    resolve_thread_group,
)

from .._temp_storage import TempStorage
from .._thread_data import ThreadData
from ._group_errors import (
    CyclicArrayProvenanceError,
    EscapingGroupDescriptorError,
    InconsistentArrayExtentError,
    InconsistentLoopPayloadExtentError,
    InconsistentLoopTupleExtentError,
    InconsistentTupleExtentError,
    InvalidGroupSelectorError,
    NonConstantGroupArgumentError,
    NonConstantThreadGroupError,
)
from ._group_planner_support import (
    _GROUP_CONSTRUCTORS,
    _NAME_COUNTER,
    _PORTABLE_GROUP_CONSTRUCTORS,
    GroupRewriteError,
    _group_operation_name,
    _is_common_root_operation,
    ir,
)
from ._group_planning import GroupPlanningContext
from ._operations import group_primitive
from ._scalar_provenance import (
    try_resolve_static_scalar,
    try_resolve_static_scalar_provenance,
)


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
        definitions = getattr(self.func_ir, "_definitions", {}).get(
            value.name, ()
        )
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

    def _reject_literal_unroll_value(self, value: Any, parameter: str) -> None:
        """Reject compile-time controls that depend on a pending literal unroll.

        Group planning needs shapes and selectors before the literal-unroll pass
        has expanded its iterations. Trace all reaching definitions and
        expression operands for a recognized ``literal_unroll`` call instead of
        trying to resolve an iteration value prematurely. Unrelated
        literal-unroll loops are allowed. Cycles terminate the search without
        establishing a dependency.

        Parameters
        ----------
        value : ir.Var or object
            Argument whose IR dependencies are inspected. Non-variables have no
            dependencies to inspect.
        parameter : str
            Description of the shape, selector, or other compile-time control
            used in the diagnostic.

        Returns
        -------
        None
            The value has no detected dependency on a recognized unroll call.

        Raises
        ------
        GroupRewriteError
            A dependency reaches the standard or vendored Numba literal-unroll
            marker; use constant controls or an ordinary loop with a fixed
            cooperative shape.
        """

        def depends_on_unroll(current, seen):
            if not isinstance(current, ir.Var) or current.name in seen:
                return False
            seen = {*seen, current.name}
            for definition in self._all_definitions(current):
                if isinstance(definition, ir.Var):
                    if depends_on_unroll(definition, seen):
                        return True
                elif isinstance(definition, ir.Expr):
                    if definition.op == "call":
                        function = self._callable(definition.func)
                        if getattr(
                            function, "__name__", None
                        ) == "literal_unroll" and getattr(
                            function, "__module__", None
                        ) in {
                            "numba.misc.special",
                            "numba_cuda_mlir.numba_cuda.misc.special",
                        }:
                            return True
                    if any(
                        depends_on_unroll(source, seen)
                        for source in definition.list_vars()
                    ):
                        return True
            return False

        if depends_on_unroll(value, set()):
            raise GroupRewriteError(
                "cuda.coop.numba_mlir does not support literal_unroll values "
                f"that determine {parameter}. Write separate "
                "cooperative calls with explicit constant shapes/selectors, "
                "or use an ordinary loop with a fixed cooperative shape."
            )

    def _reject_literal_unroll_constructors(self) -> None:
        constructors = {
            *_GROUP_CONSTRUCTORS,
            ThreadHierarchy,
            ThreadData,
            _portable_api.ThreadData,
            TempStorage,
            _portable_api.TempStorage,
        }
        for block in self.func_ir.blocks.values():
            for inst in block.body:
                call = getattr(inst, "value", None)
                if not isinstance(call, ir.Expr) or call.op != "call":
                    continue
                function = self._callable(call.func)
                if function not in constructors:
                    continue
                for argument in (*call.args, *(value for _, value in call.kws)):
                    self._reject_literal_unroll_value(
                        argument, f"{function.__name__} arguments"
                    )

    def _constant(self, value: Any) -> Any:
        """Resolve a required compile-time argument or request specialization.

        Try direct argument and constant definitions first, then reconstruct
        hierarchy or group descriptors, and finally use Numba constant
        inference. A non-literal kernel argument requests dispatcher
        specialization rather than being treated as an unsupported value.
        Descriptor reconstruction may populate the planner's caches. This is the
        required-constant path; callers classifying optional runtime controls
        use ``_try_static_scalar`` instead.

        Parameters
        ----------
        value : ir.Var or object
            IR variable to resolve, or an already-resolved value returned
            unchanged.

        Returns
        -------
        object
            Compile-time value, including ``None`` or a reconstructed
            descriptor.

        Raises
        ------
        ForceLiteralArg
            A function argument must be recompiled with a literal type.
        NonConstantGroupArgumentError
            The value cannot be resolved by descriptor or constant inference.
        GroupRewriteError
            A recognized descriptor has unsupported call syntax or arguments.
        """

        if not isinstance(value, ir.Var):
            return value
        definition = self._definition(value)
        if isinstance(definition, ir.Arg):
            position = definition.index
            argtype = self.state.args[position]
            if not isinstance(argtype, _numba_types.Literal):
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
            raise NonConstantGroupArgumentError(value.name) from exc

    def _try_constant(self, value: Any) -> tuple[bool, Any]:
        """Resolve a constant without requesting dispatcher specialization."""
        if isinstance(value, ir.Var):
            definition = self._definition(value)
            if isinstance(definition, ir.Arg):
                argtype = self.state.args[definition.index]
                if isinstance(argtype, _numba_types.Literal):
                    return (True, argtype.literal_value)
                if isinstance(argtype, _numba_types.NoneType) or (
                    isinstance(argtype, _numba_types.Omitted)
                    and argtype.value is None
                ):
                    return (True, None)
                return (False, None)
        try:
            return (True, self._constant(value))
        except (ForceLiteralArg, GroupRewriteError):
            return (False, None)

    def _try_static_scalar(
        self,
        value: Any,
    ) -> tuple[bool, Any]:
        """Resolve only values whose IR provenance is explicitly static."""

        return try_resolve_static_scalar(
            value,
            definitions=self._all_definitions,
            argument_type=lambda index: (
                self.state.args[index]
                if 0 <= index < len(self.state.args)
                else None
            ),
        )

    def _try_static_scalar_provenance(self, value: Any) -> tuple[bool, Any]:
        return try_resolve_static_scalar_provenance(
            value,
            definitions=self._all_definitions,
            argument_type=lambda index: (
                self.state.args[index]
                if 0 <= index < len(self.state.args)
                else None
            ),
        )

    def _bind(self, function: Any, call: ir.Expr) -> inspect.BoundArguments:
        if call.vararg is not None or call.varkwarg is not None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir group calls do not support *args/**kwargs"
            )
        try:
            bound = inspect.signature(function).bind(
                *call.args, **dict(call.kws)
            )
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
        self._reject_literal_unroll_value(value, f"{operation} {parameter}")
        token = self._constant(value)
        if token is None and allow_none:
            return None
        if not isinstance(token, str) or isinstance(token, Enum):
            raise TypeError(
                f"cuda.coop.{operation} {parameter} must be a string"
            )
        token = token.strip().lower().replace("-", "_")
        if token not in allowed:
            choices = ", ".join(sorted(allowed))
            raise InvalidGroupSelectorError(operation, parameter, choices)
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
        """Reconstruct a group descriptor from a supported IR definition.

        Follow aliases and casts, accept existing host descriptors, and
        interpret registered constructors and ``group_by`` calls on recognized
        parents. Constructor identity matters; matching a callable's name is
        insufficient. Arguments are resolved through ``_constant``, which may
        request literal specialization. Common-API constructors retain
        ``common_root`` provenance so later validation applies the portable
        contract.

        Cache newly constructed descriptors by variable name for this planner.
        This describes the requested group; launch-dependent resolution belongs
        to ``_resolve_group``. Unlike marker detection, this routine requires a
        single resolvable definition and does not merge phi inputs.

        Parameters
        ----------
        value : ThreadGroup, ir.Var, or object
            Existing descriptor or variable expected to name one.

        Returns
        -------
        ThreadGroup or None
            Reconstructed or cached descriptor, or ``None`` if the definition is
            not a recognized group expression.

        Raises
        ------
        ForceLiteralArg
            A constructor or subgroup argument needs literal specialization.
        GroupRewriteError
            A recognized constructor or subgroup call cannot be bound or its
            arguments cannot be resolved as required compile-time values.
        """

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
                group = group.with_hierarchy(
                    group.hierarchy, source="common_root"
                )
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
            self._reject_literal_unroll_value(count_arg, "group_by count")
            exhaustive_arg = raw_kwargs.get("exhaustive", True)
            self._reject_literal_unroll_value(
                exhaustive_arg, "group_by exhaustive"
            )
            count_value = self._constant(count_arg)
            exhaustive = self._constant(exhaustive_arg)
            group = parent.group_by(count_value, exhaustive=exhaustive)
            self._group_cache[value.name] = group
            return group
        return None

    def _resolve_group(
        self,
        group: ThreadGroup,
        *,
        feature: str,
        through_level: str | None = None,
    ) -> ThreadGroup:
        resolution = resolve_thread_group(
            group, self.launch, through_level=through_level
        )
        try:
            resolved = resolution.require_supported()
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"cuda.coop.numba_mlir.{feature} {exc}"
            ) from exc
        if group.source == "common_root":
            assert resolved.hierarchy is not None
            resolved = resolved.with_hierarchy(
                resolved.hierarchy, source="common_root"
            )
        return resolved

    def _is_none(self, value: Any) -> bool:
        resolved, constant = self._try_constant(value)
        return resolved and constant is None

    @staticmethod
    def _merge_array_states(states: tuple[bool | None, ...]) -> bool | None:
        if not states or any(state is False for state in states):
            return False
        if any(state is True for state in states):
            return True
        return None

    def _is_array_tuple_item(
        self,
        value: Any,
        index: int,
        *,
        seen: set[str],
        thread_data_only: bool = False,
    ) -> bool | None:
        if not isinstance(value, ir.Var):
            return False
        seen_key = f"{value.name}[{index}]"
        if seen_key in seen:
            return None
        seen.add(seen_key)
        return self._merge_array_states(
            tuple(
                self._is_array_tuple_item_definition(
                    definition,
                    index,
                    seen=set(seen),
                    thread_data_only=thread_data_only,
                )
                for definition in self._all_definitions(value)
            )
        )

    def _is_array_tuple_item_definition(
        self,
        definition: Any,
        index: int,
        *,
        seen: set[str],
        thread_data_only: bool,
    ) -> bool | None:
        if isinstance(definition, ir.Var):
            return self._is_array_tuple_item(
                definition, index, seen=seen, thread_data_only=thread_data_only
            )
        if not isinstance(definition, ir.Expr):
            return False
        if definition.op in {"cast", "exhaust_iter"}:
            return self._is_array_tuple_item(
                definition.value,
                index,
                seen=seen,
                thread_data_only=thread_data_only,
            )
        if definition.op == "phi":
            incoming_values = getattr(definition, "incoming_values", ())
            return self._merge_array_states(
                tuple(
                    self._is_array_tuple_item(
                        incoming,
                        index,
                        seen=set(seen),
                        thread_data_only=thread_data_only,
                    )
                    for incoming in incoming_values
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
        return False

    def _is_array_value(
        self,
        value: Any,
        *,
        seen: set[str] | None = None,
        thread_data_only: bool = False,
    ) -> bool | None:
        """Classify a value by its per-thread payload constructor provenance.

        Trace aliases, casts, phi inputs, and constant tuple projections to
        ``ThreadData`` or qualified local-array constructors. The three-state
        result lets a loop backedge coexist with a known constructor: a cycle
        alone is unresolved, a constructor plus cycles is accepted, and any
        concrete unrecognized path rejects the value. This classifies the
        payload form only; dtype and extent are inferred separately.

        Parameters
        ----------
        value : ir.Var or object
            Variable to classify. Non-variables are not recognized payloads.
        seen : set of str, optional
            Recursion-path variable names and tuple-projection keys. The current
            name is added in place; branches receive separate copies.
        thread_data_only : bool, optional
            If true, accept only common or qualified ``ThreadData``
            constructors. Otherwise, also accept the recognized CUDA local-array
            constructor.

        Returns
        -------
        bool or None
            ``True`` if at least one constructor is found and no path is
            rejected; ``False`` for an unsupported definition, missing
            definitions, or a non-variable; ``None`` when only cyclic paths
            remain. The operand validation wrappers turn that last state into a
            provenance diagnostic.
        """

        if not isinstance(value, ir.Var):
            return False
        if seen is None:
            seen = set()
        if value.name in seen:
            return None
        seen.add(value.name)
        return self._merge_array_states(
            tuple(
                self._is_array_definition(
                    definition,
                    seen=set(seen),
                    thread_data_only=thread_data_only,
                )
                for definition in self._all_definitions(value)
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
                    self._is_array_value(
                        incoming,
                        seen=set(seen),
                        thread_data_only=thread_data_only,
                    )
                    for incoming in incoming_values
                )
            )
        if definition.op in {"getitem", "static_getitem"}:
            index = getattr(definition, "index", None)
            if isinstance(index, ir.Var):
                resolved, index = self._try_constant(index)
                if not resolved:
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
        if function is _cuda_local_array:
            return not thread_data_only
        return False

    @staticmethod
    def _new_var(scope: Any, loc: ir.Loc, stem: str) -> ir.Var:
        return ir.Var(
            scope, f"__cuda_coop_group_{stem}_{next(_NAME_COUNTER)}__", loc
        )

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
            ir.Assign(
                ir.Global(function_var.name, factory, loc), function_var, loc
            )
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
        statements.append(
            ir.Assign(
                ir.Expr.call(
                    function_var, rewritten_args, rewritten_kwargs, loc
                ),
                inst.target,
                loc,
            )
        )
        return statements

    def _array_operand_state(self, operation: str, value: Any) -> bool:
        state = self._is_array_value(value)
        if state is None:
            raise CyclicArrayProvenanceError(operation)
        return state

    def _thread_data_operand_state(
        self, operation: str, parameter: str, value: Any
    ) -> bool:
        state = self._is_array_value(value, thread_data_only=True)
        if state is None:
            raise GroupRewriteError(
                f"cuda.coop.{operation} could not "
                f"resolve {parameter} payload provenance"
            )
        return state

    def _array_extent(
        self, value: Any, *, seen: set[str] | None = None
    ) -> int | None:
        """Recover one known per-thread item count from payload definitions.

        Follow aliases, casts, phi inputs, and constant tuple projections to
        ``ThreadData`` item counts or scalar local-array shapes. Gather known
        extents and require them to agree, ignoring unresolved paths and
        recursion backedges. Payload-kind validation is separate: a returned
        extent alone is not proof that every reaching definition is a valid
        payload. Required constructor dimensions may request literal argument
        specialization.

        Parameters
        ----------
        value : ir.Var or object
            Payload variable to inspect. Non-variables have no inferred extent.
        seen : set of str, optional
            Recursion-path names and tuple-projection keys. The current name is
            added in place; each reaching definition receives a separate copy.

        Returns
        -------
        int or None
            The unique known integral extent, excluding booleans, or ``None`` if
            none can be recovered. Positivity is validated elsewhere.

        Raises
        ------
        GroupRewriteError
            Known extents disagree, or a dimension depends on literal unrolling.
        ForceLiteralArg
            A constructor dimension needs literal argument specialization.
        """

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
            raise InconsistentArrayExtentError()
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
            raise InconsistentTupleExtentError()
        return next(iter(extents), None)

    def _array_extent_tuple_item_definition(
        self, definition: Any, index: int, *, seen: set[str]
    ) -> int | None:
        if isinstance(definition, ir.Var):
            return self._array_extent_tuple_item(definition, index, seen=seen)
        if not isinstance(definition, ir.Expr):
            return None
        if definition.op in {"cast", "exhaust_iter"}:
            return self._array_extent_tuple_item(
                definition.value, index, seen=seen
            )
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
                raise InconsistentLoopTupleExtentError()
            return next(iter(extents), None)
        if definition.op == "build_tuple":
            items = tuple(getattr(definition, "items", ()))
            if not -len(items) <= index < len(items):
                return None
            return self._array_extent(items[index], seen=set(seen))
        if definition.op != "call":
            return None
        return None

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
                if (extent := self._array_extent(incoming, seen=set(seen)))
                is not None
            }
            if len(extents) > 1:
                raise InconsistentLoopPayloadExtentError()
            return next(iter(extents), None)
        if definition.op in {"getitem", "static_getitem"}:
            index = getattr(definition, "index", None)
            if isinstance(index, ir.Var):
                resolved, index = self._try_constant(index)
                if not resolved:
                    return None
            if isinstance(index, Integral) and (not isinstance(index, bool)):
                return self._array_extent_tuple_item(
                    definition.value, int(index), seen=set(seen)
                )
            return None
        if definition.op != "call":
            return None
        function = self._callable(definition.func)
        if function in {ThreadData, _portable_api.ThreadData}:
            bound = self._bind(function, definition)
            extent_argument = bound.arguments["items_per_thread"]
            self._reject_literal_unroll_value(extent_argument, "payload extent")
            try:
                extent = self._constant(extent_argument)
            except GroupRewriteError:
                return None
            if isinstance(extent, Integral) and (not isinstance(extent, bool)):
                return int(extent)
            return None
        if function is _cuda_local_array:
            shape_ref = (
                definition.args[0]
                if definition.args
                else dict(definition.kws).get("shape")
            )
            if shape_ref is None:
                return None
            self._reject_literal_unroll_value(shape_ref, "payload extent")
            try:
                extent = self._constant(shape_ref)
            except GroupRewriteError:
                return None
            if isinstance(extent, Integral) and (not isinstance(extent, bool)):
                return int(extent)
            return None
        return None

    def _lower_root_operation(
        self, inst: ir.Assign, call: ir.Expr, function: Any, operation: str
    ) -> None:
        bound = self._bind(function, call)
        if bound.arguments.get("kwargs"):
            names = ", ".join(sorted(bound.arguments["kwargs"]))
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} "
                f"got unexpected keyword(s): {names}"
            )
        group = self._group(bound.arguments["group"])
        if group is None:
            raise NonConstantThreadGroupError(operation)
        is_common_root = _is_common_root_operation(function, operation)
        if is_common_root:
            _portable_dispatch._validate_portable_operation_group(
                operation, group
            )
        group = self._resolve_group(group, feature=operation)
        registration = group_primitive(operation)
        if registration is None:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir operation {operation!r} has no planner"
            )
        if (
            is_common_root
            and registration.validate_common_arguments is not None
        ):
            registration.validate_common_arguments(
                self.context, operation, bound
            )
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

    def _mark_descriptor_calls(self) -> None:
        for block in self.func_ir.blocks.values():
            for inst in block.body:
                if not isinstance(inst, ir.Assign):
                    continue
                call = inst.value
                if not isinstance(call, ir.Expr) or call.op != "call":
                    continue
                function = self._callable(call.func)
                if (
                    function is ThreadHierarchy
                    or function in _GROUP_CONSTRUCTORS
                ):
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
        descriptor_names = {
            inst.target.name for inst in self.descriptor_assigns
        }
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
        descriptor_names = {
            inst.target.name for inst in self.descriptor_assigns
        }
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
                raise EscapingGroupDescriptorError(names)

    def run(self) -> bool:
        self._reject_literal_unroll_constructors()
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
        self._validate_descriptor_uses()
        if not (
            self.descriptor_assigns or self.replacements or self.dead_func_names
        ):
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
                        ir.Assign(
                            ir.Const(None, inst.loc), inst.target, inst.loc
                        )
                    )
                    continue
                rewritten.append(inst)
            block.body = rewritten
        return True


def has_group_markers(func_ir) -> bool:
    """Return whether the current function IR still needs group planning.

    One recognized call anywhere in the function is enough: a group
    constructor such as ``this_block()``, ``ThreadHierarchy()``, a registered
    public group operation such as ``load()`` or ``store()``, or ``group_by()``
    on a recognized group descriptor. For ``group_by()``, trace the receiver
    through aliases, casts, control-flow merges, and earlier subgroup calls
    to distinguish group descriptors from unrelated objects with that method.

    Inspect only the supplied IR. Calls inside device helpers become visible
    here after inlining; this scan does not visit their bodies. ``ThreadData``
    and ``TempStorage`` constructors alone do not count as group markers.

    This check controls the handoff between group planning and provider
    rewriting. ``CoopGroupHierarchyPlanner`` uses a positive result to request
    launch metadata and resolve the group calls. ``CoopSinglePhaseRewrite``
    waits while the result is true. Successful group planning removes group
    descriptors and replaces public operations with private provider calls,
    which do not count as group markers. The result then becomes false even
    though cooperative work remains, allowing provider rewriting to proceed.

    Detection does not validate the calls or resolve their launch dimensions;
    those checks belong to the group planner.
    """
    analyzer = object.__new__(_GroupCallPlanner)
    analyzer.func_ir = func_ir
    analyzer._group_cache = {}
    analyzer._hierarchy_cache = {}

    def is_group_descriptor(value: Any, seen: set[str]) -> bool:
        if isinstance(value, ThreadGroup):
            return True
        if not isinstance(value, ir.Var) or value.name in seen:
            return False
        seen = {*seen, value.name}
        for definition in analyzer._all_definitions(value):
            if isinstance(definition, ir.Var) and is_group_descriptor(
                definition, seen
            ):
                return True
            if isinstance(definition, (ir.Global, ir.FreeVar, ir.Const)):
                if isinstance(definition.value, ThreadGroup):
                    return True
                continue
            if not isinstance(definition, ir.Expr):
                continue
            if definition.op == "cast" and is_group_descriptor(
                definition.value, seen
            ):
                return True
            if definition.op == "phi" and any(
                is_group_descriptor(incoming, seen)
                for incoming in getattr(definition, "incoming_values", ())
            ):
                return True
            if definition.op != "call":
                continue
            function = analyzer._callable(definition.func)
            if function in _GROUP_CONSTRUCTORS:
                return True
            function_definition = analyzer._definition(definition.func)
            if (
                isinstance(function_definition, ir.Expr)
                and function_definition.op == "getattr"
                and function_definition.attr == "group_by"
                and is_group_descriptor(function_definition.value, seen)
            ):
                return True
        return False

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
                and (function_definition.attr == "group_by")
                and is_group_descriptor(function_definition.value, set())
            ):
                return True
    return False


@register_planner
class CoopGroupHierarchyPlanner(WholeFunctionPlanner):
    """Resolve cooperative group calls against one exact configured launch."""

    def run(self) -> bool:
        if not has_group_markers(self.state.func_ir):
            return False
        if self.is_device_function:
            function_name = self.state.func_ir.func_id.func_qualname
            raise GroupRewriteError(
                "cuda.coop.numba_mlir cooperative calls in device function "
                f"{function_name!r} must be inlined into a kernel. Standalone "
                "collective helpers and collectives inside standalone "
                "callbacks "
                "are unsupported; use inline='always' for a kernel helper or "
                "move the cooperative calls into the kernel."
            )
        launch_config = require_launch_config(self.state)
        return _GroupCallPlanner(self.state, launch_config).run()


__all__ = [
    "CoopGroupHierarchyPlanner",
    "GroupRewriteError",
    "has_group_markers",
]
