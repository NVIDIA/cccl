# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Whole-function lowering for Numba-CUDA-MLIR thread-group markers."""

from __future__ import annotations

import inspect
from itertools import count
from numbers import Integral
from typing import Any

from numba_cuda_mlir import types
from numba_cuda_mlir.errors import ForceLiteralArg
from numba_cuda_mlir.numbair_transforms import ir

from cuda.coop._core import (
    LaunchFactOrigin,
    LaunchFacts,
    ThreadGroup,
    ThreadHierarchy,
    normalize_thread_level,
    resolve_thread_group,
)
from cuda.coop._core import root_api as _common_root_api

from . import _group_ops
from . import _thread_group as _thread_groups
from ._scan_op import ScanOp

_NAME_COUNTER = count()
_PAYLOAD_DTYPE_LIKE = "like"
_PAYLOAD_DTYPE_BOOL = "bool"
_PAYLOAD_DTYPE_INT32 = "int32"
_GROUP_CONSTRUCTORS = {
    _thread_groups.this_thread: _thread_groups.this_thread,
    _thread_groups.this_warp: _thread_groups.this_warp,
    _thread_groups.this_block: _thread_groups.this_block,
    _thread_groups.this_cluster: _thread_groups.this_cluster,
    _thread_groups.this_grid: _thread_groups.this_grid,
    _common_root_api.this_thread: _thread_groups.this_thread,
    _common_root_api.this_warp: _thread_groups.this_warp,
    _common_root_api.this_block: _thread_groups.this_block,
    _common_root_api.this_cluster: _thread_groups.this_cluster,
    _common_root_api.this_grid: _thread_groups.this_grid,
}
_ROOT_OPERATIONS = {
    getattr(_group_ops, name): name
    for name in (
        "load",
        "store",
        "reduce",
        "sum",
        "scan",
        "exclusive_sum",
        "inclusive_sum",
        "exclusive_scan",
        "inclusive_scan",
    )
}
_ROOT_OPERATIONS.update(
    {
        getattr(_common_root_api, name): name
        for name in (*_common_root_api._GROUP_OPERATIONS,)
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


def _builtin_subtract(lhs: Any, rhs: Any) -> Any:
    return lhs - rhs


def _builtin_not_equal(lhs: Any, rhs: Any) -> bool:
    return lhs != rhs


def _builtin_less(lhs: Any, rhs: Any) -> bool:
    return lhs < rhs


def _builtin_greater(lhs: Any, rhs: Any) -> bool:
    return lhs > rhs


def _histogram_provider_counter_dtype(counter_dtype: Any) -> Any:
    """Use the unsigned CUB accumulator matching the public counter width."""

    if counter_dtype in (types.int32, types.uint32):
        return types.uint32
    if counter_dtype in (types.int64, types.uint64):
        return types.uint64
    return counter_dtype


def _group_operation_name(function: Any) -> str | None:
    """Return the portable/qualified operation name for one marker callable."""

    operation = getattr(function, "__cuda_coop_backend_member__", None)
    if operation is not None:
        return operation
    if getattr(function, "__module__", None) == _group_ops.__name__:
        return getattr(function, "__name__", None)
    return None


def _is_common_root_operation(function: Any, operation: str) -> bool:
    """Return whether identity rewriting found the common-root marker."""

    member = getattr(_common_root_api, operation, None)
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
    """Compiler-only marker for one typed group result/input payload."""

    raise GroupRewriteError(
        "typed group payload markers must be lowered before device compilation"
    )


class _GroupCallPlanner:
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
            value.name,
            (),
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
                "cuda.coop.numba_mlir group arguments that shape provider "
                f"specialization must be compile-time constants; got {value.name!r}"
            ) from exc

    def _try_constant(self, value: Any) -> tuple[bool, Any]:
        """Resolve a constant without requesting dispatcher specialization."""
        if isinstance(value, ir.Var):
            definition = self._definition(value)
            if isinstance(definition, ir.Arg):
                argtype = self.state.args[definition.index]
                if isinstance(argtype, types.Literal):
                    return True, argtype.literal_value
                if isinstance(argtype, types.NoneType) or (
                    isinstance(argtype, types.Omitted) and argtype.value is None
                ):
                    return True, None
                return False, None
        try:
            return True, self._constant(value)
        except (ForceLiteralArg, GroupRewriteError):
            return False, None

    def _bind(self, function: Any, call: ir.Expr) -> inspect.BoundArguments:
        if call.vararg is not None or call.varkwarg is not None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir group calls do not support *args/**kwargs"
            )
        try:
            bound = inspect.signature(function).bind(
                *call.args,
                **dict(call.kws),
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

        token = self._constant(value)
        if token is None and allow_none:
            return None
        token = getattr(token, "value", token)
        if isinstance(token, str):
            token = token.strip().lower().replace("-", "_")
        if token not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(
                f"cuda.coop.{operation} {parameter} must be one of: {choices}; "
                "use a backend-qualified import for backend-only controls"
            )
        return token

    def _validate_common_arguments(
        self,
        operation: str,
        bound: inspect.BoundArguments,
    ) -> None:
        """Enforce the backend-neutral V1 subset before backend lowering."""

        selector_specs = {
            "load": (
                "algorithm",
                _common_root_api._LOAD_STORE_ALGORITHMS,
                False,
            ),
            "store": (
                "algorithm",
                _common_root_api._LOAD_STORE_ALGORITHMS,
                False,
            ),
            "reduce": (
                "algorithm",
                _common_root_api._REDUCE_ALGORITHMS,
                True,
            ),
            "sum": (
                "algorithm",
                _common_root_api._REDUCE_ALGORITHMS,
                True,
            ),
            "scan": (
                "algorithm",
                _common_root_api._SCAN_ALGORITHMS,
                True,
            ),
            "exclusive_sum": (
                "algorithm",
                _common_root_api._SCAN_ALGORITHMS,
                True,
            ),
            "inclusive_sum": (
                "algorithm",
                _common_root_api._SCAN_ALGORITHMS,
                True,
            ),
            "exclusive_scan": (
                "algorithm",
                _common_root_api._SCAN_ALGORITHMS,
                True,
            ),
            "inclusive_scan": (
                "algorithm",
                _common_root_api._SCAN_ALGORITHMS,
                True,
            ),
            "exchange": ("mode", _common_root_api._EXCHANGE_MODES, False),
            "adjacent_difference": (
                "direction",
                _common_root_api._ADJACENT_DIFFERENCE_DIRECTIONS,
                False,
            ),
            "discontinuity": (
                "mode",
                _common_root_api._DISCONTINUITY_MODES,
                False,
            ),
            "shuffle": ("mode", _common_root_api._SHUFFLE_MODES, False),
            "histogram": (
                "algorithm",
                _common_root_api._HISTOGRAM_ALGORITHMS,
                False,
            ),
        }
        spec = selector_specs.get(operation)
        if spec is not None:
            parameter, allowed, allow_none = spec
            bound.arguments[parameter] = self._validate_common_selector(
                operation,
                parameter,
                bound.arguments[parameter],
                allowed,
                allow_none=allow_none,
            )

        if operation == "scan":
            bound.arguments["mode"] = self._validate_common_selector(
                operation,
                "mode",
                bound.arguments["mode"],
                _common_root_api._SCAN_MODES,
            )

        if operation == "reduce":
            from ._group_provider import _normalize_reduce_operation

            binary_op = self._constant(bound.arguments["binary_op"])
            try:
                _normalize_reduce_operation(binary_op)
            except NotImplementedError as exc:
                raise ValueError(
                    "cuda.coop.reduce binary_op accepts built-in operators only; "
                    "use cuda.coop.numba_mlir for custom callback behavior"
                ) from exc

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
        # Intentionally discard the binding: this validates the zero-argument
        # marker call before launch facts resolve the hierarchy extents.
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
            if getattr(function, "__module__", None) == _common_root_api.__name__:
                assert group.hierarchy is not None
                group = group.with_hierarchy(
                    group.hierarchy,
                    source="common_root",
                )
            self._group_cache[value.name] = group
            return group

        function_definition = self._definition(definition.func)
        if (
            isinstance(function_definition, ir.Expr)
            and function_definition.op == "getattr"
            and function_definition.attr == "group_by"
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
        self,
        group: ThreadGroup,
        *,
        feature: str,
        through_level: str | None = None,
    ) -> ThreadGroup:
        resolution = resolve_thread_group(
            group,
            self.launch,
            through_level=through_level,
        )
        try:
            resolved = resolution.require_supported()
        except NotImplementedError as exc:
            raise NotImplementedError(f"cuda.coop.numba_mlir.{feature} {exc}") from exc
        if group.source == "common_root":
            assert resolved.hierarchy is not None
            resolved = resolved.with_hierarchy(
                resolved.hierarchy,
                source="common_root",
            )
        return resolved

    def _is_none(self, value: Any) -> bool:
        resolved, constant = self._try_constant(value)
        return resolved and constant is None

    @staticmethod
    def _merge_array_states(
        states: tuple[bool | None, ...],
    ) -> bool | None:
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
                definition,
                index,
                seen=seen,
                thread_data_only=thread_data_only,
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
                items[index],
                seen=set(seen),
                thread_data_only=thread_data_only,
            )
        if definition.op != "call":
            return False

        function = self._callable(definition.func)
        operation = _group_operation_name(function)
        if operation == "discontinuity":
            bound = self._bind(function, definition)
            from cuda.coop._core.block import BlockDiscontinuityMode

            try:
                mode = BlockDiscontinuityMode(self._constant(bound.arguments["mode"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "cuda.coop.numba_mlir.discontinuity mode must be "
                    "'heads', 'tails', or 'heads_and_tails'"
                ) from exc
            if mode is not BlockDiscontinuityMode.HEADS_AND_TAILS:
                return False
            if index < 0:
                index += 2
            if not 0 <= index < 2:
                return False
            return self._is_array_value(
                bound.arguments["value"],
                seen=seen,
                thread_data_only=thread_data_only,
            )

        pair_arguments = {
            "merge_sort_pairs": ("keys", "values"),
            "radix_sort_pairs": ("keys", "values"),
            "topk_max_pairs": ("keys", "values"),
            "topk_min_pairs": ("keys", "values"),
        }.get(operation)
        if pair_arguments is None:
            return False
        if index < 0:
            index += len(pair_arguments)
        if not 0 <= index < len(pair_arguments):
            return False
        bound = self._bind(function, definition)
        return self._is_array_value(
            bound.arguments[pair_arguments[index]],
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
                self._is_array_definition(
                    definition,
                    seen=set(seen),
                    thread_data_only=thread_data_only,
                )
                for definition in self._all_definitions(value)
            )
        )

    def _is_array_definition(
        self,
        definition: Any,
        *,
        seen: set[str],
        thread_data_only: bool,
    ) -> bool | None:
        if isinstance(definition, ir.Var):
            return self._is_array_value(
                definition,
                seen=seen,
                thread_data_only=thread_data_only,
            )
        if not isinstance(definition, ir.Expr):
            return False
        if definition.op == "cast":
            return self._is_array_value(
                definition.value,
                seen=seen,
                thread_data_only=thread_data_only,
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
                try:
                    index = self._constant(index)
                except GroupRewriteError:
                    return False
            if isinstance(index, Integral) and not isinstance(index, bool):
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
        operation = _group_operation_name(function)
        from . import ThreadData

        if function in {ThreadData, _common_root_api.ThreadData}:
            return True
        if function is _typed_group_payload_like:
            return self._is_array_value(
                definition.args[0],
                seen=seen,
                thread_data_only=thread_data_only,
            )
        if getattr(function, "__name__", "") == "array" and getattr(
            function,
            "__module__",
            "",
        ) in {
            "cuda.local",
            "numba_cuda_mlir.cuda.local",
        }:
            return not thread_data_only

        if operation == "discontinuity":
            bound = self._bind(function, definition)
            from cuda.coop._core.block import BlockDiscontinuityMode

            try:
                mode = BlockDiscontinuityMode(self._constant(bound.arguments["mode"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "cuda.coop.numba_mlir.discontinuity mode must be "
                    "'heads', 'tails', or 'heads_and_tails'"
                ) from exc
            if mode is BlockDiscontinuityMode.HEADS_AND_TAILS:
                return False
            return self._is_array_value(
                bound.arguments["value"],
                seen=seen,
                thread_data_only=thread_data_only,
            )

        if thread_data_only and operation in {"histogram", "run_length_decode"}:
            return True

        array_result_argument = {
            "adjacent_difference": "value",
            "discontinuity": "value",
            "exchange": "value",
            "exclusive_scan": "value",
            "exclusive_sum": "value",
            "histogram": "samples",
            "inclusive_scan": "value",
            "inclusive_sum": "value",
            "load": "output",
            "merge_sort_keys": "keys",
            "radix_rank": "keys",
            "radix_sort_keys": "keys",
            "run_length_decode": "run_values",
            "scan": "value",
            "shuffle": "value",
            "topk_max_keys": "keys",
            "topk_min_keys": "keys",
        }.get(operation)
        if array_result_argument is None:
            return False
        bound = self._bind(function, definition)
        return self._is_array_value(
            bound.arguments[array_result_argument],
            seen=seen,
            thread_data_only=thread_data_only,
        )

    @staticmethod
    def _new_var(scope: Any, loc: ir.Loc, stem: str) -> ir.Var:
        return ir.Var(
            scope,
            f"__cuda_coop_group_{stem}_{next(_NAME_COUNTER)}__",
            loc,
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
        return_alias: ir.Var | tuple[ir.Var, ...] | None = None,
        common_profile_operation: str | None = None,
    ) -> list[Any]:
        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        if common_profile_operation is not None:
            kwargs = dict(kwargs)
            kwargs.setdefault("_common_profile_operation", common_profile_operation)
        function_var = self._new_var(scope, loc, "factory")
        statements.append(
            ir.Assign(
                ir.Global(function_var.name, factory, loc),
                function_var,
                loc,
            )
        )
        rewritten_args = [
            self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"arg{idx}",
                value=value,
            )
            for idx, value in enumerate(args)
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
        call_target = (
            inst.target
            if return_alias is None
            else self._new_var(scope, loc, "ignored_result")
        )
        statements.append(
            ir.Assign(
                ir.Expr.call(
                    function_var,
                    rewritten_args,
                    rewritten_kwargs,
                    loc,
                ),
                call_target,
                loc,
            )
        )
        if isinstance(return_alias, tuple):
            statements.append(
                ir.Assign(
                    ir.Expr.build_tuple(list(return_alias), loc),
                    inst.target,
                    loc,
                )
            )
        elif return_alias is not None:
            statements.append(ir.Assign(return_alias, inst.target, loc))
        return statements

    def _array_operand_state(self, operation: str, value: Any) -> bool:
        state = self._is_array_value(value)
        if state is None:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} could not resolve cyclic "
                "array provenance to a concrete scalar or array value"
            )
        return state

    def _thread_data_operand_state(
        self,
        operation: str,
        parameter: str,
        value: Any,
    ) -> bool:
        state = self._is_array_value(value, thread_data_only=True)
        if state is None:
            raise GroupRewriteError(
                f"cuda.coop.{operation} could not resolve {parameter} "
                "payload provenance"
            )
        return state

    def _array_extent(
        self,
        value: Any,
        *,
        seen: set[str] | None = None,
    ) -> int | None:
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
                "cuda.coop.numba_mlir array aliases have inconsistent "
                "items_per_thread extents"
            )
        return next(iter(extents), None)

    def _array_extent_tuple_item(
        self,
        value: Any,
        index: int,
        *,
        seen: set[str],
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
                    definition,
                    index,
                    seen=set(seen),
                )
            )
            is not None
        }
        if len(extents) > 1:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir tuple projections have inconsistent "
                "items_per_thread extents"
            )
        return next(iter(extents), None)

    def _array_extent_tuple_item_definition(
        self,
        definition: Any,
        index: int,
        *,
        seen: set[str],
    ) -> int | None:
        if isinstance(definition, ir.Var):
            return self._array_extent_tuple_item(definition, index, seen=seen)
        if not isinstance(definition, ir.Expr):
            return None
        if definition.op in {"cast", "exhaust_iter"}:
            return self._array_extent_tuple_item(
                definition.value,
                index,
                seen=seen,
            )
        if definition.op == "phi":
            extents = {
                extent
                for incoming in getattr(definition, "incoming_values", ())
                if (
                    extent := self._array_extent_tuple_item(
                        incoming,
                        index,
                        seen=set(seen),
                    )
                )
                is not None
            }
            if len(extents) > 1:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir loop-carried tuple payloads have "
                    "inconsistent items_per_thread extents"
                )
            return next(iter(extents), None)
        if definition.op == "build_tuple":
            items = tuple(getattr(definition, "items", ()))
            if not -len(items) <= index < len(items):
                return None
            return self._array_extent(items[index], seen=set(seen))
        if definition.op != "call":
            return None

        function = self._callable(definition.func)
        operation = _group_operation_name(function)
        tuple_arguments = {
            "merge_sort_pairs": ("keys", "values"),
            "radix_sort_pairs": ("keys", "values"),
            "topk_max_pairs": ("keys", "values"),
            "topk_min_pairs": ("keys", "values"),
        }.get(operation)
        if tuple_arguments is None:
            return None
        if index < 0:
            index += len(tuple_arguments)
        if not 0 <= index < len(tuple_arguments):
            return None
        bound = self._bind(function, definition)
        return self._array_extent(
            bound.arguments[tuple_arguments[index]],
            seen=set(seen),
        )

    def _array_extent_definition(
        self,
        definition: Any,
        *,
        seen: set[str],
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
                    "cuda.coop.numba_mlir loop-carried payloads have "
                    "inconsistent items_per_thread extents"
                )
            return next(iter(extents), None)
        if definition.op in {"getitem", "static_getitem"}:
            index = getattr(definition, "index", None)
            if isinstance(index, ir.Var):
                try:
                    index = self._constant(index)
                except GroupRewriteError:
                    return None
            if isinstance(index, Integral) and not isinstance(index, bool):
                return self._array_extent_tuple_item(
                    definition.value,
                    int(index),
                    seen=set(seen),
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
                if isinstance(extent, Integral) and not isinstance(extent, bool):
                    return int(extent)
                return None
            if is_array is False:
                return 1
            return self._array_extent(definition.args[0], seen=seen)

        from . import ThreadData

        if function in {ThreadData, _common_root_api.ThreadData}:
            bound = self._bind(function, definition)
            extent_argument = bound.arguments["items_per_thread"]
            try:
                extent = self._constant(extent_argument)
            except GroupRewriteError:
                return None
            if isinstance(extent, Integral) and not isinstance(extent, bool):
                return int(extent)
            return None
        if getattr(function, "__name__", "") == "array" and getattr(
            function, "__module__", ""
        ) in {"cuda.local", "numba_cuda_mlir.cuda.local"}:
            if not definition.args:
                return None
            try:
                extent = self._constant(definition.args[0])
            except GroupRewriteError:
                return None
            if isinstance(extent, Integral) and not isinstance(extent, bool):
                return int(extent)
            return None

        operation = _group_operation_name(function)
        if operation == "histogram":
            bound = self._bind(function, definition)
            try:
                extent = self._constant(bound.arguments["bins_per_thread"])
            except GroupRewriteError:
                return None
            if isinstance(extent, Integral) and not isinstance(extent, bool):
                return int(extent)
            return None
        if operation == "run_length_decode":
            bound = self._bind(function, definition)
            try:
                extent = self._constant(bound.arguments["decoded_items_per_thread"])
            except GroupRewriteError:
                return None
            if isinstance(extent, Integral) and not isinstance(extent, bool):
                return int(extent)
            return None
        shape_argument = {
            "adjacent_difference": "value",
            "discontinuity": "value",
            "exchange": "value",
            "exclusive_scan": "value",
            "exclusive_sum": "value",
            "inclusive_scan": "value",
            "inclusive_sum": "value",
            "load": "output",
            "merge_sort_keys": "keys",
            "merge_sort_pairs": "keys",
            "radix_rank": "keys",
            "radix_sort_keys": "keys",
            "radix_sort_pairs": "keys",
            "scan": "value",
            "shuffle": "value",
            "topk_max_keys": "keys",
            "topk_max_pairs": "keys",
            "topk_min_keys": "keys",
            "topk_min_pairs": "keys",
        }.get(operation)
        if shape_argument is None:
            return None
        bound = self._bind(function, definition)
        return self._array_extent(bound.arguments[shape_argument], seen=seen)

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
        # Fresh scalar boxes live in ``statements`` until the replacement is
        # spliced into the function IR, so definition lookup cannot see them yet.
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
            statements,
            scope=scope,
            loc=loc,
            stem=f"{stem}_is_array",
            value=is_array,
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
            ir.Assign(
                ir.Expr.call(
                    function_var,
                    args,
                    (),
                    loc,
                ),
                payload,
                loc,
            )
        )
        return payload

    def _thread_data_payload(
        self,
        statements: list[Any],
        *,
        scope: Any,
        loc: ir.Loc,
        stem: str,
        items_per_thread: Any,
        dtype: Any = None,
    ) -> ir.Var:
        """Emit one explicit-size ThreadData marker for the next rewrite pass."""

        from . import ThreadData

        result = self._new_var(scope, loc, f"{stem}_payload")
        proxy = ir.Assign(ir.Const(None, loc), result, loc)
        args = [items_per_thread]
        if dtype is not None:
            args.append(dtype)
        statements.extend(
            self._rewritten_call(
                proxy,
                factory=ThreadData,
                args=args,
                kwargs={},
            )
        )
        return result

    def _emit_factory_call(
        self,
        statements: list[Any],
        *,
        scope: Any,
        loc: ir.Loc,
        stem: str,
        factory: Any,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> ir.Var:
        """Emit a call to an ordinary compiler/runtime callable."""

        result = self._new_var(scope, loc, stem)
        proxy = ir.Assign(ir.Const(None, loc), result, loc)
        statements.extend(
            self._rewritten_call(
                proxy,
                factory=factory,
                args=args,
                kwargs=kwargs,
            )
        )
        return result

    def _emit_method_call(
        self,
        statements: list[Any],
        *,
        scope: Any,
        loc: ir.Loc,
        stem: str,
        receiver: ir.Var,
        method: str,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> ir.Var:
        """Emit a method call on a compiler-only scoped parent placeholder."""

        method_var = self._new_var(scope, loc, f"{stem}_{method}")
        statements.append(
            ir.Assign(ir.Expr.getattr(receiver, method, loc), method_var, loc)
        )
        rewritten_args = [
            self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{stem}_{method}_arg{idx}",
                value=value,
            )
            for idx, value in enumerate(args)
        ]
        rewritten_kwargs = tuple(
            (
                name,
                self._value_var(
                    statements,
                    scope=scope,
                    loc=loc,
                    stem=f"{stem}_{method}_{name}",
                    value=value,
                ),
            )
            for name, value in kwargs.items()
        )
        result = self._new_var(scope, loc, f"{stem}_{method}_result")
        statements.append(
            ir.Assign(
                ir.Expr.call(method_var, rewritten_args, rewritten_kwargs, loc),
                result,
                loc,
            )
        )
        return result

    def _emit_shared_array(
        self,
        statements: list[Any],
        *,
        scope: Any,
        loc: ir.Loc,
        stem: str,
        items: Any,
        dtype: Any,
    ) -> ir.Var:
        from numba_cuda_mlir import cuda as cuda_module

        module_var = self._new_var(scope, loc, f"{stem}_cuda")
        shared_var = self._new_var(scope, loc, f"{stem}_shared")
        array_var = self._new_var(scope, loc, f"{stem}_array")
        statements.append(
            ir.Assign(ir.Global(module_var.name, cuda_module, loc), module_var, loc)
        )
        statements.append(
            ir.Assign(ir.Expr.getattr(module_var, "shared", loc), shared_var, loc)
        )
        statements.append(
            ir.Assign(ir.Expr.getattr(shared_var, "array", loc), array_var, loc)
        )
        items_var = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{stem}_items",
            value=items,
        )
        dtype_var = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{stem}_dtype",
            value=dtype,
        )
        result = self._new_var(scope, loc, f"{stem}_result")
        statements.append(
            ir.Assign(
                ir.Expr.call(array_var, [items_var, dtype_var], (), loc),
                result,
                loc,
            )
        )
        return result

    def _boxed_group_operand(
        self,
        statements: list[Any],
        *,
        operation: str,
        value: ir.Var,
        scope: Any,
        loc: ir.Loc,
    ) -> tuple[ir.Var, bool]:
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
        statements.append(
            ir.Assign(
                ir.Expr.getitem(payload, index, loc),
                result,
                loc,
            )
        )
        return result

    def _scope_factory(
        self,
        group: ThreadGroup,
        operation: str,
    ) -> tuple[Any, dict[str, Any]]:
        assert group.hierarchy is not None
        block_dim = group.hierarchy.block_dim
        assert block_dim is not None
        if group.kind == "block":
            from . import _block as block

            return getattr(block, operation), {"threads_per_block": block_dim}
        if group.kind in {"warp", "threads_within_warp"}:
            from cuda.coop._core.group_dispatch import _cub_warp_width

            from . import _warp as warp

            name = {
                "exchange": "warp_exchange",
                "load": "warp_load",
                "store": "warp_store",
                "reduce": "warp_reduce",
                "sum": "warp_sum",
                "exclusive_sum": "warp_exclusive_sum",
                "inclusive_sum": "warp_inclusive_sum",
                "exclusive_scan": "warp_exclusive_scan",
                "inclusive_scan": "warp_inclusive_scan",
                "merge_sort_keys": "warp_merge_sort_keys",
                "merge_sort_pairs": "warp_merge_sort_pairs",
            }[operation]
            threads_in_warp = _cub_warp_width(group)
            return getattr(warp, name), {
                "threads_in_warp": threads_in_warp,
                "threads_per_block": block_dim,
            }
        raise NotImplementedError(
            f"cuda.coop.numba_mlir.{operation} currently lowers only block, "
            "physical-warp, and logical-warp groups through CUB"
        )

    def _lower_load_store(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        if is_common_root:
            if operation == "load":
                if not self._thread_data_operand_state(
                    operation,
                    "output",
                    bound.arguments["output"],
                ):
                    raise TypeError(
                        "cuda.coop.load requires output to be a fixed-size "
                        "ThreadData payload in common V1; use "
                        "cuda.coop.numba_mlir for backend-qualified local-array "
                        "payload support"
                    )
            else:
                value = bound.arguments["value"]
                if self._array_operand_state(operation, value) and not (
                    self._thread_data_operand_state(operation, "value", value)
                ):
                    raise TypeError(
                        "cuda.coop.store accepts only a scalar or fixed-size "
                        "ThreadData value payload in common V1; use "
                        "cuda.coop.numba_mlir for backend-qualified local-array "
                        "payload support"
                    )

        factory, factory_kwargs = self._scope_factory(group, operation)
        factory_kwargs["algorithm"] = bound.arguments["algorithm"]
        if is_common_root:
            factory_kwargs["_common_profile_operation"] = operation
        if not self._is_none(bound.arguments["valid_items"]):
            factory_kwargs["num_valid_items"] = bound.arguments["valid_items"]
        if operation == "load" and not self._is_none(bound.arguments["oob_default"]):
            factory_kwargs["oob_default"] = bound.arguments["oob_default"]
        if group.kind in {"warp", "threads_within_warp"}:
            factory_kwargs["_physical_warp_tile_origin"] = True
            factory_kwargs["offset"] = (
                0
                if self._is_none(bound.arguments["offset"])
                else bound.arguments["offset"]
            )
        elif not self._is_none(bound.arguments["offset"]):
            factory_kwargs["offset"] = bound.arguments["offset"]
        if operation == "store":
            factory_kwargs["_group_root_store"] = True
        if not self._is_none(bound.arguments["temp_storage"]):
            factory_kwargs["temp_storage"] = bound.arguments["temp_storage"]

        if operation == "load":
            runtime_args = [
                bound.arguments["source"],
                bound.arguments["output"],
            ]
            return_alias = bound.arguments["output"]
        else:
            runtime_args = [
                bound.arguments["destination"],
                bound.arguments["value"],
            ]
            return_alias = None
        return self._rewritten_call(
            inst,
            factory=factory,
            args=runtime_args,
            kwargs=factory_kwargs,
            return_alias=return_alias,
        )

    def _lower_reduce(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        from ._group_provider import _validate_group_reduce_support

        if bound.arguments.get("args"):
            raise GroupRewriteError(
                "cuda.coop.numba_mlir.reduce accepts no extra positional arguments"
            )
        if bound.arguments.get("kwargs"):
            names = ", ".join(sorted(bound.arguments["kwargs"]))
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.reduce got unexpected keyword(s): {names}"
            )

        if is_common_root:
            value = bound.arguments["value"]
            if self._array_operand_state(operation, value) and not (
                self._thread_data_operand_state(operation, "value", value)
            ):
                raise TypeError(
                    f"cuda.coop.{operation} accepts only a scalar or fixed-size "
                    "ThreadData value payload in common V1; use "
                    "cuda.coop.numba_mlir for backend-qualified local-array "
                    "payload support"
                )

        has_valid = not self._is_none(bound.arguments["valid_items"])
        has_algorithm = not self._is_none(bound.arguments["algorithm"])
        if has_valid or has_algorithm:
            if group.kind not in {"block", "warp", "threads_within_warp"}:
                raise NotImplementedError(
                    "valid_items and explicit CUB algorithms are supported only "
                    "for physical block, physical-warp, and logical-warp groups"
                )
            broadcast = self._constant(bound.arguments["broadcast"])
            if not isinstance(broadcast, bool):
                raise TypeError(
                    "cuda.coop.numba_mlir.reduce broadcast must be a compile-time bool"
                )
            if broadcast:
                raise NotImplementedError(
                    "direct CUB reduce returns a defined value only at the group "
                    "root; it cannot satisfy broadcast=True"
                )
            if has_valid:
                is_static, valid_items = self._try_constant(
                    bound.arguments["valid_items"]
                )
                if is_static:
                    if isinstance(valid_items, bool) or not isinstance(
                        valid_items, Integral
                    ):
                        raise TypeError(
                            "cuda.coop.numba_mlir.reduce valid_items must be "
                            "an integer, not bool"
                        )
                    group_size = group.static_size
                    assert group_size is not None
                    if not 1 <= int(valid_items) <= group_size:
                        raise ValueError(
                            "cuda.coop.numba_mlir.reduce static valid_items "
                            f"must be between 1 and group size {group_size}"
                        )
            binary_op_ref = bound.arguments["binary_op"]
            binary_op = self._constant(binary_op_ref)
            from ._group_provider import _normalize_reduce_operation

            normalized_op = _normalize_reduce_operation(binary_op)
            if normalized_op == "sum":
                factory, kwargs = self._scope_factory(group, "sum")
            elif group.kind == "block":
                from ._block._block_reduce import block_reduce_builtin

                assert group.hierarchy is not None
                factory = block_reduce_builtin
                kwargs = {
                    "threads_per_block": group.hierarchy.block_dim,
                    "binary_op": normalized_op,
                }
            elif group.kind in {"warp", "threads_within_warp"}:
                from ._warp._warp_reduce import warp_reduce_builtin

                assert group.hierarchy is not None
                threads_in_warp = group.static_size
                assert threads_in_warp is not None
                factory = warp_reduce_builtin
                kwargs = {
                    "threads_in_warp": threads_in_warp,
                    "threads_per_block": group.hierarchy.block_dim,
                    "binary_op": normalized_op,
                }
            if has_valid:
                name = "num_valid" if group.kind == "block" else "valid_items"
                kwargs[name] = bound.arguments["valid_items"]
            if has_algorithm:
                if group.kind != "block":
                    raise NotImplementedError(
                        "CUB algorithm selection applies to BlockReduce, not WarpReduce"
                    )
                kwargs["algorithm"] = bound.arguments["algorithm"]
            return self._rewritten_call(
                inst,
                factory=factory,
                args=[bound.arguments["value"]],
                kwargs=kwargs,
                common_profile_operation=(operation if is_common_root else None),
            )

        _validate_group_reduce_support(group)
        if group.kind == "grid":
            raise NotImplementedError(
                "cuda.coop.numba_mlir.reduce grid groups require a hidden "
                "per-launch provider workspace; support is intentionally "
                "deferred until the backend exposes that ABI"
            )
        from ._group_provider import group_reduce

        return self._rewritten_call(
            inst,
            factory=group_reduce,
            args=[bound.arguments["value"]],
            kwargs={
                "group": group,
                "binary_op": bound.arguments["binary_op"],
                "broadcast": bound.arguments["broadcast"],
            },
            common_profile_operation=(operation if is_common_root else None),
        )

    @staticmethod
    def _reject_extra_root_arguments(
        operation: str,
        bound: inspect.BoundArguments,
    ) -> None:
        if bound.arguments.get("args"):
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} accepts no extra "
                "positional arguments"
            )
        if bound.arguments.get("kwargs"):
            names = ", ".join(sorted(bound.arguments["kwargs"]))
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} got unexpected keyword(s): {names}"
            )

    def _lower_scan(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        self._reject_extra_root_arguments("scan", bound)
        mode = self._constant(bound.arguments["mode"])
        if mode not in {"exclusive", "inclusive"}:
            raise ValueError(
                "cuda.coop.numba_mlir.scan mode must be 'exclusive' or 'inclusive'"
            )
        if mode == "inclusive" and not self._is_none(bound.arguments["initial_value"]):
            raise ValueError(
                "cuda.coop.numba_mlir.scan initial_value is not supported for "
                "inclusive scans"
            )

        if group.kind == "block":
            if not self._is_none(bound.arguments["valid_items"]):
                raise NotImplementedError(
                    "cuda.coop.numba_mlir.scan valid_items applies to physical "
                    "and logical warp groups, not block groups"
                )
            factory, factory_kwargs = self._scope_factory(group, "scan")
            factory_kwargs.update(
                {
                    "mode": mode,
                    "scan_op": (
                        "+"
                        if self._is_none(bound.arguments["scan_op"])
                        else bound.arguments["scan_op"]
                    ),
                }
            )
            if not self._is_none(bound.arguments["initial_value"]):
                factory_kwargs["initial_value"] = bound.arguments["initial_value"]
            if not self._is_none(bound.arguments["algorithm"]):
                factory_kwargs["algorithm"] = bound.arguments["algorithm"]
            if not self._is_none(bound.arguments["temp_storage"]):
                factory_kwargs["temp_storage"] = bound.arguments["temp_storage"]
            if not self._is_none(bound.arguments["aggregate_output"]):
                factory_kwargs["block_aggregate"] = bound.arguments["aggregate_output"]
            statements: list[Any] = []
            scope = inst.target.scope
            loc = inst.loc
            value = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem="scan_value",
                value=bound.arguments["value"],
            )
            if is_common_root and self._array_operand_state("scan", value):
                is_thread_data = self._is_array_value(
                    value,
                    thread_data_only=True,
                )
                if is_thread_data is None:
                    raise GroupRewriteError(
                        f"cuda.coop.{operation} could not resolve value "
                        "payload provenance"
                    )
                if not is_thread_data:
                    raise TypeError(
                        f"cuda.coop.{operation} accepts only a scalar or "
                        "fixed-size ThreadData value payload in common V1; "
                        "use cuda.coop.numba_mlir for backend-qualified "
                        "local-array payload support"
                    )
            input_payload, is_array = self._boxed_group_operand(
                statements,
                operation="scan",
                value=value,
                scope=scope,
                loc=loc,
            )
            result_payload = self._typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem="scan_result",
                prototype=value,
                is_array=is_array,
                dtype_policy=_PAYLOAD_DTYPE_LIKE,
            )
            call_statements = self._rewritten_call(
                inst,
                factory=factory,
                args=[input_payload, result_payload],
                kwargs=factory_kwargs,
                return_alias=result_payload,
                common_profile_operation=(operation if is_common_root else None),
            )
            call_statements.pop()
            statements.extend(call_statements)
            result = self._result_value(
                statements,
                payload=result_payload,
                is_array=is_array,
                scope=scope,
                loc=loc,
                stem="scan_result",
            )
            statements.append(ir.Assign(result, inst.target, loc))
            return statements

        if group.kind not in {"warp", "threads_within_warp"}:
            raise NotImplementedError(
                "cuda.coop.numba_mlir.scan currently lowers only block, "
                "physical-warp, and logical-warp groups"
            )
        if not self._is_none(bound.arguments["algorithm"]):
            raise NotImplementedError(
                "cuda.coop.numba_mlir.scan algorithm applies only to block groups"
            )
        if not self._is_none(bound.arguments["temp_storage"]):
            raise NotImplementedError(
                "cuda.coop.numba_mlir.scan temp_storage applies only to block groups"
            )
        if self._array_operand_state("scan", bound.arguments["value"]):
            raise NotImplementedError(
                "cuda.coop.numba_mlir.scan warp groups support one scalar value "
                "per lane"
            )

        scan_op = bound.arguments["scan_op"]
        default_sum = self._is_none(scan_op)
        if not default_sum:
            default_sum = ScanOp(self._constant(scan_op)).is_sum
        has_initial = not self._is_none(bound.arguments["initial_value"])
        has_valid_items = not self._is_none(bound.arguments["valid_items"])
        if has_valid_items:
            is_static, valid_items = self._try_constant(bound.arguments["valid_items"])
            if is_static:
                if isinstance(valid_items, bool) or not isinstance(
                    valid_items, Integral
                ):
                    raise TypeError(
                        "cuda.coop.numba_mlir.scan valid_items must be an "
                        "integer, not bool"
                    )
                group_size = group.static_size
                assert group_size is not None
                if not 1 <= int(valid_items) <= group_size:
                    raise ValueError(
                        "cuda.coop.numba_mlir.scan static valid_items must be "
                        f"between 1 and group size {group_size}"
                    )
        factory_operation = (
            f"{mode}_sum"
            if default_sum and not has_initial and not has_valid_items
            else f"{mode}_scan"
        )
        factory, factory_kwargs = self._scope_factory(group, factory_operation)
        if factory_operation.endswith("_scan"):
            factory_kwargs["scan_op"] = "+" if default_sum else scan_op
        if has_initial:
            factory_kwargs["initial_value"] = bound.arguments["initial_value"]
        if has_valid_items:
            factory_kwargs["valid_items"] = bound.arguments["valid_items"]
        if not self._is_none(bound.arguments["aggregate_output"]):
            factory_kwargs["warp_aggregate"] = bound.arguments["aggregate_output"]
        return self._rewritten_call(
            inst,
            factory=factory,
            args=[bound.arguments["value"]],
            kwargs=factory_kwargs,
            common_profile_operation=(operation if is_common_root else None),
        )

    def _lower_root_operation(
        self,
        inst: ir.Assign,
        call: ir.Expr,
        function: Any,
        operation: str,
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
                f"cuda.coop.numba_mlir.{operation} requires a compile-time "
                "ThreadGroup from this_*()"
            )
        is_common_root = _is_common_root_operation(function, operation)
        if is_common_root:
            _common_root_api._validate_common_v1_operation_group(operation, group)
        group = self._resolve_group(group, feature=operation)
        if operation == "sum":
            bound.arguments["binary_op"] = None
        elif operation in {
            "exclusive_sum",
            "inclusive_sum",
            "exclusive_scan",
            "inclusive_scan",
        }:
            bound.arguments["mode"] = (
                "exclusive" if operation.startswith("exclusive") else "inclusive"
            )
            if operation.endswith("_sum"):
                bound.arguments["scan_op"] = None
                bound.arguments["initial_value"] = None
            elif operation == "exclusive_scan":
                bound.arguments.setdefault("scan_op", None)
                bound.arguments.setdefault("initial_value", None)
            else:
                bound.arguments.setdefault("scan_op", None)
                bound.arguments["initial_value"] = None
        bound.arguments.setdefault("difference_op", None)
        bound.arguments.setdefault("flag_op", None)
        bound.arguments.setdefault("block_prefix", None)
        bound.arguments.setdefault("block_suffix", None)
        bound.arguments.setdefault("compare_op", None)
        bound.arguments.setdefault("valid_items", None)
        bound.arguments.setdefault("aggregate_output", None)
        bound.arguments.setdefault("ranks", None)
        bound.arguments.setdefault("valid_flags", None)
        bound.arguments.setdefault("warp_time_slicing", False)
        bound.arguments.setdefault("blocked_to_striped", False)
        bound.arguments.setdefault("exclusive_digit_prefix", None)
        bound.arguments.setdefault("relative_offsets", None)
        bound.arguments.setdefault("total_decoded_size", None)
        bound.arguments.setdefault("decoded_offset_dtype", None)

        if is_common_root:
            self._validate_common_arguments(operation, bound)

        normalized_scan_op = None
        if operation in {
            "scan",
            "exclusive_scan",
            "inclusive_scan",
        }:
            scan_op = bound.arguments.get("scan_op")
            if not self._is_none(scan_op):
                normalized_scan_op = ScanOp(self._constant(scan_op))
                mode = self._constant(bound.arguments["mode"])
                if (
                    mode == "exclusive"
                    and not normalized_scan_op.is_sum
                    and self._is_none(bound.arguments["initial_value"])
                ):
                    raise ValueError(
                        "cuda.coop.numba_mlir.scan requires initial_value for "
                        "non-default exclusive scans"
                    )

        if (
            is_common_root
            and normalized_scan_op is not None
            and normalized_scan_op.is_callable
        ):
            raise ValueError(
                "cuda.coop scan operations accept built-in operators only; "
                "use cuda.coop.numba_mlir for a custom scan callback"
            )
        if operation in {"load", "store"}:
            replacement = self._lower_load_store(
                inst,
                operation=operation,
                group=group,
                bound=bound,
                is_common_root=is_common_root,
            )
        elif operation in {"reduce", "sum"}:
            replacement = self._lower_reduce(
                inst,
                operation=operation,
                group=group,
                bound=bound,
                is_common_root=is_common_root,
            )
        elif operation in {
            "scan",
            "exclusive_sum",
            "inclusive_sum",
            "exclusive_scan",
            "inclusive_scan",
        }:
            replacement = self._lower_scan(
                inst,
                operation=operation,
                group=group,
                bound=bound,
                is_common_root=is_common_root,
            )
        else:
            raise AssertionError(f"unhandled cuda.coop operation {operation!r}")
        self.dead_func_names.add(call.func.name)
        self.replacements[inst] = replacement

    def _group_method(self, call: ir.Expr) -> tuple[str, ThreadGroup] | None:
        definition = self._definition(call.func)
        if (
            not isinstance(definition, ir.Expr)
            or definition.op != "getattr"
            or definition.attr not in _GROUP_METHODS
            or definition.attr == "group_by"
        ):
            return None
        group = self._group(definition.value)
        if group is None:
            return None
        return definition.attr, group

    def _lower_group_method(
        self,
        inst: ir.Assign,
        call: ir.Expr,
        *,
        method: str,
        group: ThreadGroup,
    ) -> None:
        if call.vararg is not None or call.varkwarg is not None:
            raise GroupRewriteError(f"ThreadGroup.{method} does not support splats")
        kwargs = dict(call.kws)
        dtype = None
        level = "thread"
        if method in {"rank", "count"}:
            if len(call.args) > 1 or any(name != "level" for name in kwargs):
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
                name not in {"dtype", "level"} for name in kwargs
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
                level,
                scope="cuda.coop.numba_mlir",
                feature=f"ThreadGroup.{operation}",
            )
            group = self._resolve_group(
                group,
                feature=f"ThreadGroup.{operation}",
                through_level=level,
            )
        else:
            group = self._resolve_group(
                group,
                feature=f"ThreadGroup.{operation}",
            )
        if group.kind == "grid" and operation in {"sync", "sync_aligned"}:
            if group.source == "common_root":
                raise NotImplementedError(
                    f"cuda.coop.ThreadGroup.{operation} does not support grid "
                    "groups in common V1; use a backend-qualified import for "
                    "backend-specific grid support"
                )
            raise NotImplementedError(
                "cuda.coop.numba_mlir grid synchronization requires a "
                "verified cooperative launch, which the current launch "
                "descriptor cannot request"
            )

        from ._group_provider import make_group_method_invocable

        invocable = make_group_method_invocable(
            group=group,
            operation=operation,
            dtype=dtype,
            level=level,
        )
        self.dead_func_names.add(call.func.name)
        self.replacements[inst] = self._rewritten_call(
            inst,
            factory=invocable,
            args=[],
            kwargs={},
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
                    and definition.attr == "group_by"
                    and self._group(definition.value) is not None
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
                        and inst.target.name not in descriptor_names
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
                        and value.value.name in used_names
                        and inst.target.name in self.dead_func_names
                    ):
                        continue
                names = ", ".join(sorted(used_names))
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir ThreadGroup/ThreadHierarchy values are "
                    "compile-time descriptors and may only feed this_*(), "
                    "group_by(), group methods, or group-first primitives; "
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
                function = self._callable(call.func)
                operation = _ROOT_OPERATIONS.get(function)
                if operation is not None:
                    self._lower_root_operation(inst, call, function, operation)
                    continue
                method = self._group_method(call)
                if method is not None:
                    method_name, group = method
                    self._lower_group_method(
                        inst,
                        call,
                        method=method_name,
                        group=group,
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
                or function in _ROOT_OPERATIONS
            ):
                return True
            function_definition = analyzer._definition(value.func)
            if (
                isinstance(function_definition, ir.Expr)
                and function_definition.op == "getattr"
                and function_definition.attr in _GROUP_METHODS
                and analyzer._group(function_definition.value) is not None
            ):
                return True
    return False


try:
    from numba_cuda_mlir.extending import (
        WholeFunctionPlanner,
        register_planner,
        require_launch_config,
    )
except ImportError:
    WholeFunctionPlanner = None
    CoopGroupHierarchyPlanner = None
else:

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
