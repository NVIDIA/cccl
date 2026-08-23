# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Whole-function lowering for Numba-CUDA-MLIR group-first primitives."""

from __future__ import annotations

import inspect
import operator
from itertools import count
from numbers import Integral
from typing import Any

import numpy as np
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
from cuda.coop._core import root_api as _common_root_api

from . import _group_ops
from . import _thread_group as _thread_groups
from ._scan_op import ScanOp

_NAME_COUNTER = count()
_PAYLOAD_DTYPE_INT32 = "int32"
_PAYLOAD_DTYPE_LIKE = "like"
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
_ROOT_OPERATIONS = {getattr(_group_ops, name): name for name in _QUALIFIED_OPERATIONS}
_ROOT_OPERATIONS.update(
    {
        getattr(_common_root_api, name): name
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

    operation = getattr(function, "__cuda_coop_backend_member__", None)
    if operation in _QUALIFIED_OPERATIONS:
        return operation
    if getattr(function, "__module__", None) == _group_ops.__name__:
        name = getattr(function, "__name__", None)
        if name in _QUALIFIED_OPERATIONS:
            return name
    return None


def _is_common_root_operation(function: Any, operation: str) -> bool:
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

    def _validate_common_arguments(
        self, operation: str, bound: inspect.BoundArguments
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

            try:
                _normalize_reduce_operation(
                    self._constant(bound.arguments["binary_op"])
                )
            except NotImplementedError as exc:
                raise ValueError(
                    "cuda.coop.reduce binary_op accepts built-in operators only; "
                    "use cuda.coop.numba_mlir for backend-specific behavior"
                ) from exc
        if operation in {"scan", "exclusive_scan", "inclusive_scan"}:
            scan_op = bound.arguments.get("scan_op")
            if not self._is_none(scan_op):
                normalized_scan_op = ScanOp(self._constant(scan_op))
                if normalized_scan_op.is_callable:
                    raise ValueError(
                        f"cuda.coop.{operation} scan_op accepts built-in "
                        "operators only; use cuda.coop.numba_mlir for "
                        "backend-specific callbacks"
                    )

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
            if getattr(function, "__module__", None) == _common_root_api.__name__:
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
        operation = _group_operation_name(function)
        from . import ThreadData

        if function in {ThreadData, _common_root_api.ThreadData}:
            return True
        if function is _typed_group_payload_like:
            return self._is_array_value(
                definition.args[0], seen=seen, thread_data_only=thread_data_only
            )
        if getattr(function, "__name__", "") == "array" and getattr(
            function, "__module__", ""
        ) in {"cuda.local", "numba_cuda_mlir.cuda.local"}:
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
            "histogram": "samples",
            "load": "output",
            "radix_rank": "keys",
            "radix_sort_keys": "keys",
            "scan": "value",
            "exclusive_sum": "value",
            "inclusive_sum": "value",
            "exclusive_scan": "value",
            "inclusive_scan": "value",
            "merge_sort_keys": "keys",
            "run_length_decode": "run_values",
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
                return None
            if index < 0:
                index += 2
            if not 0 <= index < 2:
                return None
            return self._array_extent(bound.arguments["value"], seen=set(seen))

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
        from . import ThreadData

        if function in {ThreadData, _common_root_api.ThreadData}:
            bound = self._bind(function, definition)
            extent_argument = bound.arguments["items_per_thread"]
            try:
                extent = self._constant(extent_argument)
            except GroupRewriteError:
                return None
            if isinstance(extent, Integral) and (not isinstance(extent, bool)):
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
            if isinstance(extent, Integral) and (not isinstance(extent, bool)):
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
            "load": "output",
            "radix_rank": "keys",
            "radix_sort_keys": "keys",
            "radix_sort_pairs": "keys",
            "scan": "value",
            "exclusive_sum": "value",
            "inclusive_sum": "value",
            "exclusive_scan": "value",
            "inclusive_scan": "value",
            "merge_sort_keys": "keys",
            "merge_sort_pairs": "keys",
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

    def _thread_data_payload(
        self,
        statements: list[Any],
        *,
        scope: Any,
        loc: ir.Loc,
        stem: str,
        items_per_thread: Any,
        dtype: Any,
    ) -> ir.Var:
        """Emit an explicitly typed ThreadData marker for a fresh result."""

        from . import ThreadData

        result = self._new_var(scope, loc, f"{stem}_payload")
        proxy = ir.Assign(ir.Const(None, loc), result, loc)
        statements.extend(
            self._rewritten_call(
                proxy,
                factory=ThreadData,
                args=[items_per_thread, dtype],
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
        """Emit one planner-private provider call."""

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
        """Emit a statically sized compiler-owned shared array."""

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

    def _scope_factory(
        self, group: ThreadGroup, operation: str
    ) -> tuple[Any, dict[str, Any]]:
        assert group.hierarchy is not None
        block_dim = group.hierarchy.block_dim
        assert block_dim is not None
        if group.kind == "block":
            from . import _block as block

            return (getattr(block, operation), {"threads_per_block": block_dim})
        if group.kind in {"warp", "threads_within_warp"}:
            from cuda.coop._core.group_dispatch import _cub_warp_width

            from . import _warp as warp

            name = {
                "exchange": "warp_exchange",
                "load": "warp_load",
                "merge_sort_keys": "warp_merge_sort_keys",
                "merge_sort_pairs": "warp_merge_sort_pairs",
                "store": "warp_store",
                "sum": "warp_sum",
                "exclusive_sum": "warp_exclusive_sum",
                "inclusive_sum": "warp_inclusive_sum",
                "exclusive_scan": "warp_exclusive_scan",
                "inclusive_scan": "warp_inclusive_scan",
            }[operation]
            threads_in_warp = _cub_warp_width(group)
            return (
                getattr(warp, name),
                {"threads_in_warp": threads_in_warp, "threads_per_block": block_dim},
            )
        raise NotImplementedError(
            f"cuda.coop.numba_mlir.{operation} currently lowers only block, physical-warp, and logical-warp groups through CUB"
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
                    operation, "output", bound.arguments["output"]
                ):
                    raise TypeError(
                        "cuda.coop.load requires output to be a fixed-size ThreadData payload in common V1; use cuda.coop.numba_mlir for backend-qualified local-array payload support"
                    )
            else:
                value = bound.arguments["value"]
                if self._array_operand_state(operation, value) and (
                    not self._thread_data_operand_state(operation, "value", value)
                ):
                    raise TypeError(
                        "cuda.coop.store accepts only a scalar or fixed-size ThreadData value payload in common V1; use cuda.coop.numba_mlir for backend-qualified local-array payload support"
                    )
        factory, factory_kwargs = self._scope_factory(group, operation)
        factory_kwargs["algorithm"] = bound.arguments["algorithm"]
        if is_common_root:
            factory_kwargs["_common_profile_operation"] = operation
        if not self._is_none(bound.arguments["valid_items"]):
            factory_kwargs["num_valid_items"] = bound.arguments["valid_items"]
        if operation == "load" and (not self._is_none(bound.arguments["oob_default"])):
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
            if group.kind != "block":
                raise NotImplementedError(
                    "cuda.coop.numba_mlir Load/Store TempStorage is supported only for block groups"
                )
            factory_kwargs["temp_storage"] = bound.arguments["temp_storage"]
        if operation == "load":
            runtime_args = [bound.arguments["source"], bound.arguments["output"]]
            return_alias = bound.arguments["output"]
        else:
            runtime_args = [bound.arguments["destination"], bound.arguments["value"]]
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
        self._reject_extra_root_arguments("reduce", bound)
        broadcast = self._constant(bound.arguments["broadcast"])
        if not isinstance(broadcast, bool):
            raise TypeError(
                "cuda.coop.numba_mlir.reduce broadcast must be a compile-time bool"
            )

        value = bound.arguments["value"]
        if (
            is_common_root
            and self._array_operand_state(operation, value)
            and not (self._thread_data_operand_state(operation, "value", value))
        ):
            raise TypeError(
                f"cuda.coop.{operation} accepts only a scalar or fixed-size "
                "ThreadData value payload in common V1; use "
                "cuda.coop.numba_mlir for backend-qualified local arrays"
            )

        has_valid = not self._is_none(bound.arguments["valid_items"])
        has_algorithm = not self._is_none(bound.arguments["algorithm"])
        binary_op = self._constant(bound.arguments["binary_op"])
        custom_binary_op = None
        from ._group_provider import _normalize_reduce_operation

        try:
            normalized_op = _normalize_reduce_operation(binary_op)
        except NotImplementedError:
            if is_common_root or not callable(binary_op):
                raise
            normalized_op = None
            custom_binary_op = binary_op

        if has_valid or has_algorithm or custom_binary_op is not None:
            if group.kind not in {"block", "warp", "threads_within_warp"}:
                raise NotImplementedError(
                    "valid_items, custom callbacks, and explicit CUB algorithms "
                    "are supported only for block, physical-warp, and "
                    "logical-warp groups"
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
                            "cuda.coop.numba_mlir.reduce valid_items must be an "
                            "integer, not bool"
                        )
                    group_size = group.static_size
                    assert group_size is not None
                    if not 1 <= int(valid_items) <= group_size:
                        raise ValueError(
                            "cuda.coop.numba_mlir.reduce static valid_items must "
                            f"be between 1 and group size {group_size}"
                        )
            if (
                group.kind == "block"
                and self._array_operand_state(operation, value)
                and not has_algorithm
            ):
                raise ValueError(
                    "cuda.coop.numba_mlir.reduce ThreadData BlockReduce "
                    "requires an explicit algorithm"
                )
            if normalized_op == "sum":
                factory, factory_kwargs = self._scope_factory(group, "sum")
            elif custom_binary_op is not None and group.kind == "block":
                from ._block._block_reduce import reduce

                assert group.hierarchy is not None
                factory = reduce
                factory_kwargs = {
                    "threads_per_block": group.hierarchy.block_dim,
                    "binary_op": custom_binary_op,
                }
            elif custom_binary_op is not None:
                from ._warp._warp_reduce import warp_reduce

                assert group.hierarchy is not None
                threads_in_warp = group.static_size
                assert threads_in_warp is not None
                factory = warp_reduce
                factory_kwargs = {
                    "threads_in_warp": threads_in_warp,
                    "threads_per_block": group.hierarchy.block_dim,
                    "binary_op": custom_binary_op,
                }
            elif group.kind == "block":
                from ._block._block_reduce import block_reduce_builtin

                assert group.hierarchy is not None
                factory = block_reduce_builtin
                factory_kwargs = {
                    "threads_per_block": group.hierarchy.block_dim,
                    "binary_op": normalized_op,
                }
            else:
                from ._warp._warp_reduce import warp_reduce_builtin

                assert group.hierarchy is not None
                threads_in_warp = group.static_size
                assert threads_in_warp is not None
                factory = warp_reduce_builtin
                factory_kwargs = {
                    "threads_in_warp": threads_in_warp,
                    "threads_per_block": group.hierarchy.block_dim,
                    "binary_op": normalized_op,
                }
            if has_valid:
                parameter = "num_valid" if group.kind == "block" else "valid_items"
                factory_kwargs[parameter] = bound.arguments["valid_items"]
            if has_algorithm:
                if group.kind != "block":
                    raise NotImplementedError(
                        "CUB algorithm selection applies to BlockReduce, not WarpReduce"
                    )
                factory_kwargs["algorithm"] = bound.arguments["algorithm"]
            return self._rewritten_call(
                inst,
                factory=factory,
                args=[value],
                kwargs=factory_kwargs,
                common_profile_operation=(operation if is_common_root else None),
            )

        if group.kind == "grid":
            raise NotImplementedError(
                "cuda.coop.numba_mlir.reduce grid groups require a hidden "
                "per-launch provider workspace"
            )
        from ._group_provider import group_reduce

        return self._rewritten_call(
            inst,
            factory=group_reduce,
            args=[value],
            kwargs={
                "group": group,
                "binary_op": bound.arguments["binary_op"],
                "broadcast": broadcast,
            },
            common_profile_operation=(operation if is_common_root else None),
        )

    @staticmethod
    def _reject_extra_root_arguments(
        operation: str, bound: inspect.BoundArguments
    ) -> None:
        if bound.arguments.get("args"):
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} accepts no extra positional arguments"
            )
        if bound.arguments.get("kwargs"):
            names = ", ".join(sorted(bound.arguments["kwargs"]))
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} got unexpected keyword(s): {names}"
            )

    def _validate_scan_aggregate(
        self,
        bound: inspect.BoundArguments,
    ) -> None:
        aggregate = bound.arguments["aggregate_output"]
        if self._is_none(aggregate):
            return
        if not self._array_operand_state("scan", aggregate):
            raise TypeError(
                "cuda.coop.numba_mlir.scan aggregate_output must be a "
                "single-item ThreadData or local array"
            )
        extent = self._array_extent(aggregate)
        if extent != 1:
            raise ValueError(
                "cuda.coop.numba_mlir.scan aggregate_output must contain "
                "exactly one item"
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
        self._validate_scan_aggregate(bound)

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
                if not self._thread_data_operand_state(operation, "value", value):
                    raise TypeError(
                        f"cuda.coop.{operation} accepts only a scalar or fixed-size "
                        "ThreadData value payload in common V1; use "
                        "cuda.coop.numba_mlir for backend-qualified local arrays"
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

    def _lower_exchange(
        self,
        inst: ir.Assign,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        self._reject_extra_root_arguments("exchange", bound)
        if group.kind not in {"block", "warp", "threads_within_warp"}:
            raise NotImplementedError(
                "cuda.coop.numba_mlir.exchange currently lowers only block, physical-warp, and logical-warp groups"
            )
        mode = self._constant(bound.arguments["mode"])
        if hasattr(mode, "value"):
            mode = mode.value
        if not isinstance(mode, str):
            raise TypeError(
                "cuda.coop.numba_mlir.exchange mode must be a compile-time string"
            )
        time_slicing = self._constant(bound.arguments["warp_time_slicing"])
        if not isinstance(time_slicing, bool):
            raise TypeError(
                "cuda.coop.numba_mlir.exchange warp_time_slicing must be a compile-time bool"
            )
        if group.kind == "block":
            from ._block import BlockExchangeType

            exchange_types = {
                "striped_to_blocked": BlockExchangeType.StripedToBlocked,
                "blocked_to_striped": BlockExchangeType.BlockedToStriped,
                "warp_striped_to_blocked": BlockExchangeType.WarpStripedToBlocked,
                "blocked_to_warp_striped": BlockExchangeType.BlockedToWarpStriped,
                "scatter_to_blocked": BlockExchangeType.ScatterToBlocked,
                "scatter_to_striped": BlockExchangeType.ScatterToStriped,
                "scatter_to_striped_guarded": BlockExchangeType.ScatterToStripedGuarded,
                "scatter_to_striped_flagged": BlockExchangeType.ScatterToStripedFlagged,
            }
            exchange_type_name = "block_exchange_type"
        else:
            from ._warp import WarpExchangeType

            exchange_types = {
                "striped_to_blocked": WarpExchangeType.StripedToBlocked,
                "blocked_to_striped": WarpExchangeType.BlockedToStriped,
                "scatter_to_striped": WarpExchangeType.ScatterToStriped,
            }
            exchange_type_name = "warp_exchange_type"
            if time_slicing:
                raise ValueError(
                    "cuda.coop.numba_mlir.exchange warp_time_slicing applies only to block groups"
                )
        try:
            exchange_type = exchange_types[mode]
        except KeyError as exc:
            choices = ", ".join(exchange_types)
            raise ValueError(
                f"cuda.coop.numba_mlir.exchange mode must be one of: {choices}"
            ) from exc
        uses_ranks = mode.startswith("scatter_to_")
        uses_valid_flags = mode == "scatter_to_striped_flagged"
        has_ranks = not self._is_none(bound.arguments["ranks"])
        has_valid_flags = not self._is_none(bound.arguments["valid_flags"])
        if uses_ranks != has_ranks:
            requirement = "requires" if uses_ranks else "does not accept"
            raise ValueError(
                f"cuda.coop.numba_mlir.exchange {mode} {requirement} ranks"
            )
        if uses_valid_flags != has_valid_flags:
            requirement = "requires" if uses_valid_flags else "does not accept"
            raise ValueError(
                f"cuda.coop.numba_mlir.exchange {mode} {requirement} valid_flags"
            )
        factory, factory_kwargs = self._scope_factory(group, "exchange")
        factory_kwargs[exchange_type_name] = exchange_type
        if time_slicing:
            factory_kwargs["warp_time_slicing"] = True
        if is_common_root:
            factory_kwargs["_common_profile_operation"] = "exchange"
        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        value = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="exchange_value",
            value=bound.arguments["value"],
        )
        if not self._array_operand_state("exchange", value):
            raise TypeError(
                "cuda.coop.numba_mlir.exchange requires a fixed-size ThreadData or local-array payload"
            )
        if is_common_root and (
            not self._thread_data_operand_state("exchange", "value", value)
        ):
            raise TypeError(
                "cuda.coop.exchange requires a fixed-size ThreadData payload in common V1; use cuda.coop.numba_mlir for backend-qualified local-array payload support"
            )
        result_payload = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem="exchange_result",
            prototype=value,
            is_array=True,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
        )
        runtime_args = [value, result_payload]
        if uses_ranks:
            ranks = bound.arguments["ranks"]
            if not self._array_operand_state("exchange", ranks):
                raise TypeError(
                    "cuda.coop.numba_mlir.exchange ranks must be a fixed-size ThreadData or local-array payload"
                )
            runtime_args.append(ranks)
        if uses_valid_flags:
            valid_flags = bound.arguments["valid_flags"]
            if not self._array_operand_state("exchange", valid_flags):
                raise TypeError(
                    "cuda.coop.numba_mlir.exchange valid_flags must be a fixed-size ThreadData or local-array payload"
                )
            runtime_args.append(valid_flags)
        call_statements = self._rewritten_call(
            inst,
            factory=factory,
            args=runtime_args,
            kwargs=factory_kwargs,
            return_alias=result_payload,
        )
        call_statements.pop()
        statements.extend(call_statements)
        statements.append(ir.Assign(result_payload, inst.target, loc))
        return statements

    def _lower_adjacent_difference(
        self,
        inst: ir.Assign,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        operation = "adjacent_difference"
        self._reject_extra_root_arguments(operation, bound)
        if group.kind != "block":
            raise NotImplementedError(
                "cuda.coop.numba_mlir.adjacent_difference currently lowers "
                "only complete physical block groups"
            )

        difference_argument = bound.arguments.get("difference_op")
        if self._is_none(difference_argument):
            difference_op = _builtin_subtract
        else:
            difference_op = self._constant(difference_argument)
            if not callable(difference_op):
                raise TypeError(
                    "cuda.coop.numba_mlir.adjacent_difference difference_op "
                    "must be a device callable"
                )
            if is_common_root:
                raise ValueError(
                    "cuda.coop.adjacent_difference uses built-in subtraction "
                    "in the common profile"
                )

        from cuda.coop._core.block import BlockAdjacentDifferenceDirection

        from ._block import BlockAdjacentDifferenceType

        try:
            direction = BlockAdjacentDifferenceDirection(
                self._constant(bound.arguments["direction"])
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "cuda.coop.numba_mlir.adjacent_difference direction must be "
                "'left' or 'right'"
            ) from exc
        adjacent_type = {
            BlockAdjacentDifferenceDirection.LEFT: (
                BlockAdjacentDifferenceType.SubtractLeft
            ),
            BlockAdjacentDifferenceDirection.RIGHT: (
                BlockAdjacentDifferenceType.SubtractRight
            ),
        }[direction]

        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        value = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_value",
            value=bound.arguments["value"],
        )
        if is_common_root and not self._thread_data_operand_state(
            operation,
            "value",
            value,
        ):
            raise TypeError(
                "cuda.coop.adjacent_difference requires a fixed-size "
                "ThreadData payload in the common profile; use "
                "cuda.coop.numba_mlir for qualified scalar or local-array "
                "support"
            )
        input_payload, is_array = self._boxed_group_operand(
            statements,
            operation=operation,
            value=value,
            scope=scope,
            loc=loc,
        )
        items_per_thread = self._array_extent(value) if is_array else 1
        if items_per_thread is None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir.adjacent_difference could not infer a "
                "static items_per_thread extent"
            )
        result_payload = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_result",
            prototype=value,
            is_array=is_array,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
        )

        factory, factory_kwargs = self._scope_factory(group, operation)
        factory_kwargs.update(
            {
                "block_adjacent_difference_type": adjacent_type,
                "difference_op": difference_op,
            }
        )
        runtime_args = [input_payload, result_payload]
        valid_items = bound.arguments["valid_items"]
        predecessor = bound.arguments["tile_predecessor_item"]
        successor = bound.arguments["tile_successor_item"]
        if not self._is_none(valid_items):
            is_static, static_valid_items = self._try_constant(valid_items)
            if is_static:
                if isinstance(static_valid_items, bool) or not isinstance(
                    static_valid_items, Integral
                ):
                    raise TypeError(
                        "cuda.coop.numba_mlir.adjacent_difference valid_items "
                        "must be an integer, not bool"
                    )
                assert group.static_size is not None
                tile_size = group.static_size * items_per_thread
                if not 0 <= int(static_valid_items) <= tile_size:
                    raise ValueError(
                        "cuda.coop.numba_mlir.adjacent_difference static "
                        f"valid_items must be between 0 and tile size {tile_size}"
                    )
            runtime_args.append(valid_items)
            factory_kwargs["valid_items"] = True
        if not self._is_none(predecessor):
            runtime_args.append(predecessor)
            factory_kwargs["tile_predecessor_item"] = True
        if not self._is_none(successor):
            runtime_args.append(successor)
            factory_kwargs["tile_successor_item"] = True
        if not self._is_none(bound.arguments["temp_storage"]):
            factory_kwargs["temp_storage"] = bound.arguments["temp_storage"]

        call_statements = self._rewritten_call(
            inst,
            factory=factory,
            args=runtime_args,
            kwargs=factory_kwargs,
            return_alias=result_payload,
            common_profile_operation=operation if is_common_root else None,
        )
        call_statements.pop()
        statements.extend(call_statements)
        result = self._result_value(
            statements,
            payload=result_payload,
            is_array=is_array,
            scope=scope,
            loc=loc,
            stem=f"{operation}_result",
        )
        statements.append(ir.Assign(result, inst.target, loc))
        return statements

    def _lower_discontinuity(
        self,
        inst: ir.Assign,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        operation = "discontinuity"
        self._reject_extra_root_arguments(operation, bound)
        if group.kind != "block":
            raise NotImplementedError(
                "cuda.coop.numba_mlir.discontinuity currently lowers only "
                "complete physical block groups"
            )

        flag_argument = bound.arguments.get("flag_op")
        if self._is_none(flag_argument):
            flag_op = _builtin_not_equal
        else:
            flag_op = self._constant(flag_argument)
            if not callable(flag_op):
                raise TypeError(
                    "cuda.coop.numba_mlir.discontinuity flag_op must be a "
                    "device callable"
                )
            if is_common_root:
                raise ValueError(
                    "cuda.coop.discontinuity uses built-in inequality in the "
                    "common profile"
                )

        from cuda.coop._core.block import BlockDiscontinuityMode

        from ._block import BlockDiscontinuityType

        try:
            mode = BlockDiscontinuityMode(self._constant(bound.arguments["mode"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "cuda.coop.numba_mlir.discontinuity mode must be "
                "'heads', 'tails', or 'heads_and_tails'"
            ) from exc
        discontinuity_type = {
            BlockDiscontinuityMode.HEADS: BlockDiscontinuityType.HEADS,
            BlockDiscontinuityMode.TAILS: BlockDiscontinuityType.TAILS,
            BlockDiscontinuityMode.HEADS_AND_TAILS: (
                BlockDiscontinuityType.HEADS_AND_TAILS
            ),
        }[mode]

        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        value = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_value",
            value=bound.arguments["value"],
        )
        if is_common_root and not self._thread_data_operand_state(
            operation,
            "value",
            value,
        ):
            raise TypeError(
                "cuda.coop.discontinuity requires a fixed-size ThreadData "
                "payload in the common profile; use cuda.coop.numba_mlir for "
                "qualified scalar or local-array support"
            )
        input_payload, is_array = self._boxed_group_operand(
            statements,
            operation=operation,
            value=value,
            scope=scope,
            loc=loc,
        )
        head_payload = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_head_result",
            prototype=value,
            is_array=is_array,
            dtype_policy=_PAYLOAD_DTYPE_INT32,
        )
        tail_payload = None
        if mode is BlockDiscontinuityMode.HEADS_AND_TAILS:
            tail_payload = self._typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_tail_result",
                prototype=value,
                is_array=is_array,
                dtype_policy=_PAYLOAD_DTYPE_INT32,
            )

        factory, factory_kwargs = self._scope_factory(group, operation)
        factory_kwargs.update(
            {
                "block_discontinuity_type": discontinuity_type,
                "flag_op": flag_op,
                "flag_dtype": types.int32,
            }
        )
        runtime_args = [input_payload, head_payload]
        return_payload: ir.Var | tuple[ir.Var, ...] = head_payload
        if tail_payload is not None:
            runtime_args.append(tail_payload)
            return_payload = (head_payload, tail_payload)
        predecessor = bound.arguments["tile_predecessor_item"]
        successor = bound.arguments["tile_successor_item"]
        if not self._is_none(predecessor):
            runtime_args.append(predecessor)
            factory_kwargs["tile_predecessor_item"] = True
        if not self._is_none(successor):
            runtime_args.append(successor)
            factory_kwargs["tile_successor_item"] = True
        if not self._is_none(bound.arguments["temp_storage"]):
            factory_kwargs["temp_storage"] = bound.arguments["temp_storage"]

        call_statements = self._rewritten_call(
            inst,
            factory=factory,
            args=runtime_args,
            kwargs=factory_kwargs,
            return_alias=return_payload,
            common_profile_operation=operation if is_common_root else None,
        )
        call_statements.pop()
        statements.extend(call_statements)
        head_result = self._result_value(
            statements,
            payload=head_payload,
            is_array=is_array,
            scope=scope,
            loc=loc,
            stem=f"{operation}_head_result",
        )
        if tail_payload is None:
            statements.append(ir.Assign(head_result, inst.target, loc))
            return statements
        tail_result = self._result_value(
            statements,
            payload=tail_payload,
            is_array=is_array,
            scope=scope,
            loc=loc,
            stem=f"{operation}_tail_result",
        )
        statements.append(
            ir.Assign(
                ir.Expr.build_tuple([head_result, tail_result], loc),
                inst.target,
                loc,
            )
        )
        return statements

    def _lower_shuffle(
        self,
        inst: ir.Assign,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        self._reject_extra_root_arguments("shuffle", bound)
        if group.kind != "block":
            raise NotImplementedError(
                "cuda.coop.numba_mlir.shuffle currently lowers only complete physical block groups"
            )
        if not self._is_none(bound.arguments.get("block_prefix")) or not self._is_none(
            bound.arguments.get("block_suffix")
        ):
            raise NotImplementedError(
                "cuda.coop.numba_mlir.shuffle root projection currently supports the scalar-return ABI without boundary outputs"
            )
        from ._block import BlockShuffleType

        mode = self._constant(bound.arguments["mode"])
        if hasattr(mode, "value"):
            mode = mode.value
        try:
            shuffle_type = {
                "offset": BlockShuffleType.Offset,
                "rotate": BlockShuffleType.Rotate,
                "up": BlockShuffleType.Up,
                "down": BlockShuffleType.Down,
            }[mode]
        except KeyError as exc:
            raise ValueError(
                "cuda.coop.numba_mlir.shuffle mode must be offset, rotate, up, or down"
            ) from exc
        factory, factory_kwargs = self._scope_factory(group, "shuffle")
        factory_kwargs["block_shuffle_type"] = shuffle_type
        distance = bound.arguments["distance"]
        normalized_distance = self._constant(distance)
        is_default_up_down_distance = (
            shuffle_type in {BlockShuffleType.Up, BlockShuffleType.Down}
            and normalized_distance == 1
        )
        if not is_default_up_down_distance:
            factory_kwargs["distance"] = distance
        value = bound.arguments["value"]
        array_state = self._is_array_value(value)
        if array_state is None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir.shuffle could not resolve cyclic array provenance to a concrete scalar or array value"
            )
        is_array_value = array_state
        if is_common_root:
            is_thread_data = self._is_array_value(value, thread_data_only=True)
            if is_thread_data is None:
                raise GroupRewriteError(
                    "cuda.coop.shuffle could not resolve value payload provenance"
                )
            if not is_thread_data:
                raise TypeError(
                    "cuda.coop.shuffle requires a fixed-size ThreadData payload in common V1; use cuda.coop.numba_mlir for backend-qualified scalar or local-array shuffles"
                )
            if shuffle_type not in {BlockShuffleType.Up, BlockShuffleType.Down}:
                raise ValueError(
                    "cuda.coop.shuffle mode must be 'down' or 'up' in common V1; use cuda.coop.numba_mlir for backend-qualified scalar offset/rotate shuffles"
                )
            if (
                isinstance(normalized_distance, bool)
                or not isinstance(normalized_distance, Integral)
                or int(normalized_distance) != 1
            ):
                raise ValueError(
                    "cuda.coop.shuffle distance must be exactly 1 in common V1; use cuda.coop.numba_mlir for backend-qualified scalar shuffles with other distances"
                )
        if is_array_value and shuffle_type not in {
            BlockShuffleType.Up,
            BlockShuffleType.Down,
        }:
            raise NotImplementedError(
                "cuda.coop.numba_mlir.shuffle array values currently support only 'up' and 'down' modes"
            )
        if is_array_value:
            statements: list[Any] = []
            scope = inst.target.scope
            loc = inst.loc
            result_payload = self._typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem="shuffle_result",
                prototype=value,
                is_array=True,
                dtype_policy=_PAYLOAD_DTYPE_LIKE,
            )
            call_statements = self._rewritten_call(
                inst,
                factory=factory,
                args=[value, result_payload],
                kwargs=factory_kwargs,
                return_alias=result_payload,
                common_profile_operation="shuffle" if is_common_root else None,
            )
            call_statements.pop()
            statements.extend(call_statements)
            statements.append(ir.Assign(result_payload, inst.target, loc))
            return statements
        return self._rewritten_call(
            inst,
            factory=factory,
            args=[value],
            kwargs=factory_kwargs,
            common_profile_operation="shuffle" if is_common_root else None,
        )

    def _lower_merge_sort(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        self._reject_extra_root_arguments(operation, bound)
        if group.kind not in {"block", "warp", "threads_within_warp"}:
            raise NotImplementedError(
                f"cuda.coop.numba_mlir.{operation} currently lowers only "
                "physical block, physical-warp, and logical-warp groups"
            )

        descending = self._constant(bound.arguments["descending"])
        if not isinstance(descending, bool):
            raise TypeError(
                f"cuda.coop.numba_mlir.{operation} descending must be a "
                "compile-time bool"
            )
        compare_arg = bound.arguments.get("compare_op")
        if self._is_none(compare_arg):
            compare_op = _builtin_greater if descending else _builtin_less
        else:
            if descending:
                raise ValueError(
                    f"cuda.coop.numba_mlir.{operation} custom compare_op and "
                    "descending=True are mutually exclusive"
                )
            compare_op = compare_arg

        has_valid_items = not self._is_none(bound.arguments["valid_items"])
        has_oob_default = not self._is_none(bound.arguments["oob_default"])
        if has_valid_items != has_oob_default:
            raise ValueError(
                f"cuda.coop.numba_mlir.{operation} valid_items and "
                "oob_default must be provided together"
            )
        if has_valid_items:
            is_static, valid_items = self._try_constant(bound.arguments["valid_items"])
            if is_static:
                if isinstance(valid_items, (bool, np.bool_)) or not isinstance(
                    valid_items, Integral
                ):
                    raise TypeError(
                        f"cuda.coop.numba_mlir.{operation} valid_items must be "
                        "an integer, not bool"
                        if isinstance(valid_items, (bool, np.bool_))
                        else f"cuda.coop.numba_mlir.{operation} valid_items "
                        "must be an integer"
                    )
                items_per_thread = self._array_extent(bound.arguments["keys"])
                if items_per_thread is None and not self._array_operand_state(
                    operation, bound.arguments["keys"]
                ):
                    items_per_thread = 1
                group_size = group.static_size
                if items_per_thread is not None and group_size is not None:
                    maximum = group_size * items_per_thread
                    if not 0 <= int(valid_items) <= maximum:
                        raise ValueError(
                            f"cuda.coop.numba_mlir.{operation} static "
                            f"valid_items must be in [0, {maximum}]"
                        )

        factory, factory_kwargs = self._scope_factory(group, operation)
        factory_kwargs["compare_op"] = compare_op
        if has_valid_items:
            factory_kwargs["valid_items"] = bound.arguments["valid_items"]
            factory_kwargs["oob_default"] = bound.arguments["oob_default"]
        if not self._is_none(bound.arguments["temp_storage"]):
            if group.kind != "block":
                raise NotImplementedError(
                    "cuda.coop.numba_mlir Merge Sort TempStorage is supported "
                    "only for block groups"
                )
            factory_kwargs["temp_storage"] = bound.arguments["temp_storage"]

        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        keys = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_keys",
            value=bound.arguments["keys"],
        )
        if is_common_root and not self._thread_data_operand_state(
            operation, "keys", keys
        ):
            raise TypeError(
                f"cuda.coop.{operation} requires keys to be fixed-size "
                "ThreadData in common V1; use cuda.coop.numba_mlir for "
                "backend-qualified scalar or local-array payloads"
            )
        keys_payload, keys_are_array = self._boxed_group_operand(
            statements,
            operation=operation,
            value=keys,
            scope=scope,
            loc=loc,
        )
        result_keys = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_keys_result",
            prototype=keys,
            is_array=keys_are_array,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
        )
        self._copy_array_payload(
            statements,
            operation=operation,
            source=keys_payload,
            destination=result_keys,
            scope=scope,
            loc=loc,
            known_items_per_thread=1 if not keys_are_array else None,
        )

        runtime_args = [result_keys]
        result_values = None
        values_are_array = False
        if operation == "merge_sort_pairs":
            values = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_values",
                value=bound.arguments["values"],
            )
            if is_common_root and not self._thread_data_operand_state(
                operation, "values", values
            ):
                raise TypeError(
                    "cuda.coop.merge_sort_pairs requires values to be "
                    "fixed-size ThreadData in common V1; use "
                    "cuda.coop.numba_mlir for backend-qualified scalar or "
                    "local-array payloads"
                )
            values_payload, values_are_array = self._boxed_group_operand(
                statements,
                operation=operation,
                value=values,
                scope=scope,
                loc=loc,
            )
            if values_are_array != keys_are_array:
                raise TypeError(
                    f"cuda.coop.numba_mlir.{operation} keys and values must "
                    "have the same scalar or ThreadData shape"
                )
            if self._array_extent(values_payload) != self._array_extent(keys_payload):
                raise ValueError(
                    f"cuda.coop.numba_mlir.{operation} keys and values must "
                    "have the same items_per_thread"
                )
            result_values = self._typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_values_result",
                prototype=values,
                is_array=values_are_array,
                dtype_policy=_PAYLOAD_DTYPE_LIKE,
            )
            self._copy_array_payload(
                statements,
                operation=operation,
                source=values_payload,
                destination=result_values,
                scope=scope,
                loc=loc,
                known_items_per_thread=1 if not values_are_array else None,
            )
            runtime_args.append(result_values)

        call_statements = self._rewritten_call(
            inst,
            factory=factory,
            args=runtime_args,
            kwargs=factory_kwargs,
            return_alias=(
                result_keys if result_values is None else (result_keys, result_values)
            ),
            common_profile_operation=(operation if is_common_root else None),
        )
        call_statements.pop()
        statements.extend(call_statements)
        keys_result = self._result_value(
            statements,
            payload=result_keys,
            is_array=keys_are_array,
            scope=scope,
            loc=loc,
            stem=f"{operation}_keys_result",
        )
        if result_values is None:
            statements.append(ir.Assign(keys_result, inst.target, loc))
            return statements

        values_result = self._result_value(
            statements,
            payload=result_values,
            is_array=values_are_array,
            scope=scope,
            loc=loc,
            stem=f"{operation}_values_result",
        )
        statements.append(
            ir.Assign(
                ir.Expr.build_tuple([keys_result, values_result], loc),
                inst.target,
                loc,
            )
        )
        return statements

    def _lower_radix_rank(
        self,
        inst: ir.Assign,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        operation = "radix_rank"
        scope_name = "cuda.coop" if is_common_root else "cuda.coop.numba_mlir"
        self._reject_extra_root_arguments(operation, bound)
        if group.kind != "block":
            raise NotImplementedError(
                f"{scope_name}.radix_rank currently lowers only complete "
                "physical block groups"
            )
        if is_common_root and not self._thread_data_operand_state(
            operation, "keys", bound.arguments["keys"]
        ):
            raise TypeError(
                "cuda.coop.radix_rank requires keys to be coop.ThreadData "
                "in common V1; use cuda.coop.numba_mlir for "
                "backend-qualified payloads"
            )

        begin_bit = _static_index(
            scope_name,
            operation,
            "begin_bit",
            self._constant(bound.arguments["begin_bit"]),
        )
        end_bit = (
            None
            if self._is_none(bound.arguments["end_bit"])
            else _static_index(
                scope_name,
                operation,
                "end_bit",
                self._constant(bound.arguments["end_bit"]),
            )
        )
        radix_bits = (
            None
            if self._is_none(bound.arguments["radix_bits"])
            else _static_index(
                scope_name,
                operation,
                "radix_bits",
                self._constant(bound.arguments["radix_bits"]),
            )
        )
        if begin_bit < 0:
            raise ValueError(f"{scope_name}.radix_rank begin_bit must be non-negative")
        if radix_bits is not None and radix_bits <= 0:
            raise ValueError(f"{scope_name}.radix_rank radix_bits must be positive")
        if end_bit is None:
            end_bit = begin_bit + (4 if radix_bits is None else radix_bits)
        elif radix_bits is not None and end_bit != begin_bit + radix_bits:
            raise ValueError(
                f"{scope_name}.radix_rank radix_bits must match end_bit - begin_bit"
            )
        if end_bit <= begin_bit:
            raise ValueError(
                f"{scope_name}.radix_rank end_bit must be greater than begin_bit"
            )
        if end_bit - begin_bit > 8:
            raise ValueError(f"{scope_name}.radix_rank bit width must be <= 8")
        descending = _static_bool(
            scope_name,
            operation,
            "descending",
            self._constant(bound.arguments["descending"]),
        )

        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        keys = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_keys",
            value=bound.arguments["keys"],
        )
        keys_payload, is_array = self._boxed_group_operand(
            statements,
            operation=operation,
            value=keys,
            scope=scope,
            loc=loc,
        )
        ranks_payload = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_result",
            prototype=keys,
            is_array=is_array,
            dtype_policy=_PAYLOAD_DTYPE_INT32,
        )

        factory, factory_kwargs = self._scope_factory(group, operation)
        if is_common_root:
            from ._block._block_radix_rank import _common_radix_rank

            factory = _common_radix_rank
        factory_kwargs.update(
            {
                "begin_bit": begin_bit,
                "end_bit": end_bit,
                "descending": descending,
            }
        )
        prefix = bound.arguments.get("exclusive_digit_prefix")
        if prefix is not None and not self._is_none(prefix):
            if not self._array_operand_state(operation, prefix):
                raise TypeError(
                    "cuda.coop.numba_mlir.radix_rank "
                    "exclusive_digit_prefix must be an explicit array payload"
                )
            prefix_extent = self._array_extent(prefix)
            group_size = group.static_size
            if prefix_extent is not None and group_size is not None:
                expected = max(
                    1, ((1 << (end_bit - begin_bit)) + group_size - 1) // group_size
                )
                if prefix_extent != expected:
                    raise ValueError(
                        "cuda.coop.numba_mlir.radix_rank "
                        "exclusive_digit_prefix must contain "
                        f"{expected} items per thread"
                    )
            factory_kwargs["exclusive_digit_prefix"] = prefix

        call_statements = self._rewritten_call(
            inst,
            factory=factory,
            args=[keys_payload, ranks_payload],
            kwargs=factory_kwargs,
            return_alias=ranks_payload,
        )
        call_statements.pop()
        statements.extend(call_statements)
        result = self._result_value(
            statements,
            payload=ranks_payload,
            is_array=is_array,
            scope=scope,
            loc=loc,
            stem=f"{operation}_result",
        )
        statements.append(ir.Assign(result, inst.target, loc))
        return statements

    def _lower_radix_sort(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        self._reject_extra_root_arguments(operation, bound)
        scope_name = "cuda.coop" if is_common_root else "cuda.coop.numba_mlir"
        if group.kind != "block":
            raise NotImplementedError(
                f"{scope_name}.{operation} currently lowers only complete "
                "physical block groups"
            )
        descending = _static_bool(
            scope_name,
            operation,
            "descending",
            self._constant(bound.arguments["descending"]),
        )
        blocked_to_striped = _static_bool(
            scope_name,
            operation,
            "blocked_to_striped",
            self._constant(bound.arguments.get("blocked_to_striped", False)),
        )
        if is_common_root:
            parameters = (
                ("keys", "values") if operation.endswith("_pairs") else ("keys",)
            )
            for parameter in parameters:
                if not self._thread_data_operand_state(
                    operation, parameter, bound.arguments[parameter]
                ):
                    raise TypeError(
                        f"cuda.coop.{operation} requires {parameter} to be "
                        "coop.ThreadData in common V1; use cuda.coop.numba_mlir "
                        "for backend-qualified payloads"
                    )

        factory_operation = f"{operation}_descending" if descending else operation
        factory, factory_kwargs = self._scope_factory(group, factory_operation)
        if is_common_root:
            from ._block import _block_radix_sort

            factory = getattr(_block_radix_sort, f"_common_{operation}")
            factory_kwargs["descending"] = descending
        elif blocked_to_striped:
            factory_kwargs["blocked_to_striped"] = True
        if not self._is_none(bound.arguments["temp_storage"]):
            factory_kwargs["temp_storage"] = bound.arguments["temp_storage"]

        begin_bit = bound.arguments["begin_bit"]
        end_bit = bound.arguments["end_bit"]
        begin_is_static, static_begin = self._try_constant(begin_bit)
        end_is_none = self._is_none(end_bit)
        end_is_static, static_end = (
            (True, None) if end_is_none else self._try_constant(end_bit)
        )
        if begin_is_static:
            static_begin = _static_index(
                scope_name, operation, "begin_bit", static_begin
            )
            if static_begin < 0:
                raise ValueError(
                    f"{scope_name}.{operation} begin_bit must be non-negative"
                )
        if end_is_static and not end_is_none:
            static_end = _static_index(scope_name, operation, "end_bit", static_end)
            if static_end < 1:
                raise ValueError(f"{scope_name}.{operation} end_bit must be positive")
            if begin_is_static and static_end <= static_begin:
                raise ValueError(
                    f"{scope_name}.{operation} end_bit must be greater than begin_bit"
                )
        if not end_is_none:
            factory_kwargs["begin_bit"] = begin_bit
            factory_kwargs["end_bit"] = end_bit
        elif not (begin_is_static and static_begin == 0):
            factory_kwargs["begin_bit"] = begin_bit
            factory_kwargs["end_bit"] = None

        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        keys = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_keys",
            value=bound.arguments["keys"],
        )
        keys_payload, keys_are_array = self._boxed_group_operand(
            statements,
            operation=operation,
            value=keys,
            scope=scope,
            loc=loc,
        )
        result_keys = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_keys_result",
            prototype=keys,
            is_array=keys_are_array,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
        )
        self._copy_array_payload(
            statements,
            operation=operation,
            source=keys_payload,
            destination=result_keys,
            scope=scope,
            loc=loc,
            known_items_per_thread=1 if not keys_are_array else None,
        )

        runtime_args = [result_keys]
        result_values = None
        values_are_array = False
        if operation == "radix_sort_pairs":
            values = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_values",
                value=bound.arguments["values"],
            )
            values_payload, values_are_array = self._boxed_group_operand(
                statements,
                operation=operation,
                value=values,
                scope=scope,
                loc=loc,
            )
            if values_are_array != keys_are_array:
                raise TypeError(
                    f"{scope_name}.{operation} keys and values must have "
                    "the same scalar or ThreadData shape"
                )
            if self._array_extent(values_payload) != self._array_extent(keys_payload):
                raise ValueError(
                    f"{scope_name}.{operation} keys and values must have "
                    "the same items_per_thread"
                )
            result_values = self._typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_values_result",
                prototype=values,
                is_array=values_are_array,
                dtype_policy=_PAYLOAD_DTYPE_LIKE,
            )
            self._copy_array_payload(
                statements,
                operation=operation,
                source=values_payload,
                destination=result_values,
                scope=scope,
                loc=loc,
                known_items_per_thread=1 if not values_are_array else None,
            )
            runtime_args.append(result_values)

        call_statements = self._rewritten_call(
            inst,
            factory=factory,
            args=runtime_args,
            kwargs=factory_kwargs,
            return_alias=(
                result_keys if result_values is None else (result_keys, result_values)
            ),
        )
        call_statements.pop()
        statements.extend(call_statements)
        keys_result = self._result_value(
            statements,
            payload=result_keys,
            is_array=keys_are_array,
            scope=scope,
            loc=loc,
            stem=f"{operation}_keys_result",
        )
        if result_values is None:
            statements.append(ir.Assign(keys_result, inst.target, loc))
            return statements
        values_result = self._result_value(
            statements,
            payload=result_values,
            is_array=values_are_array,
            scope=scope,
            loc=loc,
            stem=f"{operation}_values_result",
        )
        statements.append(
            ir.Assign(
                ir.Expr.build_tuple([keys_result, values_result], loc),
                inst.target,
                loc,
            )
        )
        return statements

    def _lower_topk(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        self._reject_extra_root_arguments(operation, bound)
        scope_name = "cuda.coop" if is_common_root else "cuda.coop.numba_mlir"
        if group.kind != "block":
            raise NotImplementedError(
                f"{scope_name}.{operation} currently lowers only complete "
                "physical block groups"
            )
        assert group.hierarchy is not None
        block_dim = group.hierarchy.block_dim
        if block_dim is None or block_dim[1:] != (1, 1):
            raise ValueError(
                f"{scope_name}.{operation} requires a one-dimensional block"
            )
        if group.static_size is None:
            raise GroupRewriteError(
                f"{scope_name}.{operation} requires a static block size"
            )
        if group.static_size > 1024:
            raise ValueError(
                f"{scope_name}.{operation} block thread count must be <= 1024"
            )

        parameters = ("keys", "values") if operation.endswith("_pairs") else ("keys",)
        if is_common_root:
            for parameter in parameters:
                is_thread_data = self._is_array_value(
                    bound.arguments[parameter],
                    thread_data_only=True,
                )
                if is_thread_data is None:
                    raise GroupRewriteError(
                        f"{scope_name}.{operation} could not resolve {parameter} "
                        "payload provenance"
                    )
                if not is_thread_data:
                    raise TypeError(
                        f"{scope_name}.{operation} requires {parameter} to be "
                        "coop.ThreadData in common V1; use "
                        "cuda.coop.numba_mlir for backend-qualified payloads"
                    )

        items_per_thread = self._array_extent(bound.arguments["keys"])
        if items_per_thread is None:
            raise GroupRewriteError(
                f"{scope_name}.{operation} could not infer a static "
                "items_per_thread extent"
            )
        if items_per_thread <= 0:
            raise ValueError(
                f"{scope_name}.{operation} keys.items_per_thread must be positive"
            )
        if operation.endswith("_pairs"):
            values_extent = self._array_extent(bound.arguments["values"])
            if values_extent != items_per_thread:
                raise ValueError(
                    f"{scope_name}.{operation} keys and values must have the "
                    "same items_per_thread"
                )

        def static_int(name: str, value: Any) -> int | None:
            is_static, static_value = self._try_constant(value)
            if not is_static:
                return None
            if isinstance(static_value, (bool, np.bool_)):
                raise TypeError(
                    f"{scope_name}.{operation} {name} must be an int-like scalar"
                )
            try:
                normalized = operator.index(static_value)
            except TypeError as exc:
                raise TypeError(
                    f"{scope_name}.{operation} {name} must be an int-like scalar"
                ) from exc
            if isinstance(normalized, bool):
                raise TypeError(
                    f"{scope_name}.{operation} {name} must be an int-like scalar"
                )
            return int(normalized)

        static_k = static_int("k", bound.arguments["k"])
        if static_k is not None and static_k <= 0:
            raise ValueError(f"{scope_name}.{operation} k must be positive")
        tile_size = group.static_size * items_per_thread
        if self._is_none(bound.arguments["valid_items"]):
            static_valid_items = tile_size
        else:
            static_valid_items = static_int(
                "valid_items",
                bound.arguments["valid_items"],
            )
            if static_valid_items is not None and not (
                1 <= static_valid_items <= tile_size
            ):
                raise ValueError(
                    f"{scope_name}.{operation} valid_items must be in [1, {tile_size}]"
                )
        if (
            static_k is not None
            and static_valid_items is not None
            and static_k > static_valid_items
        ):
            raise ValueError(f"{scope_name}.{operation} k must be <= valid_items")

        begin_bit = bound.arguments["begin_bit"]
        end_bit = bound.arguments["end_bit"]
        begin_is_static, static_begin_value = self._try_constant(begin_bit)
        static_begin = static_int("begin_bit", begin_bit)
        if static_begin is not None and static_begin < 0:
            raise ValueError(f"{scope_name}.{operation} begin_bit must be non-negative")
        if self._is_none(end_bit):
            static_end = None
        else:
            static_end = static_int("end_bit", end_bit)
            if static_end is not None and static_end < 1:
                raise ValueError(f"{scope_name}.{operation} end_bit must be positive")
            if (
                static_begin is not None
                and static_end is not None
                and static_end <= static_begin
            ):
                raise ValueError(
                    f"{scope_name}.{operation} end_bit must exceed begin_bit"
                )

        from ._block import _block_topk

        factory = getattr(_block_topk, operation)
        factory_kwargs = {"threads_per_block": block_dim}
        if is_common_root:
            factory = getattr(_block_topk, f"_common_{operation}")
        elif self._is_none(end_bit) and not (
            begin_is_static and static_begin_value == 0
        ):
            factory = getattr(_block_topk, f"_qualified_group_{operation}")
        if not self._is_none(bound.arguments["valid_items"]):
            factory_kwargs["num_valid"] = bound.arguments["valid_items"]
        if self._is_none(end_bit):
            if not (begin_is_static and static_begin_value == 0):
                factory_kwargs["begin_bit"] = begin_bit
        else:
            factory_kwargs["begin_bit"] = begin_bit
            factory_kwargs["end_bit"] = end_bit
        if not self._is_none(bound.arguments["temp_storage"]):
            factory_kwargs["temp_storage"] = bound.arguments["temp_storage"]

        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        keys = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_keys",
            value=bound.arguments["keys"],
        )
        if not self._array_operand_state(operation, keys):
            raise TypeError(
                f"{scope_name}.{operation} requires a fixed-size key payload"
            )
        result_keys = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_keys_result",
            prototype=keys,
            is_array=True,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
        )
        self._copy_array_payload(
            statements,
            operation=operation,
            source=keys,
            destination=result_keys,
            scope=scope,
            loc=loc,
            known_items_per_thread=items_per_thread,
        )

        runtime_args = [result_keys]
        result_values = None
        if operation.endswith("_pairs"):
            values = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_values",
                value=bound.arguments["values"],
            )
            if not self._array_operand_state(operation, values):
                raise TypeError(
                    f"{scope_name}.{operation} requires a fixed-size value payload"
                )
            result_values = self._typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_values_result",
                prototype=values,
                is_array=True,
                dtype_policy=_PAYLOAD_DTYPE_LIKE,
            )
            self._copy_array_payload(
                statements,
                operation=operation,
                source=values,
                destination=result_values,
                scope=scope,
                loc=loc,
                known_items_per_thread=items_per_thread,
            )
            runtime_args.append(result_values)
        runtime_args.append(bound.arguments["k"])

        call_statements = self._rewritten_call(
            inst,
            factory=factory,
            args=runtime_args,
            kwargs=factory_kwargs,
            return_alias=(
                result_keys if result_values is None else (result_keys, result_values)
            ),
        )
        call_statements.pop()
        statements.extend(call_statements)
        if result_values is None:
            statements.append(ir.Assign(result_keys, inst.target, loc))
        else:
            statements.append(
                ir.Assign(
                    ir.Expr.build_tuple([result_keys, result_values], loc),
                    inst.target,
                    loc,
                )
            )
        return statements

    def _lower_histogram(
        self,
        inst: ir.Assign,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        operation = "histogram"
        self._reject_extra_root_arguments(operation, bound)
        if group.kind != "block":
            raise NotImplementedError(
                "cuda.coop.numba_mlir.histogram currently lowers only "
                "complete physical block groups"
            )

        from cuda.coop._core.block import (
            normalize_block_histogram_positive_int,
            validate_block_histogram_output_capacity,
        )

        bins = normalize_block_histogram_positive_int(
            "bins",
            self._constant(bound.arguments["bins"]),
            scope="cuda.coop.numba_mlir.histogram",
        )
        bins_per_thread = normalize_block_histogram_positive_int(
            "bins_per_thread",
            self._constant(bound.arguments["bins_per_thread"]),
            scope="cuda.coop.numba_mlir.histogram",
        )
        group_size = group.static_size
        assert group_size is not None
        validate_block_histogram_output_capacity(
            bins=bins,
            bins_per_thread=bins_per_thread,
            block_threads=group_size,
            scope="cuda.coop.numba_mlir.histogram",
        )

        if is_common_root and not self._thread_data_operand_state(
            operation,
            "samples",
            bound.arguments["samples"],
        ):
            raise TypeError(
                "cuda.coop.histogram requires samples to be coop.ThreadData "
                "in common V1; use cuda.coop.numba_mlir for "
                "backend-qualified local-array payloads"
            )

        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        samples = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="histogram_samples",
            value=bound.arguments["samples"],
        )
        if not self._array_operand_state(operation, samples):
            raise TypeError(
                "cuda.coop.numba_mlir.histogram requires a fixed-size "
                "ThreadData or local-array samples payload"
            )
        provider_samples = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem="histogram_provider_samples",
            prototype=samples,
            is_array=True,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
        )
        self._copy_array_payload(
            statements,
            operation=operation,
            source=samples,
            destination=provider_samples,
            scope=scope,
            loc=loc,
        )

        counter_dtype = bound.arguments["counter_dtype"]
        if self._is_none(counter_dtype):
            counter_dtype = types.int32
        else:
            from ._common import normalize_dtype_param

            counter_dtype = self._constant(counter_dtype)
            counter_dtype = (
                types.int32
                if counter_dtype is int
                else normalize_dtype_param(counter_dtype)
            )
        provider_counter_dtype = _histogram_provider_counter_dtype(counter_dtype)
        histogram = self._emit_shared_array(
            statements,
            scope=scope,
            loc=loc,
            stem="histogram_counters",
            items=bins,
            dtype=provider_counter_dtype,
        )
        result = self._thread_data_payload(
            statements,
            scope=scope,
            loc=loc,
            stem="histogram_result",
            items_per_thread=bins_per_thread,
            dtype=counter_dtype,
        )

        from ._block._block_histogram import _group_histogram

        assert group.hierarchy is not None
        self._emit_factory_call(
            statements,
            scope=scope,
            loc=loc,
            stem="histogram_fused",
            factory=_group_histogram,
            args=[provider_samples, histogram],
            kwargs={
                "threads_per_block": group.hierarchy.block_dim,
                "bins": bins,
                "algorithm": bound.arguments["algorithm"],
                **({"_common_profile_operation": operation} if is_common_root else {}),
            },
        )
        rank = self._emit_group_method_call(
            statements,
            scope=scope,
            loc=loc,
            stem="histogram_rank",
            group=group,
            operation="rank",
        )
        bins_var = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="histogram_bins",
            value=bins,
        )
        for item_index in range(bins_per_thread):
            offset = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"histogram_offset_{item_index}",
                value=item_index * group_size,
            )
            striped_index = self._new_var(
                scope,
                loc,
                f"histogram_striped_index_{item_index}",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.binop(operator.add, rank, offset, loc),
                    striped_index,
                    loc,
                )
            )
            safe_index = self._new_var(
                scope,
                loc,
                f"histogram_safe_index_{item_index}",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.binop(operator.mod, striped_index, bins_var, loc),
                    safe_index,
                    loc,
                )
            )
            counter = self._new_var(
                scope,
                loc,
                f"histogram_counter_{item_index}",
            )
            statements.append(
                ir.Assign(ir.Expr.getitem(histogram, safe_index, loc), counter, loc)
            )
            is_valid = self._new_var(
                scope,
                loc,
                f"histogram_counter_valid_{item_index}",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.binop(operator.lt, striped_index, bins_var, loc),
                    is_valid,
                    loc,
                )
            )
            projected = self._new_var(
                scope,
                loc,
                f"histogram_projected_counter_{item_index}",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.binop(operator.mul, counter, is_valid, loc),
                    projected,
                    loc,
                )
            )
            output_index = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"histogram_output_index_{item_index}",
                value=item_index,
            )
            statements.append(ir.SetItem(result, output_index, projected, loc))
        statements.append(ir.Assign(result, inst.target, loc))
        return statements

    def _emit_group_method_call(
        self,
        statements: list[Any],
        *,
        scope: Any,
        loc: ir.Loc,
        stem: str,
        group: ThreadGroup,
        operation: str,
    ) -> ir.Var:
        """Emit one already-resolved group-method invocable call."""

        from ._group_provider import make_group_method_invocable

        invocable = make_group_method_invocable(
            group=group,
            operation=operation,
            dtype=None,
            level="thread",
        )
        return self._emit_factory_call(
            statements,
            scope=scope,
            loc=loc,
            stem=stem,
            factory=invocable,
            args=[],
            kwargs={},
        )

    def _lower_run_length_decode(
        self,
        inst: ir.Assign,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        operation = "run_length_decode"
        scope_name = "cuda.coop" if is_common_root else "cuda.coop.numba_mlir"
        self._reject_extra_root_arguments(operation, bound)
        if group.kind != "block":
            raise NotImplementedError(
                f"{scope_name}.run_length_decode currently lowers only "
                "complete physical block groups"
            )

        decoded_items_per_thread = self._constant(
            bound.arguments["decoded_items_per_thread"]
        )
        if isinstance(decoded_items_per_thread, bool) or not isinstance(
            decoded_items_per_thread,
            Integral,
        ):
            raise TypeError(
                f"{scope_name}.run_length_decode decoded_items_per_thread "
                "must be a compile-time positive integer"
            )
        decoded_items_per_thread = int(decoded_items_per_thread)
        if decoded_items_per_thread < 1:
            raise ValueError(
                f"{scope_name}.run_length_decode decoded_items_per_thread "
                "must be a compile-time positive integer"
            )

        offset_is_static, static_offset = self._try_constant(
            bound.arguments["decoded_window_offset"]
        )
        if offset_is_static:
            if isinstance(static_offset, bool) or not isinstance(
                static_offset,
                Integral,
            ):
                raise TypeError(
                    f"{scope_name}.run_length_decode decoded_window_offset "
                    "must be an integer"
                )
            if int(static_offset) < 0:
                raise ValueError(
                    f"{scope_name}.run_length_decode decoded_window_offset "
                    "must be non-negative"
                )

        if is_common_root:
            for name in ("run_values", "run_lengths"):
                if not self._thread_data_operand_state(
                    operation,
                    name,
                    bound.arguments[name],
                ):
                    raise TypeError(
                        f"cuda.coop.run_length_decode requires {name} to be "
                        "coop.ThreadData in common V1; use "
                        "cuda.coop.numba_mlir for backend-qualified payloads"
                    )

        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        run_values = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_values",
            value=bound.arguments["run_values"],
        )
        run_lengths = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_lengths",
            value=bound.arguments["run_lengths"],
        )
        if not self._array_operand_state(
            operation,
            run_values,
        ) or not self._array_operand_state(operation, run_lengths):
            raise TypeError(
                f"{scope_name}.run_length_decode requires fixed-size "
                "ThreadData or local-array run_values and run_lengths payloads"
            )
        runs_per_thread = self._array_extent(run_values)
        if runs_per_thread is None:
            raise GroupRewriteError(
                f"{scope_name}.run_length_decode could not infer runs_per_thread"
            )
        if self._array_extent(run_lengths) != runs_per_thread:
            raise ValueError(
                f"{scope_name}.run_length_decode run_values and run_lengths "
                "must have the same items_per_thread"
            )

        decoded_items = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_decoded",
            prototype=run_values,
            is_array=True,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
            items_per_thread=decoded_items_per_thread,
        )
        total_decoded_size = bound.arguments.get("total_decoded_size")
        if self._is_none(total_decoded_size):
            total_decoded_size = self._typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem="run_length_total_decoded_size",
                prototype=run_lengths,
                is_array=True,
                dtype_policy=_PAYLOAD_DTYPE_LIKE,
                items_per_thread=1,
            )
        else:
            total_decoded_size = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem="run_length_total_decoded_size_output",
                value=total_decoded_size,
            )
            if not self._array_operand_state(operation, total_decoded_size):
                raise TypeError(
                    f"{scope_name}.run_length_decode total_decoded_size "
                    "must be a single-item ThreadData or local-array output"
                )
            if self._array_extent(total_decoded_size) != 1:
                raise ValueError(
                    f"{scope_name}.run_length_decode total_decoded_size "
                    "must contain exactly one item"
                )

        relative_offsets = bound.arguments.get("relative_offsets")
        has_relative_offsets = not self._is_none(relative_offsets)
        if has_relative_offsets:
            relative_offsets = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem="run_length_relative_offsets",
                value=relative_offsets,
            )
            if not self._array_operand_state(operation, relative_offsets):
                raise TypeError(
                    f"{scope_name}.run_length_decode relative_offsets must be "
                    "a ThreadData or local-array output"
                )
            if self._array_extent(relative_offsets) != decoded_items_per_thread:
                raise ValueError(
                    f"{scope_name}.run_length_decode relative_offsets must "
                    "match decoded_items_per_thread"
                )

        decoded_offset_dtype = bound.arguments.get("decoded_offset_dtype")
        assert group.hierarchy is not None
        factory_kwargs: dict[str, Any] = {
            "threads_per_block": group.hierarchy.block_dim,
            "runs_per_thread": runs_per_thread,
            "decoded_items_per_thread": decoded_items_per_thread,
            "with_relative_offsets": has_relative_offsets,
            **({"_common_profile_operation": operation} if is_common_root else {}),
        }
        if offset_is_static:
            factory_kwargs["_static_decoded_window_offset"] = int(static_offset)
        if not self._is_none(decoded_offset_dtype):
            factory_kwargs["decoded_offset_dtype"] = decoded_offset_dtype

        mask_index = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_mask_index",
            value=0,
        )
        zero_literal = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_zero_literal",
            value=0,
        )
        statements.append(ir.SetItem(decoded_items, mask_index, zero_literal, loc))
        decoded_zero = self._new_var(scope, loc, "run_length_decoded_zero")
        statements.append(
            ir.Assign(
                ir.Expr.getitem(decoded_items, mask_index, loc),
                decoded_zero,
                loc,
            )
        )
        relative_sentinel = None
        if has_relative_offsets:
            minus_one_literal = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem="run_length_minus_one_literal",
                value=-1,
            )
            statements.append(
                ir.SetItem(relative_offsets, mask_index, minus_one_literal, loc)
            )
            relative_sentinel = self._new_var(
                scope,
                loc,
                "run_length_relative_sentinel",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.getitem(relative_offsets, mask_index, loc),
                    relative_sentinel,
                    loc,
                )
            )

        runtime_args = [
            run_values,
            run_lengths,
            total_decoded_size,
            decoded_items,
        ]
        if has_relative_offsets:
            runtime_args.append(relative_offsets)
        decoded_window_offset = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_decoded_window_offset",
            value=bound.arguments["decoded_window_offset"],
        )
        runtime_args.append(decoded_window_offset)

        from ._block._block_run_length_decode import _group_run_length_decode

        self._emit_factory_call(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_fused",
            factory=_group_run_length_decode,
            args=runtime_args,
            kwargs=factory_kwargs,
        )

        from numba_cuda_mlir import cuda as cuda_module

        rank = self._emit_group_method_call(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_rank",
            group=group,
            operation="rank",
        )
        decoded_items_per_thread_var = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_decoded_items_per_thread",
            value=decoded_items_per_thread,
        )
        rank_base = self._new_var(scope, loc, "run_length_rank_base")
        statements.append(
            ir.Assign(
                ir.Expr.binop(
                    operator.mul,
                    rank,
                    decoded_items_per_thread_var,
                    loc,
                ),
                rank_base,
                loc,
            )
        )
        total_value = self._new_var(scope, loc, "run_length_total")
        statements.append(
            ir.Assign(
                ir.Expr.getitem(total_decoded_size, mask_index, loc),
                total_value,
                loc,
            )
        )
        offset_is_in_range = self._new_var(
            scope,
            loc,
            "run_length_offset_is_in_range",
        )
        statements.append(
            ir.Assign(
                ir.Expr.binop(
                    operator.lt,
                    decoded_window_offset,
                    total_value,
                    loc,
                ),
                offset_is_in_range,
                loc,
            )
        )
        safe_offset = self._emit_factory_call(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_safe_offset",
            factory=cuda_module.selp,
            args=[offset_is_in_range, decoded_window_offset, total_value],
            kwargs={},
        )
        remaining_items = self._new_var(scope, loc, "run_length_remaining_items")
        statements.append(
            ir.Assign(
                ir.Expr.binop(operator.sub, total_value, safe_offset, loc),
                remaining_items,
                loc,
            )
        )
        for item_index in range(decoded_items_per_thread):
            item_index_var = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"run_length_item_index_{item_index}",
                value=item_index,
            )
            local_target = self._new_var(
                scope,
                loc,
                f"run_length_local_target_{item_index}",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.binop(operator.add, rank_base, item_index_var, loc),
                    local_target,
                    loc,
                )
            )
            is_valid = self._new_var(
                scope,
                loc,
                f"run_length_item_valid_{item_index}",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.binop(operator.lt, local_target, remaining_items, loc),
                    is_valid,
                    loc,
                )
            )
            decoded_value = self._new_var(
                scope,
                loc,
                f"run_length_decoded_value_{item_index}",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.getitem(decoded_items, item_index_var, loc),
                    decoded_value,
                    loc,
                )
            )
            projected = self._emit_factory_call(
                statements,
                scope=scope,
                loc=loc,
                stem=f"run_length_projected_value_{item_index}",
                factory=cuda_module.selp,
                args=[is_valid, decoded_value, decoded_zero],
                kwargs={},
            )
            statements.append(ir.SetItem(decoded_items, item_index_var, projected, loc))
            if has_relative_offsets:
                relative_value = self._new_var(
                    scope,
                    loc,
                    f"run_length_relative_value_{item_index}",
                )
                statements.append(
                    ir.Assign(
                        ir.Expr.getitem(relative_offsets, item_index_var, loc),
                        relative_value,
                        loc,
                    )
                )
                assert relative_sentinel is not None
                projected_relative = self._emit_factory_call(
                    statements,
                    scope=scope,
                    loc=loc,
                    stem=f"run_length_projected_relative_{item_index}",
                    factory=cuda_module.selp,
                    args=[is_valid, relative_value, relative_sentinel],
                    kwargs={},
                )
                statements.append(
                    ir.SetItem(
                        relative_offsets,
                        item_index_var,
                        projected_relative,
                        loc,
                    )
                )
        statements.append(ir.Assign(decoded_items, inst.target, loc))
        return statements

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
        if operation in {"scan", "exclusive_scan", "inclusive_scan"}:
            scan_op = bound.arguments.get("scan_op")
            if not self._is_none(scan_op):
                normalized_scan_op = ScanOp(self._constant(scan_op))
                if (
                    self._constant(bound.arguments["mode"]) == "exclusive"
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
        elif operation == "exchange":
            replacement = self._lower_exchange(
                inst, group=group, bound=bound, is_common_root=is_common_root
            )
        elif operation == "adjacent_difference":
            replacement = self._lower_adjacent_difference(
                inst,
                group=group,
                bound=bound,
                is_common_root=is_common_root,
            )
        elif operation == "discontinuity":
            replacement = self._lower_discontinuity(
                inst,
                group=group,
                bound=bound,
                is_common_root=is_common_root,
            )
        elif operation == "shuffle":
            replacement = self._lower_shuffle(
                inst, group=group, bound=bound, is_common_root=is_common_root
            )
        elif operation in {"merge_sort_keys", "merge_sort_pairs"}:
            replacement = self._lower_merge_sort(
                inst,
                operation=operation,
                group=group,
                bound=bound,
                is_common_root=is_common_root,
            )
        elif operation == "radix_rank":
            replacement = self._lower_radix_rank(
                inst, group=group, bound=bound, is_common_root=is_common_root
            )
        elif operation in {"radix_sort_keys", "radix_sort_pairs"}:
            replacement = self._lower_radix_sort(
                inst,
                operation=operation,
                group=group,
                bound=bound,
                is_common_root=is_common_root,
            )
        elif operation == "histogram":
            replacement = self._lower_histogram(
                inst,
                group=group,
                bound=bound,
                is_common_root=is_common_root,
            )
        elif operation in {
            "topk_max_keys",
            "topk_max_pairs",
            "topk_min_keys",
            "topk_min_pairs",
        }:
            replacement = self._lower_topk(
                inst,
                operation=operation,
                group=group,
                bound=bound,
                is_common_root=is_common_root,
            )
        elif operation == "run_length_decode":
            replacement = self._lower_run_length_decode(
                inst,
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
                    f"cuda.coop.ThreadGroup.{operation} does not support grid groups in common V1; use a backend-qualified import for backend-specific grid support"
                )
            raise NotImplementedError(
                "cuda.coop.numba_mlir grid synchronization requires a verified cooperative launch, which the current launch descriptor cannot request"
            )
        from ._group_provider import make_group_method_invocable

        invocable = make_group_method_invocable(
            group=group, operation=operation, dtype=dtype, level=level
        )
        self.dead_func_names.add(call.func.name)
        self.replacements[inst] = self._rewritten_call(
            inst, factory=invocable, args=[], kwargs={}
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
                operation = _ROOT_OPERATIONS.get(function)
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
                or function in _ROOT_OPERATIONS
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
        launch_config = require_launch_config(self.state)
        return _GroupCallPlanner(self.state, launch_config).run()


__all__ = [
    "CoopGroupHierarchyPlanner",
    "GroupRewriteError",
    "has_group_markers",
]
