# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load and store IR planning.

This mixin owns only its primitive-family IR rewrite. Shared provenance,
launch facts, caches, and final orchestration remain in the group planner.
"""

import math
from numbers import Integral, Real

import numpy as np

from cuda.coop._core import (
    ArgumentBinding,
    BindingKind,
    GroupLoadStoreAlgorithm,
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
    GroupLoweringPlan,
    GroupLoweringTarget,
    StorageOwnership,
    make_group_primitive_call,
    plan_group_primitive,
)

from ._group_planner_support import (
    Any,
    GroupRewriteError,
    ThreadGroup,
    _cuda_module,
    _group_operation_name,
    _portable_api,
    _typed_group_payload_like,
    inspect,
    ir,
    types,
)
from ._operations import (
    RewriteOperationSpec,
    register_group_primitive,
    register_rewrite_operation,
)
from ._parameters import _validate_common_numeric_dtype, normalize_dtype_param
from ._rewrite_load_store import (
    analyze_load_store_match,
    infer_load_store_payload,
    prepare_load_store_runtime_args,
    validate_load_store_runtime_controls,
)

_BLOCK_LOAD_STORE_ALGORITHMS = frozenset(
    {
        "direct",
        "striped",
        "vectorize",
        "transpose",
        "warp_transpose",
        "warp_transpose_timesliced",
    }
)


def _direct_algorithm(value: object, *, operation: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{operation} algorithm must not be bool")
    if isinstance(value, str):
        token = value.strip().lower().replace("-", "_")
    elif hasattr(value, "name"):
        token = str(getattr(value, "name")).lower()
    elif isinstance(value, int):
        token = "direct" if value == 0 else None
    else:
        token = None
    if token == "direct":
        return token
    if token in _BLOCK_LOAD_STORE_ALGORITHMS:
        raise NotImplementedError(
            f"cuda.coop.numba_mlir.{operation} algorithm {token!r} is not "
            "executable; only 'direct' is currently supported"
        )
    choices = ", ".join(sorted(_BLOCK_LOAD_STORE_ALGORITHMS))
    raise ValueError(
        f"cuda.coop.numba_mlir.{operation} algorithm must be one of: {choices}"
    )


_CUB_PLAN_ROUTES = {
    (
        "CUB",
        "cub/block/block_load.cuh",
        "cub::BlockLoad",
        "Load",
    ): "load",
    (
        "CUB",
        "cub/block/block_store.cuh",
        "cub::BlockStore",
        "Store",
    ): "store",
}


class _LoadStorePlanning:
    """Family-local planner facade over the shared IR analysis context."""

    def __init__(self, planner: Any) -> None:
        self._planner = planner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._planner, name)

    def _validate_common_arguments(
        self, operation: str, bound: inspect.BoundArguments
    ) -> None:
        bound.arguments["algorithm"] = self._validate_common_selector(
            operation,
            "algorithm",
            bound.arguments["algorithm"],
            _BLOCK_LOAD_STORE_ALGORITHMS,
        )

    def _scope_factory(
        self, group: ThreadGroup, operation: str
    ) -> tuple[Any, dict[str, Any]]:
        assert group.hierarchy is not None
        if group.kind != "block":
            raise NotImplementedError(
                f"cuda.coop.numba_mlir.{operation} currently lowers only "
                "this_block() groups through CUB"
            )
        block_dim = group.hierarchy.block_dim
        if block_dim is None:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} requires an exact block "
                "dimension before provider selection"
            )

        from .._lowering import _load_store

        return (getattr(_load_store, operation), {"threads_per_block": block_dim})

    def _planning_binding(self, value: Any) -> ArgumentBinding:
        resolved, constant = self._try_constant(value)
        if not resolved:
            return ArgumentBinding.runtime()
        if constant is None:
            return ArgumentBinding.omitted()
        return ArgumentBinding.static(constant)

    def _planning_oob_default(
        self,
        value: Any,
        *,
        payload_dtype: Any,
    ) -> ArgumentBinding:
        binding = self._planning_binding(value)
        if binding.kind is BindingKind.OMITTED:
            return binding
        if binding.kind is BindingKind.RUNTIME:
            value_dtype = self._planning_dtype(value)
            if value_dtype is None:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir.load could not infer the runtime "
                    "oob_default dtype before provider selection"
                )
            value_dtype = _validate_common_numeric_dtype(
                value_dtype,
                operation="load",
                parameter="oob_default",
            )
            if value_dtype != payload_dtype:
                raise TypeError(
                    "cuda.coop.numba_mlir.load runtime oob_default dtype "
                    f"{value_dtype} does not match payload dtype {payload_dtype}"
                )
            return binding

        scalar = binding.value
        if isinstance(scalar, np.generic):
            scalar_dtype = scalar.dtype
        elif type(scalar) in {bool, int, float, complex}:
            scalar_dtype = type(scalar)
        else:
            raise TypeError(
                "cuda.coop.numba_mlir.load static oob_default must be a "
                "portable numeric scalar"
            )
        _validate_common_numeric_dtype(
            scalar_dtype,
            operation="load",
            parameter="oob_default",
        )
        if isinstance(scalar, Integral):
            scalar = int(scalar)
            if not -(1 << 63) <= scalar <= (1 << 64) - 1:
                raise ValueError(
                    "cuda.coop.numba_mlir.load static oob_default must fit a "
                    "64-bit integer"
                )
        elif isinstance(scalar, Real) and not math.isfinite(float(scalar)):
            raise ValueError(
                "cuda.coop.numba_mlir.load static oob_default must be finite"
            )
        return binding

    @staticmethod
    def _dtype_from_numba_type(value: Any) -> Any | None:
        if isinstance(value, types.Array):
            value = value.dtype
        elif not isinstance(value, types.Type):
            return None
        return normalize_dtype_param(value)

    def _planning_dtype_definition(
        self,
        definition: Any,
        *,
        seen: set[str],
    ) -> Any | None:
        if isinstance(definition, ir.Var):
            return self._planning_dtype(definition, seen=seen)
        if isinstance(definition, ir.Arg):
            if not 0 <= definition.index < len(self.state.args):
                return None
            return self._dtype_from_numba_type(self.state.args[definition.index])
        if not isinstance(definition, ir.Expr):
            return None
        if definition.op in {"cast", "exhaust_iter"}:
            return self._planning_dtype(definition.value, seen=seen)
        if definition.op == "phi":
            candidates = {
                dtype
                for incoming in getattr(definition, "incoming_values", ())
                if (dtype := self._planning_dtype(incoming, seen=set(seen))) is not None
            }
            if len(candidates) > 1:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir Load/Store payload aliases have "
                    "inconsistent dtypes"
                )
            return next(iter(candidates), None)
        if definition.op in {"getitem", "static_getitem"}:
            return self._planning_dtype(definition.value, seen=seen)
        if definition.op != "call":
            return None
        function = self._callable(definition.func)
        from .._thread_data import ThreadData

        if function in {ThreadData, _portable_api.ThreadData}:
            bound = self._bind(function, definition)
            resolved, dtype = self._try_constant(bound.arguments["dtype"])
            if resolved and dtype is not None:
                return normalize_dtype_param(dtype)
            return None
        if function is _cuda_module.local.array:
            if len(definition.args) >= 2:
                resolved, dtype = self._try_constant(definition.args[1])
                if resolved:
                    return normalize_dtype_param(dtype)
            dtype_ref = dict(definition.kws).get("dtype")
            if dtype_ref is not None:
                resolved, dtype = self._try_constant(dtype_ref)
                if resolved:
                    return normalize_dtype_param(dtype)
            return None
        if function is _typed_group_payload_like and definition.args:
            return self._planning_dtype(definition.args[0], seen=seen)
        operation = _group_operation_name(function)
        if operation == "load":
            bound = self._bind(function, definition)
            return self._planning_dtype(bound.arguments["output"], seen=seen)
        return None

    def _planning_dtype(
        self,
        value: Any,
        *,
        seen: set[str] | None = None,
    ) -> Any | None:
        if not isinstance(value, ir.Var):
            return self._dtype_from_numba_type(value)
        if seen is None:
            seen = set()
        if value.name in seen:
            return None
        seen.add(value.name)
        candidates = {
            dtype
            for definition in self._all_definitions(value)
            if (
                dtype := self._planning_dtype_definition(
                    definition,
                    seen=set(seen),
                )
            )
            is not None
        }
        if len(candidates) > 1:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir Load/Store payload aliases have "
                "inconsistent dtypes"
            )
        return next(iter(candidates), None)

    def _planning_store_write_dtype(self, payload: Any) -> Any | None:
        """Infer an untyped Store payload from values written through its aliases."""

        if not isinstance(payload, ir.Var):
            return None
        alias_names = {payload.name}
        changed = True
        while changed:
            changed = False
            for block in self.func_ir.blocks.values():
                for stmt in block.body:
                    if not isinstance(stmt, ir.Assign):
                        continue
                    definition = stmt.value
                    sources: tuple[ir.Var, ...] = ()
                    if isinstance(definition, ir.Var):
                        sources = (definition,)
                    elif isinstance(definition, ir.Expr) and definition.op in {
                        "cast",
                        "exhaust_iter",
                    }:
                        if isinstance(definition.value, ir.Var):
                            sources = (definition.value,)
                    elif isinstance(definition, ir.Expr) and definition.op == "phi":
                        sources = tuple(
                            incoming
                            for incoming in getattr(definition, "incoming_values", ())
                            if isinstance(incoming, ir.Var)
                        )
                    source_names = {source.name for source in sources}
                    if stmt.target.name in alias_names or source_names & alias_names:
                        additions = {stmt.target.name, *source_names} - alias_names
                        if additions:
                            alias_names.update(additions)
                            changed = True

        inferred = None
        static_setitem_cls = getattr(ir, "StaticSetItem", None)
        for block in self.func_ir.blocks.values():
            for stmt in block.body:
                if isinstance(stmt, ir.SetItem) or (
                    static_setitem_cls is not None
                    and isinstance(stmt, static_setitem_cls)
                ):
                    target = getattr(stmt, "target", None)
                    value = getattr(stmt, "value", None)
                else:
                    continue
                if not isinstance(target, ir.Var) or target.name not in alias_names:
                    continue
                if not isinstance(value, ir.Var):
                    continue
                value_dtype = self._planning_dtype(value)
                if value_dtype is None:
                    continue
                if inferred is None:
                    inferred = value_dtype
                elif inferred != value_dtype:
                    raise TypeError(
                        "cuda.coop.numba_mlir.store could not infer one "
                        "consistent dtype from ThreadData writes"
                    )
        return inferred

    def _planning_items_per_thread(self, operation: str, payload: Any) -> int:
        is_array = self._array_operand_state(operation, payload)
        if not is_array:
            if operation == "load":
                raise TypeError(
                    "cuda.coop.numba_mlir.load output must be a fixed-size local array"
                )
            return 1
        extent = self._array_extent(payload)
        if extent is None:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} requires a static "
                "items_per_thread extent before provider selection"
            )
        return extent

    def _planning_temp_storage_definition(
        self,
        definition: Any,
        *,
        seen: set[str],
    ) -> tuple[int | None, int | None, bool, str] | None:
        if isinstance(definition, ir.Var):
            return self._planning_temp_storage(definition, seen=seen)
        if not isinstance(definition, ir.Expr):
            return None
        if definition.op in {"cast", "exhaust_iter"}:
            return self._planning_temp_storage(definition.value, seen=seen)
        if definition.op == "phi":
            candidates = {
                descriptor
                for incoming in getattr(definition, "incoming_values", ())
                if (
                    descriptor := self._planning_temp_storage(
                        incoming,
                        seen=set(seen),
                    )
                )
                is not None
            }
            if len(candidates) > 1:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir TempStorage aliases have "
                    "inconsistent contracts"
                )
            return next(iter(candidates), None)
        if definition.op != "call":
            return None
        function = self._callable(definition.func)
        from .._temp_storage import TempStorage

        if function not in {TempStorage, _portable_api.TempStorage}:
            return None
        bound = self._bind(function, definition)
        values = {
            name: self._constant(value) for name, value in bound.arguments.items()
        }
        descriptor = TempStorage(**values)
        return (
            descriptor.size_in_bytes,
            descriptor.alignment,
            descriptor.auto_sync,
            descriptor.sharing,
        )

    def _planning_temp_storage(
        self,
        value: Any,
        *,
        seen: set[str] | None = None,
    ) -> tuple[int | None, int | None, bool, str] | None:
        if not isinstance(value, ir.Var):
            return None
        if seen is None:
            seen = set()
        if value.name in seen:
            return None
        seen.add(value.name)
        candidates = {
            descriptor
            for definition in self._all_definitions(value)
            if (
                descriptor := self._planning_temp_storage_definition(
                    definition,
                    seen=set(seen),
                )
            )
            is not None
        }
        if len(candidates) > 1:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir TempStorage aliases have inconsistent contracts"
            )
        return next(iter(candidates), None)

    def _plan_load_store(
        self,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
    ) -> GroupLoweringPlan:
        payload_name = "output" if operation == "load" else "value"
        memory_name = "source" if operation == "load" else "destination"
        payload = bound.arguments[payload_name]
        items_per_thread = self._planning_items_per_thread(operation, payload)
        payload_dtype = self._planning_dtype(payload)
        if operation == "store" and payload_dtype is None:
            payload_dtype = self._planning_store_write_dtype(payload)
        memory_dtype = self._planning_dtype(bound.arguments[memory_name])
        dtype = payload_dtype if payload_dtype is not None else memory_dtype
        if dtype is None:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} could not infer a dtype "
                "before provider selection"
            )
        dtype = _validate_common_numeric_dtype(dtype, operation=operation)
        if (
            payload_dtype is not None
            and memory_dtype is not None
            and payload_dtype != memory_dtype
        ):
            raise TypeError(
                f"cuda.coop.numba_mlir.{operation} memory dtype "
                f"{memory_dtype} does not match payload dtype {payload_dtype}"
            )

        oob_default = (
            self._planning_oob_default(
                bound.arguments["oob_default"],
                payload_dtype=dtype,
            )
            if operation == "load"
            else ArgumentBinding.omitted()
        )

        algorithm = _direct_algorithm(
            self._constant(bound.arguments["algorithm"]),
            operation=operation,
        )
        temp_storage_value = bound.arguments["temp_storage"]
        if self._is_none(temp_storage_value):
            storage_kwargs: dict[str, Any] = {
                "storage_ownership": StorageOwnership.IMPLEMENTATION,
            }
        else:
            storage = self._planning_temp_storage(temp_storage_value)
            if storage is None:
                raise GroupRewriteError(
                    f"cuda.coop.numba_mlir.{operation} temp_storage must "
                    "resolve to a compile-time TempStorage descriptor"
                )
            size_in_bytes, alignment, auto_sync, sharing = storage
            storage_kwargs = {
                "storage_ownership": StorageOwnership.CALLER,
                "storage_sharing": sharing,
                "storage_size_in_bytes": size_in_bytes,
                "storage_alignment": alignment,
                "storage_auto_sync": auto_sync,
            }

        semantics = GroupLoadStoreSemantics(
            kind=GroupLoadStoreKind(operation),
            dtype=dtype,
            items_per_thread=items_per_thread,
            algorithm=GroupLoadStoreAlgorithm(algorithm),
            valid_items=self._planning_binding(bound.arguments["valid_items"]),
            oob_default=oob_default,
            offset=self._planning_binding(bound.arguments["offset"]),
            **storage_kwargs,
        )
        plan = plan_group_primitive(
            make_group_primitive_call(group, semantics),
            self.launch,
        )
        try:
            return plan.require_supported()
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"cuda.coop.numba_mlir.{operation} currently lowers only "
                f"this_block() groups through CUB: {exc}"
            ) from exc

    @staticmethod
    def _plan_provider_operation(plan: GroupLoweringPlan) -> str:
        if plan.target is not GroupLoweringTarget.CUB_BLOCK:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir Load/Store received an unsupported "
                f"lowering target {plan.target.value!r}"
            )
        assert plan.provenance is not None
        try:
            return _CUB_PLAN_ROUTES[plan.provenance.semantic_key]
        except KeyError as exc:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir Load/Store received an unknown CUB "
                f"implementation provenance {plan.provenance.semantic_key!r}"
            ) from exc

    @staticmethod
    def _planned_argument(
        binding: ArgumentBinding,
        runtime_value: Any,
    ) -> Any:
        return runtime_value if binding.kind is BindingKind.RUNTIME else binding

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
                        "cuda.coop.load requires output to be a fixed-size ThreadData payload in the portable API; use cuda.coop.numba_mlir for backend-qualified local-array payload support"
                    )
            else:
                value = bound.arguments["value"]
                if self._array_operand_state(operation, value) and (
                    not self._thread_data_operand_state(operation, "value", value)
                ):
                    raise TypeError(
                        "cuda.coop.store accepts only a scalar or fixed-size ThreadData value payload in the portable API; use cuda.coop.numba_mlir for backend-qualified local-array payload support"
                    )
        plan = self._plan_load_store(
            operation=operation,
            group=group,
            bound=bound,
        )
        planned_operation = self._plan_provider_operation(plan)
        if planned_operation != operation:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} selected the "
                f"{planned_operation!r} provider"
            )
        factory, factory_kwargs = self._scope_factory(
            plan.resolved_group,
            planned_operation,
        )
        assert plan.implementation is not None
        factory_kwargs.update(
            {
                "algorithm": plan.implementation.metadata["algorithm"],
                "dtype": plan.implementation.template_arguments["T"],
                "items_per_thread": plan.implementation.template_arguments[
                    "ITEMS_PER_THREAD"
                ],
            }
        )
        if is_common_root:
            factory_kwargs["_common_root_operation"] = operation
        semantics = plan.call.operation
        for public_name, factory_name in (
            ("valid_items", "num_valid_items"),
            ("oob_default", "oob_default"),
            ("offset", "offset"),
        ):
            binding = getattr(semantics, public_name)
            if binding.kind is BindingKind.OMITTED:
                continue
            factory_kwargs[factory_name] = self._planned_argument(
                binding,
                bound.arguments[public_name],
            )
        if operation == "store":
            factory_kwargs["_group_root_store"] = True
        if not self._is_none(bound.arguments["temp_storage"]):
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


def _lower_registered_load_store(planner: Any, *args: Any, **kwargs: Any) -> list[Any]:
    return _LoadStorePlanning(planner)._lower_load_store(*args, **kwargs)


def _validate_registered_common_arguments(
    planner: Any,
    operation: str,
    bound: inspect.BoundArguments,
) -> None:
    _LoadStorePlanning(planner)._validate_common_arguments(operation, bound)


for _operation in ("load", "store"):
    register_group_primitive(
        _operation,
        lower=_lower_registered_load_store,
        array_result_parameter="output" if _operation == "load" else None,
        validate_common_arguments=_validate_registered_common_arguments,
    )
del _operation

_COMMON_REWRITE_KWARGS = frozenset(
    {
        "algorithm",
        "dim",
        "dtype",
        "items_per_thread",
        "num_valid_items",
        "offset",
        "threads_per_block",
        "_common_root_operation",
    }
)
register_rewrite_operation(
    "load",
    RewriteOperationSpec(
        namespace="block",
        runtime_arg_counts=frozenset({2, 3, 4}),
        runtime_factory_kwargs=("num_valid_items", "oob_default"),
        runtime_factory_kw_prerequisites=(("oob_default", "num_valid_items"),),
        allowed_factory_kwargs=_COMMON_REWRITE_KWARGS | {"oob_default"},
        required_factory_kwargs=frozenset({"threads_per_block", "dtype"}),
        runtime_temp_storage=True,
        scalar_binding_kwargs=frozenset({"num_valid_items", "oob_default"}),
        runtime_offset_kwarg="offset",
        infer_payload=infer_load_store_payload,
        analyze_match=analyze_load_store_match,
        prepare_runtime_args=prepare_load_store_runtime_args,
        validate_runtime_controls=validate_load_store_runtime_controls,
    ),
)
register_rewrite_operation(
    "store",
    RewriteOperationSpec(
        namespace="block",
        runtime_arg_counts=frozenset({2, 3}),
        runtime_factory_kwargs=("num_valid_items",),
        runtime_factory_kw_prerequisites=(),
        allowed_factory_kwargs=_COMMON_REWRITE_KWARGS | {"_group_root_store"},
        required_factory_kwargs=frozenset({"threads_per_block", "dtype"}),
        runtime_temp_storage=True,
        scalar_binding_kwargs=frozenset({"num_valid_items"}),
        runtime_offset_kwarg="offset",
        infer_payload=infer_load_store_payload,
        analyze_match=analyze_load_store_match,
        prepare_runtime_args=prepare_load_store_runtime_args,
        validate_runtime_controls=validate_load_store_runtime_controls,
    ),
)


__all__: tuple[str, ...] = ()
