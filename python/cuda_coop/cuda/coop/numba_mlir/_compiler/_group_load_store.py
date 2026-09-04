# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load and store IR planning.

This mixin owns only its primitive-family IR rewrite. Shared provenance,
launch facts, caches, and final orchestration remain in the group planner.
"""

from cuda.coop._core import (
    ArgumentBinding,
    BindingKind,
    GroupLoadStoreAlgorithm,
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
    GroupLoweringPlan,
    GroupLoweringTarget,
    make_group_primitive_call,
    plan_group_primitive,
)

from ._group_planner_support import (
    Any,
    GroupRewriteError,
    ThreadGroup,
    inspect,
    ir,
)
from ._group_planning import GroupPlanningContext
from ._operations import (
    GroupResultSource,
    RewriteOperationSpec,
    register_group_primitive,
    register_rewrite_operation,
)
from ._parameters import _validate_common_numeric_dtype, coerce_static_scalar
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
    """Family-local semantics over the declared shared planning context."""

    def __init__(self, context: GroupPlanningContext) -> None:
        self._context = context

    def _validate_common_arguments(
        self, operation: str, bound: inspect.BoundArguments
    ) -> None:
        bound.arguments["algorithm"] = self._context.validate_common_selector(
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

    def _planning_oob_default(
        self,
        value: Any,
        *,
        payload_dtype: Any,
    ) -> ArgumentBinding:
        binding = self._context.planning_binding(value)
        if binding.kind is BindingKind.OMITTED:
            return binding
        if binding.kind is BindingKind.RUNTIME:
            value_dtype = self._context.dtype(value)
            if value_dtype is None:
                return binding
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
        resolved, provenance = self._context.try_static_scalar_provenance(value)
        assert resolved and provenance is not None
        scalar = coerce_static_scalar(
            scalar,
            payload_dtype,
            operation="load",
            parameter="oob_default",
            source_dtype=provenance.dtype,
        )
        return ArgumentBinding.static(scalar)

    def _planning_items_per_thread(self, operation: str, payload: Any) -> int:
        is_array = self._context.is_array(operation, payload)
        if not is_array:
            if operation == "load":
                raise TypeError(
                    "cuda.coop.numba_mlir.load output must be a fixed-size local array"
                )
            return 1
        extent = self._context.array_extent(payload)
        if extent is None:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} requires a static "
                "items_per_thread extent before provider selection"
            )
        return extent

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
        payload_is_array = self._context.is_array(operation, payload)
        items_per_thread = self._planning_items_per_thread(operation, payload)
        payload_dtype = self._context.dtype(payload)
        if operation == "store" and payload_dtype is None:
            payload_dtype = self._context.payload_write_dtype(payload)
        memory_dtype = self._context.dtype(bound.arguments[memory_name])
        dtype = memory_dtype if memory_dtype is not None else payload_dtype
        if dtype is None:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} could not infer a dtype "
                "before provider selection"
            )
        dtype = _validate_common_numeric_dtype(dtype, operation=operation)
        if operation == "store" and not payload_is_array:
            resolved, provenance = self._context.try_static_scalar_provenance(payload)
            if resolved:
                assert provenance is not None
                coerce_static_scalar(
                    provenance.value,
                    dtype,
                    operation="store",
                    parameter="value",
                    source_dtype=provenance.dtype,
                )
                payload_dtype = dtype
        if payload_dtype is not None:
            _validate_common_numeric_dtype(payload_dtype, operation=operation)
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
            self._context.constant(bound.arguments["algorithm"]),
            operation=operation,
        )
        temp_storage_value = bound.arguments["temp_storage"]
        if (
            not self._context.is_none(temp_storage_value)
            and self._context.temp_storage(temp_storage_value) is None
        ):
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} temp_storage must "
                "resolve to a compile-time TempStorage descriptor"
            )

        semantics = GroupLoadStoreSemantics(
            kind=GroupLoadStoreKind(operation),
            dtype=dtype,
            items_per_thread=items_per_thread,
            algorithm=GroupLoadStoreAlgorithm(algorithm),
            valid_items=self._context.planning_binding(bound.arguments["valid_items"]),
            oob_default=oob_default,
            offset=self._context.planning_binding(bound.arguments["offset"]),
        )
        plan = plan_group_primitive(
            make_group_primitive_call(group, semantics),
            self._context.launch,
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
                if not self._context.is_thread_data(
                    operation, "output", bound.arguments["output"]
                ):
                    raise TypeError(
                        "cuda.coop.load requires output to be a fixed-size ThreadData payload in the portable API; use cuda.coop.numba_mlir for backend-qualified local-array payload support"
                    )
            else:
                value = bound.arguments["value"]
                if self._context.is_array(operation, value) and (
                    not self._context.is_thread_data(operation, "value", value)
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
        if not self._context.is_none(bound.arguments["temp_storage"]):
            factory_kwargs["temp_storage"] = bound.arguments["temp_storage"]
        if operation == "load":
            runtime_args = [bound.arguments["source"], bound.arguments["output"]]
            return_alias = bound.arguments["output"]
        else:
            runtime_args = [bound.arguments["destination"], bound.arguments["value"]]
            return_alias = None
        return self._context.rewrite_call(
            inst,
            lowering_plan=plan,
            factory=factory,
            args=runtime_args,
            kwargs=factory_kwargs,
            return_alias=return_alias,
        )


def _lower_registered_load_store(
    context: GroupPlanningContext, *args: Any, **kwargs: Any
) -> list[Any]:
    return _LoadStorePlanning(context)._lower_load_store(*args, **kwargs)


def _validate_registered_common_arguments(
    context: GroupPlanningContext,
    operation: str,
    bound: inspect.BoundArguments,
) -> None:
    _LoadStorePlanning(context)._validate_common_arguments(operation, bound)


for _operation in ("load", "store"):
    register_group_primitive(
        _operation,
        lower=_lower_registered_load_store,
        results=(
            (GroupResultSource("output", "output"),) if _operation == "load" else ()
        ),
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
        factory_namespaces=frozenset({"block"}),
        dtype_factory_kwargs=frozenset({"dtype"}),
        runtime_arg_counts=frozenset({2, 3, 4}),
        runtime_factory_kwargs=("num_valid_items", "oob_default"),
        runtime_factory_kw_prerequisites=(("oob_default", "num_valid_items"),),
        allowed_factory_kwargs=_COMMON_REWRITE_KWARGS | {"oob_default"},
        required_factory_kwargs=frozenset({"threads_per_block", "dtype"}),
        accepts_temp_storage=True,
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
        factory_namespaces=frozenset({"block"}),
        dtype_factory_kwargs=frozenset({"dtype"}),
        runtime_arg_counts=frozenset({2, 3}),
        runtime_factory_kwargs=("num_valid_items",),
        runtime_factory_kw_prerequisites=(),
        allowed_factory_kwargs=_COMMON_REWRITE_KWARGS | {"_group_root_store"},
        required_factory_kwargs=frozenset({"threads_per_block", "dtype"}),
        accepts_temp_storage=True,
        scalar_binding_kwargs=frozenset({"num_valid_items"}),
        runtime_offset_kwarg="offset",
        infer_payload=infer_load_store_payload,
        analyze_match=analyze_load_store_match,
        prepare_runtime_args=prepare_load_store_runtime_args,
        validate_runtime_controls=validate_load_store_runtime_controls,
    ),
)


__all__: tuple[str, ...] = ()
