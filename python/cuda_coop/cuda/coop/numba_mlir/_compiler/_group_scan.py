# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block and Warp Scan IR planning."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from numba_cuda_mlir import types

from cuda.coop._core import (
    ArgumentBinding,
    BindingKind,
    CxxFunction,
    CxxOperator,
    Dependency,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupScanSemantics,
    PythonOperator,
    Reference,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    make_scan_semantics,
    plan_group_primitive,
)

from .._semantic import _normalize_numba_callable
from ._group_planner_support import (
    _PAYLOAD_DTYPE_LIKE,
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
from ._parameters import (
    _validate_common_numeric_dtype,
    _validate_runtime_integer_dtype,
    coerce_static_scalar,
    make_typed_cpp_literal,
)
from ._rewrite_scan import infer_scan_payload, validate_scan_runtime_controls

_PORTABLE_ALGORITHMS = frozenset({"raking", "raking_memoize", "warp_scans"})
_PORTABLE_MODES = frozenset({"exclusive", "inclusive"})
_BUILTIN_OPERATOR_CPP = {
    "multiplies": "::cuda::std::multiplies<T>",
    "min": "::cuda::minimum<T>",
    "max": "::cuda::maximum<T>",
    "bit_and": "::cuda::std::bit_and<T>",
    "bit_or": "::cuda::std::bit_or<T>",
    "bit_xor": "::cuda::std::bit_xor<T>",
}
_PLAN_ROUTES = {
    ("CUB", "cub/block/block_scan.cuh", "cub::BlockScan"): "block",
    ("CUB", "cub/warp/warp_scan.cuh", "cub::WarpScan"): "warp",
}


class _ScanPlanning:
    """Family-local Scan semantics over the declared planning context."""

    def __init__(self, context: GroupPlanningContext) -> None:
        self._context = context

    def _validate_common_arguments(
        self,
        operation: str,
        bound: inspect.BoundArguments,
    ) -> None:
        if "mode" in bound.arguments:
            bound.arguments["mode"] = self._context.validate_common_selector(
                operation,
                "mode",
                bound.arguments["mode"],
                _PORTABLE_MODES,
            )
        if "algorithm" in bound.arguments:
            bound.arguments["algorithm"] = self._context.validate_common_selector(
                operation,
                "algorithm",
                bound.arguments["algorithm"],
                _PORTABLE_ALGORITHMS,
                allow_none=True,
            )

    @staticmethod
    def _operation_options(
        operation: str,
        bound: inspect.BoundArguments,
    ) -> tuple[str, Any]:
        if operation == "scan":
            return bound.arguments["mode"], bound.arguments["scan_op"]
        if operation == "exclusive_scan":
            return "exclusive", bound.arguments["scan_op"]
        if operation == "inclusive_scan":
            return "inclusive", bound.arguments["scan_op"]
        if operation == "exclusive_sum":
            return "exclusive", None
        if operation == "inclusive_sum":
            return "inclusive", None
        raise GroupRewriteError(
            f"Scan planner received unexpected operation {operation!r}"
        )

    @staticmethod
    def _operator(
        scan_op: Any,
        *,
        dtype: Any,
        is_common_root: bool,
    ) -> tuple[str, CxxOperator | PythonOperator | None]:
        from .._lowering._scan import (
            normalize_scan_operation,
            validate_scan_operator_dtype,
        )

        operation = normalize_scan_operation(scan_op)
        validate_scan_operator_dtype(scan_op, dtype)
        if operation == "sum":
            return "sum", None
        if operation is not None:
            return (
                operation,
                CxxOperator(
                    cpp=_BUILTIN_OPERATOR_CPP[operation],
                    dtype=Dependency("T"),
                    name="scan_op",
                ),
            )
        if is_common_root:
            raise NotImplementedError(
                "portable cuda.coop Scan supports built-in operators only; "
                "use cuda.coop.numba_mlir for a stateless device callback"
            )
        return (
            "callback",
            PythonOperator(
                ret_dtype=Dependency("T"),
                arg_dtypes=(Dependency("T"), Dependency("T")),
                op=_normalize_numba_callable(scan_op),
                name="scan_op",
            ),
        )

    def _initial_binding(
        self,
        value: Any,
        *,
        dtype: Any,
        mode: str,
    ) -> tuple[ArgumentBinding, CxxFunction | Reference | None]:
        binding = self._context.planning_binding(value)
        if mode == "inclusive" and binding.kind is not BindingKind.OMITTED:
            raise ValueError(
                "cuda.coop.numba_mlir inclusive scans do not accept initial_value"
            )
        if binding.kind is BindingKind.OMITTED:
            return binding, None
        if binding.kind is BindingKind.RUNTIME:
            value_dtype = self._context.dtype(value)
            if value_dtype is None:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir.scan could not infer the runtime "
                    "initial_value dtype"
                )
            value_dtype = _validate_common_numeric_dtype(
                value_dtype,
                operation="scan",
                parameter="initial_value",
            )
            if value_dtype != dtype:
                raise TypeError(
                    "cuda.coop.numba_mlir.scan runtime initial_value dtype "
                    f"{value_dtype} does not match value dtype {dtype}"
                )
            return binding, Reference(Dependency("T"), name="initial_value")

        resolved, provenance = self._context.try_static_scalar_provenance(value)
        assert resolved and provenance is not None
        scalar = coerce_static_scalar(
            binding.value,
            dtype,
            operation="scan",
            parameter="initial_value",
            source_dtype=provenance.dtype,
        )
        return (
            ArgumentBinding.static(scalar),
            CxxFunction(
                make_typed_cpp_literal(scalar, dtype),
                Dependency("T"),
                name="initial_value",
            ),
        )

    def _aggregate_output(self, operation: str, value: Any, dtype: Any) -> bool:
        if self._context.is_none(value):
            return False
        if not self._context.is_array(operation, value):
            raise TypeError(
                "cuda.coop.numba_mlir scan aggregate_output must be a "
                "one-item ThreadData or local array"
            )
        if self._context.array_extent(value) != 1:
            raise ValueError(
                "cuda.coop.numba_mlir scan aggregate_output must contain "
                "exactly one item"
            )
        aggregate_dtype = self._context.dtype(value)
        if aggregate_dtype is None:
            aggregate_dtype = self._context.payload_write_dtype(value)
        if aggregate_dtype is not None:
            aggregate_dtype = _validate_common_numeric_dtype(
                aggregate_dtype,
                operation="scan",
                parameter="aggregate_output",
            )
            if aggregate_dtype != dtype:
                raise TypeError(
                    "cuda.coop.numba_mlir scan aggregate_output dtype "
                    f"{aggregate_dtype} does not match value dtype {dtype}"
                )
        return True

    @staticmethod
    def _caller_storage_plan(
        plan: GroupLoweringPlan,
        descriptor: tuple[int | None, int | None, bool, str],
    ) -> GroupLoweringPlan:
        if plan.target is not GroupLoweringTarget.CUB_BLOCK:
            raise ValueError(
                "cuda.coop.numba_mlir scan temp_storage applies only to block groups"
            )
        storage = plan.temp_storage
        synchronization = plan.synchronization
        assert storage is not None and synchronization is not None
        size_in_bytes, alignment, auto_sync, sharing = descriptor
        return replace(
            plan,
            temp_storage=replace(
                storage,
                ownership=StorageOwnership.CALLER,
                exact_layout_required=True,
                sharing=sharing,
                requested_size_in_bytes=size_in_bytes,
                requested_alignment=alignment,
                auto_sync=auto_sync,
            ),
            synchronization=replace(
                synchronization,
                storage_reuse_barrier=(
                    SynchronizationScope.BLOCK
                    if auto_sync
                    else SynchronizationScope.NONE
                ),
            ),
        )

    def _plan(
        self,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> tuple[GroupLoweringPlan, str, Any, ArgumentBinding, bool]:
        mode, raw_scan_op = self._operation_options(operation, bound)
        from .._lowering._scan import _block_scan_algorithm, _scan_mode

        mode = _scan_mode(self._context.constant(mode))
        scan_op = None if raw_scan_op is None else self._context.constant(raw_scan_op)

        value = bound.arguments["value"]
        is_array = self._context.is_array(operation, value)
        if (
            is_common_root
            and is_array
            and not self._context.is_thread_data(
                operation,
                "value",
                value,
            )
        ):
            raise TypeError(
                f"cuda.coop.{operation} accepts only a scalar or fixed-size "
                "ThreadData value payload; use cuda.coop.numba_mlir for "
                "backend-qualified local arrays"
            )
        if group.kind in {"warp", "threads_within_warp"} and is_array:
            raise TypeError(
                "cuda.coop.numba_mlir WarpScan supports one scalar value per lane"
            )
        items_per_thread = 1
        if is_array:
            extent = self._context.array_extent(value)
            if extent is None:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir.scan requires a static "
                    "items_per_thread extent"
                )
            items_per_thread = extent
        dtype = self._context.dtype(value)
        if dtype is None and is_array:
            dtype = self._context.payload_write_dtype(value)
        if dtype is None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir.scan could not infer value dtype"
            )
        dtype = _validate_common_numeric_dtype(
            dtype,
            operation="scan",
            parameter="value",
        )
        operator_kind, scan_operator = self._operator(
            scan_op,
            dtype=dtype,
            is_common_root=is_common_root,
        )

        initial_raw = bound.arguments.get("initial_value")
        initial_binding, initial_value = self._initial_binding(
            initial_raw,
            dtype=dtype,
            mode=mode,
        )
        if (
            mode == "exclusive"
            and operator_kind != "sum"
            and initial_binding.kind is BindingKind.OMITTED
        ):
            raise ValueError(
                "cuda.coop.numba_mlir non-sum exclusive scans require initial_value"
            )
        aggregate_raw = bound.arguments.get("aggregate_output")
        aggregate = self._aggregate_output(operation, aggregate_raw, dtype)

        valid_raw = bound.arguments.get("valid_items")
        valid_items = self._context.planning_binding(valid_raw)
        if valid_items.kind is BindingKind.RUNTIME:
            valid_dtype = self._context.dtype(valid_raw)
            if valid_dtype is None:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir.scan could not infer the runtime "
                    "valid_items dtype"
                )
            _validate_runtime_integer_dtype(
                valid_dtype,
                operation="scan",
                parameter="valid_items",
            )

        algorithm_raw = bound.arguments.get("algorithm")
        algorithm = None
        if not self._context.is_none(algorithm_raw):
            algorithm = _block_scan_algorithm(self._context.constant(algorithm_raw))

        semantics = GroupScanSemantics(
            make_scan_semantics(
                dtype=dtype,
                mode=mode,
                value_kind="array" if is_array else "scalar",
                items_per_thread=items_per_thread,
                scan_operator=scan_operator,
                initial_value=initial_value,
                aggregate=aggregate,
            ),
            cub_algorithm=algorithm,
            valid_items=valid_items,
        )
        plan = plan_group_primitive(
            make_group_primitive_call(group, semantics),
            self._context.launch,
        ).require_supported()

        temp_storage = bound.arguments.get("temp_storage")
        if not self._context.is_none(temp_storage):
            descriptor = self._context.temp_storage(temp_storage)
            if descriptor is None:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir.scan temp_storage must resolve to "
                    "a compile-time TempStorage descriptor"
                )
            plan = self._caller_storage_plan(plan, descriptor)
        return plan, operator_kind, scan_op, initial_binding, is_array

    @staticmethod
    def _provider(plan: GroupLoweringPlan, *, is_array: bool) -> Any:
        provenance = plan.provenance
        if provenance is None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir.scan requires provider provenance"
            )
        route = _PLAN_ROUTES.get(provenance.semantic_key[:3])
        from .._lowering import _scan

        if route == "block":
            return _scan.block_scan_array if is_array else _scan.block_scan_scalar
        if route == "warp":
            return _scan.warp_scan
        raise GroupRewriteError(
            "cuda.coop.numba_mlir.scan received unknown provider provenance "
            f"{provenance.semantic_key!r}"
        )

    def _runtime_valid_items(
        self,
        statements: list[Any],
        inst: ir.Assign,
        binding: ArgumentBinding,
        value: Any,
    ) -> Any:
        if binding.kind is not BindingKind.RUNTIME:
            return binding
        scope = inst.target.scope
        loc = inst.loc
        cast = self._context.value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="scan_valid_items_type",
            value=types.int64,
        )
        result = self._context.new_var(scope, loc, "scan_valid_items_i64")
        statements.append(
            ir.Assign(
                ir.Expr.call(cast, [value], (), loc),
                result,
                loc,
            )
        )
        return result

    @staticmethod
    def _planned_argument(binding: ArgumentBinding, runtime_value: Any) -> Any:
        return runtime_value if binding.kind is BindingKind.RUNTIME else binding

    def _lower_scan(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        plan, operator_kind, scan_op, initial_binding, is_array = self._plan(
            operation=operation,
            group=group,
            bound=bound,
            is_common_root=is_common_root,
        )
        primitive = plan.call.operation.primitive
        factory = self._provider(plan, is_array=is_array)
        block_dim = plan.participation.exact_block_dim
        assert block_dim is not None
        scope = inst.target.scope
        loc = inst.loc
        statements: list[Any] = []
        value = self._context.value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="scan_value",
            value=bound.arguments["value"],
        )
        runtime_args = [value]
        result_payload = None
        if is_array:
            result_payload = self._context.typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem="scan_result",
                prototype=value,
                is_array=True,
                dtype_policy=_PAYLOAD_DTYPE_LIKE,
            )
            runtime_args.append(result_payload)

        provider_scan_op = operator_kind
        if operator_kind == "sum":
            provider_scan_op = None
        elif operator_kind == "callback":
            provider_scan_op = scan_op

        factory_kwargs: dict[str, Any] = {
            "dtype": primitive.dtype,
            "mode": primitive.mode.value,
            "scan_op": provider_scan_op,
        }
        if plan.target is GroupLoweringTarget.CUB_BLOCK:
            algorithm = plan.call.operation.cub_algorithm
            assert algorithm is not None
            factory_kwargs.update(
                {
                    "algorithm": algorithm.name.lower(),
                    "items_per_thread": primitive.items_per_thread,
                    "threads_per_block": block_dim,
                    "value_kind": primitive.value_kind.value,
                }
            )
        else:
            assert plan.topology is not None
            factory_kwargs.update(
                {
                    "threads_in_warp": plan.topology.logical_width,
                    "threads_per_block": block_dim,
                }
            )
            valid_items = plan.call.operation.valid_items
            if valid_items.kind is not BindingKind.OMITTED:
                factory_kwargs["valid_items"] = self._runtime_valid_items(
                    statements,
                    inst,
                    valid_items,
                    bound.arguments.get("valid_items"),
                )

        if initial_binding.kind is not BindingKind.OMITTED:
            factory_kwargs["initial_value"] = self._planned_argument(
                initial_binding,
                bound.arguments.get("initial_value"),
            )
        aggregate_output = bound.arguments.get("aggregate_output")
        if not self._context.is_none(aggregate_output):
            aggregate_name = (
                "block_aggregate"
                if plan.target is GroupLoweringTarget.CUB_BLOCK
                else "warp_aggregate"
            )
            factory_kwargs[aggregate_name] = aggregate_output
        temp_storage = bound.arguments.get("temp_storage")
        if not self._context.is_none(temp_storage):
            factory_kwargs["temp_storage"] = temp_storage

        statements.extend(
            self._context.rewrite_call(
                inst,
                lowering_plan=plan,
                factory=factory,
                args=runtime_args,
                kwargs=factory_kwargs,
                return_alias=result_payload,
            )
        )
        return statements


def _lower_registered_scan(
    context: GroupPlanningContext,
    *args: Any,
    **kwargs: Any,
) -> list[Any]:
    return _ScanPlanning(context)._lower_scan(*args, **kwargs)


def _validate_registered_common_arguments(
    context: GroupPlanningContext,
    operation: str,
    bound: inspect.BoundArguments,
) -> None:
    _ScanPlanning(context)._validate_common_arguments(operation, bound)


for _operation in (
    "scan",
    "exclusive_scan",
    "inclusive_scan",
    "exclusive_sum",
    "inclusive_sum",
):
    register_group_primitive(
        _operation,
        lower=_lower_registered_scan,
        results=(GroupResultSource("value", "value"),),
        validate_common_arguments=_validate_registered_common_arguments,
    )
del _operation

_BLOCK_REWRITE_KWARGS = frozenset(
    {
        "algorithm",
        "block_aggregate",
        "dtype",
        "initial_value",
        "items_per_thread",
        "mode",
        "scan_op",
        "threads_per_block",
        "value_kind",
    }
)
for _operation, _runtime_counts in (
    ("block_scan_scalar", frozenset({1, 2, 3})),
    ("block_scan_array", frozenset({2, 3, 4})),
):
    register_rewrite_operation(
        _operation,
        RewriteOperationSpec(
            factory_namespaces=frozenset({"block"}),
            dtype_factory_kwargs=frozenset({"dtype"}),
            runtime_arg_counts=_runtime_counts,
            runtime_factory_kwargs=("initial_value", "block_aggregate"),
            runtime_factory_kw_prerequisites=(),
            allowed_factory_kwargs=_BLOCK_REWRITE_KWARGS,
            required_factory_kwargs=frozenset(
                {
                    "dtype",
                    "items_per_thread",
                    "mode",
                    "threads_per_block",
                    "value_kind",
                }
            ),
            accepts_temp_storage=True,
            scalar_binding_kwargs=frozenset({"initial_value"}),
            runtime_offset_kwarg=None,
            infer_payload=infer_scan_payload,
        ),
    )
del _operation, _runtime_counts

register_rewrite_operation(
    "warp_scan",
    RewriteOperationSpec(
        factory_namespaces=frozenset({"warp"}),
        dtype_factory_kwargs=frozenset({"dtype"}),
        runtime_arg_counts=frozenset({1, 2, 3, 4}),
        runtime_factory_kwargs=("initial_value", "valid_items", "warp_aggregate"),
        runtime_factory_kw_prerequisites=(),
        allowed_factory_kwargs=frozenset(
            {
                "dtype",
                "initial_value",
                "mode",
                "scan_op",
                "threads_in_warp",
                "threads_per_block",
                "valid_items",
                "warp_aggregate",
            }
        ),
        required_factory_kwargs=frozenset(
            {"dtype", "mode", "threads_in_warp", "threads_per_block"}
        ),
        accepts_temp_storage=False,
        scalar_binding_kwargs=frozenset({"initial_value", "valid_items"}),
        runtime_offset_kwarg=None,
        infer_payload=infer_scan_payload,
        validate_runtime_controls=validate_scan_runtime_controls,
    ),
)


__all__: tuple[str, ...] = ()
