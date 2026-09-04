# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Hierarchy-aware Reduce and Sum IR planning."""

from __future__ import annotations

from numba_cuda_mlir import types

from cuda.coop._core import (
    ArgumentBinding,
    BindingKind,
    CxxOperator,
    Dependency,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupReduceSemantics,
    PythonOperator,
    SynchronizationScope,
    make_group_primitive_call,
    make_reduce_semantics,
    plan_group_primitive,
)

from .._semantic import _normalize_numba_callable
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
from ._parameters import (
    _validate_common_numeric_dtype,
    _validate_runtime_integer_dtype,
)
from ._rewrite_reduce import (
    infer_reduce_payload,
    validate_block_reduce_runtime_controls,
    validate_warp_reduce_runtime_controls,
)

_PORTABLE_ALGORITHMS = frozenset(
    {"raking", "raking_commutative_only", "warp_reductions"}
)
_BUILTIN_OPERATOR_CPP = {
    "multiplies": "::cuda::std::multiplies<T>",
    "min": "::cuda::minimum<T>",
    "max": "::cuda::maximum<T>",
    "bit_and": "::cuda::std::bit_and<T>",
    "bit_or": "::cuda::std::bit_or<T>",
    "bit_xor": "::cuda::std::bit_xor<T>",
}


class _ReducePlanning:
    """Family-local Reduce semantics over the declared planning context."""

    def __init__(self, context: GroupPlanningContext) -> None:
        self._context = context

    def _validate_common_arguments(
        self,
        operation: str,
        bound: inspect.BoundArguments,
    ) -> None:
        bound.arguments["algorithm"] = self._context.validate_common_selector(
            operation,
            "algorithm",
            bound.arguments["algorithm"],
            _PORTABLE_ALGORITHMS,
            allow_none=True,
        )

    @staticmethod
    def _operator(
        operation: str,
        binary_op: Any,
        *,
        dtype: Any,
        is_common_root: bool,
    ) -> tuple[str, str, CxxOperator | PythonOperator | None]:
        from .._lowering._reduce import (
            normalize_reduce_operation,
            validate_reduce_operator_dtype,
        )

        if operation == "sum":
            validate_reduce_operator_dtype("sum", dtype)
            return "sum", "sum", None
        try:
            canonical = normalize_reduce_operation(binary_op)
        except NotImplementedError:
            if is_common_root or not callable(binary_op):
                raise
            return (
                "reduce",
                "callback",
                PythonOperator(
                    ret_dtype=Dependency("T"),
                    arg_dtypes=(Dependency("T"), Dependency("T")),
                    op=_normalize_numba_callable(binary_op),
                    name="binary_op",
                ),
            )
        validate_reduce_operator_dtype(canonical, dtype)
        if canonical == "sum":
            return "sum", canonical, None
        return (
            "reduce",
            canonical,
            CxxOperator(
                cpp=_BUILTIN_OPERATOR_CPP[canonical],
                dtype=Dependency("T"),
                name="binary_op",
            ),
        )

    @staticmethod
    def _provider(plan: GroupLoweringPlan, *, operator_kind: str):
        if plan.provenance is None or plan.topology is None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir.reduce requires provider provenance and topology"
            )
        provenance = plan.provenance
        if (
            plan.target is GroupLoweringTarget.CUDAX_GROUP
            and provenance.library == "CUDAX"
            and provenance.header == "cuda/experimental/coop.cuh"
        ):
            from .._lowering import _reduce

            return {
                SynchronizationScope.NONE: _reduce.group_reduce_none,
                SynchronizationScope.WARP: _reduce.group_reduce_warp,
                SynchronizationScope.BLOCK: _reduce.group_reduce_block,
                SynchronizationScope.GROUP: _reduce.group_reduce_group,
            }[plan.topology.execution_scope]
        if (
            plan.target is GroupLoweringTarget.CUB_BLOCK
            and provenance.library == "CUB"
            and provenance.header == "cub/block/block_reduce.cuh"
            and provenance.cpp_class == "cub::BlockReduce"
        ):
            from .._lowering import _reduce

            if operator_kind == "sum":
                return _reduce.sum
            if operator_kind == "callback":
                return _reduce.reduce
            return _reduce.block_reduce_builtin
        if (
            plan.target is GroupLoweringTarget.CUB_WARP
            and provenance.library == "CUB"
            and provenance.header == "cub/warp/warp_reduce.cuh"
            and provenance.cpp_class == "cub::WarpReduce"
        ):
            from .._lowering import _reduce

            if operator_kind == "sum":
                return _reduce.warp_sum
            if operator_kind == "callback":
                return _reduce.warp_reduce
            return _reduce.warp_reduce_builtin
        raise GroupRewriteError(
            "cuda.coop.numba_mlir.reduce received unknown provider provenance "
            f"{provenance.semantic_key!r}"
        )

    def _plan(
        self,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> tuple[GroupLoweringPlan, str, Any, bool]:
        if operation not in {"reduce", "sum"}:
            raise GroupRewriteError(
                f"Reduce planner received unexpected operation {operation!r}"
            )
        broadcast = self._context.constant(bound.arguments["broadcast"])
        if not isinstance(broadcast, bool):
            raise TypeError(
                f"cuda.coop.numba_mlir.{operation} broadcast must be a "
                "compile-time bool"
            )

        value = bound.arguments["value"]
        is_array = self._context.is_array(operation, value)
        if (
            is_common_root
            and is_array
            and not self._context.is_thread_data(operation, "value", value)
        ):
            raise TypeError(
                f"cuda.coop.{operation} accepts only a scalar or fixed-size "
                "ThreadData value payload; use cuda.coop.numba_mlir for "
                "backend-qualified local arrays"
            )
        items_per_thread = 1
        if is_array:
            extent = self._context.array_extent(value)
            if extent is None:
                raise GroupRewriteError(
                    f"cuda.coop.numba_mlir.{operation} requires a static "
                    "items_per_thread extent"
                )
            items_per_thread = extent

        dtype = self._context.dtype(value)
        if dtype is None and is_array:
            dtype = self._context.payload_write_dtype(value)
        if dtype is None:
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} could not infer value dtype"
            )
        dtype = _validate_common_numeric_dtype(
            dtype,
            operation=operation,
            parameter="value",
        )

        binary_op = (
            None
            if operation == "sum"
            else self._context.constant(bound.arguments["binary_op"])
        )
        semantic_operation, operator_kind, reduce_operator = self._operator(
            operation,
            binary_op,
            dtype=dtype,
            is_common_root=is_common_root,
        )

        valid_items = self._context.planning_binding(bound.arguments["valid_items"])
        if valid_items.kind is BindingKind.RUNTIME:
            valid_dtype = self._context.dtype(bound.arguments["valid_items"])
            if valid_dtype is None:
                raise GroupRewriteError(
                    f"cuda.coop.numba_mlir.{operation} could not infer the "
                    "runtime valid_items dtype"
                )
            _validate_runtime_integer_dtype(
                valid_dtype,
                operation=operation,
                parameter="valid_items",
            )
        algorithm = (
            None
            if self._context.is_none(bound.arguments["algorithm"])
            else self._context.constant(bound.arguments["algorithm"])
        )
        semantics = GroupReduceSemantics(
            make_reduce_semantics(
                dtype=dtype,
                items_per_thread=items_per_thread,
                operation=semantic_operation,
                value_kind="array" if is_array else "scalar",
                reduce_operator=reduce_operator,
                valid_items=valid_items,
            ),
            broadcast=broadcast,
            cub_algorithm=algorithm,
        )
        plan = plan_group_primitive(
            make_group_primitive_call(group, semantics),
            self._context.launch,
        ).require_supported()
        return plan, operator_kind, binary_op, is_array

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
            stem="reduce_valid_items_type",
            value=types.int64,
        )
        result = self._context.new_var(scope, loc, "reduce_valid_items_i64")
        statements.append(
            ir.Assign(
                ir.Expr.call(cast, [value], (), loc),
                result,
                loc,
            )
        )
        return result

    def _lower_reduce(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        plan, operator_kind, binary_op, is_array = self._plan(
            operation=operation,
            group=group,
            bound=bound,
            is_common_root=is_common_root,
        )
        primitive = plan.call.operation.primitive
        factory = self._provider(plan, operator_kind=operator_kind)
        block_dim = plan.participation.exact_block_dim
        assert block_dim is not None
        statements: list[Any] = []
        factory_kwargs: dict[str, Any] = {"dtype": primitive.dtype}
        if plan.target is GroupLoweringTarget.CUDAX_GROUP:
            factory_kwargs.update(
                {
                    "group": plan.resolved_group,
                    "binary_op": None if operator_kind == "sum" else operator_kind,
                    "broadcast": plan.call.operation.broadcast,
                    "items_per_thread": primitive.items_per_thread,
                    "value_kind": primitive.value_kind.value,
                }
            )
        elif plan.target is GroupLoweringTarget.CUB_BLOCK:
            factory_kwargs.update(
                {
                    "threads_per_block": block_dim,
                    "algorithm": plan.call.operation.cub_algorithm,
                    "items_per_thread": primitive.items_per_thread,
                    "value_kind": primitive.value_kind.value,
                }
            )
            if operator_kind != "sum":
                factory_kwargs["binary_op"] = binary_op
            if primitive.valid_items.kind is not BindingKind.OMITTED:
                factory_kwargs["num_valid"] = self._runtime_valid_items(
                    statements,
                    inst,
                    primitive.valid_items,
                    bound.arguments["valid_items"],
                )
        else:
            assert plan.topology is not None
            factory_kwargs.update(
                {
                    "threads_per_block": block_dim,
                    "threads_in_warp": plan.topology.logical_width,
                }
            )
            if operator_kind != "sum":
                factory_kwargs["binary_op"] = binary_op
            if primitive.valid_items.kind is not BindingKind.OMITTED:
                factory_kwargs["valid_items"] = self._runtime_valid_items(
                    statements,
                    inst,
                    primitive.valid_items,
                    bound.arguments["valid_items"],
                )

        statements.extend(
            self._context.rewrite_call(
                inst,
                lowering_plan=plan,
                factory=factory,
                args=[bound.arguments["value"]],
                kwargs=factory_kwargs,
            )
        )
        return statements


def _lower_registered_reduce(
    context: GroupPlanningContext,
    *args: Any,
    **kwargs: Any,
) -> list[Any]:
    return _ReducePlanning(context)._lower_reduce(*args, **kwargs)


def _validate_registered_common_arguments(
    context: GroupPlanningContext,
    operation: str,
    bound: inspect.BoundArguments,
) -> None:
    _ReducePlanning(context)._validate_common_arguments(operation, bound)


for _operation in ("reduce", "sum"):
    register_group_primitive(
        _operation,
        lower=_lower_registered_reduce,
        results=(GroupResultSource("value", None),),
        validate_common_arguments=_validate_registered_common_arguments,
    )
del _operation

_BLOCK_REWRITE_KWARGS = frozenset(
    {
        "algorithm",
        "binary_op",
        "dtype",
        "items_per_thread",
        "num_valid",
        "threads_per_block",
        "value_kind",
    }
)
for _operation in (
    "block_sum",
    "block_reduce_builtin",
    "block_reduce_callback",
):
    register_rewrite_operation(
        _operation,
        RewriteOperationSpec(
            factory_namespaces=frozenset({"block"}),
            dtype_factory_kwargs=frozenset({"dtype"}),
            runtime_arg_counts=frozenset({1, 2}),
            runtime_factory_kwargs=("num_valid",),
            runtime_factory_kw_prerequisites=(),
            allowed_factory_kwargs=_BLOCK_REWRITE_KWARGS,
            required_factory_kwargs=frozenset({"dtype", "threads_per_block"}),
            accepts_temp_storage=False,
            scalar_binding_kwargs=frozenset({"num_valid"}),
            runtime_offset_kwarg=None,
            infer_payload=infer_reduce_payload,
            validate_runtime_controls=validate_block_reduce_runtime_controls,
        ),
    )
del _operation

_WARP_REWRITE_KWARGS = frozenset(
    {
        "binary_op",
        "dtype",
        "threads_in_warp",
        "threads_per_block",
        "valid_items",
    }
)
for _operation in ("warp_sum", "warp_reduce_builtin", "warp_reduce_callback"):
    register_rewrite_operation(
        _operation,
        RewriteOperationSpec(
            factory_namespaces=frozenset({"warp"}),
            dtype_factory_kwargs=frozenset({"dtype"}),
            runtime_arg_counts=frozenset({1, 2}),
            runtime_factory_kwargs=("valid_items",),
            runtime_factory_kw_prerequisites=(),
            allowed_factory_kwargs=_WARP_REWRITE_KWARGS,
            required_factory_kwargs=frozenset(
                {"dtype", "threads_in_warp", "threads_per_block"}
            ),
            accepts_temp_storage=False,
            scalar_binding_kwargs=frozenset({"valid_items"}),
            runtime_offset_kwarg=None,
            infer_payload=infer_reduce_payload,
            validate_runtime_controls=validate_warp_reduce_runtime_controls,
        ),
    )
del _operation

register_rewrite_operation(
    "group_reduce",
    RewriteOperationSpec(
        factory_namespaces=frozenset(
            {"cudax_block", "cudax_group", "cudax_none", "cudax_warp"}
        ),
        dtype_factory_kwargs=frozenset({"dtype"}),
        runtime_arg_counts=frozenset({1}),
        runtime_factory_kwargs=(),
        runtime_factory_kw_prerequisites=(),
        allowed_factory_kwargs=frozenset(
            {
                "_compile_context",
                "binary_op",
                "broadcast",
                "dtype",
                "group",
                "items_per_thread",
                "value_kind",
            }
        ),
        required_factory_kwargs=frozenset(
            {"broadcast", "dtype", "group", "items_per_thread", "value_kind"}
        ),
        accepts_temp_storage=False,
        scalar_binding_kwargs=frozenset(),
        runtime_offset_kwarg=None,
        infer_payload=infer_reduce_payload,
    ),
)


__all__: tuple[str, ...] = ()
