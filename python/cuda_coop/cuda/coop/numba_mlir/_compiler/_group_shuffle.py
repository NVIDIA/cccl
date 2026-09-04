# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete-block Shuffle IR planning."""

from __future__ import annotations

from numbers import Integral

from numba_cuda_mlir import types

from cuda.coop._core import (
    ArgumentBinding,
    BindingKind,
    BlockShuffleMode,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupShuffleSemantics,
    make_block_shuffle_semantics,
    make_group_primitive_call,
    plan_group_primitive,
)

from ._group_planner_support import (
    _PAYLOAD_DTYPE_LIKE,
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
from ._rewrite_shuffle import (
    infer_shuffle_array_payload,
    infer_shuffle_scalar_payload,
    validate_shuffle_array_runtime_controls,
    validate_shuffle_scalar_runtime_controls,
)

_ARRAY_MODES = frozenset({BlockShuffleMode.UP.value, BlockShuffleMode.DOWN.value})
_SCALAR_MODES = frozenset(
    {BlockShuffleMode.OFFSET.value, BlockShuffleMode.ROTATE.value}
)
_ALL_MODES = _ARRAY_MODES | _SCALAR_MODES


def _mode_token(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(
            "cuda.coop.numba_mlir.shuffle mode must be a compile-time string"
        )
    token = value.strip().lower().replace("-", "_")
    if token not in _ALL_MODES:
        choices = ", ".join(sorted(_ALL_MODES))
        raise ValueError(f"cuda.coop.numba_mlir.shuffle mode must be one of: {choices}")
    return token


class _ShufflePlanning:
    """Family-local Shuffle semantics over the declared planning context."""

    def __init__(self, context: GroupPlanningContext) -> None:
        self._context = context

    def _validate_common_arguments(
        self,
        operation: str,
        bound: inspect.BoundArguments,
    ) -> None:
        bound.arguments["mode"] = self._context.validate_common_selector(
            operation,
            "mode",
            bound.arguments["mode"],
            _ARRAY_MODES,
        )

    @staticmethod
    def _provider(plan: GroupLoweringPlan, *, is_array: bool):
        if (
            plan.target is not GroupLoweringTarget.CUB_BLOCK
            or plan.provenance is None
            or plan.provenance.header != "cub/block/block_shuffle.cuh"
            or plan.provenance.cpp_class != "cub::BlockShuffle"
        ):
            provenance = (
                None if plan.provenance is None else plan.provenance.semantic_key
            )
            raise GroupRewriteError(
                "cuda.coop.numba_mlir.shuffle received unknown CUB provider "
                f"provenance {provenance!r}"
            )
        if is_array:
            from .._lowering._shuffle import shuffle_array

            return shuffle_array
        from .._lowering._shuffle import shuffle_scalar

        return shuffle_scalar

    @staticmethod
    def _planned_argument(binding: ArgumentBinding, runtime_value: Any) -> Any:
        return runtime_value if binding.kind is BindingKind.RUNTIME else binding

    def _plan(
        self,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> tuple[GroupLoweringPlan, bool]:
        value = bound.arguments["value"]
        is_array = self._context.is_array("shuffle", value)
        mode = _mode_token(self._context.constant(bound.arguments["mode"]))
        distance_value = bound.arguments["distance"]

        if is_common_root:
            if not self._context.is_thread_data("shuffle", "value", value):
                raise TypeError(
                    "cuda.coop.shuffle requires a fixed-size ThreadData payload; "
                    "use cuda.coop.numba_mlir for backend-qualified scalar or "
                    "local-array shuffle support"
                )
            if mode not in _ARRAY_MODES:
                raise ValueError(
                    "cuda.coop.shuffle mode must be 'down' or 'up'; use "
                    "cuda.coop.numba_mlir for backend-qualified scalar "
                    "Offset or Rotate"
                )

        if is_array:
            if mode not in _ARRAY_MODES:
                raise ValueError(
                    "cuda.coop.numba_mlir.shuffle array values support only "
                    "'up' and 'down' modes"
                )
            resolved, distance = self._context.try_static_scalar(distance_value)
            if (
                not resolved
                or isinstance(distance, bool)
                or not isinstance(distance, Integral)
                or int(distance) != 1
            ):
                raise ValueError(
                    "cuda.coop.numba_mlir.shuffle array Up and Down use a "
                    "compile-time unit distance"
                )
            items_per_thread = self._context.array_extent(value)
            if items_per_thread is None:
                raise GroupRewriteError(
                    "cuda.coop.numba_mlir.shuffle requires a static "
                    "items_per_thread extent"
                )
            distance_binding = ArgumentBinding.omitted()
        else:
            if mode not in _SCALAR_MODES:
                raise ValueError(
                    "cuda.coop.numba_mlir.shuffle scalar values support only "
                    "'offset' and 'rotate' modes"
                )
            items_per_thread = None
            distance_binding = self._context.planning_binding(distance_value)
            if distance_binding.kind is BindingKind.OMITTED:
                raise ValueError(
                    "cuda.coop.numba_mlir.shuffle scalar values require distance"
                )
            if distance_binding.kind is BindingKind.STATIC:
                distance = distance_binding.value
                if isinstance(distance, bool) or not isinstance(distance, Integral):
                    raise TypeError(
                        "cuda.coop.numba_mlir.shuffle distance must be an integer"
                    )
                if mode == BlockShuffleMode.ROTATE.value:
                    block_threads = self._context.launch.exact_block_threads
                    if block_threads is not None and block_threads <= 1:
                        raise ValueError(
                            "cuda.coop.numba_mlir.shuffle rotate requires at "
                            "least two threads per block"
                        )
                    if (
                        block_threads is not None
                        and not 1 <= int(distance) < block_threads
                    ):
                        raise ValueError(
                            "cuda.coop.numba_mlir.shuffle rotate distance must "
                            "satisfy 1 <= distance < block_threads"
                        )
            else:
                distance_dtype = self._context.dtype(distance_value)
                if distance_dtype is None:
                    raise GroupRewriteError(
                        "cuda.coop.numba_mlir.shuffle could not infer the "
                        "runtime distance dtype"
                    )
                _validate_runtime_integer_dtype(
                    distance_dtype,
                    operation="shuffle",
                    parameter="distance",
                )
            if (
                distance_binding.kind is BindingKind.RUNTIME
                and mode == BlockShuffleMode.ROTATE.value
                and self._context.launch.exact_block_threads is not None
                and self._context.launch.exact_block_threads <= 1
            ):
                raise ValueError(
                    "cuda.coop.numba_mlir.shuffle rotate requires at least "
                    "two threads per block"
                )

        dtype = self._context.dtype(value)
        if dtype is None and is_array:
            dtype = self._context.payload_write_dtype(value)
        if dtype is None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir.shuffle could not infer value dtype"
            )
        dtype = _validate_common_numeric_dtype(
            dtype,
            operation="shuffle",
            parameter="value",
        )
        semantics = GroupShuffleSemantics(
            make_block_shuffle_semantics(
                dtype=dtype,
                mode=mode,
                items_per_thread=items_per_thread,
                distance=distance_binding,
            )
        )
        plan = plan_group_primitive(
            make_group_primitive_call(group, semantics),
            self._context.launch,
        ).require_supported()
        return plan, is_array

    def _lower_shuffle(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        if operation != "shuffle":
            raise GroupRewriteError(
                f"Shuffle planner received unexpected operation {operation!r}"
            )
        plan, is_array = self._plan(
            group=group,
            bound=bound,
            is_common_root=is_common_root,
        )
        primitive = plan.call.operation.primitive
        block_dim = plan.participation.exact_block_dim
        assert block_dim is not None
        factory = self._provider(plan, is_array=is_array)
        statements: list[Any] = []
        factory_kwargs: dict[str, Any] = {
            "dtype": primitive.dtype,
            "threads_per_block": block_dim,
            "mode": primitive.mode.value,
        }
        if is_array:
            factory_kwargs["items_per_thread"] = primitive.items_per_thread
        else:
            distance = self._planned_argument(
                primitive.distance,
                bound.arguments["distance"],
            )
            if primitive.distance.kind is BindingKind.RUNTIME:
                scope = inst.target.scope
                loc = inst.loc
                cast = self._context.value_var(
                    statements,
                    scope=scope,
                    loc=loc,
                    stem="shuffle_distance_type",
                    value=types.int64,
                )
                cast_distance = self._context.new_var(
                    scope,
                    loc,
                    "shuffle_distance_i64",
                )
                statements.append(
                    ir.Assign(
                        ir.Expr.call(
                            cast,
                            [bound.arguments["distance"]],
                            (),
                            loc,
                        ),
                        cast_distance,
                        loc,
                    )
                )
                distance = cast_distance
            factory_kwargs["distance"] = distance

        if is_array:
            scope = inst.target.scope
            loc = inst.loc
            value = self._context.value_var(
                statements,
                scope=scope,
                loc=loc,
                stem="shuffle_value",
                value=bound.arguments["value"],
            )
            result = self._context.typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem="shuffle_result",
                prototype=value,
                is_array=True,
                dtype_policy=_PAYLOAD_DTYPE_LIKE,
                items_per_thread=primitive.items_per_thread,
            )
            statements.extend(
                self._context.rewrite_call(
                    inst,
                    lowering_plan=plan,
                    factory=factory,
                    args=[value, result],
                    kwargs=factory_kwargs,
                    return_alias=result,
                )
            )
            return statements

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


def _lower_registered_shuffle(
    context: GroupPlanningContext,
    *args: Any,
    **kwargs: Any,
) -> list[Any]:
    return _ShufflePlanning(context)._lower_shuffle(*args, **kwargs)


def _validate_registered_common_arguments(
    context: GroupPlanningContext,
    operation: str,
    bound: inspect.BoundArguments,
) -> None:
    _ShufflePlanning(context)._validate_common_arguments(operation, bound)


register_group_primitive(
    "shuffle",
    lower=_lower_registered_shuffle,
    results=(GroupResultSource("value", "value"),),
    validate_common_arguments=_validate_registered_common_arguments,
)

_SCALAR_REWRITE_KWARGS = frozenset(
    {
        "distance",
        "dtype",
        "mode",
        "threads_per_block",
    }
)
register_rewrite_operation(
    "shuffle_scalar",
    RewriteOperationSpec(
        factory_namespaces=frozenset({"block"}),
        dtype_factory_kwargs=frozenset({"dtype"}),
        runtime_arg_counts=frozenset({1, 2}),
        runtime_factory_kwargs=("distance",),
        runtime_factory_kw_prerequisites=(),
        allowed_factory_kwargs=_SCALAR_REWRITE_KWARGS,
        required_factory_kwargs=frozenset({"distance", "dtype", "threads_per_block"}),
        accepts_temp_storage=False,
        scalar_binding_kwargs=frozenset({"distance"}),
        runtime_offset_kwarg=None,
        infer_payload=infer_shuffle_scalar_payload,
        validate_runtime_controls=validate_shuffle_scalar_runtime_controls,
    ),
)

_ARRAY_REWRITE_KWARGS = frozenset(
    {
        "dtype",
        "items_per_thread",
        "mode",
        "threads_per_block",
    }
)
register_rewrite_operation(
    "shuffle_array",
    RewriteOperationSpec(
        factory_namespaces=frozenset({"block"}),
        dtype_factory_kwargs=frozenset({"dtype"}),
        runtime_arg_counts=frozenset({2}),
        runtime_factory_kwargs=(),
        runtime_factory_kw_prerequisites=(),
        allowed_factory_kwargs=_ARRAY_REWRITE_KWARGS,
        required_factory_kwargs=frozenset(
            {"dtype", "items_per_thread", "threads_per_block"}
        ),
        accepts_temp_storage=False,
        scalar_binding_kwargs=frozenset(),
        runtime_offset_kwarg=None,
        infer_payload=infer_shuffle_array_payload,
        validate_runtime_controls=validate_shuffle_array_runtime_controls,
    ),
)


__all__: tuple[str, ...] = ()
