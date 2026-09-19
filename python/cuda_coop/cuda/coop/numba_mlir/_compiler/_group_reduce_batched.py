# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan independent batches and their distributed warp result payloads."""

from cuda.coop._core import (
    make_group_primitive_call,
    plan_group_primitive,
    resolve_thread_group,
)
from cuda.coop._core.group.reduce_batched import GroupReduceBatchedSemantics
from cuda.coop._core.warp.reduce_batched import WarpReduceBatchedSemantics

from .._lowering import _reduce_batched
from ._group_planner_support import _PAYLOAD_DTYPE_LIKE, GroupRewriteError
from ._operations import (
    GroupResultSource,
    RewriteOperationSpec,
    register_group_primitive,
    register_rewrite_operation,
)
from ._parameters import _validate_common_numeric_dtype
from ._rewrite_reduce_batched import infer_reduce_batched_payload


def _result_extent(context, bound):
    batches = context.array_extent(bound.arguments["value"])
    if batches is None:
        return None
    group = context.constant(bound.arguments["group"])
    group = resolve_thread_group(group, context.launch).require_supported()
    width = group.static_size
    if width is None:
        return None
    return (batches + width - 1) // width


def _lower_reduce_batched(context, inst, *, operation, group, bound, is_common_root):
    value = bound.arguments["value"]
    if not context.is_array(operation, value):
        raise TypeError(
            "reduce_batched requires a fixed-size payload of independent batches"
        )
    if is_common_root and not context.is_thread_data(operation, "value", value):
        raise TypeError("cuda.coop.reduce_batched requires a ThreadData payload")
    batches = context.array_extent(value)
    if batches is None:
        raise GroupRewriteError("reduce_batched requires a static batch count")
    dtype = context.dtype(value)
    if dtype is None:
        dtype = context.payload_write_dtype(value)
    if dtype is None:
        raise GroupRewriteError("reduce_batched could not infer value dtype")
    dtype = _validate_common_numeric_dtype(
        dtype, operation=operation, parameter="value"
    )
    context.record_thread_data_dtype(value, dtype)
    output_layout = context.validate_common_selector(
        operation,
        "output_layout",
        bound.arguments["output_layout"],
        {"striped", "blocked"},
    )
    binary_op = context.constant(bound.arguments["binary_op"])
    reduce_operator = _reduce_batched.reduction_operator(
        binary_op, dtype, is_common_root=is_common_root
    )
    semantics = GroupReduceBatchedSemantics(
        WarpReduceBatchedSemantics(dtype, batches, reduce_operator, output_layout)
    )
    plan = plan_group_primitive(
        make_group_primitive_call(group, semantics), context.launch
    ).require_supported()
    width = plan.topology.logical_width
    statements = []
    result = context.typed_payload_like(
        statements,
        scope=inst.target.scope,
        loc=inst.loc,
        stem="reduce_batched_result",
        prototype=value,
        is_array=True,
        dtype_policy=_PAYLOAD_DTYPE_LIKE,
        items_per_thread=(batches + width - 1) // width,
    )
    context.record_thread_data_dtype(result, dtype)
    statements.extend(
        context.rewrite_call(
            inst,
            lowering_plan=plan,
            factory=_reduce_batched.reduce_batched,
            args=[value, result],
            kwargs={
                "dtype": dtype,
                "threads_per_block": plan.participation.exact_block_dim,
                "threads_in_warp": width,
                "batches": batches,
                "binary_op": binary_op,
                "output_layout": output_layout,
            },
            return_alias=result,
        )
    )
    return statements


register_group_primitive(
    "reduce_batched",
    lower=_lower_reduce_batched,
    results=(GroupResultSource("value", None, extent_resolver=_result_extent),),
)

register_rewrite_operation(
    "reduce_batched",
    RewriteOperationSpec(
        factory_namespaces=frozenset({"warp"}),
        dtype_factory_kwargs=frozenset({"dtype"}),
        runtime_arg_counts=frozenset({2}),
        runtime_factory_kwargs=(),
        runtime_factory_kw_prerequisites=(),
        allowed_factory_kwargs=frozenset(
            {
                "dtype",
                "threads_per_block",
                "threads_in_warp",
                "batches",
                "binary_op",
                "output_layout",
            }
        ),
        required_factory_kwargs=frozenset(
            {"dtype", "threads_per_block", "threads_in_warp", "batches"}
        ),
        accepts_temp_storage=False,
        scalar_binding_kwargs=frozenset(),
        runtime_offset_kwarg=None,
        infer_payload=infer_reduce_batched_payload,
    ),
)

__all__: tuple[str, ...] = ()
