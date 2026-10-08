# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan independent reductions of payload slots across a selected warp.

Each input slot is one batch with one item from each lane. CUB distributes
its aggregates across the warp, so the result extent is ceil(batches / width)
instead of the input extent. This rewrite resolves that shape, allocates a
separate payload, and preserves the item dtype for later cooperative calls.
Shared planning validates the warp and owns its scratch contract.

The module also registers the follow-up rewrite for the generated provider
call. That call takes the input and result arrays, requires static dtype,
block shape, warp width, and batch count, and accepts no temp_storage. Its
payload hook checks the two array extents.
"""

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
    RewriteOperationSpecification,
    register_group_primitive,
    register_rewrite_operation,
)
from ._parameters import _validate_common_numeric_dtype
from ._rewrite_reduce_batched import infer_reduce_batched_payload


def _result_extent(context, bound):
    """Infer result capacity from the batch count and resolved warp width.

    Each input slot represents one independent reduction across the
    warp. If there are ``batches`` slots and ``width`` lanes,
    distributing the results needs ``ceil(batches / width)`` slots per
    lane. That output item count is the extent returned here.

    When a later call asks for the extent of this call's result, the group
    planner traces back to this call and uses this value before the call is
    rewritten and its output is allocated. Return None while the payload
    extent or selected group's static size is unknown; do not guess the
    input shape.

    Parameters
    ----------
    context : GroupPlanningContext
        Access to launch dimensions, constant controls, payload
        facts, and IR builders for this group-planning attempt.
    bound : inspect.BoundArguments
        Public call arguments after signature binding and default
        application. Runtime values remain IR variables; selectors
        are resolved through the context.
    """

    batches = context.array_extent(bound.arguments["value"])
    if batches is None:
        return None
    group = context.constant(bound.arguments["group"])
    group = resolve_thread_group(group, context.launch).require_supported()
    width = group.static_size
    if width is None:
        return None
    return (batches + width - 1) // width


def _lower_reduce_batched(
    context, inst, *, operation, group, bound, is_common_root
):
    """Rewrite a batch reduction to a native input/output array call.

    The group-operation registry calls this during whole-function
    planning, before ordinary typing. It returns pending replacement IR;
    the owning planner installs it only after the function's group
    operations validate.

    Require fixed array input and infer its item type. Common calls require
    ThreadData; qualified calls also accept local arrays. Resolve the static
    layout and operator, then ask shared planning to validate the warp.

    Allocate ceil(batches / width) slots per lane with the input dtype and
    record that shape for later uses. The provider writes a separate result,
    so the input stays intact. Layout assigns batch indices to result slots;
    slots with no batch remain unspecified. Scratch comes from the plan and
    has no caller-supplied descriptor in this API.

    Parameters
    ----------
    context : GroupPlanningContext
        Access to launch dimensions, constant controls, payload
        facts, and IR builders for this group-planning attempt.
    inst : ir.Assign
        Original public call assignment. Its target, scope, and
        source location identify the replacement result and
        generated temporaries.
    operation : str
        Canonical public operation name, used in diagnostics and
        generated temporary names.
    group : ThreadGroup
        Group resolved by the owning planner against the current
        kernel launch.
    bound : inspect.BoundArguments
        Public call arguments after signature binding and default
        application. Runtime values remain IR variables; selectors
        are resolved through the context.
    is_common_root : bool
        Whether the call came through the common ``cuda.coop`` API
        and must satisfy its narrower operand and selector rules.
    """

    value = bound.arguments["value"]
    if not context.is_array(operation, value):
        raise TypeError(
            "reduce_batched requires a fixed-size "
            "payload of independent batches"
        )
    if is_common_root and not context.is_thread_data(operation, "value", value):
        raise TypeError(
            "cuda.coop.reduce_batched requires a ThreadData payload"
        )
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
    from .._lowering._core import NumbaMlirCoreAdapter

    adapter = NumbaMlirCoreAdapter()
    semantics = GroupReduceBatchedSemantics(
        WarpReduceBatchedSemantics(
            adapter.core_dtype(dtype), batches, reduce_operator, output_layout
        )
    )
    plan = plan_group_primitive(
        make_group_primitive_call(group, semantics), context.launch
    ).require_supported()
    assert plan.participation is not None
    assert plan.topology is not None
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
    RewriteOperationSpecification(
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
