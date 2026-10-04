# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan block histograms and allocate an independent counter result.

Samples describe the input tile; counter dtype and bins_per_thread describe
the output. This rewrite validates both, boxes a scalar sample from a
qualified ``cuda.coop.numba_mlir`` call into a one-item array, and allocates a
fresh counter payload. Shared planning supplies the exact block and
bin-capacity constraints. The C++ provider fills the result in striped bin
order while preserving the samples.
"""

from dataclasses import replace

import numba_cuda_mlir.numba_cuda.types as numba_types

from cuda.coop._core import (
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block._common import normalize_positive_int
from cuda.coop._core.block.histogram import (
    normalize_histogram_algorithm,
    validate_histogram_dtype,
)
from cuda.coop._core.group.histogram import GroupHistogramSemantics

from .._thread_data import ThreadData
from ._group_planner_support import GroupRewriteError, ir
from ._operations import (
    GroupResultSource,
    RewriteOperationSpecification,
    register_group_primitive,
    register_rewrite_operation,
)
from ._parameters import normalize_dtype_param


def _extent(context, bound):
    """Resolve the static counter extent independently of sample shape.

    Here, extent is the counter slots owned by one thread. The
    registered result policy calls this when a later operation needs the
    Histogram output shape, and Histogram lowering uses it to allocate
    that same output. Sample items per thread do not determine counter
    capacity.

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

    return normalize_positive_int(
        "bins_per_thread", context.constant(bound.arguments["bins_per_thread"])
    )


def _infer_payload(context, inference):
    """Infer separate sample and counter shapes for the provider call.

    The provider takes an input array and an output array whose dtypes and
    extents can differ. Recover each dtype from array evidence or its factory
    keyword, validate the corresponding allowed type set, and record the type
    for later ThreadData uses. Never infer counter shape from sample shape.
    """

    for index, dtype_name, extent_name in (
        (0, "sample_dtype", "items_per_thread"),
        (1, "counter_dtype", "bins_per_thread"),
    ):
        value, specification = inference.array_candidate(index)
        if (
            value is None
            or specification is None
            or specification.items_per_thread is None
        ):
            raise GroupRewriteError(
                "histogram requires fixed-size per-thread payloads"
            )
        dtype = inference.inferred_array_dtype(value, specification)
        if dtype is None:
            dtype = inference.factory_value(dtype_name)
        dtype = normalize_dtype_param(dtype)
        validate_histogram_dtype(dtype, counter=index == 1)
        inference.infer_kwarg(dtype_name, dtype)
        inference.infer_kwarg(extent_name, specification.items_per_thread)
        context.record_thread_data_dtype(value, dtype)


def _lower_histogram(context, inst, *, operation, group, bound, is_common_root):
    """Rewrite a histogram call to fill a new counter payload.

    Common calls require ThreadData samples; qualified calls also accept fixed
    local arrays or scalars. Recover the sample type, resolve static counting
    options, and ask shared planning to validate the complete one-dimensional
    block and output capacity. Every sample contributes; there is no partial
    input count to pass to the provider.

    Box a scalar sample for the array ABI. Allocate bins_per_thread counters
    with the selected dtype, int32 by default, and record that independent
    result type. Return the statements that call the provider and bind the
    public result to this payload. The provider fills fresh striped counts and
    zero padding; scalar input still produces an array result.

    Explicit scratch replaces the plan's default storage ownership. Its
    reuse barrier follows auto_sync. The provider's internal barrier before
    reading shared counters is required for both scratch policies.
    """

    from .._lowering import _histogram
    from .._lowering._core import NumbaMlirCoreAdapter

    value = bound.arguments["samples"]
    if is_common_root and not context.is_thread_data(
        operation, "samples", value
    ):
        raise TypeError("cuda.coop.histogram samples must be ThreadData")
    is_array = context.is_array(operation, value)
    extent = context.array_extent(value) if is_array else 1
    if extent is None:
        raise GroupRewriteError("histogram samples require a static extent")
    dtype = context.dtype(value)
    if dtype is None:
        dtype = context.payload_write_dtype(value)
    if dtype is None:
        raise GroupRewriteError("histogram could not infer samples dtype")
    dtype = normalize_dtype_param(dtype)
    validate_histogram_dtype(dtype)
    context.record_thread_data_dtype(value, dtype)
    counter = (
        numba_types.int32
        if context.is_none(bound.arguments["counter_dtype"])
        else normalize_dtype_param(
            context.constant(bound.arguments["counter_dtype"])
        )
    )
    validate_histogram_dtype(counter, counter=True)
    bins = normalize_positive_int(
        "bins", context.constant(bound.arguments["bins"])
    )
    bins_per_thread = _extent(context, bound)
    algorithm = normalize_histogram_algorithm(
        context.constant(bound.arguments["algorithm"])
    )
    adapter = NumbaMlirCoreAdapter()
    semantics = GroupHistogramSemantics(
        sample_dtype=adapter.core_dtype(dtype),
        items_per_thread=extent,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=adapter.core_dtype(counter),
        algorithm=algorithm,
    )
    plan = plan_group_primitive(
        make_group_primitive_call(group, semantics), context.launch
    ).require_supported()
    participation = plan.participation
    assert participation is not None
    assert plan.temp_storage is not None
    assert plan.synchronization is not None
    storage = bound.arguments.get("temp_storage")
    if not context.is_none(storage):
        descriptor = context.temp_storage(storage)
        if descriptor is None:
            raise TypeError(
                "histogram temp_storage must "
                "resolve to a TempStorage descriptor"
            )
        size, alignment, auto_sync, sharing = descriptor
        plan = replace(
            plan,
            temp_storage=replace(
                plan.temp_storage,
                ownership=StorageOwnership.CALLER,
                exact_layout_required=True,
                requested_size_in_bytes=size,
                requested_alignment=alignment,
                auto_sync=auto_sync,
                sharing=sharing,
            ),
            synchronization=replace(
                plan.synchronization,
                storage_reuse_barrier=SynchronizationScope.BLOCK
                if auto_sync
                else SynchronizationScope.NONE,
            ),
        )
    statements = []
    scope, loc = inst.target.scope, inst.loc
    value = context.value_var(
        statements, scope=scope, loc=loc, stem="histogram_samples", value=value
    )
    samples, _ = context.box_group_operand(
        statements, operation=operation, value=value, scope=scope, loc=loc
    )
    constructor = context.value_var(
        statements,
        scope=scope,
        loc=loc,
        stem="histogram_payload",
        value=ThreadData,
    )
    count = context.value_var(
        statements,
        scope=scope,
        loc=loc,
        stem="histogram_extent",
        value=bins_per_thread,
    )
    counter_type = context.value_var(
        statements, scope=scope, loc=loc, stem="histogram_dtype", value=counter
    )
    output = context.new_var(scope, loc, "histogram_counts")
    statements.append(
        ir.Assign(
            ir.Expr.call(constructor, [count], (("dtype", counter_type),), loc),
            output,
            loc,
        )
    )
    context.record_thread_data_dtype(output, counter)
    kwargs = {
        "sample_dtype": dtype,
        "counter_dtype": counter,
        "threads_per_block": participation.exact_block_dim,
        "items_per_thread": extent,
        "bins": bins,
        "bins_per_thread": bins_per_thread,
        "algorithm": algorithm,
    }
    if not context.is_none(storage):
        kwargs["temp_storage"] = storage
    statements.extend(
        context.rewrite_call(
            inst,
            lowering_plan=plan,
            factory=_histogram.histogram,
            args=[samples, output],
            kwargs=kwargs,
            return_alias=output,
        )
    )
    return statements


register_group_primitive(
    "histogram",
    lower=_lower_histogram,
    results=(
        GroupResultSource(
            None,
            None,
            fixed_dtype=numba_types.int32,
            dtype_keyword="counter_dtype",
            extent_resolver=_extent,
        ),
    ),
)
register_rewrite_operation(
    "histogram",
    RewriteOperationSpecification(
        factory_namespaces=frozenset({"block"}),
        dtype_factory_kwargs=frozenset({"sample_dtype", "counter_dtype"}),
        runtime_arg_counts=frozenset({2}),
        runtime_factory_kwargs=(),
        runtime_factory_kw_prerequisites=(),
        allowed_factory_kwargs=frozenset(
            {
                "sample_dtype",
                "counter_dtype",
                "threads_per_block",
                "items_per_thread",
                "bins",
                "bins_per_thread",
                "algorithm",
            }
        ),
        required_factory_kwargs=frozenset(
            {
                "sample_dtype",
                "counter_dtype",
                "threads_per_block",
                "items_per_thread",
                "bins",
                "bins_per_thread",
            }
        ),
        accepts_temp_storage=True,
        scalar_binding_kwargs=frozenset(),
        runtime_offset_kwarg=None,
        infer_payload=_infer_payload,
    ),
)
