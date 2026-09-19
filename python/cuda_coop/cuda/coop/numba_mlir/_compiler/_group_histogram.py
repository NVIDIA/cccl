# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Whole-function planning for fresh block histogram results."""

from dataclasses import replace

from numba_cuda_mlir import types

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
    RewriteOperationSpec,
    register_group_primitive,
    register_rewrite_operation,
)
from ._parameters import normalize_dtype_param


def _extent(context, bound):
    return normalize_positive_int(
        "bins_per_thread", context.constant(bound.arguments["bins_per_thread"])
    )


def _infer_payload(context, inference):
    for index, dtype_name, extent_name in (
        (0, "sample_dtype", "items_per_thread"),
        (1, "counter_dtype", "bins_per_thread"),
    ):
        value, spec = inference.array_candidate(index)
        if value is None or spec is None or spec.items_per_thread is None:
            raise GroupRewriteError("histogram requires fixed-size per-thread payloads")
        dtype = inference.inferred_array_dtype(value, spec)
        if dtype is None:
            dtype = inference.factory_value(dtype_name)
        dtype = normalize_dtype_param(dtype)
        validate_histogram_dtype(dtype, counter=index == 1)
        inference.infer_kwarg(dtype_name, dtype)
        inference.infer_kwarg(extent_name, spec.items_per_thread)
        context.record_thread_data_dtype(value, dtype)


def _lower_histogram(context, inst, *, operation, group, bound, is_common_root):
    from .._lowering import _histogram
    from .._lowering._core import NumbaMlirCoreAdapter

    value = bound.arguments["samples"]
    if is_common_root and not context.is_thread_data(operation, "samples", value):
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
        types.int32
        if context.is_none(bound.arguments["counter_dtype"])
        else normalize_dtype_param(context.constant(bound.arguments["counter_dtype"]))
    )
    validate_histogram_dtype(counter, counter=True)
    bins = normalize_positive_int("bins", context.constant(bound.arguments["bins"]))
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
    storage = bound.arguments.get("temp_storage")
    if not context.is_none(storage):
        descriptor = context.temp_storage(storage)
        if descriptor is None:
            raise TypeError(
                "histogram temp_storage must resolve to a TempStorage descriptor"
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
        statements, scope=scope, loc=loc, stem="histogram_payload", value=ThreadData
    )
    count = context.value_var(
        statements, scope=scope, loc=loc, stem="histogram_extent", value=bins_per_thread
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
    kwargs = dict(
        sample_dtype=dtype,
        counter_dtype=counter,
        threads_per_block=plan.participation.exact_block_dim,
        items_per_thread=extent,
        bins=bins,
        bins_per_thread=bins_per_thread,
        algorithm=algorithm,
    )
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
            fixed_dtype=types.int32,
            dtype_keyword="counter_dtype",
            extent_resolver=_extent,
        ),
    ),
)
register_rewrite_operation(
    "histogram",
    RewriteOperationSpec(
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
