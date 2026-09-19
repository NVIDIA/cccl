# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Family-local planning for windowed and bulk Run Length Decode."""

from dataclasses import replace

from numba_cuda_mlir import types

from cuda.coop._core import (
    BindingKind,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block._common import normalize_positive_int
from cuda.coop._core.group.run_length import GroupRunLengthDecodeSemantics

from .._thread_data import ThreadData
from ._group_planner_support import GroupRewriteError, ir
from ._operations import (
    GroupResultSource,
    RewriteOperationSpec,
    register_group_primitive,
    register_rewrite_operation,
)
from ._parameters import _validate_common_numeric_dtype, normalize_dtype_param
from ._rewrite_support import CoopSinglePhaseRewriteError


def _integer_dtype(dtype, name):
    dtype = normalize_dtype_param(dtype)
    if not isinstance(dtype, types.Integer) or dtype.bitwidth > 64:
        raise TypeError(
            f"run_length_decode {name} must have an integer dtype up to 64 bits"
        )
    return dtype


def _decode_extent(context, bound):
    return normalize_positive_int(
        "decoded_items_per_thread",
        context.constant(bound.arguments["decoded_items_per_thread"]),
    )


def _infer_payload(context, inference):
    for index, name in enumerate(("item_dtype", "run_length_dtype")):
        value, spec = inference.array_candidate(index)
        if spec is None or spec.items_per_thread is None:
            raise CoopSinglePhaseRewriteError(
                "run_length_decode requires fixed per-thread run arrays"
            )
        dtype = inference.inferred_array_dtype(value, spec)
        inference.infer_kwarg(name, dtype)
        inference.infer_kwarg("runs_per_thread", spec.items_per_thread)
    if inference.op_name == "run_length_decode":
        decoded_extent = inference.factory_value("decoded_items_per_thread")
        for index, name, extent, dtype in (
            (2, "decoded", decoded_extent, inference.factory_value("item_dtype")),
            (
                3,
                "total_decoded_size",
                1,
                inference.factory_value("decoded_offset_dtype"),
            ),
            (
                4,
                "relative_offsets",
                decoded_extent,
                inference.factory_value("decoded_offset_dtype"),
            ),
        ):
            value, spec = inference.array_candidate(index)
            if spec is None or spec.items_per_thread != extent:
                raise CoopSinglePhaseRewriteError(
                    f"{name} requires a per-thread array of extent {extent}"
                )
            actual_dtype = inference.inferred_array_dtype(value, spec)
            if actual_dtype is not None and actual_dtype != dtype:
                raise CoopSinglePhaseRewriteError(f"{name} dtype must match {dtype}")
            context.record_thread_data_dtype(value, dtype)
    else:
        arrays = [(2, "destination", inference.factory_value("item_dtype"))]
        if inference.factory_value("relative_offsets"):
            arrays.append(
                (4, "relative_offsets", inference.factory_value("decoded_offset_dtype"))
            )
        for index, name, dtype in arrays:
            actual = context.numba_type(inference.runtime_args[index])
            if (
                not isinstance(actual, types.Array)
                or actual.ndim != 1
                or actual.layout != "C"
                or not actual.mutable
                or actual.dtype != dtype
            ):
                raise TypeError(
                    f"run_length_decode_into {name} must be a writable contiguous "
                    f"one-dimensional array with dtype {dtype}"
                )


def _allocate(context, statements, scope, loc, name, extent, dtype):
    constructor = context.value_var(
        statements, scope=scope, loc=loc, stem="rld_ThreadData", value=ThreadData
    )
    size = context.value_var(
        statements, scope=scope, loc=loc, stem="rld_extent", value=extent
    )
    item_type = context.value_var(
        statements, scope=scope, loc=loc, stem="rld_dtype", value=dtype
    )
    output = context.new_var(scope, loc, name)
    statements.append(
        ir.Assign(
            ir.Expr.call(constructor, [size], (("dtype", item_type),), loc), output, loc
        )
    )
    context.record_thread_data_dtype(output, dtype)
    return output


def _lower(context, inst, *, operation, group, bound, is_common_root):
    from .._lowering import _run_length

    bulk = operation == "run_length_decode_into"
    extent = None
    dtypes = []
    for name in ("run_values", "run_lengths"):
        value = bound.arguments[name]
        if not context.is_array(operation, value):
            raise TypeError(f"{operation} {name} must be a fixed per-thread array")
        size = context.array_extent(value)
        if size is None or (extent is not None and extent != size):
            raise ValueError(
                f"{operation} run values and lengths require matching fixed extents"
            )
        extent = size
        if is_common_root and not context.is_thread_data(operation, name, value):
            raise TypeError(f"cuda.coop.{operation} requires {name} to be ThreadData")
        dtype = context.dtype(value) or context.payload_write_dtype(value)
        dtype = (
            _integer_dtype(dtype, name)
            if name == "run_lengths"
            else _validate_common_numeric_dtype(
                dtype, operation=operation, parameter=name
            )
        )
        context.record_thread_data_dtype(value, dtype)
        dtypes.append(dtype)
    decoded_extent = _decode_extent(context, bound)
    offset_dtype = bound.arguments.get("decoded_offset_dtype")
    offset_dtype = (
        types.uint32
        if context.is_none(offset_dtype)
        else normalize_dtype_param(context.constant(offset_dtype))
    )
    if offset_dtype not in (types.uint32, types.uint64):
        raise TypeError("decoded_offset_dtype must be uint32 or uint64")
    control_name = "destination_offset" if bulk else "decoded_window_offset"
    control = bound.arguments[control_name]
    binding = context.planning_binding(control)
    control_dtype = types.uint64
    if binding.kind is BindingKind.RUNTIME:
        control_dtype = _integer_dtype(context.dtype(control), control_name)
    relative = bound.arguments.get("relative_offsets")
    has_relative = not context.is_none(relative)
    semantics = GroupRunLengthDecodeSemantics(
        item_dtype=dtypes[0],
        run_length_dtype=dtypes[1],
        runs_per_thread=extent,
        decoded_items_per_thread=decoded_extent,
        offset=binding,
        decoded_offset_dtype=offset_dtype,
        control_dtype=control_dtype,
        bulk=bulk,
        relative_offsets=has_relative,
    )
    plan = plan_group_primitive(
        make_group_primitive_call(group, semantics), context.launch
    ).require_supported()
    storage = bound.arguments.get("temp_storage")
    if not context.is_none(storage):
        descriptor = context.temp_storage(storage)
        if descriptor is None:
            raise GroupRewriteError(
                "run_length_decode temp_storage requires a TempStorage descriptor"
            )
        size, alignment, auto_sync, sharing = descriptor
        plan = replace(
            plan,
            temp_storage=replace(
                plan.temp_storage,
                ownership=StorageOwnership.CALLER,
                exact_layout_required=True,
                sharing=sharing,
                requested_size_in_bytes=size,
                requested_alignment=alignment,
                auto_sync=auto_sync,
            ),
            synchronization=replace(
                plan.synchronization,
                storage_reuse_barrier=(
                    SynchronizationScope.BLOCK
                    if auto_sync
                    else SynchronizationScope.NONE
                ),
            ),
        )
    statements = []
    scope, loc = inst.target.scope, inst.loc
    args = [bound.arguments[name] for name in ("run_values", "run_lengths")]
    if bulk:
        for name in ("destination", "relative_offsets"):
            if name == "relative_offsets" and not has_relative:
                continue
            array = bound.arguments[name]
            args.append(array)
            length = context.value_var(
                statements, scope=scope, loc=loc, stem="rld_len", value=len
            )
            capacity = context.new_var(scope, loc, "rld_capacity")
            statements.append(
                ir.Assign(ir.Expr.call(length, [array], (), loc), capacity, loc)
            )
            args.append(capacity)
        output = None
    else:
        output = _allocate(
            context, statements, scope, loc, "rld_decoded", decoded_extent, dtypes[0]
        )
        args.append(output)
        for name, size in (
            ("total_decoded_size", 1),
            ("relative_offsets", decoded_extent),
        ):
            value = bound.arguments.get(name)
            if context.is_none(value):
                value = _allocate(
                    context, statements, scope, loc, "rld_" + name, size, offset_dtype
                )
            else:
                if (
                    not context.is_array(operation, value)
                    or context.array_extent(value) != size
                ):
                    raise TypeError(
                        f"{name} requires a fixed per-thread payload of extent {size}"
                    )
                dtype = context.dtype(value) or context.payload_write_dtype(value)
                if dtype is not None and dtype != offset_dtype:
                    raise TypeError(f"{name} dtype must match decoded_offset_dtype")
                context.record_thread_data_dtype(value, offset_dtype)
            args.append(value)
    kwargs = dict(
        item_dtype=dtypes[0],
        run_length_dtype=dtypes[1],
        decoded_offset_dtype=offset_dtype,
        control_dtype=control_dtype,
        threads_per_block=plan.participation.exact_block_dim,
        runs_per_thread=extent,
        decoded_items_per_thread=decoded_extent,
        offset=control if binding.kind is BindingKind.RUNTIME else binding,
        relative_offsets=has_relative,
    )
    if not context.is_none(storage):
        kwargs["temp_storage"] = storage
    statements.extend(
        context.rewrite_call(
            inst,
            lowering_plan=plan,
            factory=getattr(
                _run_length, operation + ("_offsets" if bulk and has_relative else "")
            ),
            args=args,
            kwargs=kwargs,
            return_alias=output,
        )
    )
    return statements


register_group_primitive(
    "run_length_decode",
    lower=_lower,
    results=(GroupResultSource("run_values", None, extent_resolver=_decode_extent),),
)
register_group_primitive(
    "run_length_decode_into",
    lower=_lower,
    results=(
        GroupResultSource(
            None, None, fixed_dtype=types.uint32, dtype_keyword="decoded_offset_dtype"
        ),
    ),
)
for _name, _count in (
    ("run_length_decode", 5),
    ("run_length_decode_into", 4),
    ("run_length_decode_into_offsets", 6),
):
    register_rewrite_operation(
        _name,
        RewriteOperationSpec(
            factory_namespaces=frozenset({"block"}),
            dtype_factory_kwargs=frozenset(
                {
                    "item_dtype",
                    "run_length_dtype",
                    "decoded_offset_dtype",
                    "control_dtype",
                }
            ),
            runtime_arg_counts=frozenset({_count}),
            runtime_factory_kwargs=("offset",),
            runtime_factory_kw_prerequisites=(),
            allowed_factory_kwargs=frozenset(
                {
                    "item_dtype",
                    "run_length_dtype",
                    "decoded_offset_dtype",
                    "control_dtype",
                    "threads_per_block",
                    "runs_per_thread",
                    "decoded_items_per_thread",
                    "offset",
                    "relative_offsets",
                }
            ),
            required_factory_kwargs=frozenset(
                {
                    "item_dtype",
                    "run_length_dtype",
                    "decoded_offset_dtype",
                    "control_dtype",
                    "threads_per_block",
                    "runs_per_thread",
                    "decoded_items_per_thread",
                    "offset",
                }
            ),
            accepts_temp_storage=True,
            scalar_binding_kwargs=frozenset({"offset"}),
            runtime_offset_kwarg=None,
            infer_payload=_infer_payload,
        ),
    )
del _name, _count
