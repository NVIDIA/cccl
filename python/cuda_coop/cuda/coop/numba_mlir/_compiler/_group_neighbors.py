# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan and lower input-preserving block neighbor operations."""

from dataclasses import replace

from numba_cuda_mlir import types

from cuda.coop._core import (
    BindingKind,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block.neighbors import BlockNeighborSemantics
from cuda.coop._core.group.neighbors import GroupNeighborSemantics

from ._group_planner_support import (
    _PAYLOAD_DTYPE_INT32,
    _PAYLOAD_DTYPE_LIKE,
    GroupRewriteError,
    ir,
)
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
)
from ._rewrite_support import CoopSinglePhaseRewriteError


def _infer_payload(context, inference):
    both = inference.factory_value("mode") == "heads_and_tails"
    extent = None
    for index in range(3 if both else 2):
        value, spec = inference.array_candidate(index)
        if spec is None or spec.items_per_thread is None:
            raise CoopSinglePhaseRewriteError(
                "neighbor operations require fixed-size arrays"
            )
        if extent is not None and spec.items_per_thread != extent:
            raise CoopSinglePhaseRewriteError(
                "neighbor input and output extents must match"
            )
        extent = spec.items_per_thread
        dtype = inference.inferred_array_dtype(value, spec)
        expected = (
            types.int32
            if index > 0 and inference.op_name.startswith("discontinuity")
            else inference.factory_value("dtype")
        )
        if dtype is None:
            dtype = expected
        dtype = _validate_common_numeric_dtype(
            dtype, operation=inference.op_name, parameter="values"
        )
        if expected is not None and dtype != expected:
            raise CoopSinglePhaseRewriteError(
                "neighbor payload dtype disagrees with its specialization"
            )
        if index == 0:
            inference.infer_kwarg("dtype", dtype)
        context.record_thread_data_dtype(value, dtype)
    inference.infer_kwarg("items_per_thread", extent)


def _cast(context, statements, inst, value, dtype, name):
    scope, loc = inst.target.scope, inst.loc
    cast = context.value_var(
        statements, scope=scope, loc=loc, stem="neighbor_type", value=dtype
    )
    argument = context.value_var(
        statements, scope=scope, loc=loc, stem=name, value=value
    )
    result = context.new_var(scope, loc, name + "_cast")
    statements.append(ir.Assign(ir.Expr.call(cast, [argument], (), loc), result, loc))
    return result


def _lower_neighbors(context, inst, *, operation, group, bound, is_common_root):
    from .._lowering import _neighbors

    arguments = bound.arguments
    value = arguments["values"]
    if not context.is_array(operation, value):
        raise TypeError(
            f"{operation} values must be fixed-size ThreadData or a local array"
        )
    if is_common_root and not context.is_thread_data(operation, "values", value):
        raise TypeError(f"cuda.coop.{operation} requires ThreadData")
    extent = context.array_extent(value)
    if extent is None:
        raise GroupRewriteError(f"{operation} could not infer input extent")
    dtype = context.dtype(value)
    if dtype is None:
        dtype = context.payload_write_dtype(value)
    dtype = _validate_common_numeric_dtype(
        dtype, operation=operation, parameter="values"
    )
    context.record_thread_data_dtype(value, dtype)
    adjacent = operation == "adjacent_difference"
    mode = context.constant(arguments["direction" if adjacent else "mode"])
    op_raw = arguments.get("difference_op" if adjacent else "flag_op")
    op = None if context.is_none(op_raw) else context.constant(op_raw)
    valid_raw = arguments.get("valid_items")
    valid = context.planning_binding(valid_raw)
    partial = valid.kind is not BindingKind.OMITTED
    if valid.kind is BindingKind.RUNTIME:
        _validate_runtime_integer_dtype(
            context.dtype(valid_raw), operation=operation, parameter="valid_items"
        )
    boundaries = {}
    for name in ("tile_predecessor_item", "tile_successor_item"):
        raw = arguments[name]
        if context.is_none(raw):
            boundaries[name] = 0
        else:
            binding = context.planning_binding(raw)
            if binding.kind is BindingKind.STATIC:
                boundaries[name] = coerce_static_scalar(
                    binding.value, dtype, operation=operation, parameter=name
                )
            else:
                actual = context.dtype(raw)
                if actual != dtype:
                    raise TypeError(
                        f"{operation} {name} dtype must match input dtype {dtype}"
                    )
                boundaries[name] = raw
    primitive = BlockNeighborSemantics(
        operation=operation,
        dtype=dtype,
        items_per_thread=extent,
        mode=mode,
        operator=_neighbors.neighbor_operator(operation, op),
        partial=partial,
        predecessor=not context.is_none(arguments["tile_predecessor_item"]),
        successor=not context.is_none(arguments["tile_successor_item"]),
    )
    plan = plan_group_primitive(
        make_group_primitive_call(
            group, GroupNeighborSemantics(primitive, valid_items=valid)
        ),
        context.launch,
    ).require_supported()
    storage = arguments["temp_storage"]
    if not context.is_none(storage):
        descriptor = context.temp_storage(storage)
        if descriptor is None:
            raise GroupRewriteError(
                f"{operation} temp_storage must resolve to a TempStorage descriptor"
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
                storage_reuse_barrier=SynchronizationScope.BLOCK
                if auto_sync
                else SynchronizationScope.NONE,
            ),
        )
    statements = []
    scope, loc = inst.target.scope, inst.loc
    source = context.value_var(
        statements, scope=scope, loc=loc, stem="neighbor_input", value=value
    )
    outputs = []
    for name in primitive.result_names:
        output = context.typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem="neighbor_" + name,
            prototype=source,
            is_array=True,
            dtype_policy=_PAYLOAD_DTYPE_LIKE if adjacent else _PAYLOAD_DTYPE_INT32,
            items_per_thread=extent,
        )
        context.record_thread_data_dtype(output, dtype if adjacent else types.int32)
        outputs.append(output)
    count = (
        valid_raw
        if valid.kind is BindingKind.RUNTIME
        else valid.value
        if partial
        else 0
    )
    runtime_args = [
        source,
        *outputs,
        _cast(context, statements, inst, count, types.int64, "neighbor_count"),
    ]
    runtime_args.extend(
        _cast(context, statements, inst, raw, dtype, name)
        for name, raw in boundaries.items()
    )
    kwargs = dict(
        dtype=dtype,
        threads_per_block=plan.participation.exact_block_dim,
        items_per_thread=extent,
        mode=mode,
        partial=partial,
        predecessor=primitive.predecessor,
        successor=primitive.successor,
        op=op_raw,
    )
    if not context.is_none(storage):
        kwargs["temp_storage"] = storage
    statements.extend(
        context.rewrite_call(
            inst,
            lowering_plan=plan,
            factory=getattr(
                _neighbors,
                "block_" + operation + ("_both" if len(outputs) == 2 else ""),
            ),
            args=runtime_args,
            kwargs=kwargs,
            return_alias=tuple(outputs) if len(outputs) == 2 else outputs[0],
        )
    )
    return statements


def _discontinuity_results(context, bound):
    mode = context.constant(bound.arguments["mode"])
    result = GroupResultSource(None, "values", fixed_dtype=types.int32)
    return (result, result) if mode == "heads_and_tails" else (result,)


register_group_primitive(
    "adjacent_difference",
    lower=_lower_neighbors,
    results=(GroupResultSource("values", "values"),),
)
register_group_primitive(
    "discontinuity",
    lower=_lower_neighbors,
    result_resolver=_discontinuity_results,
)

for _name in ("adjacent_difference", "discontinuity", "discontinuity_both"):
    register_rewrite_operation(
        _name,
        RewriteOperationSpec(
            factory_namespaces=frozenset({"block"}),
            dtype_factory_kwargs=frozenset({"dtype"}),
            runtime_arg_counts=frozenset({6} if _name == "discontinuity_both" else {5}),
            runtime_factory_kwargs=(),
            runtime_factory_kw_prerequisites=(),
            allowed_factory_kwargs=frozenset(
                {
                    "dtype",
                    "threads_per_block",
                    "items_per_thread",
                    "mode",
                    "partial",
                    "predecessor",
                    "successor",
                    "op",
                }
            ),
            required_factory_kwargs=frozenset(
                {"dtype", "threads_per_block", "items_per_thread", "mode"}
            ),
            accepts_temp_storage=True,
            scalar_binding_kwargs=frozenset(),
            runtime_offset_kwarg=None,
            infer_payload=_infer_payload,
        ),
    )
del _name

__all__: tuple[str, ...] = ()
