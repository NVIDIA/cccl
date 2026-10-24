# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan block neighbor calls and emit markers for their result payloads.

Difference results retain the input dtype; discontinuity flags use int32.
Record that distinction before ordinary typing so later group operations can
consume either result. The provider rewrite materializes the marked arrays.
"""

from dataclasses import replace

import numba_cuda_mlir.numba_cuda.types as numba_types

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
    RewriteOperationSpecification,
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
    """Infer matching payload extents and the provider's element dtype.

    All input and output arrays have the same fixed extent. Difference outputs
    use the input dtype; discontinuity outputs use int32. Record each dtype
    so inferred ThreadData can be materialized consistently.
    """

    both = inference.factory_value("mode") == "heads_and_tails"
    extent = None
    for index in range(3 if both else 2):
        value, specification = inference.array_candidate(index)
        if specification is None or specification.items_per_thread is None:
            raise CoopSinglePhaseRewriteError(
                "neighbor operations require fixed-size arrays"
            )
        if extent is not None and specification.items_per_thread != extent:
            raise CoopSinglePhaseRewriteError(
                "neighbor input and output extents must match"
            )
        extent = specification.items_per_thread
        dtype = inference.inferred_array_dtype(value, specification)
        expected = (
            numba_types.int32
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
    """Append an IR cast and return its temporary for a provider argument.

    Counts use int64 before range checks; boundary items use the input dtype.
    The cast runs in the kernel after compilation, not during group planning.
    """

    scope, loc = inst.target.scope, inst.loc
    cast = context.value_var(
        statements, scope=scope, loc=loc, stem="neighbor_type", value=dtype
    )
    argument = context.value_var(
        statements, scope=scope, loc=loc, stem=name, value=value
    )
    result = context.new_var(scope, loc, name + "_cast")
    statements.append(
        ir.Assign(ir.Expr.call(cast, [argument], (), loc), result, loc)
    )
    return result


def _lower_neighbors(context, inst, *, operation, group, bound, is_common_root):
    """Validate a neighbor call and emit its provider call and result markers.

    Resolve the fixed input extent, dtype, selector, count binding, and
    boundary values, then build the core plan. An explicit ``temp_storage``
    descriptor then replaces implementation-owned scratch with caller-owned
    storage and selects the reuse barrier from ``auto_sync``.

    Static boundary values go through ``coerce_static_scalar``. Finite,
    in-range Python literals take the input dtype; float literals cannot
    become integers. NumPy scalars must already match the input dtype. Runtime
    boundaries need an inferred dtype that exactly matches the input dtype.

    Each result marker describes a fresh array with the input extent. Preserve
    the source payload and record int32 flag results for later consumers. The
    provider rewrite allocates these arrays and lowers the emitted call.
    """

    from .._lowering import _neighbors

    arguments = bound.arguments
    value = arguments["values"]
    if not context.is_array(operation, value):
        raise TypeError(
            f"{operation} values must be fixed-size ThreadData or a local array"
        )
    if is_common_root and not context.is_thread_data(
        operation, "values", value
    ):
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
            context.dtype(valid_raw),
            operation=operation,
            parameter="valid_items",
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
                        f"{operation} {name} dtype "
                        f"must match input dtype {dtype}"
                    )
                boundaries[name] = raw
    from .._lowering._core import NumbaMlirCoreAdapter

    adapter = NumbaMlirCoreAdapter()
    primitive = BlockNeighborSemantics(
        operation=operation,
        dtype=adapter.core_dtype(dtype),
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
    participation = plan.participation
    assert participation is not None
    assert plan.temp_storage is not None
    assert plan.synchronization is not None
    storage = arguments["temp_storage"]
    if not context.is_none(storage):
        descriptor = context.temp_storage(storage)
        if descriptor is None:
            raise GroupRewriteError(
                f"{operation} temp_storage must "
                f"resolve to a TempStorage descriptor"
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
            dtype_policy=_PAYLOAD_DTYPE_LIKE
            if adjacent
            else _PAYLOAD_DTYPE_INT32,
            items_per_thread=extent,
        )
        context.record_thread_data_dtype(
            output, dtype if adjacent else numba_types.int32
        )
        outputs.append(output)
    # Full-tile overloads ignore the count slot; zero is an ABI placeholder.
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
        _cast(
            context,
            statements,
            inst,
            count,
            numba_types.int64,
            "neighbor_count",
        ),
    ]
    runtime_args.extend(
        _cast(context, statements, inst, raw, dtype, name)
        for name, raw in boundaries.items()
    )
    kwargs = {
        "dtype": dtype,
        "threads_per_block": participation.exact_block_dim,
        "items_per_thread": extent,
        "mode": mode,
        "partial": partial,
        "predecessor": primitive.predecessor,
        "successor": primitive.successor,
        "op": op_raw,
    }
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
    """Resolve the selector to one flag payload or a heads/tails pair.

    The registered result resolver is queried by
    ``_GroupCallPlanner._result_source`` when another operation consumes
    a Discontinuity result. It describes the return shape before
    lowering allocates either flag array; ``heads_and_tails`` therefore
    exposes two results while each other mode exposes one.

    Both results inherit the input extent and use int32. Resolve the mode
    before ordinary typing so tuple projections and chained calls see the
    correct number of results.

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

    mode = context.constant(bound.arguments["mode"])
    result = GroupResultSource(None, "values", fixed_dtype=numba_types.int32)
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
        RewriteOperationSpecification(
            factory_namespaces=frozenset({"block"}),
            dtype_factory_kwargs=frozenset({"dtype"}),
            runtime_arg_counts=frozenset(
                {6} if _name == "discontinuity_both" else {5}
            ),
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
