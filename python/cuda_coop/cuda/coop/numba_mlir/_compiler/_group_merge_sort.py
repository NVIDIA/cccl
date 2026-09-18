# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Family-local Merge Sort group planning and input-preserving lowering."""

from dataclasses import replace

from numba_cuda_mlir import types

from cuda.coop._core import (
    BindingKind,
    GroupLoweringTarget,
    GroupMergeSortSemantics,
    StorageOwnership,
    SynchronizationScope,
    make_block_merge_sort_semantics,
    make_group_primitive_call,
    plan_group_primitive,
)

from ._group_planner_support import _PAYLOAD_DTYPE_LIKE, GroupRewriteError, ir
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
from ._rewrite_merge_sort import infer_merge_sort_payload


def _payload(context, operation, name, value, is_common_root):
    if not context.is_array(operation, value):
        raise TypeError(
            f"{operation} {name} must be a fixed-size ThreadData or local array"
        )
    if is_common_root and not context.is_thread_data(operation, name, value):
        raise TypeError(
            f"cuda.coop.{operation} {name} requires ThreadData; use cuda.coop.numba_mlir for local arrays"
        )
    extent = context.array_extent(value)
    if extent is None:
        raise GroupRewriteError(f"{operation} could not infer {name} extent")
    dtype = context.dtype(value)
    if dtype is None:
        dtype = context.payload_write_dtype(value)
    if dtype is None:
        raise GroupRewriteError(f"{operation} could not infer {name} dtype")
    dtype = _validate_common_numeric_dtype(dtype, operation=operation, parameter=name)
    context.record_thread_data_dtype(value, dtype)
    return extent, dtype


def _cast(context, statements, inst, value, dtype, name):
    kwargs = dict(scope=inst.target.scope, loc=inst.loc)
    cast = context.value_var(
        statements, stem=f"merge_sort_{name}_type", value=dtype, **kwargs
    )
    value = context.value_var(
        statements, stem=f"merge_sort_{name}", value=value, **kwargs
    )
    result = context.new_var(inst.target.scope, inst.loc, f"merge_sort_{name}_cast")
    statements.append(
        ir.Assign(ir.Expr.call(cast, [value], (), inst.loc), result, inst.loc)
    )
    return result


def _lower_merge_sort(context, inst, *, operation, group, bound, is_common_root):
    from .._lowering import _merge_sort

    arguments = bound.arguments
    pairs = operation == "merge_sort_pairs"
    names = ("keys", "values") if pairs else ("keys",)
    payloads = [
        _payload(context, operation, name, arguments[name], is_common_root)
        for name in names
    ]
    extent, key_dtype = payloads[0]
    if pairs and payloads[1][0] != extent:
        raise ValueError(
            "Merge Sort keys and values must have matching items_per_thread extents"
        )
    value_dtype = payloads[1][1] if pairs else None
    descending = context.constant(arguments["descending"])
    compare_raw = arguments.get("compare_op")
    compare_op = None if context.is_none(compare_raw) else context.constant(compare_raw)
    compare_operator = _merge_sort.comparison_operator(descending, compare_op)
    valid_raw = arguments["valid_items"]
    sentinel_raw = arguments["oob_default"]
    valid = context.planning_binding(valid_raw)
    partial = valid.kind is not BindingKind.OMITTED
    if partial == context.is_none(sentinel_raw):
        raise ValueError(
            "Merge Sort valid_items and oob_default must be provided together"
        )
    if valid.kind is BindingKind.RUNTIME:
        _validate_runtime_integer_dtype(
            context.dtype(valid_raw), operation=operation, parameter="valid_items"
        )
    sentinel = None
    if partial:
        binding = context.planning_binding(sentinel_raw)
        if binding.kind is BindingKind.STATIC:
            sentinel = coerce_static_scalar(
                binding.value, key_dtype, operation=operation, parameter="oob_default"
            )
        else:
            sentinel_dtype = _validate_common_numeric_dtype(
                context.dtype(sentinel_raw),
                operation=operation,
                parameter="oob_default",
            )
            if sentinel_dtype != key_dtype:
                raise TypeError(
                    f"Merge Sort oob_default dtype {sentinel_dtype} does not match keys dtype {key_dtype}"
                )
            sentinel = sentinel_raw
    semantics = GroupMergeSortSemantics(
        make_block_merge_sort_semantics(
            key_dtype=key_dtype,
            value_dtype=value_dtype,
            items_per_thread=extent,
            compare_operator=compare_operator,
            valid_items=0 if partial else None,
            oob_default=0 if partial else None,
        ),
        valid_items=valid,
    )
    plan = plan_group_primitive(
        make_group_primitive_call(group, semantics), context.launch
    ).require_supported()
    temp_storage = arguments["temp_storage"]
    if not context.is_none(temp_storage):
        if plan.target is not GroupLoweringTarget.CUB_BLOCK:
            raise ValueError("Merge Sort temp_storage applies only to block groups")
        descriptor = context.temp_storage(temp_storage)
        if descriptor is None:
            raise GroupRewriteError(
                "Merge Sort temp_storage must resolve to a compile-time TempStorage descriptor"
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
    namespace = "block" if plan.target is GroupLoweringTarget.CUB_BLOCK else "warp"
    assert plan.provenance is not None
    expected_class = (
        "cub::BlockMergeSort" if namespace == "block" else "cub::WarpMergeSort"
    )
    if plan.provenance.cpp_class != expected_class:
        raise GroupRewriteError("Merge Sort received unknown provider provenance")
    factory = getattr(
        _merge_sort, f"{namespace}_{operation}" + ("_partial" if partial else "")
    )
    kwargs = dict(
        key_dtype=key_dtype,
        items_per_thread=extent,
        threads_per_block=plan.participation.exact_block_dim,
        descending=descending,
        compare_op=compare_raw,
    )
    if pairs:
        kwargs["value_dtype"] = value_dtype
    if namespace == "warp":
        kwargs["threads_in_warp"] = plan.topology.logical_width
    if not context.is_none(temp_storage):
        kwargs["temp_storage"] = temp_storage
    statements = []
    results = []
    for name in names:
        value = context.value_var(
            statements,
            scope=inst.target.scope,
            loc=inst.loc,
            stem=f"merge_sort_{name}",
            value=arguments[name],
        )
        result = context.typed_payload_like(
            statements,
            scope=inst.target.scope,
            loc=inst.loc,
            stem=f"merge_sort_{name}_result",
            prototype=value,
            is_array=True,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
            items_per_thread=extent,
        )
        context.copy_array_payload(
            statements,
            operation=operation,
            source=value,
            destination=result,
            scope=inst.target.scope,
            loc=inst.loc,
            known_items_per_thread=extent,
        )
        results.append(result)
    runtime_args = list(results)
    if partial:
        count = valid.value if valid.kind is BindingKind.STATIC else valid_raw
        runtime_args.extend(
            (
                _cast(context, statements, inst, count, types.int64, "valid_items"),
                _cast(context, statements, inst, sentinel, key_dtype, "oob_default"),
            )
        )
    statements.extend(
        context.rewrite_call(
            inst,
            lowering_plan=plan,
            factory=factory,
            args=runtime_args,
            kwargs=kwargs,
            return_alias=tuple(results) if pairs else results[0],
        )
    )
    return statements


for _name, _results in (
    ("merge_sort_keys", (GroupResultSource("keys", "keys"),)),
    (
        "merge_sort_pairs",
        (GroupResultSource("keys", "keys"), GroupResultSource("values", "values")),
    ),
):
    register_group_primitive(_name, lower=_lower_merge_sort, results=_results)
    for _partial in (False, True):
        register_rewrite_operation(
            _name + ("_partial" if _partial else ""),
            RewriteOperationSpec(
                factory_namespaces=frozenset({"block", "warp"}),
                dtype_factory_kwargs=frozenset({"key_dtype", "value_dtype"}),
                runtime_arg_counts=frozenset({len(_results) + (2 if _partial else 0)}),
                runtime_factory_kwargs=(),
                runtime_factory_kw_prerequisites=(),
                allowed_factory_kwargs=frozenset(
                    {
                        "key_dtype",
                        "value_dtype",
                        "items_per_thread",
                        "threads_per_block",
                        "threads_in_warp",
                        "descending",
                        "compare_op",
                    }
                ),
                required_factory_kwargs=frozenset(
                    {"key_dtype", "items_per_thread", "threads_per_block"}
                ),
                accepts_temp_storage=True,
                scalar_binding_kwargs=frozenset(),
                runtime_offset_kwarg=None,
                infer_payload=infer_merge_sort_payload,
            ),
        )
del _name, _results, _partial

__all__: tuple[str, ...] = ()
