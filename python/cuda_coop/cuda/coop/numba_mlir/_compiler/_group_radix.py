# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Family-local whole-function planning for block radix operations."""

from dataclasses import replace
from numbers import Integral

from numba_cuda_mlir import types

from cuda.coop._core import (
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.api.radix import _radix_bounds
from cuda.coop._core.block.radix import make_radix_bit_range
from cuda.coop._core.block.radix_rank import (
    block_radix_rank_bins_per_thread,
    make_block_radix_rank_semantics,
)
from cuda.coop._core.block.radix_sort import make_block_radix_sort_semantics
from cuda.coop._core.group.radix import GroupRadixRankSemantics, GroupRadixSortSemantics

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
    normalize_dtype_param,
)
from ._rewrite_radix import infer_radix_payload


def _dtype(context, operation, parameter, value):
    dtype = context.dtype(value)
    if dtype is None:
        dtype = context.payload_write_dtype(value)
    if dtype is None:
        raise GroupRewriteError(
            f"cuda.coop.{operation} could not infer {parameter} dtype"
        )
    return normalize_dtype_param(dtype)


def _bool(context, operation, name, value):
    value = context.constant(value)
    if not isinstance(value, bool):
        raise TypeError(f"cuda.coop.{operation} {name} must be a compile-time bool")
    return value


def _sort_bit(context, operation, name, value):
    known, constant = context.try_static_scalar(value)
    if known:
        if isinstance(constant, bool) or not isinstance(constant, Integral):
            raise TypeError(f"cuda.coop.{operation} {name} must be an integer")
        return int(constant)
    dtype = context.dtype(value)
    if dtype is None:
        raise GroupRewriteError(f"cuda.coop.{operation} could not infer {name} dtype")
    _validate_runtime_integer_dtype(dtype, operation=operation, parameter=name)
    return value


def _lower(context, inst, *, operation, group, bound, is_common_root):
    from .._lowering import _radix
    from .._lowering._core import NumbaMlirCoreAdapter

    if group.kind != "block":
        raise NotImplementedError(
            "cuda.coop radix operations require a complete physical block"
        )
    rank = operation == "radix_rank"
    pairs = operation == "radix_sort_pairs"
    parameters = ("keys", "values") if pairs else ("keys",)
    dtype_map = {}
    array_map = {}
    extent = None
    for name in parameters:
        value = bound.arguments[name]
        if is_common_root and not context.is_thread_data(operation, name, value):
            raise TypeError(f"cuda.coop.{operation} requires {name} to be ThreadData")
        is_array = context.is_array(operation, value)
        count = context.array_extent(value) if is_array else 1
        if count is None:
            raise GroupRewriteError(
                f"cuda.coop.{operation} requires a static items_per_thread extent"
            )
        if extent is not None and (extent != count or is_array != array_map["keys"]):
            raise ValueError(
                "keys and values must have the same shape and items_per_thread"
            )
        extent = count
        array_map[name] = is_array
        dtype_map[name] = _dtype(context, operation, name, value)
    key_dtype = dtype_map["keys"]
    integral = {types.int32, types.uint32, types.int64, types.uint64}
    allowed = (
        integral
        if rank or is_common_root
        else integral | {types.float32, types.float64}
    )
    if key_dtype not in allowed:
        raise TypeError(
            f"cuda.coop.{operation} keys require "
            + (
                "int32, uint32, int64, or uint64"
                if rank or is_common_root
                else "32- or 64-bit integer or float dtype"
            )
        )
    if pairs:
        dtype_map["values"] = _validate_common_numeric_dtype(
            dtype_map["values"], operation=operation, parameter="values"
        )
    descending = _bool(context, operation, "descending", bound.arguments["descending"])
    adapter = NumbaMlirCoreAdapter()
    kwargs = {
        "dtype": key_dtype,
        "threads_per_block": context.launch.exact_block_dim,
        "items_per_thread": extent,
        "descending": descending,
    }
    core_kwargs = {
        "key_dtype": adapter.core_dtype(key_dtype),
        "items_per_thread": extent,
        "descending": descending,
    }
    if rank:
        begin_bit, end_bit = _radix_bounds(
            operation,
            key_dtype.bitwidth,
            context.constant(bound.arguments["begin_bit"]),
            context.constant(bound.arguments["end_bit"]),
            context.constant(bound.arguments["radix_bits"]),
        )
        prefix = bound.arguments.get("exclusive_digit_prefix")
        prefix_extent = None
        if not context.is_none(prefix):
            expected = block_radix_rank_bins_per_thread(
                end_bit - begin_bit, context.launch.exact_block_threads
            )
            prefix_extent = context.array_extent(prefix)
            if not context.is_array(operation, prefix) or prefix_extent != expected:
                raise ValueError(
                    f"exclusive_digit_prefix must contain {expected} items per thread"
                )
            dtype = context.dtype(prefix)
            if dtype is not None and normalize_dtype_param(dtype) != types.int32:
                raise TypeError("exclusive_digit_prefix must have int32 dtype")
            context.record_thread_data_dtype(prefix, types.int32)
        primitive = make_block_radix_rank_semantics(
            **core_kwargs,
            begin_bit=begin_bit,
            end_bit=end_bit,
            key_bit_width=key_dtype.bitwidth,
            block_threads=context.launch.exact_block_threads,
            exclusive_digit_prefix_items_per_thread=prefix_extent,
        )
        semantics = GroupRadixRankSemantics(
            primitive, operand_kind="array" if array_map["keys"] else "scalar"
        )
        kwargs.update(begin_bit=begin_bit, end_bit=end_bit)
    else:
        begin_bit = _sort_bit(
            context, operation, "begin_bit", bound.arguments["begin_bit"]
        )
        end_bit = (
            key_dtype.bitwidth
            if context.is_none(bound.arguments["end_bit"])
            else _sort_bit(context, operation, "end_bit", bound.arguments["end_bit"])
        )
        begin_binding = context.planning_binding(begin_bit)
        end_binding = context.planning_binding(end_bit)
        make_radix_bit_range(
            begin_bit=begin_binding, end_bit=end_binding, bit_width=key_dtype.bitwidth
        )
        striped = _bool(
            context,
            operation,
            "blocked_to_striped",
            bound.arguments.get("blocked_to_striped", False),
        )
        primitive = make_block_radix_sort_semantics(
            **core_kwargs,
            value_dtype=None if not pairs else adapter.core_dtype(dtype_map["values"]),
            begin_bit=begin_binding,
            end_bit=end_binding,
            key_bit_width=key_dtype.bitwidth,
            blocked_to_striped=striped,
            bit_policy="both",
        )
        semantics = GroupRadixSortSemantics(
            primitive, operand_kind="array" if array_map["keys"] else "scalar"
        )
        kwargs["blocked_to_striped"] = striped
        if pairs:
            kwargs["value_dtype"] = dtype_map["values"]
    plan = plan_group_primitive(
        make_group_primitive_call(group, semantics), context.launch
    ).require_supported()
    temp_storage = bound.arguments.get("temp_storage")
    if not context.is_none(temp_storage):
        descriptor = context.temp_storage(temp_storage)
        if descriptor is None:
            raise TypeError(
                "radix sort temp_storage must resolve to a TempStorage descriptor"
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
        kwargs["temp_storage"] = temp_storage
    statements = []
    scope, loc = inst.target.scope, inst.loc
    outputs = []
    runtime_args = []
    for name in parameters:
        value = context.value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_{name}",
            value=bound.arguments[name],
        )
        payload, is_array = context.box_group_operand(
            statements, operation=operation, value=value, scope=scope, loc=loc
        )
        result = context.typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_{name}_result",
            prototype=value,
            is_array=is_array,
            dtype_policy=_PAYLOAD_DTYPE_INT32 if rank else _PAYLOAD_DTYPE_LIKE,
            items_per_thread=extent,
        )
        if rank:
            runtime_args.extend((payload, result))
        else:
            context.copy_array_payload(
                statements,
                operation=operation,
                source=payload,
                destination=result,
                scope=scope,
                loc=loc,
                known_items_per_thread=extent,
            )
            runtime_args.append(result)
        outputs.append(result)
    if rank:
        if prefix_extent is not None:
            runtime_args.append(prefix)
    else:
        runtime_args.extend((begin_bit, end_bit))
    rewritten = context.rewrite_call(
        inst,
        lowering_plan=plan,
        factory=getattr(_radix, operation),
        args=runtime_args,
        kwargs=kwargs,
        return_alias=outputs[0] if len(outputs) == 1 else tuple(outputs),
    )
    # Preserve the scalar/array shape selected by the qualified frontend.
    statements.extend(rewritten[:-1])
    results = [
        context.result_value(
            statements,
            payload=output,
            is_array=array_map[name],
            scope=scope,
            loc=loc,
            stem=f"{operation}_{name}_value",
        )
        for name, output in zip(parameters, outputs)
    ]
    if len(results) == 1:
        statements.append(ir.Assign(results[0], inst.target, loc))
    else:
        statements.append(
            ir.Assign(ir.Expr.build_tuple(results, loc), inst.target, loc)
        )
    return statements


for _operation, _results, _count in (
    ("radix_rank", (GroupResultSource(None, "keys", fixed_dtype=types.int32),), 2),
    ("radix_sort_keys", (GroupResultSource("keys", "keys"),), 3),
    (
        "radix_sort_pairs",
        (GroupResultSource("keys", "keys"), GroupResultSource("values", "values")),
        4,
    ),
):
    register_group_primitive(_operation, lower=_lower, results=_results)
    register_rewrite_operation(
        _operation,
        RewriteOperationSpec(
            factory_namespaces=frozenset({"block"}),
            dtype_factory_kwargs=frozenset({"dtype", "value_dtype"}),
            runtime_arg_counts=frozenset({2, 3})
            if _operation == "radix_rank"
            else frozenset({_count}),
            runtime_factory_kwargs=("with_exclusive_digit_prefix",)
            if _operation == "radix_rank"
            else (),
            runtime_factory_kw_prerequisites=(),
            allowed_factory_kwargs=frozenset(
                {
                    "dtype",
                    "value_dtype",
                    "threads_per_block",
                    "items_per_thread",
                    "begin_bit",
                    "end_bit",
                    "descending",
                    "blocked_to_striped",
                    "with_exclusive_digit_prefix",
                }
            ),
            required_factory_kwargs=frozenset(
                {"dtype", "threads_per_block", "items_per_thread"}
            ),
            accepts_temp_storage=_operation != "radix_rank",
            scalar_binding_kwargs=frozenset(),
            runtime_offset_kwarg=None,
            infer_payload=infer_radix_payload,
        ),
    )
del _operation, _results, _count

__all__: tuple[str, ...] = ()
