# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Family-local planning and payload inference for block TopK."""

from __future__ import annotations

from dataclasses import replace

from numba_cuda_mlir import types

from cuda.coop._core import (
    BindingKind,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.group.topk import GroupTopKSemantics

from ._group_planner_support import _PAYLOAD_DTYPE_LIKE, GroupRewriteError, ir
from ._operations import (
    GroupResultSource,
    RewriteOperationSpec,
    register_group_primitive,
    register_rewrite_operation,
)
from ._parameters import _validate_common_numeric_dtype, _validate_runtime_integer_dtype
from ._rewrite_support import CoopSinglePhaseRewriteError


def _infer_payload(context, inference):
    names = (
        ("key_dtype", "value_dtype")
        if inference.op_name == "topk_pairs"
        else ("key_dtype",)
    )
    extent = None
    for index, name in enumerate(names):
        value, spec = inference.array_candidate(index)
        if value is None or spec is None or spec.items_per_thread is None:
            raise CoopSinglePhaseRewriteError(
                "topk requires fixed-size per-thread arrays"
            )
        if extent is not None and extent != spec.items_per_thread:
            raise CoopSinglePhaseRewriteError(
                "topk keys and values must have matching extents"
            )
        extent = spec.items_per_thread
        dtype = inference.inferred_array_dtype(value, spec)
        if dtype is None:
            dtype = inference.factory_value(name)
        dtype = _validate_common_numeric_dtype(dtype, operation="topk", parameter=name)
        inference.infer_kwarg(name, dtype)
        context.record_thread_data_dtype(value, dtype)
    inference.infer_kwarg("items_per_thread", extent)


def _lower_topk(context, inst, *, operation, group, bound, is_common_root):
    from .._lowering import _topk

    payload_names = ("keys", "values") if operation.endswith("pairs") else ("keys",)
    extent = None
    dtypes = []
    for name in payload_names:
        value = bound.arguments[name]
        if not context.is_array(operation, value):
            raise TypeError(
                f"{operation} {name} must be a fixed-size ThreadData or local array"
            )
        size = context.array_extent(value)
        if size is None:
            raise GroupRewriteError(f"{operation} requires a static payload extent")
        if extent is not None and extent != size:
            raise ValueError(f"{operation} keys and values must have matching extents")
        extent = size
        if is_common_root and not context.is_thread_data(operation, name, value):
            raise TypeError(f"cuda.coop.{operation} requires {name} to be ThreadData")
        dtype = context.dtype(value)
        if dtype is None:
            dtype = context.payload_write_dtype(value)
        dtype = _validate_common_numeric_dtype(
            dtype, operation=operation, parameter=name
        )
        context.record_thread_data_dtype(value, dtype)
        dtypes.append(dtype)
    bindings = {}
    for name in ("k", "valid_items"):
        binding = context.planning_binding(bound.arguments[name])
        if binding.kind is BindingKind.RUNTIME:
            dtype = context.dtype(bound.arguments[name])
            if dtype is None:
                raise GroupRewriteError(
                    f"{operation} could not infer runtime {name} dtype"
                )
            _validate_runtime_integer_dtype(dtype, operation=operation, parameter=name)
        bindings[name] = binding
    semantics = GroupTopKSemantics(
        key_dtype=dtypes[0],
        value_dtype=dtypes[1] if len(dtypes) > 1 else None,
        items_per_thread=extent,
        selection=operation.split("_")[1],
        k=bindings["k"],
        valid_items=bindings["valid_items"],
    )
    plan = plan_group_primitive(
        make_group_primitive_call(group, semantics), context.launch
    ).require_supported()
    storage = bound.arguments.get("temp_storage")
    if not context.is_none(storage):
        descriptor = context.temp_storage(storage)
        if descriptor is None:
            raise GroupRewriteError(
                "topk temp_storage must resolve to a TempStorage descriptor"
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
    outputs = []
    scope, loc = inst.target.scope, inst.loc
    for name, dtype in zip(payload_names, dtypes):
        value = context.value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"topk_{name}",
            value=bound.arguments[name],
        )
        output = context.typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"topk_result_{name}",
            prototype=value,
            is_array=True,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
            items_per_thread=extent,
        )
        context.copy_array_payload(
            statements,
            operation=operation,
            source=value,
            destination=output,
            scope=scope,
            loc=loc,
            known_items_per_thread=extent,
        )
        context.record_thread_data_dtype(output, dtype)
        outputs.append(output)
    kwargs = dict(
        key_dtype=dtypes[0],
        threads_per_block=plan.participation.exact_block_dim,
        items_per_thread=extent,
        selection=semantics.selection,
    )
    if len(dtypes) > 1:
        kwargs["value_dtype"] = dtypes[1]
    for name, binding in bindings.items():
        if binding.kind is BindingKind.RUNTIME:
            cast = context.value_var(
                statements,
                scope=scope,
                loc=loc,
                stem="topk_count_type",
                value=types.int64,
            )
            value = context.new_var(scope, loc, f"topk_{name}_i64")
            statements.append(
                ir.Assign(
                    ir.Expr.call(cast, [bound.arguments[name]], (), loc), value, loc
                )
            )
        else:
            value = binding
        kwargs["num_valid" if name == "valid_items" else name] = value
    if not context.is_none(storage):
        kwargs["temp_storage"] = storage
    factory = _topk.topk_pairs if len(outputs) > 1 else _topk.topk_keys
    statements.extend(
        context.rewrite_call(
            inst,
            lowering_plan=plan,
            factory=factory,
            args=outputs,
            kwargs=kwargs,
            return_alias=tuple(outputs) if len(outputs) > 1 else outputs[0],
        )
    )
    return statements


for _selection in ("min", "max"):
    for _payload in ("keys", "pairs"):
        _names = ("keys", "values") if _payload == "pairs" else ("keys",)
        register_group_primitive(
            f"topk_{_selection}_{_payload}",
            lower=_lower_topk,
            results=tuple(GroupResultSource(name, name) for name in _names),
        )

for _payload in ("keys", "pairs"):
    register_rewrite_operation(
        f"topk_{_payload}",
        RewriteOperationSpec(
            factory_namespaces=frozenset({"block"}),
            dtype_factory_kwargs=frozenset({"key_dtype", "value_dtype"}),
            runtime_arg_counts=frozenset({2 if _payload == "pairs" else 1}),
            runtime_factory_kwargs=("k", "num_valid"),
            runtime_factory_kw_prerequisites=(),
            allowed_factory_kwargs=frozenset(
                {
                    "key_dtype",
                    "value_dtype",
                    "threads_per_block",
                    "items_per_thread",
                    "selection",
                    "k",
                    "num_valid",
                }
            ),
            required_factory_kwargs=frozenset(
                {"key_dtype", "threads_per_block", "items_per_thread", "k"}
            ),
            accepts_temp_storage=True,
            scalar_binding_kwargs=frozenset({"k", "num_valid"}),
            runtime_offset_kwarg=None,
            infer_payload=_infer_payload,
        ),
    )
del _selection, _payload, _names
