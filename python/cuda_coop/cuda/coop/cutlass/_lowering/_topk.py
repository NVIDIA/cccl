# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed TopK provider using the shared checked CUB compatibility shim."""

import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Integral

import numpy as np
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Int64, Uint32
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    ArgumentBinding,
    BindingKind,
    GroupLoweringPlan,
    GroupLoweringTarget,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.api._dispatch import _common_root_operation_name
from cuda.coop._core.group.topk import GroupTopKSemantics

from .._compiler import _rendering, _state, _storage, _types
from .._temp_storage import TempStorage
from .._thread_data import ThreadData, _make_rmem_tensor

_SCOPE = "cuda.coop.cutlass"
_HEADER = "cub/block/block_topk.cuh"
_resolve_type = _types.make_provider_type_resolver(
    scope=_SCOPE, root_scope=_SCOPE, namespace="thread_group"
)


def _count_binding(value, name, *, optional=False):
    if optional and value is None:
        return ArgumentBinding.omitted()
    if isinstance(value, (bool, np.bool_, Enum)):
        raise TypeError(f"TopK {name} must be an integer, not bool or Enum")
    if isinstance(value, Integral):
        return ArgumentBinding.static(int(value))
    spec = _types.TYPE_SPECS.get(_types.canonical_dsl_type(value))
    if spec is None or spec.token[0] not in {"i", "u"} or spec.token == "u64":
        raise TypeError(
            f"TopK {name} requires a signed integer up to 64 bits or an unsigned integer up to 32 bits"
        )
    return ArgumentBinding.runtime()


def _make_topk_plan(
    *,
    group,
    launch,
    key_type,
    value_type,
    items,
    selection,
    k,
    valid_items,
    temp_storage=None,
):
    operation = GroupTopKSemantics(
        key_dtype=key_type,
        value_dtype=value_type,
        items_per_thread=items,
        selection=selection,
        k=_count_binding(k, "k"),
        valid_items=_count_binding(valid_items, "valid_items", optional=True),
    )
    source = (
        "common_root" if _common_root_operation_name() is not None else "cutlass_root"
    )
    plan = plan_group_primitive(
        make_group_primitive_call(group, operation, source=source), launch
    ).require_supported()
    return replace(
        plan,
        temp_storage=replace(
            plan.temp_storage,
            ownership=StorageOwnership.IMPLEMENTATION
            if temp_storage is None
            else StorageOwnership.CALLER,
            exact_layout_required=True,
            sharing=None if temp_storage is None else temp_storage.sharing,
            requested_size_in_bytes=None
            if temp_storage is None
            else temp_storage.size_in_bytes,
            requested_alignment=None
            if temp_storage is None
            else temp_storage.alignment,
            auto_sync=True if temp_storage is None else temp_storage.auto_sync,
        ),
        synchronization=replace(
            plan.synchronization,
            storage_reuse_barrier=SynchronizationScope.BLOCK
            if temp_storage is None or temp_storage.auto_sync
            else SynchronizationScope.NONE,
        ),
    )


@dataclass(frozen=True, eq=False)
class _CubTopKRequest:
    plan: GroupLoweringPlan
    kind: str = "cub_group_topk"

    def __post_init__(self):
        self.plan.require_supported()
        if (
            self.plan.target is not GroupLoweringTarget.CUB_BLOCK
            or not isinstance(self.plan.call.operation, GroupTopKSemantics)
            or not isinstance(self.plan.implementation, AlgorithmSpec)
        ):
            raise ValueError("TopK requires a shared CUB block plan")
        p, spec = self.operation, self.implementation
        if p.key_dtype not in _types.ALL_PROVIDER_TYPES or (
            p.value_dtype is not None and p.value_dtype not in _types.ALL_PROVIDER_TYPES
        ):
            raise TypeError("TopK requires supported numeric dtypes")
        payload = "keys" if p.value_dtype is None else "pairs"
        tile = "full" if p.valid_items.kind is BindingKind.OMITTED else "partial"
        if (
            spec.struct_name != "BlockTopKCoop"
            or spec.method_name != f"{p.selection}_{payload}_{tile}"
        ):
            raise ValueError("TopK implementation does not match its plan")
        args = spec.template_arguments
        if (
            args.get("KeyT") is not p.key_dtype
            or args.get("ITEMS_PER_THREAD") != p.items_per_thread
            or args.get("ValueT")
            != (p.value_dtype if p.value_dtype is not None else "::cub::NullType")
        ):
            raise ValueError("TopK template payload does not match its plan")
        dimensions = self.plan.participation.exact_block_dim
        if (
            dimensions is None
            or dimensions[1:] != (1, 1)
            or args.get("BLOCK_DIM_X") != dimensions[0]
        ):
            raise ValueError("TopK requires matching one-dimensional block dimensions")
        expected = (
            (p.key_dtype,) if p.value_dtype is None else (p.key_dtype, p.value_dtype)
        )
        if (
            self.plan.result is None
            or tuple(item.dtype for item in self.plan.result.values) != expected
        ):
            raise ValueError("TopK result dtypes do not match its plan")
        if any(
            item.items_per_member != p.items_per_thread
            for item in self.plan.result.values
        ):
            raise ValueError("TopK result extent does not match its plan")
        storage, sync = self.plan.temp_storage, self.plan.synchronization
        if (
            storage is None
            or not storage.exact_layout_required
            or storage.instances != 1
            or sync is None
        ):
            raise ValueError("TopK requires one exact block scratch layout")
        if sync.storage_reuse_barrier is not (
            SynchronizationScope.BLOCK
            if storage.auto_sync
            else SynchronizationScope.NONE
        ):
            raise ValueError("TopK scratch synchronization does not match its plan")

    @property
    def operation(self):
        return self.plan.call.operation

    @property
    def implementation(self):
        return self.plan.implementation

    @property
    def cpp_type(self):
        values = []
        for name, value in self.implementation.ordered_template_arguments:
            if name in {"KeyT", "ValueT"} and value in _types.TYPE_SPECS:
                values.append(_types.TYPE_SPECS[value].cpp_type)
            elif isinstance(value, (int, str)) and not isinstance(value, bool):
                values.append(str(value))
            else:
                raise TypeError(f"Unsupported TopK template argument {name}")
        return f"::cub::BlockTopKCoop<{', '.join(values)}>"

    @property
    def scratch_requirement_key(self):
        return "cub_topk_storage", self.cpp_type

    @property
    def symbol_name(self):
        digest = hashlib.sha256(repr(self.plan.artifact_key).encode()).hexdigest()[:16]
        return f"cuda_coop_cutlass_topk_{digest}"

    def __eq__(self, other):
        return (
            isinstance(other, _CubTopKRequest)
            and self.plan.artifact_key == other.plan.artifact_key
        )

    def __hash__(self):
        return hash(self.plan.artifact_key)


def _render_topk(request):
    request.__post_init__()
    p = request.operation
    params, inputs, outputs = [], [], []
    payloads = [("keys", p.key_dtype)]
    if p.value_dtype is not None:
        payloads.append(("values", p.value_dtype))
    for name, dtype in payloads:
        cpp = _types.TYPE_SPECS[dtype].cpp_type
        params.extend(f"{cpp} {name}{i}" for i in range(p.items_per_thread))
        inputs.append(
            f"  {cpp} {name}[{p.items_per_thread}] = {{{', '.join(f'{name}{i}' for i in range(p.items_per_thread))}}};"
        )
        outputs.extend(
            f"  result_{name}[{i}] = {name}[{i}];" for i in range(p.items_per_thread)
        )
    args = [name for name, _ in payloads]
    for name, binding in (("k", p.k), ("valid_items", p.valid_items)):
        if binding.kind is BindingKind.RUNTIME:
            params.append(f"long long {name}")
            args.append(name)
        else:
            value = (
                p.items_per_thread * request.plan.participation.exact_block_dim[0]
                if binding.kind is BindingKind.OMITTED
                else binding.value
            )
            args.append(f"{value}ll")
    params.extend(
        ("unsigned int storage_address", "int storage_bytes", "int storage_auto_sync")
    )
    params.extend(
        f"{_types.TYPE_SPECS[dtype].cpp_type}* result_{name}"
        for name, dtype in payloads
    )
    return [
        f"void {request.symbol_name}({', '.join(params)}) {{",
        f"  using implementation_type = {request.cpp_type};",
        "  using storage_type = typename implementation_type::TempStorage;",
        "  if (storage_bytes <= 0 || (unsigned long long)storage_bytes < sizeof(storage_type) ||",
        "      (storage_address & (alignof(storage_type) - 1u)) != 0u) {",
        '    asm volatile("trap;");',
        "  }",
        "  unsigned long long generic_address;",
        '  asm("cvta.shared.u64 %0, %1;" : "=l"(generic_address) : "l"((unsigned long long)storage_address));',
        "  auto& storage = *reinterpret_cast<storage_type*>(generic_address);",
        *inputs,
        f"  implementation_type(storage).{request.implementation.method_name}({', '.join(args)});",
        "  if (storage_auto_sync != 0) { __syncthreads(); }",
        *outputs,
        "}",
    ]


_rendering.register_bundle_renderer(
    "cub_group_topk",
    render=_render_topk,
    include_lines=(f"#include <{_HEADER}>",),
    cccl_headers=((f"#include <{_HEADER}>", _HEADER),),
    scratch_layout_probe=lambda request: _rendering.make_scratch_layout_probe(
        request.scratch_requirement_key, f"typename {request.cpp_type}::TempStorage"
    ),
)


def _typed_item(value, dtype):
    if isinstance(value, np.generic):
        value = value.item()
    converted = _types.coerce_plain_scalar(
        value, dtype, name="TopK item", scope=_SCOPE, allow_nonfinite=True
    )
    return dtype(value) if converted is _types._NOT_PLAIN_SCALAR else converted


def provider_topk(
    *, group, launch, keys, values, selection, k, valid_items, temp_storage
):
    payloads = [keys] if values is None else [keys, values]
    resolved = [
        _types.resolve_thread_data_value_type(
            payload,
            allowed=_types.ALL_PROVIDER_TYPES,
            feature="topk",
            scope=_SCOPE,
            resolve_type=_resolve_type,
        )
        for payload in payloads
    ]
    plan = _make_topk_plan(
        group=group,
        launch=launch,
        key_type=resolved[0][0],
        value_type=None if values is None else resolved[1][0],
        items=keys.items_per_thread,
        selection=selection,
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )
    request = _CubTopKRequest(plan)
    arguments, parameter_types = [], []
    for dtype, items in resolved:
        arguments.extend(_typed_item(item, dtype) for item in items)
        parameter_types.extend([dtype] * len(items))
    for value, binding in (
        (k, request.operation.k),
        (valid_items, request.operation.valid_items),
    ):
        if binding.kind is BindingKind.RUNTIME:
            arguments.append(Int64(value))
            parameter_types.append(Int64)
    tensors = [
        _make_rmem_tensor(payload.items_per_thread, dtype, payload.alignment)
        for payload, (dtype, _) in zip(payloads, resolved)
    ]
    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(request)
        descriptor = TempStorage() if temp_storage is None else temp_storage
        arguments.extend(
            _storage.register_deferred_temp_storage_event(
                descriptor,
                primitive_name="topk",
                requirement_key=request.scratch_requirement_key,
            )
        )
        parameter_types.extend((Uint32, Int32, Int32))
        arguments.extend(tensor.iterator.llvm_ptr for tensor in tensors)
        parameter_types.extend([llvm.PointerType.get(0)] * len(tensors))
        ffi(name=request.symbol_name, params_types=parameter_types, return_type=None)(
            *arguments
        )
        results = tuple(
            ThreadData(
                payload.items_per_thread,
                dtype=_types.thread_data_output_dtype(payload, dtype),
                values=[dtype(tensor[i]) for i in range(payload.items_per_thread)],
                alignment=payload.alignment,
            )
            for payload, (dtype, _), tensor in zip(payloads, resolved, tensors)
        )
        return results[0] if values is None else results
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise
