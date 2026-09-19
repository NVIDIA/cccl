# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed CUB Merge Sort requests from the shared block and warp planner."""

import hashlib
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Int64, Uint32
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    ArgumentBinding,
    BindingKind,
    CxxOperator,
    Dependency,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupMergeSortSemantics,
    StorageOwnership,
    SynchronizationScope,
    make_block_merge_sort_semantics,
    make_group_primitive_call,
    plan_group_primitive,
)

from .._compiler import _rendering, _state, _storage, _types
from .._compiler._types import ALL_PROVIDER_TYPES, TYPE_SPECS
from .._temp_storage import TempStorage
from .._thread_data import ThreadData, _make_rmem_tensor

_SCOPE = "cuda.coop.cutlass"
_LESS = "::cuda::std::less<KeyT>"
_GREATER = "::cuda::std::greater<KeyT>"
_resolve_type = _types.make_provider_type_resolver(
    scope=_SCOPE, root_scope=_SCOPE, namespace="thread_group"
)


def _valid_binding(value):
    if value is None:
        return ArgumentBinding.omitted()
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Merge Sort valid_items must be an integer")
    if isinstance(value, Integral):
        return ArgumentBinding.static(int(value))
    spec = TYPE_SPECS.get(_types.canonical_dsl_type(value))
    if spec is None or spec.token[0] not in {"i", "u"} or spec.token == "u64":
        raise TypeError(
            "Merge Sort valid_items requires a signed integer up to 64 bits "
            "or an unsigned integer up to 32 bits"
        )
    return ArgumentBinding.runtime()


def _make_merge_sort_plan(
    *,
    group,
    launch,
    key_type,
    value_type,
    items_per_thread,
    descending,
    valid_items,
    temp_storage=None,
):
    if not isinstance(descending, bool):
        raise TypeError("Merge Sort descending must be a compile-time bool")
    valid = _valid_binding(valid_items)
    partial = valid.kind is not BindingKind.OMITTED
    primitive = make_block_merge_sort_semantics(
        key_dtype=key_type,
        value_dtype=value_type,
        items_per_thread=items_per_thread,
        compare_operator=CxxOperator(
            _GREATER if descending else _LESS, Dependency("KeyT"), "compare_op"
        ),
        valid_items=0 if partial else None,
        oob_default=0 if partial else None,
    )
    plan = plan_group_primitive(
        make_group_primitive_call(group, GroupMergeSortSemantics(primitive, valid)),
        launch,
    ).require_supported()
    if temp_storage is not None and plan.target is not GroupLoweringTarget.CUB_BLOCK:
        raise ValueError("Merge Sort temp_storage applies only to block groups")
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError("Merge Sort temp_storage must be CUTLASS TempStorage")
    if plan.target is GroupLoweringTarget.CUB_BLOCK:
        plan = replace(
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
    return plan


@dataclass(frozen=True, eq=False)
class _CubMergeSortRequest:
    plan: GroupLoweringPlan
    kind: str = "cub_group_merge_sort"

    def __post_init__(self):
        self.plan.require_supported()
        if self.plan.target not in {
            GroupLoweringTarget.CUB_BLOCK,
            GroupLoweringTarget.CUB_WARP,
        }:
            raise ValueError("Merge Sort requires a CUB block or warp plan")
        if not isinstance(self.plan.call.operation, GroupMergeSortSemantics):
            raise TypeError("Merge Sort request requires shared Merge Sort semantics")
        if not isinstance(self.plan.implementation, AlgorithmSpec):
            raise TypeError("Merge Sort request requires an AlgorithmSpec")
        if self.key_type not in ALL_PROVIDER_TYPES or (
            self.value_type is not None and self.value_type not in ALL_PROVIDER_TYPES
        ):
            raise TypeError("Merge Sort requires supported numeric dtypes")
        comparator = self.operation.primitive.compare_operator
        if not isinstance(comparator, CxxOperator) or comparator.cpp not in {
            _LESS,
            _GREATER,
        }:
            raise TypeError("Merge Sort supports built-in ordering only")
        expected = "BlockMergeSort" if self.is_block else "WarpMergeSort"
        if self.partial:
            expected = "CudaCoop" + expected
        if (
            self.implementation.struct_name != expected
            or self.implementation.method_name != "Sort"
        ):
            raise ValueError("Merge Sort implementation does not match its plan")
        arguments = self.implementation.template_arguments
        if (
            arguments.get("KeyT") is not self.key_type
            or arguments.get("ITEMS_PER_THREAD") != self.items
        ):
            raise ValueError("Merge Sort template payload does not match its plan")
        if arguments.get("ValueT") != (
            self.value_type if self.value_type else "::cub::NullType"
        ):
            raise ValueError("Merge Sort value dtype does not match its plan")
        dimensions = self.plan.participation.exact_block_dim
        if self.is_block:
            if (
                tuple(arguments.get(f"BLOCK_DIM_{axis}") for axis in "XYZ")
                != dimensions
            ):
                raise ValueError("Merge Sort block dimensions do not match its plan")
            if not self.plan.temp_storage.exact_layout_required:
                raise ValueError("Block Merge Sort requires exact scratch layout")
        elif (
            arguments.get("VIRTUAL_WARP_THREADS")
            != self.plan.resolved_group.static_size
        ):
            raise ValueError("Merge Sort warp width does not match its plan")
        expected_dtypes = (
            (self.key_type,)
            if self.value_type is None
            else (self.key_type, self.value_type)
        )
        result = self.plan.result
        if (
            result is None
            or tuple(item.dtype for item in result.values) != expected_dtypes
        ):
            raise ValueError("Merge Sort result dtypes do not match its plan")
        if any(item.items_per_member != self.items for item in result.values):
            raise ValueError("Merge Sort result extents do not match its plan")

    @property
    def operation(self):
        return self.plan.call.operation

    @property
    def implementation(self):
        return self.plan.implementation

    @property
    def key_type(self):
        return self.operation.primitive.key_dtype

    @property
    def value_type(self):
        return self.operation.primitive.value_dtype

    @property
    def items(self):
        return self.operation.primitive.items_per_thread

    @property
    def partial(self):
        return self.operation.primitive.has_partial_tile

    @property
    def is_block(self):
        return self.plan.target is GroupLoweringTarget.CUB_BLOCK

    @property
    def cpp_type(self):
        arguments = []
        for name, value in self.implementation.ordered_template_arguments:
            if name in {"KeyT", "ValueT"} and value in TYPE_SPECS:
                arguments.append(TYPE_SPECS[value].cpp_type)
            elif isinstance(value, (int, str)) and not isinstance(value, bool):
                arguments.append(str(value))
            else:
                raise TypeError(f"Unsupported Merge Sort template argument {name}")
        return f"::cub::{self.implementation.struct_name}<{', '.join(arguments)}>"

    @property
    def scratch_requirement_key(self):
        return "cub_merge_sort_storage", self.cpp_type

    @property
    def symbol_name(self):
        digest = hashlib.sha256(repr(self.plan.artifact_key).encode()).hexdigest()[:16]
        return f"cuda_coop_cutlass_merge_sort_{digest}"

    def __eq__(self, other):
        return (
            isinstance(other, _CubMergeSortRequest)
            and self.plan.artifact_key == other.plan.artifact_key
        )

    def __hash__(self):
        return hash(self.plan.artifact_key)


def _render_merge_sort(request):
    request.__post_init__()
    key_cpp = TYPE_SPECS[request.key_type].cpp_type
    params, inputs, outputs = [], [], []
    names = [("keys", request.key_type)]
    if request.value_type is not None:
        names.append(("values", request.value_type))
    for name, dtype in names:
        cpp = TYPE_SPECS[dtype].cpp_type
        params.extend(f"{cpp} {name}{i}" for i in range(request.items))
        items = ", ".join(f"{name}{i}" for i in range(request.items))
        inputs.append(f"  {cpp} {name}[{request.items}] = {{{items}}};")
        outputs.extend(
            f"  result_{name}[{i}] = {name}[{i}];" for i in range(request.items)
        )
    args = [name for name, _ in names]
    args.append(request.operation.primitive.compare_operator.cpp + "{}")
    if request.partial:
        if request.operation.valid_items.kind is BindingKind.RUNTIME:
            params.append("long long valid_items")
            args.append("valid_items")
        else:
            args.append(f"{request.operation.valid_items.value}ll")
        params.append(f"{key_cpp} oob_default")
        args.append("oob_default")
    if request.is_block:
        params.extend(
            (
                "unsigned int storage_address",
                "int storage_bytes",
                "int storage_auto_sync",
            )
        )
        storage_lines = [
            "  using storage_type = typename implementation_type::TempStorage;",
            "  if (storage_bytes <= 0 || (unsigned long long)storage_bytes < sizeof(storage_type) ||",
            "      (storage_address & (alignof(storage_type) - 1u)) != 0u) {",
            '    asm volatile("trap;");',
            "  }",
            "  unsigned long long generic_address;",
            '  asm("cvta.shared.u64 %0, %1;" : "=l"(generic_address) : "l"((unsigned long long)storage_address));',
            "  auto& storage = *reinterpret_cast<storage_type*>(generic_address);",
        ]
        barrier = ["  if (storage_auto_sync != 0) { __syncthreads(); }"]
    else:
        x, y, z = request.plan.participation.exact_block_dim
        width = request.plan.resolved_group.static_size
        storage_lines = [
            f"  __shared__ typename implementation_type::TempStorage scratch[{x * y * z // width}];",
            "  unsigned int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);",
            f"  auto& storage = scratch[tid / {width}u];",
        ]
        mask = (
            "0xffffffffu"
            if width == 32
            else f"{(1 << width) - 1}u << (tid % 32u / {width}u * {width}u)"
        )
        barrier = [f"  __syncwarp({mask});"]
    params.extend(
        f"{TYPE_SPECS[dtype].cpp_type}* result_{name}" for name, dtype in names
    )
    return [
        f"void {request.symbol_name}({', '.join(params)}) {{",
        f"  using KeyT = {key_cpp};",
        f"  using implementation_type = {request.cpp_type};",
        *storage_lines,
        *inputs,
        f"  implementation_type(storage).Sort({', '.join(args)});",
        *barrier,
        *outputs,
        "}",
    ]


def _scratch_probe(request):
    if not request.is_block:
        return None
    return _rendering.make_scratch_layout_probe(
        request.scratch_requirement_key, f"typename {request.cpp_type}::TempStorage"
    )


_rendering.register_bundle_renderer(
    "cub_group_merge_sort",
    render=_render_merge_sort,
    include_lines=(
        "#include <cub/block/block_merge_sort.cuh>",
        "#include <cub/warp/warp_merge_sort.cuh>",
        "#include <cuda/std/functional>",
    ),
    cccl_headers=(
        ("#include <cub/block/block_merge_sort.cuh>", "cub/block/block_merge_sort.cuh"),
        ("#include <cub/warp/warp_merge_sort.cuh>", "cub/warp/warp_merge_sort.cuh"),
    ),
    scratch_layout_probe=_scratch_probe,
)


def _typed_value(value, dtype, *, sentinel=False):
    if isinstance(value, np.generic):
        if _types.canonical_dsl_type(value) is not dtype:
            raise TypeError("Merge Sort scalar dtype does not match payload dtype")
        value = value.item()
    converted = _types.coerce_plain_scalar(
        value,
        dtype,
        name="Merge Sort oob_default" if sentinel else "Merge Sort item",
        scope=_SCOPE,
        allow_nonfinite=not sentinel,
    )
    if converted is not _types._NOT_PLAIN_SCALAR:
        return converted
    if _types.canonical_dsl_type(value) is not dtype:
        raise TypeError(
            "Merge Sort oob_default dtype does not match keys dtype"
            if sentinel
            else "Merge Sort item dtype mismatch"
        )
    return dtype(value)


def provider_merge_sort(
    *, group, launch, keys, values, descending, valid_items, oob_default, temp_storage
):
    payloads = [keys] if values is None else [keys, values]
    resolved = [
        _types.resolve_thread_data_value_type(
            payload,
            allowed=ALL_PROVIDER_TYPES,
            feature="merge_sort",
            scope=_SCOPE,
            resolve_type=_resolve_type,
        )
        for payload in payloads
    ]
    key_type = resolved[0][0]
    plan = _make_merge_sort_plan(
        group=group,
        launch=launch,
        key_type=key_type,
        value_type=None if values is None else resolved[1][0],
        items_per_thread=keys.items_per_thread,
        descending=descending,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )
    request = _CubMergeSortRequest(plan)
    arguments, parameter_types = [], []
    for dtype, items in resolved:
        arguments.extend(_typed_value(item, dtype) for item in items)
        parameter_types.extend([dtype] * len(items))
    if request.partial:
        if request.operation.valid_items.kind is BindingKind.RUNTIME:
            arguments.append(Int64(valid_items))
            parameter_types.append(Int64)
        arguments.append(_typed_value(oob_default, key_type, sentinel=True))
        parameter_types.append(key_type)
    tensors = [
        _make_rmem_tensor(payload.items_per_thread, dtype, payload.alignment)
        for payload, (dtype, _) in zip(payloads, resolved)
    ]
    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(request)
        if request.is_block:
            descriptor = TempStorage() if temp_storage is None else temp_storage
            arguments.extend(
                _storage.register_deferred_temp_storage_event(
                    descriptor,
                    primitive_name="merge_sort",
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
