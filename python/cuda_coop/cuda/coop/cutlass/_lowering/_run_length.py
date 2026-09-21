# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed Run Length Decode providers using the shared checked CUB driver."""

import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Integral

import numpy as np
from cutlass import cute
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Int64, Uint32, Uint64
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
from cuda.coop._core.group.run_length import GroupRunLengthDecodeSemantics

from .._compiler import _rendering, _state, _storage, _types
from .._temp_storage import TempStorage
from .._thread_data import ThreadData, _make_rmem_tensor
from ._load_store import _try_raw_memory_pointer

_SCOPE = "cuda.coop.cutlass"
_HEADERS = (
    "cub/block/block_run_length_decode.cuh",
    "cub/block/block_scan.cuh",
    "cuda/std/limits",
    "cuda/std/type_traits",
)
_resolve_type = _types.make_provider_type_resolver(
    scope=_SCOPE, root_scope=_SCOPE, namespace="thread_group"
)


def _offset_binding(value):
    if isinstance(value, (bool, np.bool_, Enum)):
        raise TypeError("run_length_decode offset must be an integer, not bool or Enum")
    if isinstance(value, Integral):
        return ArgumentBinding.static(int(value)), Uint64
    dtype = _types.canonical_dsl_type(value)
    if dtype not in _types.INTEGER_VALUE_TYPES:
        raise TypeError("run_length_decode offset must be an integer up to 64 bits")
    return ArgumentBinding.runtime(), dtype


def _make_run_length_plan(
    *,
    group,
    launch,
    value_type,
    length_type,
    runs,
    decoded,
    offset=0,
    bulk=False,
    temp_storage=None,
):
    binding, control_type = _offset_binding(offset)
    operation = GroupRunLengthDecodeSemantics(
        item_dtype=value_type,
        run_length_dtype=length_type,
        runs_per_thread=runs,
        decoded_items_per_thread=decoded,
        offset=binding,
        decoded_offset_dtype=Uint32,
        control_dtype=control_type,
        bulk=bulk,
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
            storage_reuse_barrier=(
                SynchronizationScope.BLOCK
                if temp_storage is None or temp_storage.auto_sync
                else SynchronizationScope.NONE
            ),
        ),
    )


@dataclass(frozen=True, eq=False)
class _CubRunLengthRequest:
    plan: GroupLoweringPlan
    kind: str = "cub_group_run_length"

    def __post_init__(self):
        self.plan.require_supported()
        if (
            self.plan.target is not GroupLoweringTarget.CUB_BLOCK
            or not isinstance(self.plan.call.operation, GroupRunLengthDecodeSemantics)
            or not isinstance(self.plan.implementation, AlgorithmSpec)
        ):
            raise ValueError("run_length_decode requires a shared CUB block plan")
        p, spec = self.operation, self.implementation
        if (
            p.item_dtype not in _types.ALL_PROVIDER_TYPES
            or p.run_length_dtype not in _types.INTEGER_VALUE_TYPES
            or p.control_dtype not in _types.INTEGER_VALUE_TYPES
        ):
            raise TypeError(
                "run_length_decode requires numeric values and integer lengths/offsets"
            )
        if p.decoded_offset_dtype is not Uint32 or p.relative_offsets:
            raise ValueError(
                "run_length_decode requires the common uint32 result profile"
            )
        if spec.struct_name != "BlockRunLengthDecodeCoop" or spec.method_name != (
            "Into" if p.bulk else "Window"
        ):
            raise ValueError("run_length_decode implementation does not match its plan")
        dimensions = self.plan.participation.exact_block_dim
        if dimensions is None or dimensions[1:] != (1, 1):
            raise ValueError(
                "run_length_decode requires exact one-dimensional block dimensions"
            )
        expected = dict(
            ItemT=p.item_dtype,
            LengthT=p.run_length_dtype,
            OffsetT=Uint32,
            ControlT=p.control_dtype,
            BlockThreads=dimensions[0],
            RunsPerThread=p.runs_per_thread,
            DecodedItemsPerThread=p.decoded_items_per_thread,
        )
        if dict(spec.template_arguments) != expected:
            raise ValueError(
                "run_length_decode template arguments do not match its plan"
            )
        results = self.plan.result.values if self.plan.result is not None else ()
        if (
            len(results) != 1
            or results[0].dtype is not (Uint32 if p.bulk else p.item_dtype)
            or results[0].items_per_member
            != (1 if p.bulk else p.decoded_items_per_thread)
        ):
            raise ValueError("run_length_decode result does not match its plan")
        storage, sync = self.plan.temp_storage, self.plan.synchronization
        if (
            storage is None
            or not storage.exact_layout_required
            or storage.instances != 1
            or sync is None
        ):
            raise ValueError(
                "run_length_decode requires one exact block scratch layout"
            )
        if sync.storage_reuse_barrier is not (
            SynchronizationScope.BLOCK
            if storage.auto_sync
            else SynchronizationScope.NONE
        ):
            raise ValueError(
                "run_length_decode storage synchronization does not match its plan"
            )

    @property
    def operation(self):
        return self.plan.call.operation

    @property
    def implementation(self):
        return self.plan.implementation

    @property
    def cpp_type(self):
        args = [
            _types.TYPE_SPECS[value].cpp_type
            if value in _types.TYPE_SPECS
            else str(value)
            for _, value in self.implementation.ordered_template_arguments
        ]
        return f"::cub::BlockRunLengthDecodeCoop<{', '.join(args)}>"

    @property
    def scratch_requirement_key(self):
        return "cub_run_length_storage", self.cpp_type

    @property
    def symbol_name(self):
        digest = hashlib.sha256(repr(self.plan.artifact_key).encode()).hexdigest()[:16]
        return f"cuda_coop_cutlass_run_length_{digest}"

    def __eq__(self, other):
        return (
            isinstance(other, _CubRunLengthRequest)
            and self.plan.artifact_key == other.plan.artifact_key
        )

    def __hash__(self):
        return hash(self.plan.artifact_key)


def _render_run_length(request):
    request.__post_init__()
    p = request.operation
    item_cpp = _types.TYPE_SPECS[p.item_dtype].cpp_type
    params, inputs = [], []
    for name, dtype in (("values", p.item_dtype), ("lengths", p.run_length_dtype)):
        cpp = _types.TYPE_SPECS[dtype].cpp_type
        params.extend(f"{cpp} {name}{i}" for i in range(p.runs_per_thread))
        inputs.append(
            f"  {cpp} {name}[{p.runs_per_thread}] = {{{', '.join(f'{name}{i}' for i in range(p.runs_per_thread))}}};"
        )
    if p.bulk:
        params.extend((f"{item_cpp}* destination", "long long capacity"))
    if p.offset.kind is BindingKind.RUNTIME:
        params.append(f"{_types.TYPE_SPECS[p.control_dtype].cpp_type} offset")
        offset = "offset"
    else:
        offset = f"{p.offset.value}ULL"
    params.extend(
        ("unsigned int storage_address", "int storage_bytes", "int storage_auto_sync")
    )
    if p.bulk:
        result_type = "unsigned int"
        body = [
            "  unsigned int total;",
            f"  implementation_type(storage).Into(values, lengths, destination, capacity, {offset}, total);",
        ]
        outputs = ["  return total;"]
    else:
        result_type = "void"
        params.append(f"{item_cpp}* result")
        body = [
            f"  {item_cpp} decoded[{p.decoded_items_per_thread}];",
            "  unsigned int total[1];",
            f"  unsigned int relative[{p.decoded_items_per_thread}];",
            f"  implementation_type(storage).Window(values, lengths, decoded, total, relative, {offset});",
        ]
        outputs = [
            f"  result[{i}] = decoded[{i}];" for i in range(p.decoded_items_per_thread)
        ]
    return [
        f"{result_type} {request.symbol_name}({', '.join(params)}) {{",
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
        *body,
        "  if (storage_auto_sync != 0) { __syncthreads(); }",
        *outputs,
        "}",
    ]


_rendering.register_bundle_renderer(
    "cub_group_run_length",
    render=_render_run_length,
    include_lines=tuple(f"#include <{header}>" for header in _HEADERS),
    cccl_headers=tuple((f"#include <{header}>", header) for header in _HEADERS),
    scratch_layout_probe=lambda request: _rendering.make_scratch_layout_probe(
        request.scratch_requirement_key, f"typename {request.cpp_type}::TempStorage"
    ),
)


def _destination(destination, value_type):
    message = "run_length_decode_into destination must be a contiguous one-dimensional global-memory tensor with the run-value dtype"
    if (
        not isinstance(destination, cute.Tensor)
        or destination.element_type is not value_type
    ):
        raise TypeError(message)
    if destination.iterator.memspace != cute.AddressSpace.gmem:
        raise TypeError(message)
    shape, stride = destination.shape, destination.stride
    if isinstance(shape, tuple):
        if len(shape) != 1 or isinstance(shape[0], tuple):
            raise TypeError(message)
        shape, stride = shape[0], stride[0]
    if not isinstance(stride, Integral) or stride != 1:
        raise TypeError(message)
    if isinstance(shape, Integral):
        if not 0 <= shape <= (1 << 63) - 1:
            raise ValueError(
                "run_length_decode_into destination capacity must fit a nonnegative int64"
            )
    elif _types.canonical_dsl_type(shape) not in _types.INTEGER_VALUE_TYPES:
        raise TypeError(message)
    pointer = _try_raw_memory_pointer(destination)
    if pointer is None:
        raise TypeError(message)
    return pointer, Int64(shape)


def _typed_item(value, dtype):
    if isinstance(value, np.generic):
        value = value.item()
    converted = _types.coerce_plain_scalar(
        value, dtype, name="run_length_decode item", scope=_SCOPE, allow_nonfinite=True
    )
    return dtype(value) if converted is _types._NOT_PLAIN_SCALAR else converted


def provider_run_length_decode(
    *,
    group,
    launch,
    values,
    lengths,
    decoded_items_per_thread,
    offset,
    destination,
    bulk,
    temp_storage,
):
    resolved = [
        _types.resolve_thread_data_value_type(
            payload,
            allowed=allowed,
            feature="run_length_decode",
            scope=_SCOPE,
            resolve_type=_resolve_type,
        )
        for payload, allowed in (
            (values, _types.ALL_PROVIDER_TYPES),
            (lengths, _types.INTEGER_VALUE_TYPES),
        )
    ]
    plan = _make_run_length_plan(
        group=group,
        launch=launch,
        value_type=resolved[0][0],
        length_type=resolved[1][0],
        runs=values.items_per_thread,
        decoded=decoded_items_per_thread,
        offset=offset,
        bulk=bulk,
        temp_storage=temp_storage,
    )
    request = _CubRunLengthRequest(plan)
    arguments, parameter_types = [], []
    for dtype, items in resolved:
        arguments.extend(_typed_item(item, dtype) for item in items)
        parameter_types.extend([dtype] * len(items))
    if bulk:
        arguments.extend(_destination(destination, resolved[0][0]))
        parameter_types.extend((llvm.PointerType.get(0), Int64))
    if request.operation.offset.kind is BindingKind.RUNTIME:
        dtype = request.operation.control_dtype
        arguments.append(dtype(offset))
        parameter_types.append(dtype)
    tensor = (
        None
        if bulk
        else _make_rmem_tensor(
            decoded_items_per_thread, resolved[0][0], values.alignment
        )
    )
    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(request)
        descriptor = TempStorage() if temp_storage is None else temp_storage
        arguments.extend(
            _storage.register_deferred_temp_storage_event(
                descriptor,
                primitive_name="run_length_decode",
                requirement_key=request.scratch_requirement_key,
            )
        )
        parameter_types.extend((Uint32, Int32, Int32))
        if not bulk:
            arguments.append(tensor.iterator.llvm_ptr)
            parameter_types.append(llvm.PointerType.get(0))
        result = ffi(
            name=request.symbol_name,
            params_types=parameter_types,
            return_type=Uint32 if bulk else None,
        )(*arguments)
        if bulk:
            return Uint32(result)
        dtype = resolved[0][0]
        return ThreadData(
            decoded_items_per_thread,
            dtype=_types.thread_data_output_dtype(values, dtype),
            values=[dtype(tensor[i]) for i in range(decoded_items_per_thread)],
            alignment=values.alignment,
        )
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise
