# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed CUB Histogram calls with deferred scratch and fresh counters."""

import hashlib
from dataclasses import dataclass, replace

from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    GroupLoweringPlan,
    GroupLoweringTarget,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.group.histogram import GroupHistogramSemantics

from .._compiler import _rendering, _state, _storage, _types
from .._temp_storage import TempStorage
from .._thread_data import ThreadData, _make_rmem_tensor

_SAMPLES = frozenset({Uint8, Int32, Uint32, Int64, Uint64})
_COUNTERS = frozenset({Int32, Uint32, Int64, Uint64})
_resolve_type = _types.make_provider_type_resolver(
    scope="cuda.coop.cutlass", root_scope="cuda.coop.cutlass", namespace="histogram"
)


def _make_histogram_plan(
    *,
    group,
    launch,
    sample_type,
    items,
    bins,
    bins_per_thread,
    counter_type,
    algorithm,
    temp_storage=None,
):
    operation = GroupHistogramSemantics(
        sample_type, items, bins, bins_per_thread, counter_type, algorithm
    )
    plan = plan_group_primitive(
        make_group_primitive_call(group, operation), launch
    ).require_supported()
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError("histogram temp_storage must be CUTLASS TempStorage")
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
class _CubHistogramRequest:
    plan: GroupLoweringPlan
    kind: str = "cub_group_histogram"

    def __post_init__(self):
        self.plan.require_supported()
        if (
            self.plan.target is not GroupLoweringTarget.CUB_BLOCK
            or not isinstance(self.plan.call.operation, GroupHistogramSemantics)
            or not isinstance(self.implementation, AlgorithmSpec)
        ):
            raise TypeError("histogram request requires a shared CUB block plan")
        if (
            self.operation.sample_dtype not in _SAMPLES
            or self.operation.counter_dtype not in _COUNTERS
        ):
            raise TypeError(
                "histogram request has an unsupported sample or counter dtype"
            )
        expected = dict(
            SampleT=self.operation.sample_dtype,
            BLOCK_DIM_X=self.plan.participation.exact_block_dim[0],
            ITEMS_PER_THREAD=self.operation.items_per_thread,
            BINS=self.operation.bins,
            BINS_PER_THREAD=self.operation.bins_per_thread,
            CounterT=self.operation.counter_dtype,
            ALGORITHM="::cub::BLOCK_HISTO_ATOMIC"
            if self.operation.algorithm == "atomic"
            else "::cub::BLOCK_HISTO_SORT",
        )
        if (
            self.implementation.method_name != "Histogram"
            or self.implementation.template_arguments != expected
        ):
            raise ValueError("histogram implementation does not match its plan")
        if not self.plan.temp_storage.exact_layout_required:
            raise ValueError("histogram requires exact scratch layout")
        result = self.plan.result
        if (
            result is None
            or len(result.values) != 1
            or result.values[0].dtype is not self.operation.counter_dtype
            or result.values[0].items_per_member != self.operation.bins_per_thread
        ):
            raise ValueError("histogram result does not match its plan")

    @property
    def operation(self):
        return self.plan.call.operation

    @property
    def implementation(self):
        return self.plan.implementation

    @property
    def cpp_type(self):
        arguments = [
            _types.TYPE_SPECS[value].cpp_type
            if name in {"SampleT", "CounterT"}
            else str(value)
            for name, value in self.implementation.ordered_template_arguments
        ]
        return f"::cub::{self.implementation.struct_name}<{', '.join(arguments)}>"

    @property
    def scratch_requirement_key(self):
        return "cub_histogram_storage", self.cpp_type

    @property
    def symbol_name(self):
        digest = hashlib.sha256(repr(self.plan.artifact_key).encode()).hexdigest()[:16]
        return f"cuda_coop_cutlass_histogram_{digest}"

    def __eq__(self, other):
        return (
            isinstance(other, _CubHistogramRequest)
            and self.plan.artifact_key == other.plan.artifact_key
        )

    def __hash__(self):
        return hash(self.plan.artifact_key)


def _render_histogram(request):
    request.__post_init__()
    operation = request.operation
    sample_cpp = _types.TYPE_SPECS[operation.sample_dtype].cpp_type
    counter_cpp = _types.TYPE_SPECS[operation.counter_dtype].cpp_type
    params = [f"{sample_cpp} item{i}" for i in range(operation.items_per_thread)]
    params.extend(
        (
            "unsigned int storage_address",
            "int storage_bytes",
            "int storage_auto_sync",
            f"{counter_cpp}* result",
        )
    )
    values = ", ".join(f"item{i}" for i in range(operation.items_per_thread))
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
        f"  {sample_cpp} samples[{operation.items_per_thread}] = {{{values}}};",
        f"  {counter_cpp} counts[{operation.bins_per_thread}];",
        "  implementation_type(storage).Histogram(samples, counts);",
        "  if (storage_auto_sync != 0) { __syncthreads(); }",
        *(f"  result[{i}] = counts[{i}];" for i in range(operation.bins_per_thread)),
        "}",
    ]


def _scratch_probe(request):
    return _rendering.make_scratch_layout_probe(
        request.scratch_requirement_key, f"typename {request.cpp_type}::TempStorage"
    )


_HEADERS = ("cub/block/block_histogram.cuh", "cuda/std/__type_traits/conditional.h")
_rendering.register_bundle_renderer(
    "cub_group_histogram",
    render=_render_histogram,
    include_lines=tuple(f"#include <{header}>" for header in _HEADERS),
    cccl_headers=tuple((f"#include <{header}>", header) for header in _HEADERS),
    scratch_layout_probe=_scratch_probe,
)


def provider_histogram(
    *,
    group,
    launch,
    samples,
    bins,
    bins_per_thread,
    counter_dtype,
    algorithm,
    temp_storage,
):
    sample_type, values = _types.resolve_thread_data_value_type(
        samples,
        allowed=_SAMPLES,
        feature="samples",
        scope="cuda.coop.cutlass",
        resolve_type=_resolve_type,
    )
    counter_type = _resolve_type(
        Int32 if counter_dtype is None else counter_dtype,
        allowed=_COUNTERS,
        feature="counter_dtype",
    )
    request = _CubHistogramRequest(
        _make_histogram_plan(
            group=group,
            launch=launch,
            sample_type=sample_type,
            items=len(values),
            bins=bins,
            bins_per_thread=bins_per_thread,
            counter_type=counter_type,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )
    )
    arguments = [sample_type(value) for value in values]
    parameter_types = [sample_type] * len(values)
    tensor = _make_rmem_tensor(bins_per_thread, counter_type, samples.alignment)
    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(request)
        descriptor = TempStorage() if temp_storage is None else temp_storage
        arguments.extend(
            _storage.register_deferred_temp_storage_event(
                descriptor,
                primitive_name="histogram",
                requirement_key=request.scratch_requirement_key,
            )
        )
        parameter_types.extend((Uint32, Int32, Int32))
        arguments.append(tensor.iterator.llvm_ptr)
        parameter_types.append(llvm.PointerType.get(0))
        ffi(name=request.symbol_name, params_types=parameter_types, return_type=None)(
            *arguments
        )
        return ThreadData(
            bins_per_thread,
            dtype=counter_type
            if counter_dtype is None or counter_dtype is int
            else counter_dtype,
            values=[counter_type(tensor[i]) for i in range(bins_per_thread)],
            alignment=samples.alignment,
        )
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise
