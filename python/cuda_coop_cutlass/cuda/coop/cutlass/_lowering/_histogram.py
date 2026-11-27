# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generate CUB calls for fresh and cumulative block histograms.

CUB writes a block-wide histogram into shared memory. ``cuda.coop`` returns
counts in registers instead. The C++ adapter therefore keeps CUB scratch and
one shared counter per bin in a single storage type. Each call zeroes those
counters, counts the samples, and distributes bins round-robin across threads.
A C++ probe reports the storage type's size and alignment so the compiler can
allocate matching scratch. Composite instead passes the caller's shared counter
address to CUB and probes the primitive's own scratch type.
"""

import hashlib
from dataclasses import dataclass, replace

from cuda.coop.cutlass._planning import (
    Algorithm,
    GroupLoweringPlan,
    GroupLoweringTarget,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop.cutlass._planning.group.histogram import (
    GroupHistogramCompositeSemantics,
    GroupHistogramSemantics,
)
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cute.ffi import ffi

from .._compiler import _rendering, _state, _storage, _types
from .._temp_storage import TempStorage
from .._thread_data import ThreadData, _make_rmem_tensor

_SAMPLES = frozenset({Uint8, Int32, Uint32, Int64, Uint64})
_COUNTERS = frozenset({Int32, Uint32, Int64, Uint64})
_resolve_type = _types.make_provider_type_resolver(
    scope="cuda.coop.cutlass",
    root_scope="cuda.coop.cutlass",
    namespace="histogram",
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
    """Plan sample/counter types, bin coverage, and exact scratch.

    Shared planning checks the one-dimensional block and static capacity
    limits. Attach requested scratch size, alignment, sharing, and reuse
    policy, or retain automatic allocation with a trailing block barrier.
    """

    operation = GroupHistogramSemantics(
        sample_type, items, bins, bins_per_thread, counter_type, algorithm
    )
    return _plan_with_storage(group, launch, operation, temp_storage)


def _plan_with_storage(group, launch, operation, temp_storage):
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
    """Describe one specialized CUB histogram wrapper call.

    The plan artifact key controls equality, hashing, and the wrapper symbol.
    Block width, items per thread, sample and counter dtypes, bin count,
    output extent, and algorithm select the C++ type, its scratch layout, and
    the result contract.
    """

    plan: GroupLoweringPlan

    @property
    def composite(self):
        return isinstance(self.operation, GroupHistogramCompositeSemantics)

    @property
    def kind(self):
        return (
            "cub_group_histogram_composite"
            if self.composite
            else "cub_group_histogram"
        )

    def __post_init__(self):
        """Check that the plan matches the CUB histogram wrapper contract.

        Require a CUB block plan, supported sample and counter dtypes,
        matching template arguments and method, exact scratch, and one counter
        result of the planned extent. These checks cannot see runtime values.
        Callers must keep each sample in [0, bins).
        """

        self.plan.require_supported()
        if (
            self.plan.target is not GroupLoweringTarget.CUB_BLOCK
            or not isinstance(
                self.operation,
                (GroupHistogramSemantics, GroupHistogramCompositeSemantics),
            )
            or not isinstance(self.implementation, Algorithm)
        ):
            raise TypeError(
                "histogram request requires a shared CUB block plan"
            )
        if (
            self.operation.sample_dtype not in _SAMPLES
            or self.operation.counter_dtype not in _COUNTERS
        ):
            raise TypeError(
                "histogram request has an unsupported sample or counter dtype"
            )
        expected = {
            "SampleT": self.operation.sample_dtype,
            "BLOCK_DIM_X": self.plan.participation.exact_block_dim[0],
            "ITEMS_PER_THREAD": self.operation.items_per_thread,
            "BINS": self.operation.bins,
            "ALGORITHM": "::cub::BLOCK_HISTO_ATOMIC"
            if self.operation.algorithm == "atomic"
            else "::cub::BLOCK_HISTO_SORT",
        }
        if not self.composite:
            expected.update(
                BINS_PER_THREAD=self.operation.bins_per_thread,
                CounterT=self.operation.counter_dtype,
            )
        method = "Composite" if self.composite else "Histogram"
        if (
            self.implementation.method_name != method
            or self.implementation.template_arguments != expected
        ):
            raise ValueError("histogram implementation does not match its plan")
        if not self.plan.temp_storage.exact_layout_required:
            raise ValueError("histogram requires exact scratch layout")
        result = self.plan.result
        if self.composite:
            if result is not None:
                raise ValueError(
                    "histogram_composite must update counters in place"
                )
            return
        if (
            result is None
            or len(result.values) != 1
            or result.values[0].dtype is not self.operation.counter_dtype
            or result.values[0].items_per_member
            != self.operation.bins_per_thread
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
        """Spell the C++ type used by calls and layout probes."""

        arguments = [
            _types.TYPE_SPECIFICATIONS[value].cpp_type
            if name in {"SampleT", "CounterT"}
            else str(value)
            for name, value in self.implementation.ordered_template_arguments
        ]
        return (
            f"::cub::{self.implementation.struct_name}<{', '.join(arguments)}>"
        )

    @property
    def scratch_requirement_key(self):
        """Key scratch uses by adapter type to share one layout probe."""

        if self.composite:
            return "cub_histogram_composite_storage", self.cpp_type
        return "cub_histogram_storage", self.cpp_type

    @property
    def symbol_name(self):
        """Hash the complete plan identity into a wrapper symbol."""

        digest = hashlib.sha256(
            repr(self.plan.artifact_key).encode()
        ).hexdigest()[:16]
        if self.composite:
            return f"cuda_coop_cutlass_histogram_composite_{digest}"
        return f"cuda_coop_cutlass_histogram_{digest}"

    def __eq__(self, other):
        return (
            isinstance(other, _CubHistogramRequest)
            and self.plan.artifact_key == other.plan.artifact_key
        )

    def __hash__(self):
        return hash(self.plan.artifact_key)


def _render_histogram(request):
    """Generate a fresh histogram or an update of caller-owned shared bins.

    Both wrappers copy the sample values into the mutable array expected by
    CUB and honor the requested trailing barrier. Fresh histograms return a
    per-thread array. Composite instead receives the CTA-local address of
    the caller's bins and updates them directly.
    """

    request.__post_init__()
    operation = request.operation
    sample_cpp = _types.TYPE_SPECIFICATIONS[operation.sample_dtype].cpp_type
    counter_cpp = _types.TYPE_SPECIFICATIONS[operation.counter_dtype].cpp_type
    params = [
        f"{sample_cpp} item{i}" for i in range(operation.items_per_thread)
    ]
    params.extend(
        (
            "unsigned int storage_address",
            "int storage_bytes",
            "int storage_auto_sync",
            "unsigned int counters_address"
            if request.composite
            else f"{counter_cpp}* result",
        )
    )
    values = ", ".join(f"item{i}" for i in range(operation.items_per_thread))
    lines = [
        f"void {request.symbol_name}({', '.join(params)}) {{",
        f"  using implementation_type = {request.cpp_type};",
        "  using storage_type = typename implementation_type::TempStorage;",
        (
            "  if (storage_bytes <= 0 || (unsigned long long)storage_bytes "
            "< sizeof(storage_type) ||"
        ),
        "      (storage_address & (alignof(storage_type) - 1u)) != 0u) {",
        '    asm volatile("trap;");',
        "  }",
        "  auto* const generic_address =",
        "      __cvta_shared_to_generic(storage_address);",
        "  auto& storage = *reinterpret_cast<storage_type*>(generic_address);",
        f"  {sample_cpp} samples[{operation.items_per_thread}] = {{{values}}};",
    ]
    if request.composite:
        # Unsigned arithmetic preserves the bit pattern for signed counters.
        # CUDA atomicAdd supports unsigned long long for 64-bit counters.
        unsigned = (
            "unsigned long long"
            if _types.TYPE_SPECIFICATIONS[operation.counter_dtype].width_bits
            == 64
            else "unsigned int"
        )
        return [
            *lines,
            f"  using counter_type = {unsigned};",
            "  if ((counters_address & (alignof(counter_type) - 1u)) != 0u) {",
            '    asm volatile("trap;");',
            "  }",
            f"  using counters_type = counter_type[{operation.bins}];",
            "  auto& counters = *reinterpret_cast<counters_type*>(",
            "      __cvta_shared_to_generic(counters_address));",
            "  implementation_type(storage).Composite(samples, counters);",
            "  if (storage_auto_sync != 0) { __syncthreads(); }",
            "}",
        ]
    return [
        *lines,
        f"  {counter_cpp} counts[{operation.bins_per_thread}];",
        "  implementation_type(storage).Histogram(samples, counts);",
        "  if (storage_auto_sync != 0) { __syncthreads(); }",
        *(
            f"  result[{i}] = counts[{i}];"
            for i in range(operation.bins_per_thread)
        ),
        "}",
    ]


def _scratch_probe(request):
    """Probe the storage for the selected operation and specialization."""

    return _rendering.make_scratch_layout_probe(
        request.scratch_requirement_key,
        f"typename {request.cpp_type}::TempStorage",
    )


_HEADERS = (
    "cub/block/block_histogram.cuh",
    "cuda/std/__type_traits/conditional.h",
)
for _kind, _headers in (
    ("cub_group_histogram", _HEADERS),
    ("cub_group_histogram_composite", ("cub/block/block_histogram.cuh",)),
):
    _rendering.register_bundle_renderer(
        _kind,
        render=_render_histogram,
        include_lines=tuple(f"#include <{header}>" for header in _headers),
        cccl_headers=tuple(
            (f"#include <{header}>", header) for header in _headers
        ),
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
    """Emit a fresh histogram with independently typed counters.

    Resolve sample items and default the counter dtype to Int32. Allocate
    register counters with bins_per_thread slots and the samples' alignment.
    Record the wrapper request and its scratch use in the compile session,
    which builds the C++ later, then emit the call. Keep an explicit counter
    selector as the returned ThreadData dtype.

    If recording or emission fails, restore the session's earlier requests and
    scratch uses. Emitted IR and the register tensor are not undone.
    """

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
    tensor = _make_rmem_tensor(bins_per_thread, counter_type, samples.alignment)
    snapshot = _state.snapshot_active_session_state()
    try:
        _emit_histogram_call(
            request,
            sample_type,
            values,
            temp_storage,
            tensor,
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


def _emit_histogram_call(request, sample_type, values, temp_storage, counters):
    _state.register_request(request)
    descriptor = (
        TempStorage(auto_sync=True) if temp_storage is None else temp_storage
    )
    storage_arguments = _storage.register_deferred_temp_storage_event(
        descriptor,
        primitive_name="histogram_composite"
        if request.composite
        else "histogram",
        requirement_key=request.scratch_requirement_key,
    )
    if request.composite:
        counters_address = Uint32(counters.iterator.toint())
        counter_parameter = Uint32
    else:
        counters_address = counters.iterator.llvm_ptr
        counter_parameter = llvm.PointerType.get(0)
    parameter_types = [sample_type] * len(values) + [
        Uint32,
        Int32,
        Int32,
        counter_parameter,
    ]
    ffi(
        name=request.symbol_name, params_types=parameter_types, return_type=None
    )(
        *(sample_type(value) for value in values),
        *storage_arguments,
        counters_address,
    )


def provider_histogram_composite(
    *, group, launch, samples, counters, bins, algorithm, temp_storage
):
    """Add samples directly into the caller's shared counter array."""

    sample_type, values = _types.resolve_thread_data_value_type(
        samples,
        allowed=_SAMPLES,
        feature="samples",
        scope="cuda.coop.cutlass",
        resolve_type=_resolve_type,
    )
    counter_type = _resolve_type(
        counters.element_type, allowed=_COUNTERS, feature="counters"
    )
    operation = GroupHistogramCompositeSemantics(
        sample_type, len(values), bins, counter_type, algorithm
    )
    request = _CubHistogramRequest(
        _plan_with_storage(group, launch, operation, temp_storage)
    )
    snapshot = _state.snapshot_active_session_state()
    try:
        _emit_histogram_call(
            request,
            sample_type,
            values,
            temp_storage,
            counters,
        )
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise
