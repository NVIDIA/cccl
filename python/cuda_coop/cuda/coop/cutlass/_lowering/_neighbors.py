# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Turn neighbor plans into C++ calls for the CUTLASS backend.

This backend supports built-in subtraction for differences and inequality for
discontinuity. Each call copies its input items into a generated C++ function.
A wrapper struct from the common planner selects the CUB method for the
direction or flag mode, partial tile, and outside-tile neighbors. Results go
to separate value or Int32 flag arrays. Shared-memory scratch is recorded now
and sized later from the exact CUB TempStorage layout.
"""

import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Integral

import numpy as np
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Int64, Uint32
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    Algorithm,
    ArgumentBinding,
    BindingKind,
    CxxOperator,
    Dependency,
    GroupLoweringPlan,
    GroupLoweringTarget,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block.neighbors import BlockNeighborSemantics
from cuda.coop._core.group.neighbors import GroupNeighborSemantics

from .._compiler import _rendering, _state, _storage, _types
from .._temp_storage import TempStorage
from .._thread_data import ThreadData, _make_rmem_tensor

_SCOPE = "cuda.coop.cutlass"
_resolve_type = _types.make_provider_type_resolver(
    scope=_SCOPE, root_scope=_SCOPE, namespace="neighbors"
)


def _valid_binding(value):
    """Classify a full tile or static/runtime valid prefix count.

    Static counts stay in the plan. Reject Boolean and Enum values because
    they are not counts. Accept runtime signed integers up to 64 bits and
    unsigned integers up to 32 bits: the C++ wrapper takes a signed 64-bit
    count. Shared planning checks static bounds. The partial-tile wrapper
    checks runtime bounds before it narrows the count to int.
    """

    if value is None:
        return ArgumentBinding.omitted()
    if isinstance(value, (bool, np.bool_, Enum)):
        raise TypeError("neighbor valid_items must be an integer")
    if isinstance(value, Integral):
        return ArgumentBinding.static(int(value))
    type_specification = _types.TYPE_SPECIFICATIONS.get(
        _types.canonical_dsl_type(value)
    )
    if (
        type_specification is None
        or type_specification.token[0] not in {"i", "u"}
        or type_specification.token == "u64"
    ):
        raise TypeError(
            "neighbor valid_items requires a signed integer up to 64 bits "
            "or an unsigned integer up to 32 bits"
        )
    return ArgumentBinding.runtime()


def _make_neighbor_plan(
    *,
    group,
    launch,
    dtype,
    items,
    operation,
    mode,
    valid_items=None,
    predecessor=False,
    successor=False,
    temp_storage=None,
):
    """Select built-in neighbor semantics and exact block scratch.

    Use subtraction for differences and inequality for discontinuity. Preserve
    mode and boundary-presence flags so shared planning selects the correct
    CUB overload and result arrays. Use the caller TempStorage size,
    alignment, sharing, and auto_sync settings. Without TempStorage, allocate
    scratch automatically and add a trailing block barrier.
    """

    valid = _valid_binding(valid_items)
    operator = "minus" if operation == "adjacent_difference" else "not_equal_to"
    primitive = BlockNeighborSemantics(
        operation,
        dtype,
        items,
        mode,
        CxxOperator(f"::cuda::std::{operator}<T>", Dependency("T"), name="op"),
        partial=valid.kind is not BindingKind.OMITTED,
        predecessor=predecessor,
        successor=successor,
    )
    plan = plan_group_primitive(
        make_group_primitive_call(
            group, GroupNeighborSemantics(primitive, valid)
        ),
        launch,
    ).require_supported()
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError("neighbor temp_storage must be CUTLASS TempStorage")
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
class _CubNeighborRequest:
    """Describe one C++ neighbor wrapper and its result and storage types.

    The bundle session keeps requests in a set. Equality, hashing, and symbol
    names use the plan artifact key, so equal plans render one wrapper. The
    operation fixes the result names, dtype, and extent. Differences use the
    input type, and discontinuity uses one or two Int32 arrays.
    """

    plan: GroupLoweringPlan
    kind: str = "cub_group_neighbors"

    def __post_init__(self):
        """Match the shared plan to its template and result contracts.

        Require a supported CUB block plan, numeric input, Apply method, exact
        block dimensions and item count, and an exact scratch layout. Check
        every named result against the operation before rendering.
        """

        self.plan.require_supported()
        if (
            self.plan.target is not GroupLoweringTarget.CUB_BLOCK
            or not isinstance(self.plan.call.operation, GroupNeighborSemantics)
            or not isinstance(self.plan.implementation, Algorithm)
        ):
            raise TypeError("neighbor request requires a shared CUB block plan")
        if self.value_type not in _types.ALL_PROVIDER_TYPES:
            raise TypeError(
                "neighbor request requires a supported numeric dtype"
            )
        if self.implementation.method_name != "Apply":
            raise ValueError("neighbor implementation does not match its plan")
        arguments = self.implementation.template_arguments
        if (
            arguments.get("T") is not self.value_type
            or arguments.get("ItemsPerThread") != self.items
            or tuple(arguments.get(f"BlockDim{axis}") for axis in "XYZ")
            != self.plan.participation.exact_block_dim
        ):
            raise ValueError(
                "neighbor template arguments do not match its plan"
            )
        if not self.plan.temp_storage.exact_layout_required:
            raise ValueError("neighbor request requires exact scratch layout")
        result = self.plan.result
        if (
            result is None
            or tuple(item.name for item in result.values)
            != self.operation.result_names
            or any(
                item.dtype != self.operation.result_dtype
                or item.items_per_member != self.items
                for item in result.values
            )
        ):
            raise ValueError("neighbor result does not match its plan")

    @property
    def operation(self):
        return self.plan.call.operation.primitive

    @property
    def implementation(self):
        return self.plan.implementation

    @property
    def value_type(self):
        return self.operation.dtype

    @property
    def output_type(self):
        return (
            self.value_type
            if self.operation.operation == "adjacent_difference"
            else Int32
        )

    @property
    def items(self):
        return self.operation.items_per_thread

    @property
    def cpp_type(self):
        """Spell the C++ type used by calls and layout probes."""

        arguments = [
            _types.TYPE_SPECIFICATIONS[value].cpp_type
            if name == "T"
            else str(value)
            for name, value in self.implementation.ordered_template_arguments
        ]
        return (
            f"::cub::{self.implementation.struct_name}<{', '.join(arguments)}>"
        )

    @property
    def scratch_requirement_key(self):
        """Key layout probes by their CUB storage type."""

        return "cub_neighbors_storage", self.cpp_type

    @property
    def symbol_name(self):
        """Name the operation and hash its complete plan identity."""

        digest = hashlib.sha256(
            repr(self.plan.artifact_key).encode()
        ).hexdigest()[:16]
        return f"cuda_coop_cutlass_{self.operation.operation}_{digest}"

    def __eq__(self, other):
        return (
            isinstance(other, _CubNeighborRequest)
            and self.plan.artifact_key == other.plan.artifact_key
        )

    def __hash__(self):
        return hash(self.plan.artifact_key)


def _render_neighbors(request):
    """Render copied values, boundary arguments, and typed outputs.

    Pass runtime counts as signed 64-bit arguments and embed static counts.
    Both boundary slots are always in the C++ signature. The common planner's
    wrapper struct is chosen by the operation, mode, and partial, predecessor,
    and successor flags. It passes to CUB only the boundary slots whose flags
    are set. Partial wrappers check counts before they narrow them to int.

    Pass scratch address, size, and synchronization control before the output
    pointers. Check storage size and alignment, call Apply, perform the
    configured reuse barrier, and copy each result array out.
    """

    request.__post_init__()
    cpp = _types.TYPE_SPECIFICATIONS[request.value_type].cpp_type
    output_cpp = _types.TYPE_SPECIFICATIONS[request.output_type].cpp_type
    names = request.operation.result_names
    params = [f"{cpp} item{i}" for i in range(request.items)]
    valid = request.plan.call.operation.valid_items
    if valid.kind is BindingKind.RUNTIME:
        params.append("long long valid_items")
        count = "valid_items"
    else:
        count = f"{valid.value if valid.kind is BindingKind.STATIC else 0}ll"
    params.extend(
        (
            f"{cpp} predecessor",
            f"{cpp} successor",
            "unsigned int storage_address",
            "int storage_bytes",
            "int storage_auto_sync",
        )
    )
    params.extend(f"{output_cpp}* result_{name}" for name in names)
    values = ", ".join(f"item{i}" for i in range(request.items))
    arguments = [
        "values",
        *names,
        request.operation.operator.cpp + "{}",
        count,
        "predecessor",
        "successor",
    ]
    return [
        f"void {request.symbol_name}({', '.join(params)}) {{",
        f"  using T = {cpp};",
        f"  using implementation_type = {request.cpp_type};",
        "  using storage_type = typename implementation_type::TempStorage;",
        (
            "  if (storage_bytes <= 0 || (unsigned long long)storage_bytes "
            "< sizeof(storage_type) ||"
        ),
        "      (storage_address & (alignof(storage_type) - 1u)) != 0u) {",
        '    asm volatile("trap;");',
        "  }",
        "  unsigned long long generic_address;",
        (
            '  asm("cvta.shared.u64 %0, %1;" : "=l"(generic_address) : '
            '"l"((unsigned long long)storage_address));'
        ),
        "  auto& storage = *reinterpret_cast<storage_type*>(generic_address);",
        f"  T values[{request.items}] = {{{values}}};",
        *(f"  {output_cpp} {name}[{request.items}] = {{}};" for name in names),
        f"  implementation_type(storage).Apply({', '.join(arguments)});",
        "  if (storage_auto_sync != 0) { __syncthreads(); }",
        *(
            f"  result_{name}[{i}] = {name}[{i}];"
            for name in names
            for i in range(request.items)
        ),
        "}",
    ]


def _scratch_probe(request):
    """Request the C++ scratch size and alignment for deferred storage."""

    return _rendering.make_scratch_layout_probe(
        request.scratch_requirement_key,
        f"typename {request.cpp_type}::TempStorage",
    )


_HEADERS = (
    "cub/block/block_adjacent_difference.cuh",
    "cub/block/block_discontinuity.cuh",
    "cuda/std/functional",
)
_rendering.register_bundle_renderer(
    "cub_group_neighbors",
    render=_render_neighbors,
    include_lines=tuple(f"#include <{header}>" for header in _HEADERS),
    cccl_headers=tuple((f"#include <{header}>", header) for header in _HEADERS),
    scratch_layout_probe=_scratch_probe,
)


def _typed_value(value, dtype, *, name):
    """Convert items and external neighbors to the verified input dtype.

    NumPy and runtime CuTe scalars must already match the dtype. Plain
    Python integers convert within its range. Python floats require a
    floating-point dtype, and finite values must fit its range. Infinite
    values and NaNs are accepted for floating-point dtypes.
    """

    if isinstance(value, np.generic):
        if _types.canonical_dsl_type(value) is not dtype:
            raise TypeError(f"neighbor {name} dtype must match input dtype")
        value = value.item()
    converted = _types.coerce_plain_scalar(
        value, dtype, name=name, scope=_SCOPE, allow_nonfinite=True
    )
    if converted is not _types._NOT_PLAIN_SCALAR:
        return converted
    if _types.canonical_dsl_type(value) is not dtype:
        raise TypeError(f"neighbor {name} dtype must match input dtype")
    return dtype(value)


def provider_neighbors(
    *,
    group,
    launch,
    values,
    operation,
    mode,
    valid_items,
    tile_predecessor_item,
    tile_successor_item,
    temp_storage,
):
    """Emit one neighbor call with aligned, independently owned results.

    Resolve every initialized input item, including the invalid suffix. Pass
    typed boundary scalars and use zero placeholders for omitted neighbors.
    The wrapper struct's predecessor and successor flags keep CUB from reading
    these placeholders. Allocate value arrays for differences or Int32 arrays
    for flags.

    Register exact deferred scratch and emit the call. On failure, restore
    queued session state. Emitted IR and register allocations remain.
    """

    dtype, items = _types.resolve_thread_data_value_type(
        values,
        allowed=_types.ALL_PROVIDER_TYPES,
        feature=operation,
        scope=_SCOPE,
        resolve_type=_resolve_type,
    )
    request = _CubNeighborRequest(
        _make_neighbor_plan(
            group=group,
            launch=launch,
            dtype=dtype,
            items=len(items),
            operation=operation,
            mode=mode,
            valid_items=valid_items,
            predecessor=tile_predecessor_item is not None,
            successor=tile_successor_item is not None,
            temp_storage=temp_storage,
        )
    )
    arguments = [_typed_value(item, dtype, name="values") for item in items]
    parameter_types = [dtype] * len(items)
    if request.plan.call.operation.valid_items.kind is BindingKind.RUNTIME:
        arguments.append(Int64(valid_items))
        parameter_types.append(Int64)
    arguments.extend(
        _typed_value(value if value is not None else 0, dtype, name=name)
        for value, name in (
            (tile_predecessor_item, "tile_predecessor_item"),
            (tile_successor_item, "tile_successor_item"),
        )
    )
    parameter_types.extend((dtype, dtype))
    tensors = [
        _make_rmem_tensor(len(items), request.output_type, values.alignment)
        for _ in request.operation.result_names
    ]
    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(request)
        descriptor = (
            TempStorage(auto_sync=True)
            if temp_storage is None
            else temp_storage
        )
        arguments.extend(
            _storage.register_deferred_temp_storage_event(
                descriptor,
                primitive_name=operation,
                requirement_key=request.scratch_requirement_key,
            )
        )
        parameter_types.extend((Uint32, Int32, Int32))
        arguments.extend(tensor.iterator.llvm_ptr for tensor in tensors)
        parameter_types.extend([llvm.PointerType.get(0)] * len(tensors))
        ffi(
            name=request.symbol_name,
            params_types=parameter_types,
            return_type=None,
        )(*arguments)
        output_dtype = (
            _types.thread_data_output_dtype(values, dtype)
            if operation == "adjacent_difference"
            else Int32
        )
        results = tuple(
            ThreadData(
                len(items),
                dtype=output_dtype,
                values=[
                    request.output_type(tensor[i]) for i in range(len(items))
                ],
                alignment=values.alignment,
            )
            for tensor in tensors
        )
        return results[0] if len(results) == 1 else results
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise
