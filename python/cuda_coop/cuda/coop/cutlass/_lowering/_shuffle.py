# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed shared-planner provider for CUB BlockShuffle."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from numbers import Integral

from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int64
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    ArgumentBinding,
    BindingKind,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupShuffleSemantics,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block.shuffle import (
    BlockShuffleMode,
    make_block_shuffle_semantics,
)

from .._compiler import _rendering, _state, _types
from .._thread_data import ThreadData, _make_rmem_tensor

_SCOPE = "cuda.coop.cutlass"
_HEADER = "cub/block/block_shuffle.cuh"
_I32_MIN, _I32_MAX = -(1 << 31), (1 << 31) - 1
_resolve_type = _types.make_provider_type_resolver(
    scope=_SCOPE, root_scope=_SCOPE, namespace="thread_group"
)


def _distance_binding(distance, *, array):
    if isinstance(distance, (bool, Enum)):
        raise TypeError(
            f"{_SCOPE}.shuffle distance must be an integer, not bool or Enum"
        )
    if isinstance(distance, Integral):
        if array:
            if int(distance) != 1:
                raise ValueError(f"{_SCOPE}.shuffle array distance must be exactly 1")
            return ArgumentBinding.omitted()
        return ArgumentBinding.static(int(distance))
    if array:
        raise TypeError(f"{_SCOPE}.shuffle array distance must be a compile-time 1")
    dtype = _types.canonical_dsl_type(distance)
    spec = _types.TYPE_SPECS.get(dtype)
    if spec is None or spec.token[0] not in {"i", "u"} or spec.token == "u64":
        raise TypeError(
            f"{_SCOPE}.shuffle distance requires a signed integer up to 64 bits "
            "or an unsigned integer up to 32 bits"
        )
    return ArgumentBinding.runtime()


def _make_shuffle_plan(*, group, launch, dtype, items_per_thread, mode, distance):
    primitive = make_block_shuffle_semantics(
        dtype=dtype,
        mode=mode,
        items_per_thread=items_per_thread,
        distance=_distance_binding(distance, array=items_per_thread is not None),
    )
    return plan_group_primitive(
        make_group_primitive_call(
            group, GroupShuffleSemantics(primitive), source="cutlass_root"
        ),
        launch,
    ).require_supported()


@dataclass(frozen=True, eq=False)
class _CubShuffleRequest:
    plan: GroupLoweringPlan
    value_type: type
    kind: str = "cub_group_shuffle"

    def __post_init__(self):
        self.plan.require_supported()
        operation = self.plan.call.operation
        implementation = self.plan.implementation
        if (
            self.plan.target is not GroupLoweringTarget.CUB_BLOCK
            or not isinstance(operation, GroupShuffleSemantics)
            or not isinstance(implementation, AlgorithmSpec)
        ):
            raise ValueError("BlockShuffle requires a CUB block Shuffle plan")
        if (
            operation.dtype is not self.value_type
            or self.value_type not in _types.TYPE_SPECS
        ):
            raise ValueError("BlockShuffle dtype does not match its plan")
        if (
            implementation.struct_name != "BlockShuffle"
            or implementation.method_name != operation.primitive.method_name
        ):
            raise ValueError("BlockShuffle implementation does not match its plan")
        participation = self.plan.participation
        if participation is None:
            raise ValueError("BlockShuffle requires exact block participation")
        arguments = implementation.template_arguments
        if (
            arguments.get("T") is not self.value_type
            or tuple(arguments.get(f"BLOCK_DIM_{axis}") for axis in "XYZ")
            != participation.exact_block_dim
        ):
            raise ValueError("BlockShuffle template arguments do not match its plan")
        if (
            operation.primitive.is_array
            and arguments.get("ITEMS_PER_THREAD") != operation.items_per_thread
        ):
            raise ValueError("BlockShuffle extent does not match its plan")
        storage, synchronization = self.plan.temp_storage, self.plan.synchronization
        if (
            storage is None
            or storage.ownership is not StorageOwnership.IMPLEMENTATION
            or storage.instances != 1
            or synchronization is None
            or synchronization.storage_reuse_barrier is not SynchronizationScope.BLOCK
        ):
            raise ValueError(
                "BlockShuffle requires owned block scratch and reuse synchronization"
            )

    @property
    def operation(self):
        return self.plan.call.operation.primitive

    @property
    def semantic_key(self):
        return self.plan.artifact_key

    def __hash__(self):
        return hash(self.semantic_key)

    def __eq__(self, other):
        if not isinstance(other, _CubShuffleRequest):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    @property
    def symbol_name(self):
        digest = hashlib.sha256(repr(self.semantic_key).encode()).hexdigest()[:16]
        return f"cuda_coop_cutlass_shuffle_{self.operation.mode.value}_{digest}"


def _render_shuffle(request):
    request.__post_init__()
    primitive = request.operation
    spec = _types.TYPE_SPECS[request.value_type]
    block_dim = request.plan.participation.exact_block_dim
    params, setup, output = [], [], []
    if primitive.is_array:
        count = primitive.items_per_thread
        params.extend(f"{spec.cpp_type} item{i}" for i in range(count))
        params.append(f"{spec.cpp_type}* result_items")
        values = ", ".join(f"item{i}" for i in range(count))
        setup.extend(
            [
                f"  {spec.cpp_type} input_items[{count}] = {{{values}}};",
                f"  {spec.cpp_type} output_items[{count}] = {{}};",
            ]
        )
        arguments = "input_items, output_items"
        output.extend(f"  result_items[{i}] = output_items[{i}];" for i in range(count))
    else:
        params.append(f"{spec.cpp_type} value")
        setup.append(f"  {spec.cpp_type} result = value;")
        binding = primitive.distance
        is_rotate = primitive.mode is BlockShuffleMode.ROTATE
        cast = "unsigned int" if is_rotate else "int"
        if binding.kind is BindingKind.RUNTIME:
            params.append("long long distance")
            lower, upper = (
                (1, block_dim[0] * block_dim[1] * block_dim[2] - 1)
                if is_rotate
                else (_I32_MIN, _I32_MAX)
            )
            setup.extend(
                [
                    f"  if (distance < {lower}ll || distance > {upper}ll) {{",
                    '    asm volatile("trap;" : : :);',
                    "  }",
                ]
            )
            distance = "distance"
        else:
            distance = f"{binding.value if binding.kind is BindingKind.STATIC else 1}ll"
        arguments = f"value, result, static_cast<{cast}>({distance})"
        output.append("  return result;")
    template_arguments = ", ".join((spec.cpp_type, *(str(d) for d in block_dim)))
    return [
        f"{'void' if primitive.is_array else spec.cpp_type} {request.symbol_name}({', '.join(params)}) {{",
        f"  using implementation_type = ::cub::BlockShuffle<{template_arguments}>;",
        "  __shared__ typename implementation_type::TempStorage storage;",
        *setup,
        f"  implementation_type(storage).{primitive.method_name}({arguments});",
        "  __syncthreads();",
        *output,
        "}",
    ]


_rendering.register_bundle_renderer(
    "cub_group_shuffle",
    render=_render_shuffle,
    include_lines=(f"#include <{_HEADER}>",),
    cccl_headers=((f"#include <{_HEADER}>", _HEADER),),
)


def provider_shuffle(*, group, launch, value, mode, distance):
    """Lower a current shared Shuffle plan without an alternate compiler adapter."""
    array = isinstance(value, ThreadData)
    if array:
        value_type, values = _types.resolve_thread_data_value_type(
            value,
            allowed=_types.ALL_PROVIDER_TYPES,
            feature="shuffle",
            scope=_SCOPE,
            resolve_type=_resolve_type,
        )
    else:
        value_type = _resolve_type(
            value, allowed=_types.ALL_PROVIDER_TYPES, feature="shuffle"
        )
        values = (value,)
    plan = _make_shuffle_plan(
        group=group,
        launch=launch,
        dtype=value_type,
        items_per_thread=len(values) if array else None,
        mode=mode,
        distance=distance,
    )
    request = _CubShuffleRequest(plan, value_type)
    typed_values = []
    for item in values:
        converted = _types.coerce_plain_scalar(
            item, value_type, name="shuffle value", scope=_SCOPE, allow_nonfinite=True
        )
        typed_values.append(
            value_type(item) if converted is _types._NOT_PLAIN_SCALAR else converted
        )
    runtime_distance = request.operation.distance.kind is BindingKind.RUNTIME
    result_tensor = (
        _make_rmem_tensor(len(values), value_type, value.alignment) if array else None
    )
    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(request)
        result = ffi(
            name=request.symbol_name,
            params_types=[
                *([value_type] * len(values)),
                *([Int64] if runtime_distance else []),
                *([llvm.PointerType.get(0)] if array else []),
            ],
            return_type=None if array else value_type,
        )(
            *typed_values,
            *([Int64(distance)] if runtime_distance else []),
            *([result_tensor.iterator.llvm_ptr] if array else []),
        )
        if array:
            return ThreadData(
                len(values),
                dtype=_types.thread_data_output_dtype(value, value_type),
                values=[value_type(result_tensor[i]) for i in range(len(values))],
                alignment=value.alignment,
            )
        return value_type(result)
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise


__all__ = ["provider_shuffle"]
