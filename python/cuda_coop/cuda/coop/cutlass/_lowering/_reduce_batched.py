# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed CUB WarpReduceBatched requests using the shared group planner."""

import hashlib
from dataclasses import dataclass

import numpy as np
from cutlass._mlir.dialects import llvm
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    CxxOperator,
    Dependency,
    GroupLoweringPlan,
    GroupLoweringTarget,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.api._dispatch import _common_root_operation_name
from cuda.coop._core.group.reduce_batched import GroupReduceBatchedSemantics
from cuda.coop._core.warp.reduce_batched import WarpReduceBatchedSemantics

from .._compiler import _rendering, _state, _types
from .._operators import OPERATOR_CPP, operator_expression, validate_operator_dtype
from .._thread_data import ThreadData, _make_rmem_tensor

_SCOPE = "cuda.coop.cutlass"
_HEADER = "cub/warp/warp_reduce_batched.cuh"
_resolve_type = _types.make_provider_type_resolver(
    scope=_SCOPE, root_scope=_SCOPE, namespace="thread_group"
)


def _make_reduce_batched_plan(
    *, group, launch, dtype, batches, op="sum", output_layout="striped"
):
    validate_operator_dtype(op, dtype, primitive="reduce_batched")
    primitive = WarpReduceBatchedSemantics(
        dtype,
        batches,
        CxxOperator(cpp=OPERATOR_CPP[op], dtype=Dependency("T"), name="binary_op"),
        output_layout,
    )
    source = (
        "common_root" if _common_root_operation_name() is not None else "cutlass_root"
    )
    return plan_group_primitive(
        make_group_primitive_call(
            group, GroupReduceBatchedSemantics(primitive), source=source
        ),
        launch,
    ).require_supported()


@dataclass(frozen=True, eq=False)
class _CubReduceBatchedRequest:
    plan: GroupLoweringPlan
    op: str
    kind: str = "cub_group_reduce_batched"

    def __post_init__(self):
        self.plan.require_supported()
        if (
            self.plan.target is not GroupLoweringTarget.CUB_WARP
            or not isinstance(self.plan.call.operation, GroupReduceBatchedSemantics)
            or not isinstance(self.plan.implementation, AlgorithmSpec)
        ):
            raise ValueError("reduce_batched requires a shared CUB warp plan")
        p, spec = self.operation, self.implementation
        if p.dtype not in _types.ALL_PROVIDER_TYPES:
            raise TypeError("reduce_batched requires a supported numeric dtype")
        validate_operator_dtype(self.op, p.dtype, primitive="reduce_batched")
        if p.reduce_operator.cpp != OPERATOR_CPP[self.op]:
            raise ValueError("reduce_batched operator does not match its plan")
        args = spec.template_arguments
        width = self.plan.resolved_group.static_size
        if (
            spec.struct_name != "WarpReduceBatched"
            or spec.method_name
            != (
                "ReduceToStriped" if p.output_layout == "striped" else "ReduceToBlocked"
            )
            or args.get("T") is not p.dtype
            or args.get("BATCHES") != p.batches
            or args.get("LOGICAL_WARP_THREADS") != width
            or args.get("SYNC_PHYSICAL_WARP") != "false"
        ):
            raise ValueError("reduce_batched implementation does not match its plan")
        if (
            self.plan.result is None
            or len(self.plan.result.values) != 1
            or self.plan.result.values[0].dtype is not p.dtype
            or self.plan.result.values[0].items_per_member
            != (p.batches + width - 1) // width
        ):
            raise ValueError("reduce_batched result does not match its plan")

    @property
    def operation(self):
        return self.plan.call.operation.primitive

    @property
    def implementation(self):
        return self.plan.implementation

    @property
    def outputs_per_thread(self):
        return self.plan.result.values[0].items_per_member

    @property
    def cpp_type(self):
        p = self.operation
        width = self.plan.resolved_group.static_size
        return (
            f"::cub::WarpReduceBatched<{_types.TYPE_SPECS[p.dtype].cpp_type}, "
            f"{p.batches}, {width}, false>"
        )

    @property
    def symbol_name(self):
        digest = hashlib.sha256(repr(self.plan.artifact_key).encode()).hexdigest()[:16]
        return f"cuda_coop_cutlass_reduce_batched_{digest}"

    def __eq__(self, other):
        return (
            isinstance(other, _CubReduceBatchedRequest)
            and self.plan.artifact_key == other.plan.artifact_key
        )

    def __hash__(self):
        return hash(self.plan.artifact_key)


def _render_reduce_batched(request):
    request.__post_init__()
    p = request.operation
    cpp = _types.TYPE_SPECS[p.dtype].cpp_type
    parameters = [f"{cpp} item{i}" for i in range(p.batches)]
    parameters.append(f"{cpp}* result")
    inputs = ", ".join(f"item{i}" for i in range(p.batches))
    return [
        f"void {request.symbol_name}({', '.join(parameters)}) {{",
        f"  using implementation_type = {request.cpp_type};",
        # CUB's batched implementation uses NullType storage. Keep its reference
        # local; no shared allocation or storage pointer belongs in this ABI.
        "  typename implementation_type::TempStorage storage;",
        f"  const {cpp} inputs[{p.batches}] = {{{inputs}}};",
        f"  {cpp} outputs[{request.outputs_per_thread}] = {{}};",
        f"  implementation_type(storage).{request.implementation.method_name}("
        f"inputs, outputs, {operator_expression(request.op)});",
        *(f"  result[{i}] = outputs[{i}];" for i in range(request.outputs_per_thread)),
        "}",
    ]


_INCLUDES = (_HEADER, "cuda/std/functional", "cuda/functional")
_rendering.register_bundle_renderer(
    "cub_group_reduce_batched",
    render=_render_reduce_batched,
    include_lines=tuple(f"#include <{header}>" for header in _INCLUDES),
    cccl_headers=tuple((f"#include <{header}>", header) for header in _INCLUDES),
)


def provider_reduce_batched(*, group, launch, value, op, output_layout):
    dtype, items = _types.resolve_thread_data_value_type(
        value,
        allowed=_types.ALL_PROVIDER_TYPES,
        feature="reduce_batched",
        scope=_SCOPE,
        resolve_type=_resolve_type,
    )
    request = _CubReduceBatchedRequest(
        _make_reduce_batched_plan(
            group=group,
            launch=launch,
            dtype=dtype,
            batches=value.items_per_thread,
            op=op,
            output_layout=output_layout,
        ),
        op,
    )
    arguments = []
    for item in items:
        if isinstance(item, np.generic):
            item = item.item()
        converted = _types.coerce_plain_scalar(
            item, dtype, name="reduce_batched item", scope=_SCOPE, allow_nonfinite=True
        )
        arguments.append(
            dtype(item) if converted is _types._NOT_PLAIN_SCALAR else converted
        )
    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(request)
        result = _make_rmem_tensor(request.outputs_per_thread, dtype, value.alignment)
        ffi(
            name=request.symbol_name,
            params_types=[dtype] * len(items) + [llvm.PointerType.get(0)],
            return_type=None,
        )(*arguments, result.iterator.llvm_ptr)
        return ThreadData(
            request.outputs_per_thread,
            dtype=_types.thread_data_output_dtype(value, dtype),
            values=[dtype(result[i]) for i in range(request.outputs_per_thread)],
            alignment=value.alignment,
        )
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise
