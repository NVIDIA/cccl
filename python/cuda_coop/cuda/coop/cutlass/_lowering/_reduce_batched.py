# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lower independent batches to CUB register-shuffle reductions.

Each lane supplies one value for every batch. CUB exchanges values with warp
shuffles and returns each batch total in a blocked or striped layout. CUB's
TempStorage is an empty type, so the wrapper declares a local one only to
satisfy the constructor; it needs no shared memory. Results go through a
pointer into a new register tensor, and the input stays unchanged.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
from cutlass._mlir.dialects import llvm
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    Algorithm,
    CxxOperator,
    Dependency,
    GroupLoweringPlan,
    GroupLoweringTarget,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.group.reduce_batched import GroupReduceBatchedSemantics
from cuda.coop._core.warp.reduce_batched import WarpReduceBatchedSemantics

from .._compiler import _rendering, _state, _types
from .._operators import (
    OPERATOR_CPP,
    operator_expression,
    validate_operator_dtype,
)
from .._thread_data import ThreadData, _make_rmem_tensor

_SCOPE = "cuda.coop.cutlass"
_HEADER = "cub/warp/warp_reduce_batched.cuh"
_resolve_type = _types.make_provider_type_resolver(
    scope=_SCOPE, root_scope=_SCOPE, namespace="thread_group"
)


def _make_reduce_batched_plan(
    *, group, launch, dtype, batches, op="sum", output_layout="striped"
):
    """Plan the built-in operator and distributed result layout.

    Validate the operator/dtype combination, then let shared planning resolve
    the warp width and ceil(batches / width) result extent. Each input slot
    remains an independent batch across the participating lanes.
    """

    validate_operator_dtype(op, dtype, primitive="reduce_batched")
    primitive = WarpReduceBatchedSemantics(
        dtype,
        batches,
        CxxOperator(
            cpp=OPERATOR_CPP[op], dtype=Dependency("T"), name="binary_op"
        ),
        output_layout,
    )
    return plan_group_primitive(
        make_group_primitive_call(
            group,
            GroupReduceBatchedSemantics(primitive),
        ),
        launch,
    ).require_supported()


@dataclass(frozen=True, eq=False)
class _CubReduceBatchedRequest:
    """Identify one batched warp wrapper and its output contract.

    The plan artifact key controls equality, so equal plans share one wrapper
    in the session. Validate that the operator, method, warp width, batch
    count, and result extent agree.
    """

    plan: GroupLoweringPlan
    op: str
    kind: str = "cub_group_reduce_batched"

    def __post_init__(self):
        """Require a matching CUB warp specialization and result extent.

        Check that the plan uses CUB WarpReduceBatched with the method for the
        chosen layout, the same dtype, batch count and warp width, and
        SYNC_PHYSICAL_WARP set to false. That setting lets each logical warp
        call independently, even from a different branch. Also check that the
        operator matches the plan and is valid for the dtype, and that the
        result has ceil(batches / width) slots.
        """

        self.plan.require_supported()
        if (
            self.plan.target is not GroupLoweringTarget.CUB_WARP
            or not isinstance(
                self.plan.call.operation, GroupReduceBatchedSemantics
            )
            or not isinstance(self.plan.implementation, Algorithm)
        ):
            raise ValueError("reduce_batched requires a shared CUB warp plan")
        p, specialization = self.operation, self.implementation
        if p.dtype not in _types.ALL_PROVIDER_TYPES:
            raise TypeError("reduce_batched requires a supported numeric dtype")
        validate_operator_dtype(self.op, p.dtype, primitive="reduce_batched")
        if p.reduce_operator.cpp != OPERATOR_CPP[self.op]:
            raise ValueError("reduce_batched operator does not match its plan")
        args = specialization.template_arguments
        width = self.plan.resolved_group.static_size
        if (
            specialization.struct_name != "WarpReduceBatched"
            or specialization.method_name
            != (
                "ReduceToStriped"
                if p.output_layout == "striped"
                else "ReduceToBlocked"
            )
            or args.get("T") is not p.dtype
            or args.get("BATCHES") != p.batches
            or args.get("LOGICAL_WARP_THREADS") != width
            or args.get("SYNC_PHYSICAL_WARP") != "false"
        ):
            raise ValueError(
                "reduce_batched implementation does not match its plan"
            )
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
        """Return the ceil(batches / warp width) output capacity."""

        return self.plan.result.values[0].items_per_member

    @property
    def cpp_type(self):
        """Spell the CUB type with physical-warp synchronization disabled.

        The final false template argument lets logical warps call
        independently.
        """

        p = self.operation
        width = self.plan.resolved_group.static_size
        return (
            f"::cub::WarpReduceBatched<"
            f"{_types.TYPE_SPECIFICATIONS[p.dtype].cpp_type}, "
            f"{p.batches}, {width}, false>"
        )

    @property
    def symbol_name(self):
        """Hash the complete plan identity into a wrapper symbol."""

        digest = hashlib.sha256(
            repr(self.plan.artifact_key).encode()
        ).hexdigest()[:16]
        return f"cuda_coop_cutlass_reduce_batched_{digest}"

    def __eq__(self, other):
        return (
            isinstance(other, _CubReduceBatchedRequest)
            and self.plan.artifact_key == other.plan.artifact_key
        )

    def __hash__(self):
        return hash(self.plan.artifact_key)


def _render_reduce_batched(request):
    """Render a scalar-input, result-pointer ABI without shared scratch.

    Copy per-batch inputs into a const local array and invoke the selected CUB
    layout method with a built-in functor. CUB's storage is NullType, so this
    wrapper needs neither a storage pointer nor a reuse barrier.

    Copy all output slots to the result pointer. The zero initializer does not
    make extra slots valid: CUB may overwrite slots that have no batch.
    Callers must skip them according to the chosen layout.
    """

    request.__post_init__()
    p = request.operation
    cpp = _types.TYPE_SPECIFICATIONS[p.dtype].cpp_type
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
        (
            "  implementation_type(storage)."
            f"{request.implementation.method_name}("
            f"inputs, outputs, {operator_expression(request.op)});"
        ),
        *(
            f"  result[{i}] = outputs[{i}];"
            for i in range(request.outputs_per_thread)
        ),
        "}",
    ]


_INCLUDES = (_HEADER, "cuda/std/functional", "cuda/functional")
_rendering.register_bundle_renderer(
    "cub_group_reduce_batched",
    render=_render_reduce_batched,
    include_lines=tuple(f"#include <{header}>" for header in _INCLUDES),
    cccl_headers=tuple(
        (f"#include <{header}>", header) for header in _INCLUDES
    ),
)


def provider_reduce_batched(*, group, launch, value, op, output_layout):
    """Emit independent batch reductions into a fresh aligned payload.

    The qualified ``reduce_batched`` entry point calls this during tracing.
    Input slot i in every participating lane belongs to batch i. Reducing
    those values yields one total per batch, distributed across lanes
    according to output_layout.

    Resolve and convert initialized input items, register the planned wrapper,
    and allocate only the distributed output extent. The call passes scalar
    items and a result pointer. Preserve the input dtype and minimum alignment
    in the returned ThreadData.

    Restore queued session state if allocation or emission fails. Emitted IR
    and register allocations are outside that rollback.

    Parameters
    ----------
    group : ThreadGroup
        Complete physical or logical warp participating in every batch.
    launch : LaunchFacts
        Exact enclosing block dimensions used to check complete warps.
    value : ThreadData
        Initialized per-thread payload with one item for each batch.
    op : str
        Normalized built-in reduction operator.
    output_layout : str
        Blocked or striped placement of batch totals across lanes.

    Returns
    -------
    ThreadData
        Fresh payload with ceil(batch_count / warp_width) slots per lane.
        Only slots assigned to an existing batch contain a defined result.
    """

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
            item,
            dtype,
            name="reduce_batched item",
            scope=_SCOPE,
            allow_nonfinite=True,
        )
        arguments.append(
            dtype(item) if converted is _types._NOT_PLAIN_SCALAR else converted
        )
    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(request)
        result = _make_rmem_tensor(
            request.outputs_per_thread, dtype, value.alignment
        )
        ffi(
            name=request.symbol_name,
            params_types=[dtype] * len(items) + [llvm.PointerType.get(0)],
            return_type=None,
        )(*arguments, result.iterator.llvm_ptr)
        return ThreadData(
            request.outputs_per_thread,
            dtype=_types.thread_data_output_dtype(value, dtype),
            values=[
                dtype(result[i]) for i in range(request.outputs_per_thread)
            ],
            alignment=value.alignment,
        )
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise
