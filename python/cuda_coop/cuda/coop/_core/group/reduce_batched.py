# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group planning for independent batched warp reductions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._types import ArgumentKind, ParameterClassification, ParameterRole
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ..warp.reduce_batched import (
    WarpReduceBatchedSemantics,
    make_warp_reduce_batched_spec,
)
from ._contracts import _contracts, _unsupported_cub_warp_width
from ._dispatch import _register_group_operation_family
from ._model import (
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupPrimitiveCall,
    ImplementationProvenance,
    LogicalResultContract,
    ResultContract,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
)


@dataclass(frozen=True)
class GroupReduceBatchedSemantics:
    primitive: WarpReduceBatchedSemantics

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, WarpReduceBatchedSemantics):
            raise TypeError("primitive must be WarpReduceBatchedSemantics")

    @property
    def dtype(self) -> Any:
        return self.primitive.dtype

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    @property
    def result_visibility(self) -> ResultVisibility:
        return ResultVisibility.PER_MEMBER

    @property
    def returns_value(self) -> bool:
        return True


def _call_classifications(operation):
    del operation
    return (
        ParameterClassification("value", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ParameterClassification(
            "binary_op", ArgumentKind.STATIC, ParameterRole.CONSTANT
        ),
        ParameterClassification(
            "output_layout", ArgumentKind.STATIC, ParameterRole.CONSTANT
        ),
    )


def _plan_reduce_batched(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupReduceBatchedSemantics,
) -> GroupLoweringPlan:
    warp_width, error = _unsupported_cub_warp_width(call, resolved)
    if error is not None:
        return error
    assert warp_width is not None
    primitive = operation.primitive
    spec = make_warp_reduce_batched_spec(
        dtype=primitive.dtype,
        batches=primitive.batches,
        threads_in_warp=warp_width,
        reduce_operator=primitive.reduce_operator,
        output_layout=primitive.output_layout,
    )
    result = ResultContract(
        (
            LogicalResultContract(
                name="value",
                dtype=operation.dtype,
                visibility=ResultVisibility.PER_MEMBER,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=GroupOperandKind.ARRAY,
                items_per_member=spec.outputs_per_thread,
            ),
        )
    )
    contracts = _contracts(
        resolved,
        launch,
        result=result,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_WARP,
        call=call,
        resolved_group=resolved,
        implementation=spec.specialization,
        topology=contracts[0],
        participation=contracts[1],
        result=result,
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/warp/warp_reduce_batched.cuh",
            cpp_class="cub::WarpReduceBatched",
            method=spec.specialization.method_name,
        ),
    )


_register_group_operation_family(
    GroupReduceBatchedSemantics,
    classifications=_call_classifications,
    planner=_plan_reduce_batched,
    group_kinds=frozenset({"warp", "threads_within_warp"}),
    unsupported_group_message=(
        "cuda.coop.reduce_batched supports complete physical warps and "
        "power-of-two logical-warp groups"
    ),
)


__all__ = ["GroupReduceBatchedSemantics"]
