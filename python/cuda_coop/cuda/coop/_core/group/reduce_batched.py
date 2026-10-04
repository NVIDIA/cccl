# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan independent reductions with results distributed across warp lanes.

The operation records batch count and layout independently of group width.
Planning binds that width and selects a CUB method. It tells the backend how
many output slots and temporary-storage instances each group needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._types import ArgumentKind, ParameterClassification, ParameterRole
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ..warp.reduce_batched import (
    WarpReduceBatchedSemantics,
    make_warp_reduce_batched_specialization,
)
from ._dispatch import _register_group_operation_family
from ._execution_requirements import (
    _build_execution_requirements,
    _unsupported_cub_warp_width,
)
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
    """Expose batched reduction semantics to the common group dispatcher.

    Every member receives an array of result slots. The selected layout
    assigns each batch aggregate to one slot in one member. A slot is invalid
    if its assigned batch index is at least the batch count, even though
    result visibility is per member.

    Attributes
    ----------
    primitive : WarpReduceBatchedSemantics
        Dtype, batch count, reduction operator, and output layout. The group
        descriptor supplies the warp width separately during planning.
    """

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
    """Keep payload data at runtime and choose operator and layout statically.

    The dispatcher passes an operation record to every family. These argument
    roles are the same for every batched reduction, so its value is unused.
    """

    del operation
    return (
        ParameterClassification(
            "value", ArgumentKind.RUNTIME, ParameterRole.INPUT
        ),
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
    """Bind a warp width and describe the output arrays a backend must create.

    The shared dispatcher checks the group kind and resolves launch dimensions
    before calling this planner. The CUB width check then restricts logical
    warps to powers of two. Each member needs ``ceil(batches / width)`` result
    slots, including any slots that have no corresponding batch.

    Parameters
    ----------
    call : GroupPrimitiveCall
        Original group and operation record, retained for diagnostics.
    resolved : ThreadGroup
        Physical or logical warp with dimensions resolved from the launch.
    launch : LaunchFacts
        Exact block dimensions used to count group instances and describe
        participation and temporary-storage ownership.
    operation : GroupReduceBatchedSemantics
        Batch count, dtype, operator, and output layout to specialize.

    Returns
    -------
    GroupLoweringPlan
        CUB warp plan with one array result per member. Temporary storage is
        implementation-owned, with one instance per group and a barrier over
        that group's lanes before storage reuse. Its C++ layout is left to
        backend materialization. An unsupported plan is returned if CUB
        cannot use the group width.
    """

    warp_width, error = _unsupported_cub_warp_width(call, resolved)
    if error is not None:
        return error
    assert warp_width is not None
    primitive = operation.primitive
    specialization = make_warp_reduce_batched_specialization(
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
                items_per_member=specialization.outputs_per_thread,
            ),
        )
    )
    requirements = _build_execution_requirements(
        resolved,
        launch,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_WARP,
        call=call,
        resolved_group=resolved,
        implementation=specialization.specialization,
        topology=requirements.topology,
        participation=requirements.participation,
        result=result,
        synchronization=requirements.synchronization,
        temp_storage=requirements.temp_storage,
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/warp/warp_reduce_batched.cuh",
            cpp_class="cub::WarpReduceBatched",
            method=specialization.specialization.method_name,
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


__all__ = [
    "GroupReduceBatchedSemantics",
]
