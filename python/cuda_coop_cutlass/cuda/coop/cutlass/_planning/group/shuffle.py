# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan block Shuffle calls and describe their returned values.

Scalar Offset and Rotate return one value per member. Array Up and Down return
a fixed-size payload per member. The planner checks that the resolved group
and operand form match a CUB BlockShuffle overload, then records its result,
scratch, synchronization, and any distance precondition for lowering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._bindings import BindingKind
from .._types import ArgumentKind, ParameterClassification, ParameterRole
from ..block.shuffle import (
    BlockShuffleMode,
    BlockShuffleSemantics,
    BlockShuffleValueKind,
    make_block_shuffle_specialization,
)
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._dispatch import _register_group_operation_family
from ._execution_requirements import _build_execution_requirements, _unsupported
from ._model import (
    ArgumentPrecondition,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupPrimitiveCall,
    ImplementationProvenance,
    LogicalResultContract,
    PreconditionEnforcement,
    ResultContract,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    UnsupportedReasonCode,
)


@dataclass(frozen=True, eq=False)
class GroupShuffleSemantics:
    """Describe the per-member result of a scalar or array Shuffle.

    ``primitive`` retains the mode, dtype, extent, and distance binding before
    block dimensions are known. The group planner checks the mode against its
    operand form. Scalar calls return one item; array calls return a new
    payload with the input extent, leaving the input array unchanged.
    """

    primitive: BlockShuffleSemantics

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockShuffleSemantics):
            raise TypeError("primitive must be BlockShuffleSemantics")

    @property
    def dtype(self) -> Any:
        return self.primitive.dtype

    @property
    def mode(self) -> BlockShuffleMode:
        return self.primitive.mode

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread or 1

    @property
    def result_visibility(self) -> ResultVisibility:
        return ResultVisibility.PER_MEMBER

    @property
    def returns_value(self) -> bool:
        return True

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupShuffleSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _call_classifications(
    operation: GroupShuffleSemantics,
) -> tuple[ParameterClassification, ...]:
    """Classify Shuffle data, mode, and any explicit distance binding.

    Value is a runtime operand and mode is a static selector. An omitted
    distance adds no classification; an explicit distance retains its static
    or runtime binding so lowering can choose the correct provider ABI.
    """

    classifications = [
        ParameterClassification(
            "value",
            ArgumentKind.RUNTIME,
            ParameterRole.INPUT,
        )
    ]
    if operation.primitive.distance.kind is not BindingKind.OMITTED:
        argument_kind = operation.primitive.distance.argument_kind
        assert argument_kind is not None
        classifications.append(
            ParameterClassification(
                "distance",
                argument_kind,
                (
                    ParameterRole.CONSTANT
                    if argument_kind is ArgumentKind.STATIC
                    else ParameterRole.INPUT
                ),
            )
        )
    classifications.append(
        ParameterClassification(
            "mode",
            ArgumentKind.STATIC,
            ParameterRole.CONSTANT,
        )
    )
    return tuple(classifications)


def _plan_shuffle(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupShuffleSemantics,
) -> GroupLoweringPlan:
    """Choose CUB BlockShuffle and record its result and distance rules.

    The dispatcher rejects non-block groups before this planner runs.
    Return an unsupported plan for Rotate on a one-thread block, for array
    calls other than Up/Down or with a distance, and for scalar calls other
    than Offset/Rotate. Planning uses the exact block dimensions.

    A supported plan has implementation-owned scratch and a scalar or array
    result owned by each member. Rotate also records the allowed distance
    range. A static bound is checked during specialization; a runtime bound is
    recorded as a caller precondition, not an inserted guard at this stage.
    """

    assert resolved.kind == "block"
    assert launch.exact_block_dim is not None
    block_threads = launch.exact_block_threads
    assert block_threads is not None
    primitive = operation.primitive
    if primitive.mode is BlockShuffleMode.ROTATE and block_threads < 2:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "public CUB Rotate requires a block with at least two threads",
        )
    if (
        primitive.value_kind is BlockShuffleValueKind.ARRAY
        and primitive.mode
        not in {
            BlockShuffleMode.UP,
            BlockShuffleMode.DOWN,
        }
    ):
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "public CUB ThreadData shuffle supports only Up and Down",
        )
    if (
        primitive.value_kind is BlockShuffleValueKind.SCALAR
        and primitive.mode
        not in {
            BlockShuffleMode.OFFSET,
            BlockShuffleMode.ROTATE,
        }
    ):
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "public CUB scalar shuffle supports only Offset and Rotate",
        )
    if (
        primitive.value_kind is BlockShuffleValueKind.ARRAY
        and primitive.distance.kind is not BindingKind.OMITTED
    ):
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "public CUB ThreadData shuffle uses a unit shift",
        )
    specialization = make_block_shuffle_specialization(
        dtype=operation.dtype,
        block_dim=launch.exact_block_dim,
        mode=primitive.mode,
        items_per_thread=primitive.items_per_thread,
        distance=primitive.distance,
    ).specialization
    result = ResultContract(
        (
            LogicalResultContract(
                name="value",
                dtype=operation.dtype,
                visibility=ResultVisibility.PER_MEMBER,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=(
                    GroupOperandKind.ARRAY
                    if primitive.value_kind is BlockShuffleValueKind.ARRAY
                    else GroupOperandKind.SCALAR
                ),
                items_per_member=operation.items_per_thread,
            ),
        )
    )
    requirements = _build_execution_requirements(
        resolved,
        launch,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        argument_preconditions=(
            (
                ArgumentPrecondition(
                    name="distance",
                    minimum=1,
                    maximum=block_threads - 1,
                    enforcement=(
                        PreconditionEnforcement.CALLER
                        if primitive.distance.kind is BindingKind.RUNTIME
                        else PreconditionEnforcement.PLANNER_VALIDATED
                    ),
                ),
            )
            if primitive.mode is BlockShuffleMode.ROTATE
            else ()
        ),
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=specialization,
        topology=requirements.topology,
        participation=requirements.participation,
        result=result,
        synchronization=requirements.synchronization,
        temp_storage=requirements.temp_storage,
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_shuffle.cuh",
            cpp_class="cub::BlockShuffle",
            method=specialization.method_name,
        ),
    )


_register_group_operation_family(
    GroupShuffleSemantics,
    classifications=_call_classifications,
    planner=_plan_shuffle,
    group_kinds=frozenset({"block"}),
    unsupported_group_message=(
        "cuda.coop Shuffle supports complete physical this_block() groups"
    ),
)


__all__ = [
    "GroupShuffleSemantics",
]
