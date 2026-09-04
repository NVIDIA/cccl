# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shuffle semantics and complete-block lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._bindings import BindingKind
from .._types import ArgumentKind, ParameterClassification, ParameterRole
from ..block.shuffle import (
    BlockShuffleMode,
    BlockShuffleSemantics,
    BlockShuffleValueKind,
    make_block_shuffle_spec,
)
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._contracts import _contracts, _unsupported
from ._dispatch import _register_group_operation_family
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
    """Scalar or array BlockShuffle operation selected after group resolution."""

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
    if primitive.value_kind is BlockShuffleValueKind.ARRAY and primitive.mode not in {
        BlockShuffleMode.UP,
        BlockShuffleMode.DOWN,
    }:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "public CUB ThreadData shuffle supports only Up and Down",
        )
    if primitive.value_kind is BlockShuffleValueKind.SCALAR and primitive.mode not in {
        BlockShuffleMode.OFFSET,
        BlockShuffleMode.ROTATE,
    }:
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
    spec = make_block_shuffle_spec(
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
    contracts = _contracts(
        resolved,
        launch,
        result=result,
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
        implementation=spec,
        topology=contracts[0],
        participation=contracts[1],
        result=result,
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_shuffle.cuh",
            cpp_class="cub::BlockShuffle",
            method=spec.method_name,
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


__all__ = ["GroupShuffleSemantics"]
