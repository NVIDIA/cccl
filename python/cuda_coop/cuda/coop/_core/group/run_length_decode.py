# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run-length-decode semantics and complete-block lowering.

This family owns the fused public-CUB result contract, including optional
relative offsets and the total decoded size. Backend code generation and
compiler lifecycle stay outside the portable planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..block.run_length import (
    BlockRunLengthDecodeSemantics,
    BlockRunLengthDecodeStage,
    make_block_run_length_decode_spec,
)
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._contracts import _contracts, _unsupported
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
class GroupRunLengthDecodeSemantics:
    """Fused public-CUB run-length decode semantics for one block group."""

    primitive: BlockRunLengthDecodeSemantics

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockRunLengthDecodeSemantics):
            raise TypeError("primitive must be BlockRunLengthDecodeSemantics")
        if self.primitive.run_length_dtype is None:
            raise ValueError("group run-length decode requires a run-length dtype")
        if self.primitive.total_decoded_size_dtype is None:
            raise ValueError(
                "group run-length decode requires a total decoded-size dtype"
            )
        if not self.primitive.returns_total_decoded_size:
            raise ValueError(
                "group run-length decode requires the fused total-size result"
            )

    @property
    def dtype(self) -> Any:
        return self.primitive.item_dtype

    @property
    def items_per_thread(self) -> int:
        return self.primitive.decoded_items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupRunLengthDecodeSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _plan_run_length_decode(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupRunLengthDecodeSemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group run-length decode supports complete physical block groups",
        )
    assert launch.exact_block_dim is not None
    primitive = operation.primitive
    assert primitive.run_length_dtype is not None
    assert primitive.total_decoded_size_dtype is not None
    spec = make_block_run_length_decode_spec(
        item_dtype=primitive.item_dtype,
        run_length_dtype=primitive.run_length_dtype,
        decoded_offset_dtype=primitive.decoded_offset_dtype,
        total_decoded_size_dtype=primitive.total_decoded_size_dtype,
        relative_offset_dtype=primitive.relative_offset_dtype,
        block_dim=launch.exact_block_dim,
        runs_per_thread=primitive.runs_per_thread,
        decoded_items_per_thread=primitive.decoded_items_per_thread,
        stage=BlockRunLengthDecodeStage.FUSED,
        with_relative_offsets=primitive.has_relative_offsets,
        with_decoded_window_offset=True,
        returns_total_decoded_size=True,
    ).specialization
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=("decoded_window_offset",),
        argument_preconditions=(
            ArgumentPrecondition(
                name="run_lengths",
                minimum=0,
                maximum=None,
                enforcement=PreconditionEnforcement.CALLER,
            ),
            ArgumentPrecondition(
                name="sum(run_lengths)",
                minimum=1,
                maximum=None,
                enforcement=PreconditionEnforcement.CALLER,
            ),
            ArgumentPrecondition(
                name="decoded_window_offset",
                minimum=0,
                maximum=None,
                enforcement=PreconditionEnforcement.CALLER,
            ),
        ),
    )
    results = [
        LogicalResultContract(
            name="decoded_items",
            dtype=primitive.item_dtype,
            visibility=ResultVisibility.PER_MEMBER,
            ownership=ResultOwnership.EACH_MEMBER,
            operand_kind=GroupOperandKind.ARRAY,
            items_per_member=primitive.decoded_items_per_thread,
        )
    ]
    if primitive.has_relative_offsets:
        results.append(
            LogicalResultContract(
                name="relative_offsets",
                dtype=primitive.relative_offset_dtype,
                visibility=ResultVisibility.PER_MEMBER,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=GroupOperandKind.ARRAY,
                items_per_member=primitive.decoded_items_per_thread,
            )
        )
    results.append(
        LogicalResultContract(
            name="total_decoded_size",
            dtype=primitive.total_decoded_size_dtype,
            visibility=ResultVisibility.ALL_MEMBERS,
            ownership=ResultOwnership.EACH_MEMBER,
            operand_kind=GroupOperandKind.SCALAR,
            items_per_member=1,
        )
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=ResultContract(tuple(results)),
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_run_length_decode.cuh",
            cpp_class="cub::BlockRunLengthDecode",
            method=primitive.decode_method_name,
        ),
    )


__all__ = ["GroupRunLengthDecodeSemantics"]
