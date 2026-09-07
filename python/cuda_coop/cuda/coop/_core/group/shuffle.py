# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shuffle semantics and complete-block lowering.

This module owns scalar/array shuffle result semantics and their CUB block
specialization. Backend compiler and renderer lifecycle are intentionally
outside the portable planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._bindings import BindingKind
from ..block.shuffle import (
    BlockShuffleMode,
    BlockShuffleSemantics,
    BlockShuffleValueKind,
    make_block_shuffle_spec,
)
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._contracts import _contracts, _unsupported
from ._model import (
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupPrimitiveCall,
    ImplementationProvenance,
    ResultVisibility,
    StorageOwnership,
    UnsupportedReasonCode,
)


@dataclass(frozen=True, eq=False)
class GroupShuffleSemantics:
    """Public-CUB-compatible block-shuffle semantics for an explicit group."""

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
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupShuffleSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _plan_shuffle(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupShuffleSemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group shuffle supports complete physical block groups",
        )
    primitive = operation.primitive
    is_array = primitive.value_kind is BlockShuffleValueKind.ARRAY
    if is_array and primitive.mode not in {
        BlockShuffleMode.UP,
        BlockShuffleMode.DOWN,
    }:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "public CUB ThreadData shuffle supports only unit-shift Up and Down",
        )
    if not is_array and primitive.mode not in {
        BlockShuffleMode.OFFSET,
        BlockShuffleMode.ROTATE,
    }:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "public CUB scalar shuffle supports only Offset and Rotate",
        )
    assert launch.exact_block_dim is not None
    spec = make_block_shuffle_spec(
        dtype=operation.dtype,
        block_dim=launch.exact_block_dim,
        mode=primitive.mode,
        items_per_thread=primitive.items_per_thread,
        distance=primitive.distance,
        block_prefix=primitive.block_prefix,
        block_suffix=primitive.block_suffix,
    ).specialization
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=(
            ("distance",) if primitive.distance.kind is BindingKind.RUNTIME else ()
        ),
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=contracts[1],
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_shuffle.cuh",
            cpp_class="cub::BlockShuffle",
            method=spec.method_name,
        ),
    )


__all__ = ["GroupShuffleSemantics"]
