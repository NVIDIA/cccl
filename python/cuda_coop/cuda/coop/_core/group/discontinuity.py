# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Discontinuity semantics and complete-block lowering.

The family preserves head/tail flag semantics and boundary operands while
selecting the CUB block implementation. Public dispatch and backend rendering
remain separate responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._types import RuntimeValue
from ..block.discontinuity import (
    BlockDiscontinuityMode,
    BlockDiscontinuitySemantics,
    make_block_discontinuity_spec,
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
class GroupDiscontinuitySemantics:
    """Block-discontinuity semantics attached to an explicit group."""

    primitive: BlockDiscontinuitySemantics

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockDiscontinuitySemantics):
            raise TypeError("primitive must be BlockDiscontinuitySemantics")

    @property
    def dtype(self) -> Any:
        return self.primitive.dtype

    @property
    def flag_dtype(self) -> Any:
        return self.primitive.flag_dtype

    @property
    def mode(self) -> BlockDiscontinuityMode:
        return self.primitive.mode

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupDiscontinuitySemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _plan_discontinuity(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupDiscontinuitySemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group discontinuity supports complete physical block groups",
        )
    assert launch.exact_block_dim is not None
    primitive = operation.primitive
    tile_predecessor_item = (
        RuntimeValue("tile_predecessor_item")
        if primitive.has_tile_predecessor
        else None
    )
    tile_successor_item = (
        RuntimeValue("tile_successor_item") if primitive.has_tile_successor else None
    )
    spec = make_block_discontinuity_spec(
        dtype=operation.dtype,
        flag_dtype=operation.flag_dtype,
        block_dim=launch.exact_block_dim,
        items_per_thread=operation.items_per_thread,
        mode=operation.mode,
        flag_operator=primitive.flag_operator,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
    ).specialization
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=(
            *(("tile_predecessor_item",) if primitive.has_tile_predecessor else ()),
            *(("tile_successor_item",) if primitive.has_tile_successor else ()),
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
            header="cub/block/block_discontinuity.cuh",
            cpp_class="cub::BlockDiscontinuity",
            method=spec.method_name,
        ),
    )


__all__ = ["GroupDiscontinuitySemantics"]
