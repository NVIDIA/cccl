# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Adjacent-difference semantics and block-group lowering.

The portable planner records boundary and direction semantics here and lowers
supported complete block groups to CUB. Backend request collection and code
generation are intentionally separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._types import RuntimeValue
from ..block.adjacent_difference import (
    BlockAdjacentDifferenceBoundary,
    BlockAdjacentDifferenceDirection,
    BlockAdjacentDifferenceSemantics,
    make_block_adjacent_difference_spec,
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
class GroupAdjacentDifferenceSemantics:
    """Block-adjacent-difference semantics attached to an explicit group."""

    primitive: BlockAdjacentDifferenceSemantics

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockAdjacentDifferenceSemantics):
            raise TypeError("primitive must be BlockAdjacentDifferenceSemantics")

    @property
    def dtype(self) -> Any:
        return self.primitive.dtype

    @property
    def direction(self) -> BlockAdjacentDifferenceDirection:
        return self.primitive.direction

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupAdjacentDifferenceSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _plan_adjacent_difference(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupAdjacentDifferenceSemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group adjacent_difference supports complete physical block groups",
        )
    assert launch.exact_block_dim is not None
    primitive = operation.primitive
    valid_items = RuntimeValue("valid_items") if primitive.has_partial_tile else None
    tile_predecessor_item = (
        RuntimeValue("tile_predecessor_item")
        if primitive.boundary is BlockAdjacentDifferenceBoundary.PREDECESSOR
        else None
    )
    tile_successor_item = (
        RuntimeValue("tile_successor_item")
        if primitive.boundary is BlockAdjacentDifferenceBoundary.SUCCESSOR
        else None
    )
    spec = make_block_adjacent_difference_spec(
        dtype=operation.dtype,
        block_dim=launch.exact_block_dim,
        items_per_thread=operation.items_per_thread,
        direction=operation.direction,
        difference_operator=primitive.difference_operator,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
    ).specialization
    uniform_arguments = []
    if primitive.has_partial_tile:
        uniform_arguments.append("valid_items")
    if primitive.boundary is BlockAdjacentDifferenceBoundary.PREDECESSOR:
        uniform_arguments.append("tile_predecessor_item")
    elif primitive.boundary is BlockAdjacentDifferenceBoundary.SUCCESSOR:
        uniform_arguments.append("tile_successor_item")
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=tuple(uniform_arguments),
        valid_member_selection=(
            "first valid_items tile elements" if primitive.has_partial_tile else None
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
            header="cub/block/block_adjacent_difference.cuh",
            cpp_class="cub::BlockAdjacentDifference",
            method=spec.method_name,
        ),
    )


__all__ = ["GroupAdjacentDifferenceSemantics"]
