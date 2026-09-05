# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockAdjacentDifference semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._symbols import semantic_token
from .._types import (
    INT32,
    Array,
    CxxOperator,
    Dependency,
    PythonOperator,
    Reference,
    StatefulOperator,
    TemplateParameter,
    TempStorageParameter,
    Value,
)
from ._common import BlockTileBoundary


class BlockAdjacentDifferenceDirection(str, Enum):
    """Neighbor selected by a block adjacent-difference call."""

    LEFT = "left"
    RIGHT = "right"

    @property
    def cub_method_prefix(self) -> str:
        return (
            "SubtractLeft"
            if self is BlockAdjacentDifferenceDirection.LEFT
            else "SubtractRight"
        )


class BlockAdjacentDifferenceTilePolicy(str, Enum):
    """Whether every item in the blocked tile participates."""

    FULL = "full"
    PARTIAL = "partial"


# Compatibility alias for the boundary subset accepted by this primitive.
BlockAdjacentDifferenceBoundary = BlockTileBoundary


_DIFFERENCE_OPERATORS = (CxxOperator, PythonOperator, StatefulOperator)
_T = Dependency("T")
_ITEMS_PER_THREAD = Dependency("ITEMS_PER_THREAD")


@dataclass(frozen=True)
class BlockAdjacentDifferenceSemantics:
    """Dimension-independent adjacent-difference call contract.

    Runtime ``valid_items`` and tile-boundary payloads are deliberately not
    retained. Their presence selects a public CUB overload and participates in
    semantic identity without making runtime values part of cache identity.
    """

    dtype: Any
    direction: BlockAdjacentDifferenceDirection
    tile_policy: BlockAdjacentDifferenceTilePolicy
    boundary: BlockAdjacentDifferenceBoundary
    items_per_thread: int
    difference_operator: CxxOperator | PythonOperator | StatefulOperator
    parameters: tuple[Any, ...]

    @property
    def method_name(self) -> str:
        suffix = (
            "PartialTile"
            if self.tile_policy is BlockAdjacentDifferenceTilePolicy.PARTIAL
            else ""
        )
        return f"{self.direction.cub_method_prefix}{suffix}"

    @property
    def has_partial_tile(self) -> bool:
        return self.tile_policy is BlockAdjacentDifferenceTilePolicy.PARTIAL

    @property
    def has_boundary_item(self) -> bool:
        return self.boundary is not BlockAdjacentDifferenceBoundary.NONE

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "block_adjacent_difference",
            semantic_token(self.dtype),
            self.direction.value,
            self.tile_policy.value,
            self.boundary.value,
            self.items_per_thread,
            semantic_token(self.difference_operator),
            semantic_token(self.parameters),
        )


@dataclass(frozen=True)
class BlockAdjacentDifferenceSpec:
    """Fully specialized CUB BlockAdjacentDifference semantics."""

    specialization: AlgorithmSpec
    call: BlockAdjacentDifferenceSemantics
    block_dim: tuple[int, int, int]

    @property
    def direction(self) -> BlockAdjacentDifferenceDirection:
        return self.call.direction

    @property
    def tile_policy(self) -> BlockAdjacentDifferenceTilePolicy:
        return self.call.tile_policy

    @property
    def boundary(self) -> BlockAdjacentDifferenceBoundary:
        return self.call.boundary

    @property
    def items_per_thread(self) -> int:
        return self.call.items_per_thread

    @property
    def difference_operator(self) -> CxxOperator | PythonOperator | StatefulOperator:
        return self.call.difference_operator

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def has_partial_tile(self) -> bool:
        return self.call.has_partial_tile

    @property
    def has_boundary_item(self) -> bool:
        return self.call.has_boundary_item

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_block_adjacent_difference_semantics(
    *,
    dtype: Any,
    items_per_thread: int,
    direction: str | BlockAdjacentDifferenceDirection,
    difference_operator: CxxOperator | PythonOperator | StatefulOperator,
    valid_items: Any = None,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
) -> BlockAdjacentDifferenceSemantics:
    """Build the normalized dimension-independent call contract."""

    if dtype is None:
        raise ValueError("dtype must be provided")
    direction = BlockAdjacentDifferenceDirection(direction)
    if (
        not isinstance(items_per_thread, int)
        or isinstance(items_per_thread, bool)
        or items_per_thread < 1
    ):
        raise ValueError("items_per_thread must be a positive integer")
    if not isinstance(difference_operator, _DIFFERENCE_OPERATORS):
        raise TypeError("BlockAdjacentDifference requires a difference operator")
    if isinstance(difference_operator, (PythonOperator, StatefulOperator)) and (
        difference_operator.op is None
    ):
        raise ValueError("difference_op must be provided")
    if tile_predecessor_item is not None and tile_successor_item is not None:
        raise ValueError(
            "tile_predecessor_item and tile_successor_item are mutually exclusive"
        )
    if (
        direction is BlockAdjacentDifferenceDirection.LEFT
        and tile_successor_item is not None
    ):
        raise ValueError("tile_successor_item is not valid for SubtractLeft")
    if (
        direction is BlockAdjacentDifferenceDirection.RIGHT
        and tile_predecessor_item is not None
    ):
        raise ValueError("tile_predecessor_item is not valid for SubtractRight")
    if (
        direction is BlockAdjacentDifferenceDirection.RIGHT
        and valid_items is not None
        and tile_successor_item is not None
    ):
        # CUB intentionally has no overload combining a partial right tile
        # with an external successor: the final valid item is copied through.
        raise ValueError(
            "tile_successor_item is not valid for SubtractRightPartialTile"
        )

    tile_policy = (
        BlockAdjacentDifferenceTilePolicy.PARTIAL
        if valid_items is not None
        else BlockAdjacentDifferenceTilePolicy.FULL
    )
    boundary = BlockTileBoundary.from_presence(
        predecessor=tile_predecessor_item is not None,
        successor=tile_successor_item is not None,
    )

    parameters: list[Any] = [
        TempStorageParameter(),
        Array(_T, _ITEMS_PER_THREAD, name="input_items"),
        Array(
            _T,
            _ITEMS_PER_THREAD,
            name="output_items",
            is_output=True,
            is_return=False,
        ),
        difference_operator,
    ]
    if tile_policy is BlockAdjacentDifferenceTilePolicy.PARTIAL:
        parameters.append(Value(INT32, name="valid_items"))
    if boundary is BlockAdjacentDifferenceBoundary.PREDECESSOR:
        parameters.append(Reference(_T, name="tile_predecessor_item"))
    elif boundary is BlockAdjacentDifferenceBoundary.SUCCESSOR:
        parameters.append(Reference(_T, name="tile_successor_item"))

    return BlockAdjacentDifferenceSemantics(
        dtype=dtype,
        direction=direction,
        tile_policy=tile_policy,
        boundary=boundary,
        items_per_thread=items_per_thread,
        difference_operator=difference_operator,
        parameters=tuple(parameters),
    )


def make_block_adjacent_difference_spec(
    *,
    dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    direction: str | BlockAdjacentDifferenceDirection,
    difference_operator: CxxOperator | PythonOperator | StatefulOperator,
    valid_items: Any = None,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
) -> BlockAdjacentDifferenceSpec:
    """Build a fully specialized CUB BlockAdjacentDifference description."""

    block_dim = tuple(block_dim)
    if len(block_dim) != 3 or any(dim < 1 for dim in block_dim):
        raise ValueError("block_dim must contain three positive dimensions")
    call = make_block_adjacent_difference_semantics(
        dtype=dtype,
        items_per_thread=items_per_thread,
        direction=direction,
        difference_operator=difference_operator,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
    )
    specialization = Algorithm(
        struct_name="BlockAdjacentDifference",
        method_name=call.method_name,
        c_name="block_adjacent_difference",
        includes=("cub/block/block_adjacent_difference.cuh",),
        template_parameters=(
            TemplateParameter("T"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ),
        parameters=(call.parameters,),
    ).specialize(
        {
            "T": dtype,
            "BLOCK_DIM_X": block_dim[0],
            "BLOCK_DIM_Y": block_dim[1],
            "BLOCK_DIM_Z": block_dim[2],
            "ITEMS_PER_THREAD": items_per_thread,
        },
        metadata={
            "scope": "block",
            "primitive": "adjacent_difference",
            "direction": call.direction,
            "tile_policy": call.tile_policy,
            "boundary": call.boundary,
            "operator": type(call.difference_operator).__qualname__,
        },
    )
    return BlockAdjacentDifferenceSpec(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
    )


__all__ = [
    "BlockAdjacentDifferenceBoundary",
    "BlockAdjacentDifferenceDirection",
    "BlockAdjacentDifferenceSemantics",
    "BlockAdjacentDifferenceSpec",
    "BlockAdjacentDifferenceTilePolicy",
    "make_block_adjacent_difference_semantics",
    "make_block_adjacent_difference_spec",
]
