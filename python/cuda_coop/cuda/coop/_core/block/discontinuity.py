# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockDiscontinuity semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._symbols import semantic_token
from .._types import (
    Array,
    CxxOperator,
    Dependency,
    PythonOperator,
    Reference,
    StatefulOperator,
    TemplateParameter,
    TempStorageParameter,
)
from ._common import BlockTileBoundary


class BlockDiscontinuityMode(str, Enum):
    """Flag arrays produced by a block discontinuity call."""

    HEADS = "heads"
    TAILS = "tails"
    HEADS_AND_TAILS = "heads_and_tails"

    @property
    def cub_method_name(self) -> str:
        return {
            BlockDiscontinuityMode.HEADS: "FlagHeads",
            BlockDiscontinuityMode.TAILS: "FlagTails",
            BlockDiscontinuityMode.HEADS_AND_TAILS: "FlagHeadsAndTails",
        }[self]

    @property
    def has_heads(self) -> bool:
        return self in {
            BlockDiscontinuityMode.HEADS,
            BlockDiscontinuityMode.HEADS_AND_TAILS,
        }

    @property
    def has_tails(self) -> bool:
        return self in {
            BlockDiscontinuityMode.TAILS,
            BlockDiscontinuityMode.HEADS_AND_TAILS,
        }


_FLAG_OPERATORS = (CxxOperator, PythonOperator, StatefulOperator)
_T = Dependency("T")
_FLAG_T = Dependency("FlagT")
_ITEMS_PER_THREAD = Dependency("ITEMS_PER_THREAD")


def _flag_array(name: str) -> Array:
    return Array(
        _FLAG_T,
        _ITEMS_PER_THREAD,
        name=name,
        is_output=True,
        is_return=False,
    )


@dataclass(frozen=True)
class BlockDiscontinuitySemantics:
    """Dimension-independent block-discontinuity call contract."""

    dtype: Any
    flag_dtype: Any
    mode: BlockDiscontinuityMode
    boundary: BlockTileBoundary
    items_per_thread: int
    flag_operator: CxxOperator | PythonOperator | StatefulOperator
    parameters: tuple[Any, ...]

    @property
    def method_name(self) -> str:
        return self.mode.cub_method_name

    @property
    def has_heads(self) -> bool:
        return self.mode.has_heads

    @property
    def has_tails(self) -> bool:
        return self.mode.has_tails

    @property
    def has_tile_predecessor(self) -> bool:
        return self.boundary.has_predecessor

    @property
    def has_tile_successor(self) -> bool:
        return self.boundary.has_successor

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "block_discontinuity",
            semantic_token(self.dtype),
            semantic_token(self.flag_dtype),
            self.mode.value,
            self.boundary.value,
            self.items_per_thread,
            semantic_token(self.flag_operator),
            semantic_token(self.parameters),
        )


@dataclass(frozen=True)
class BlockDiscontinuitySpec:
    """Fully specialized CUB BlockDiscontinuity semantics."""

    specialization: AlgorithmSpec
    call: BlockDiscontinuitySemantics
    block_dim: tuple[int, int, int]

    @property
    def dtype(self) -> Any:
        return self.call.dtype

    @property
    def flag_dtype(self) -> Any:
        return self.call.flag_dtype

    @property
    def mode(self) -> BlockDiscontinuityMode:
        return self.call.mode

    @property
    def boundary(self) -> BlockTileBoundary:
        return self.call.boundary

    @property
    def items_per_thread(self) -> int:
        return self.call.items_per_thread

    @property
    def flag_operator(self) -> CxxOperator | PythonOperator | StatefulOperator:
        return self.call.flag_operator

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def has_heads(self) -> bool:
        return self.call.has_heads

    @property
    def has_tails(self) -> bool:
        return self.call.has_tails

    @property
    def has_tile_predecessor(self) -> bool:
        return self.call.has_tile_predecessor

    @property
    def has_tile_successor(self) -> bool:
        return self.call.has_tile_successor

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_block_discontinuity_semantics(
    *,
    dtype: Any,
    flag_dtype: Any,
    items_per_thread: int,
    mode: str | BlockDiscontinuityMode,
    flag_operator: CxxOperator | PythonOperator | StatefulOperator,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
) -> BlockDiscontinuitySemantics:
    """Build the normalized dimension-independent call contract."""

    if dtype is None:
        raise ValueError("dtype must be provided")
    if flag_dtype is None:
        raise ValueError("flag dtype must be provided")
    mode = BlockDiscontinuityMode(mode)
    if (
        not isinstance(items_per_thread, int)
        or isinstance(items_per_thread, bool)
        or items_per_thread < 1
    ):
        raise ValueError("items_per_thread must be a positive integer")
    if not isinstance(flag_operator, _FLAG_OPERATORS):
        raise TypeError("BlockDiscontinuity requires a flag operator")
    if isinstance(flag_operator, (PythonOperator, StatefulOperator)) and (
        flag_operator.op is None
    ):
        raise ValueError("flag_op must be provided for block discontinuity")

    boundary = BlockTileBoundary.from_presence(
        predecessor=tile_predecessor_item is not None,
        successor=tile_successor_item is not None,
    )
    if mode is BlockDiscontinuityMode.HEADS and boundary.has_successor:
        raise ValueError("tile_successor_item is not valid for HEADS")
    if mode is BlockDiscontinuityMode.TAILS and boundary.has_predecessor:
        raise ValueError("tile_predecessor_item is not valid for TAILS")

    input_items = Array(_T, _ITEMS_PER_THREAD, name="input_items")
    predecessor = Reference(_T, name="tile_predecessor_item")
    successor = Reference(_T, name="tile_successor_item")
    parameters: list[Any] = [TempStorageParameter()]

    if mode is BlockDiscontinuityMode.HEADS:
        head_flags = _flag_array("head_flags")
        parameters.extend((head_flags, input_items, flag_operator))
        if boundary.has_predecessor:
            parameters.append(predecessor)
    elif mode is BlockDiscontinuityMode.TAILS:
        tail_flags = _flag_array("tail_flags")
        parameters.extend((tail_flags, input_items, flag_operator))
        if boundary.has_successor:
            parameters.append(successor)
    else:
        head_flags = _flag_array("head_flags")
        tail_flags = _flag_array("tail_flags")
        if boundary is BlockTileBoundary.BOTH:
            parameters.extend(
                (
                    head_flags,
                    predecessor,
                    tail_flags,
                    successor,
                    input_items,
                    flag_operator,
                )
            )
        elif boundary is BlockTileBoundary.PREDECESSOR:
            parameters.extend(
                (head_flags, predecessor, tail_flags, input_items, flag_operator)
            )
        elif boundary is BlockTileBoundary.SUCCESSOR:
            parameters.extend(
                (head_flags, tail_flags, successor, input_items, flag_operator)
            )
        else:
            parameters.extend((head_flags, tail_flags, input_items, flag_operator))

    return BlockDiscontinuitySemantics(
        dtype=dtype,
        flag_dtype=flag_dtype,
        mode=mode,
        boundary=boundary,
        items_per_thread=items_per_thread,
        flag_operator=flag_operator,
        parameters=tuple(parameters),
    )


def make_block_discontinuity_spec(
    *,
    dtype: Any,
    flag_dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    mode: str | BlockDiscontinuityMode,
    flag_operator: CxxOperator | PythonOperator | StatefulOperator,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
) -> BlockDiscontinuitySpec:
    """Build a fully specialized CUB BlockDiscontinuity description."""

    block_dim = tuple(block_dim)
    if len(block_dim) != 3 or any(dim < 1 for dim in block_dim):
        raise ValueError("block_dim must contain three positive dimensions")
    call = make_block_discontinuity_semantics(
        dtype=dtype,
        flag_dtype=flag_dtype,
        items_per_thread=items_per_thread,
        mode=mode,
        flag_operator=flag_operator,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
    )
    specialization = Algorithm(
        struct_name="BlockDiscontinuity",
        method_name=call.method_name,
        c_name="block_discontinuity",
        includes=("cub/block/block_discontinuity.cuh",),
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
            "FlagT": flag_dtype,
            "ITEMS_PER_THREAD": items_per_thread,
        },
        metadata={
            "scope": "block",
            "primitive": "discontinuity",
            "mode": call.mode,
            "boundary": call.boundary,
            "operator": type(call.flag_operator).__qualname__,
        },
    )
    return BlockDiscontinuitySpec(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
    )


__all__ = [
    "BlockDiscontinuityMode",
    "BlockDiscontinuitySemantics",
    "BlockDiscontinuitySpec",
    "make_block_discontinuity_semantics",
    "make_block_discontinuity_spec",
]
