# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe input-preserving CUB operations on neighboring tile items.

Adjacent Difference writes differences; Discontinuity writes head or tail
flags. Both use separate input and output arrays. This module validates the
operation choices and describes a C++ adapter for the selected CUB overload.
The adapter has one argument shape per output count, even when a selected
overload ignores the valid count or outside-tile neighbor arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._algorithm import Algorithm, TypeDefinition
from .._symbols import semantic_token
from .._types import (
    INT32,
    INT64,
    Array,
    CxxOperator,
    Dependency,
    PythonOperator,
    Reference,
    TemplateParameter,
    TempStorageParameter,
    Value,
)


def validate_neighbor_options(
    operation, mode, *, partial=False, predecessor=False, successor=False
):
    """Reject mode and boundary combinations that CUB cannot provide.

    Left differences and heads can use a predecessor; right differences and
    tails can use a successor. Discontinuity requires a full tile. Partial
    right differences have no successor overload. These checks select a valid
    CUB call; the frontend separately checks the runtime values and their
    dtypes.
    """

    modes = (
        ("left", "right")
        if operation == "adjacent_difference"
        else ("heads", "tails", "heads_and_tails")
    )
    if operation not in {"adjacent_difference", "discontinuity"}:
        raise ValueError("unknown block neighbor operation")
    if not isinstance(mode, str) or mode not in modes:
        raise ValueError(f"{operation} mode must be one of {modes}")
    if predecessor and mode in {"right", "tails"}:
        raise ValueError(f"{mode} does not accept tile_predecessor_item")
    if successor and mode in {"left", "heads"}:
        raise ValueError(f"{mode} does not accept tile_successor_item")
    if operation == "discontinuity" and partial:
        raise ValueError("discontinuity requires a full tile")
    if mode == "right" and partial and successor:
        raise ValueError(
            "right partial tiles do not support tile_successor_item"
        )


@dataclass(frozen=True)
class BlockNeighborSemantics:
    """Keep neighbor-operation choices separate from runtime tile values.

    The record fixes the operation, mode, input dtype, per-thread extent, and
    binary operator. The three flags select overloads with a partial tile or
    outside-tile neighbors; they do not store the count or neighbor values.
    Differences keep the input dtype. Discontinuity returns one or two int32
    flag arrays. The semantic key identifies these choices for provider
    caching.
    """

    operation: str
    dtype: Any
    items_per_thread: int
    mode: str
    operator: CxxOperator | PythonOperator
    partial: bool = False
    predecessor: bool = False
    successor: bool = False

    def __post_init__(self):
        """Check overload choices, payload extent and the operator record."""

        validate_neighbor_options(
            self.operation,
            self.mode,
            partial=self.partial,
            predecessor=self.predecessor,
            successor=self.successor,
        )
        if self.dtype is None:
            raise ValueError("neighbor input dtype is required")
        if (
            not isinstance(self.items_per_thread, int)
            or isinstance(self.items_per_thread, bool)
            or self.items_per_thread < 1
        ):
            raise ValueError("items_per_thread must be a positive integer")
        if not isinstance(self.operator, (CxxOperator, PythonOperator)):
            raise TypeError("neighbor operation requires a binary operator")

    @property
    def result_dtype(self):
        return self.dtype if self.operation == "adjacent_difference" else INT32

    @property
    def result_names(self):
        if self.operation == "adjacent_difference":
            return ("differences",)
        return (
            ("heads", "tails")
            if self.mode == "heads_and_tails"
            else (self.mode,)
        )

    @property
    def semantic_key(self):
        return (
            "block_neighbors",
            self.operation,
            semantic_token(self.dtype),
            self.items_per_thread,
            self.mode,
            semantic_token(self.operator),
            self.partial,
            self.predecessor,
            self.successor,
        )


def make_block_neighbor_specialization(
    call: BlockNeighborSemantics, *, block_dim: tuple[int, int, int]
) -> Algorithm:
    """Describe the CUB adapter and its arguments for one fixed block shape.

    Select the CUB class and method from the operation, mode, and boundary
    flags. The wrapper always accepts a count, predecessor, and successor so
    frontends can use a regular call shape; only selected arguments reach the
    CUB overload. Partial calls check the wide count before converting it to
    CUB's integer.

    The input array remains separate from result arrays. Result parameters are
    inout wrapper arguments, so frontends allocate them and expose them as the
    public return value. Creating this Algorithm does not compile GPU code.
    """

    block_dim = tuple(block_dim)
    if len(block_dim) != 3 or any(
        not isinstance(d, int) or isinstance(d, bool) or d < 1
        for d in block_dim
    ):
        raise ValueError("block_dim must contain three positive dimensions")
    dtype = Dependency("T")
    extent = Dependency("ItemsPerThread")
    parameters = [
        TempStorageParameter(),
        Array(dtype, extent, name="values", is_inout=False, is_return=False),
    ]
    parameters.extend(
        Array(
            dtype if call.operation == "adjacent_difference" else INT32,
            extent,
            name=name,
            is_inout=True,
            is_return=False,
        )
        for name in call.result_names
    )
    parameters.extend(
        (
            call.operator,
            Value(INT64, name="valid_items"),
            Reference(dtype, name="tile_predecessor_item"),
            Reference(dtype, name="tile_successor_item"),
        )
    )
    adjacent = call.operation == "adjacent_difference"
    primitive = "BlockAdjacentDifference" if adjacent else "BlockDiscontinuity"
    output_decls = ", ".join(
        f"{'T' if adjacent else 'int'} (&{name})[ItemsPerThread]"
        for name in call.result_names
    )
    if adjacent:
        method = "SubtractLeft" if call.mode == "left" else "SubtractRight"
        args = ["values", "differences", "op"]
        if call.partial:
            method += "PartialTile"
            args.append("static_cast<int>(valid_items)")
        if call.predecessor:
            args.append("predecessor")
        if call.successor:
            args.append("successor")
    else:
        method = {
            "heads": "FlagHeads",
            "tails": "FlagTails",
            "heads_and_tails": "FlagHeadsAndTails",
        }[call.mode]
        if call.mode == "heads_and_tails":
            args = ["heads"]
            if call.predecessor:
                args.append("predecessor")
            args.append("tails")
            if call.successor:
                args.append("successor")
            args.extend(("values", "op"))
        else:
            args = [call.mode, "values", "op"]
            if call.predecessor:
                args.append("predecessor")
            if call.successor:
                args.append("successor")
    wrapper_name = "CudaCoop" + primitive + "_" + call.mode
    wrapper_name += (
        f"_{int(call.partial)}{int(call.predecessor)}{int(call.successor)}"
    )
    guard = (
        """
    if (valid_items < 0 || valid_items > BlockDimX * BlockDimY * BlockDimZ * ItemsPerThread)
    {
      asm volatile("trap;");
    }
"""  # noqa: E501 - Embedded C++ source.
        if call.partial
        else ""
    )
    definition = TypeDefinition(
        name=wrapper_name,
        code=f"""
namespace cub {{
template <typename T, int BlockDimX, int BlockDimY, int BlockDimZ, int ItemsPerThread>
struct {wrapper_name} : {primitive}<T, BlockDimX, BlockDimY, BlockDimZ>
{{
  using Base = {primitive}<T, BlockDimX, BlockDimY, BlockDimZ>;
  using Base::Base;
  template <typename Op>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Apply(
    T (&values)[ItemsPerThread], {output_decls}, Op op,
    long long valid_items, T predecessor, T successor)
  {{{guard}
    Base::{method}({", ".join(args)});
  }}
}};
}}
""",  # noqa: E501 - Embedded C++ source.
    )
    return Algorithm(
        struct_name=wrapper_name,
        method_name="Apply",
        c_name="block_" + call.operation,
        includes=(
            f"cub/block/block_{call.operation}.cuh",
            "cuda/std/functional",
        ),
        type_definitions=(definition,),
        template_parameters=tuple(
            TemplateParameter(name)
            for name in (
                "T",
                "BlockDimX",
                "BlockDimY",
                "BlockDimZ",
                "ItemsPerThread",
            )
        ),
        parameters=(tuple(parameters),),
        template_arguments={
            "T": call.dtype,
            "BlockDimX": block_dim[0],
            "BlockDimY": block_dim[1],
            "BlockDimZ": block_dim[2],
            "ItemsPerThread": call.items_per_thread,
        },
        metadata={
            "scope": "block",
            "primitive": call.operation,
            "method": method,
        },
    )


__all__ = ["BlockNeighborSemantics", "make_block_neighbor_specialization"]
