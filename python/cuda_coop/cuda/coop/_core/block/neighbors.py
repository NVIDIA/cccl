# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Input-preserving CUB block neighbor operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec, TypeDefinition
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
        raise ValueError("right partial tiles do not support tile_successor_item")


@dataclass(frozen=True)
class BlockNeighborSemantics:
    operation: str
    dtype: Any
    items_per_thread: int
    mode: str
    operator: CxxOperator | PythonOperator
    partial: bool = False
    predecessor: bool = False
    successor: bool = False

    def __post_init__(self):
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
        return ("heads", "tails") if self.mode == "heads_and_tails" else (self.mode,)

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


def make_block_neighbor_spec(
    call: BlockNeighborSemantics, *, block_dim: tuple[int, int, int]
) -> AlgorithmSpec:
    block_dim = tuple(block_dim)
    if len(block_dim) != 3 or any(
        not isinstance(d, int) or isinstance(d, bool) or d < 1 for d in block_dim
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
    wrapper_name += f"_{int(call.partial)}{int(call.predecessor)}{int(call.successor)}"
    guard = (
        """
    if (valid_items < 0 || valid_items > BlockDimX * BlockDimY * BlockDimZ * ItemsPerThread)
    {
      asm volatile("trap;");
    }
"""
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
""",
    )
    return Algorithm(
        struct_name=wrapper_name,
        method_name="Apply",
        c_name="block_" + call.operation,
        includes=(f"cub/block/block_{call.operation}.cuh", "cuda/std/functional"),
        type_definitions=(definition,),
        template_parameters=tuple(
            TemplateParameter(name)
            for name in ("T", "BlockDimX", "BlockDimY", "BlockDimZ", "ItemsPerThread")
        ),
        parameters=(tuple(parameters),),
    ).specialize(
        dict(
            T=call.dtype,
            BlockDimX=block_dim[0],
            BlockDimY=block_dim[1],
            BlockDimZ=block_dim[2],
            ItemsPerThread=call.items_per_thread,
        ),
        metadata={"scope": "block", "primitive": call.operation, "method": method},
    )


__all__ = ["BlockNeighborSemantics", "make_block_neighbor_spec"]
