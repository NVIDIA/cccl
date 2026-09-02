# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB WarpReduce descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._symbols import semantic_token
from ..block.reduce import BlockReduceOperator, normalize_block_reduce_operator
from ..thread_group import (
    PHYSICAL_WARP_THREADS,
    Dim3,
    normalize_thread_dim,
)


class WarpReduceOperation(str, Enum):
    """Supported CUB WarpReduce operations."""

    REDUCE = "reduce"
    SUM = "sum"


@dataclass(frozen=True, eq=False)
class WarpReduceSpec:
    """Fully specialized scalar CUB WarpReduce description."""

    dtype: Any
    block_dim: Dim3
    operation: WarpReduceOperation = WarpReduceOperation.REDUCE
    binary_op: BlockReduceOperator = BlockReduceOperator.SUM
    valid_items: bool = False
    threads_in_warp: int = PHYSICAL_WARP_THREADS

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", WarpReduceOperation(self.operation))
        object.__setattr__(
            self,
            "binary_op",
            normalize_block_reduce_operator(self.binary_op),
        )
        object.__setattr__(
            self,
            "block_dim",
            normalize_thread_dim(
                self.block_dim,
                scope="cuda.coop warp reduction",
                label="block",
            ),
        )
        if self.operation is WarpReduceOperation.SUM and (
            self.binary_op is not BlockReduceOperator.SUM
        ):
            raise ValueError("cuda.coop.sum requires the sum operator")
        if not isinstance(self.valid_items, bool):
            raise TypeError("valid_items must describe whether an argument is present")
        if self.threads_in_warp != PHYSICAL_WARP_THREADS:
            raise ValueError(
                "WarpReduce support requires a 32-thread physical CUDA warp"
            )
        block_threads = self.block_dim[0] * self.block_dim[1] * self.block_dim[2]
        if block_threads % PHYSICAL_WARP_THREADS != 0:
            raise ValueError(
                "WarpReduce requires an enclosing block composed of complete "
                "32-thread physical warps"
            )

    @property
    def method_name(self) -> str:
        """Return the CUB method selected by this description."""

        return "Sum" if self.binary_op is BlockReduceOperator.SUM else "Reduce"

    @property
    def warp_count(self) -> int:
        """Return the number of complete physical warps in the block."""

        x, y, z = self.block_dim
        return x * y * z // self.threads_in_warp

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "warp_reduce",
            semantic_token(self.dtype),
            self.block_dim,
            self.threads_in_warp,
            self.operation.value,
            self.binary_op.value,
            self.valid_items,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WarpReduceSpec):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def make_warp_reduce_spec(
    *,
    dtype: Any,
    block_dim: Dim3 | int | tuple[int, ...] | list[int],
    operation: str | WarpReduceOperation = WarpReduceOperation.REDUCE,
    binary_op: Any = None,
    valid_items: bool = False,
) -> WarpReduceSpec:
    """Build one scalar physical-warp CUB reduction description."""

    operation = WarpReduceOperation(operation)
    normalized_operator = normalize_block_reduce_operator(binary_op)
    if operation is WarpReduceOperation.SUM and (
        normalized_operator is not BlockReduceOperator.SUM
    ):
        raise ValueError("cuda.coop.sum requires the sum operator")
    return WarpReduceSpec(
        dtype=dtype,
        block_dim=normalize_thread_dim(
            block_dim,
            scope="make_warp_reduce_spec",
            label="block",
        ),
        operation=operation,
        binary_op=normalized_operator,
        valid_items=valid_items,
    )


__all__ = [
    "WarpReduceOperation",
    "WarpReduceSpec",
    "make_warp_reduce_spec",
]
