# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockReduce descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._symbols import semantic_token
from ..thread_group import Dim3, normalize_thread_dim


class BlockReduceOperation(str, Enum):
    """Supported CUB BlockReduce operations."""

    REDUCE = "reduce"
    SUM = "sum"


class BlockReduceAlgorithm(str, Enum):
    """Deterministic CUB BlockReduce algorithm selectors."""

    RAKING_COMMUTATIVE_ONLY = "raking_commutative_only"
    RAKING = "raking"
    WARP_REDUCTIONS = "warp_reductions"


class BlockReduceOperator(str, Enum):
    """Built-in binary operators supported by the portable API."""

    SUM = "sum"
    MULTIPLIES = "multiplies"
    MIN = "min"
    MAX = "max"
    BIT_AND = "bit_and"
    BIT_OR = "bit_or"
    BIT_XOR = "bit_xor"


_OPERATOR_ALIASES = {
    None: BlockReduceOperator.SUM,
    "+": BlockReduceOperator.SUM,
    "sum": BlockReduceOperator.SUM,
    "add": BlockReduceOperator.SUM,
    "plus": BlockReduceOperator.SUM,
    "*": BlockReduceOperator.MULTIPLIES,
    "mul": BlockReduceOperator.MULTIPLIES,
    "multiply": BlockReduceOperator.MULTIPLIES,
    "multiplies": BlockReduceOperator.MULTIPLIES,
    "min": BlockReduceOperator.MIN,
    "minimum": BlockReduceOperator.MIN,
    "max": BlockReduceOperator.MAX,
    "maximum": BlockReduceOperator.MAX,
    "&": BlockReduceOperator.BIT_AND,
    "bit_and": BlockReduceOperator.BIT_AND,
    "|": BlockReduceOperator.BIT_OR,
    "bit_or": BlockReduceOperator.BIT_OR,
    "^": BlockReduceOperator.BIT_XOR,
    "bit_xor": BlockReduceOperator.BIT_XOR,
}


def normalize_block_reduce_operator(value: Any) -> BlockReduceOperator:
    """Normalize one portable built-in reduction selector."""

    if isinstance(value, BlockReduceOperator):
        return value
    token = getattr(value, "value", value)
    if isinstance(token, str):
        token = token.strip().lower().replace("-", "_")
    try:
        return _OPERATOR_ALIASES[token]
    except (KeyError, TypeError):
        choices = ", ".join(operator.value for operator in BlockReduceOperator)
        raise ValueError(
            f"unsupported cuda.coop.reduce binary_op {value!r}; expected one of: "
            f"{choices}"
        ) from None


def normalize_block_reduce_algorithm(value: Any) -> BlockReduceAlgorithm:
    """Normalize one deterministic CUB BlockReduce algorithm selector."""

    if value is None:
        return BlockReduceAlgorithm.WARP_REDUCTIONS
    if isinstance(value, BlockReduceAlgorithm):
        return value
    token = getattr(value, "value", value)
    if isinstance(token, str):
        token = token.strip().lower().replace("-", "_")
    try:
        return BlockReduceAlgorithm(token)
    except (TypeError, ValueError):
        choices = ", ".join(algorithm.value for algorithm in BlockReduceAlgorithm)
        raise ValueError(
            f"unsupported cuda.coop reduction algorithm {value!r}; expected one of: "
            f"{choices}"
        ) from None


@dataclass(frozen=True, eq=False)
class BlockReduceSpec:
    """Fully specialized scalar CUB BlockReduce description."""

    dtype: Any
    block_dim: Dim3
    operation: BlockReduceOperation = BlockReduceOperation.REDUCE
    binary_op: BlockReduceOperator = BlockReduceOperator.SUM
    algorithm: BlockReduceAlgorithm = BlockReduceAlgorithm.WARP_REDUCTIONS
    valid_items: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", BlockReduceOperation(self.operation))
        object.__setattr__(
            self,
            "binary_op",
            normalize_block_reduce_operator(self.binary_op),
        )
        object.__setattr__(
            self,
            "algorithm",
            normalize_block_reduce_algorithm(self.algorithm),
        )
        object.__setattr__(
            self,
            "block_dim",
            normalize_thread_dim(
                self.block_dim,
                scope="cuda.coop reduction",
                label="block",
            ),
        )
        if self.operation is BlockReduceOperation.SUM and (
            self.binary_op is not BlockReduceOperator.SUM
        ):
            raise ValueError("cuda.coop.sum requires the sum operator")
        if not isinstance(self.valid_items, bool):
            raise TypeError("valid_items must describe whether an argument is present")

    @property
    def method_name(self) -> str:
        """Return the CUB method selected by this description."""

        return "Sum" if self.operation is BlockReduceOperation.SUM else "Reduce"

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "block_reduce",
            semantic_token(self.dtype),
            self.block_dim,
            self.operation.value,
            self.binary_op.value,
            self.algorithm.value,
            self.valid_items,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlockReduceSpec):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def make_block_reduce_spec(
    *,
    dtype: Any,
    block_dim: Dim3 | int | tuple[int, ...] | list[int],
    operation: str | BlockReduceOperation = BlockReduceOperation.REDUCE,
    binary_op: Any = None,
    algorithm: Any = None,
    valid_items: bool = False,
) -> BlockReduceSpec:
    """Build one scalar CUB BlockReduce description."""

    operation = BlockReduceOperation(operation)
    normalized_operator = normalize_block_reduce_operator(binary_op)
    if operation is BlockReduceOperation.SUM and (
        normalized_operator is not BlockReduceOperator.SUM
    ):
        raise ValueError("cuda.coop.sum requires the sum operator")
    return BlockReduceSpec(
        dtype=dtype,
        block_dim=normalize_thread_dim(
            block_dim,
            scope="make_block_reduce_spec",
            label="block",
        ),
        operation=operation,
        binary_op=normalized_operator,
        algorithm=normalize_block_reduce_algorithm(algorithm),
        valid_items=valid_items,
    )


__all__ = [
    "BlockReduceAlgorithm",
    "BlockReduceOperation",
    "BlockReduceOperator",
    "BlockReduceSpec",
    "make_block_reduce_spec",
    "normalize_block_reduce_algorithm",
    "normalize_block_reduce_operator",
]
