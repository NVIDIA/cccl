# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared static contracts for compiler-selected ``cuda.coop`` values."""

from typing import Any, Literal, Protocol, TypeAlias

import numpy

ReduceAlgorithm: TypeAlias = Literal[
    "raking_commutative_only",
    "raking",
    "warp_reductions",
]
ReduceOperator: TypeAlias = Literal[
    "+",
    "sum",
    "add",
    "plus",
    "*",
    "mul",
    "multiply",
    "multiplies",
    "min",
    "minimum",
    "max",
    "maximum",
    "&",
    "bit_and",
    "|",
    "bit_or",
    "^",
    "bit_xor",
]

class CompilerScalarLike(Protocol):
    """Structural view of one compiler-owned numeric scalar."""

    width: int

    @property
    def dtype(self) -> object:
        """Return this value's compiler dtype."""

    def ir_value(self) -> object:
        """Return this scalar's compiler IR value."""

class CompilerIntegerLike(CompilerScalarLike, Protocol):
    """Compiler scalar carrying integer signedness metadata."""

    signed: bool

PortableNumericScalar: TypeAlias = (
    int
    | float
    | numpy.int8
    | numpy.uint8
    | numpy.int16
    | numpy.uint16
    | numpy.int32
    | numpy.uint32
    | numpy.int64
    | numpy.uint64
    | numpy.float32
    | numpy.float64
    | CompilerScalarLike
)
ValidItems: TypeAlias = int | numpy.integer[Any] | CompilerIntegerLike

__all__ = [
    "PortableNumericScalar",
    "ReduceAlgorithm",
    "ReduceOperator",
    "ValidItems",
]
