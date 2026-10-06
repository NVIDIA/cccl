# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provide compiler-neutral types for public runtime annotations.

The adjacent stubs retain overload and dtype relationships. These aliases
also let documentation tools and runtime introspection resolve signatures
without importing an optional GPU compiler.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypeAlias, TypeVar

try:
    import numpy
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    numpy = None

from ._core.api._payload import (
    TempStorageLike,
    ThreadDataLike,
    _ReadableThreadDataLike,
)

BlockLoadStoreAlgorithm: TypeAlias = Literal[
    "direct",
    "striped",
    "vectorize",
    "transpose",
    "warp_transpose",
    "warp_transpose_timesliced",
]

WarpLoadStoreAlgorithm: TypeAlias = Literal[
    "direct",
    "striped",
    "vectorize",
    "transpose",
]


class CompilerScalarLike(Protocol):
    """Describe a compiler scalar without importing its concrete type."""

    width: int

    @property
    def dtype(self) -> object:
        """Return this value's compiler dtype."""

    def ir_value(self) -> object:
        """Return this scalar's compiler IR value."""


class CompilerIntegerLike(CompilerScalarLike, Protocol):
    """Compiler scalar carrying the signedness metadata of an integer."""

    signed: bool


if numpy is None:
    PortableNumericScalar: TypeAlias = int | float | CompilerScalarLike
    IntegerValue: TypeAlias = int | CompilerIntegerLike
else:
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

    IntegerValue: TypeAlias = int | numpy.integer[Any] | CompilerIntegerLike

ValidItems: TypeAlias = IntegerValue

_PortableNumericT = TypeVar("_PortableNumericT", bound=PortableNumericScalar)

PortableThreadDataLike = _ReadableThreadDataLike

__all__ = [
    "PortableThreadDataLike",
    "TempStorageLike",
    "ThreadDataLike",
    "_PortableNumericT",
]
