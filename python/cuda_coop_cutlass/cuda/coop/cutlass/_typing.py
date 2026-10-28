# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provide CUTLASS types for public runtime annotations.

The adjacent stubs retain overload and dtype relationships. These aliases
also let documentation tools and runtime introspection resolve signatures
while the CuTe compiler owns runtime validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, TypeVar

from ._planning.thread_group import ThreadGroup

if TYPE_CHECKING:
    import numpy
else:
    try:
        import numpy
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        numpy = None

from ._validation import (
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

ReduceAlgorithm: TypeAlias = Literal[
    "raking_commutative_only",
    "raking",
    "warp_reductions",
]

ScanAlgorithm: TypeAlias = Literal["raking", "raking_memoize", "warp_scans"]

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

SumScanOperator: TypeAlias = Literal["+", "sum", "add", "plus"]

NonSumScanOperator: TypeAlias = Literal[
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

ScanOperator: TypeAlias = SumScanOperator | NonSumScanOperator

ExchangeMode: TypeAlias = Literal[
    "striped_to_blocked",
    "blocked_to_striped",
]

BlockExchangeMode: TypeAlias = (
    ExchangeMode
    | Literal[
        "warp_striped_to_blocked",
        "blocked_to_warp_striped",
        "scatter_to_blocked",
        "scatter_to_striped",
        "scatter_to_striped_guarded",
        "scatter_to_striped_flagged",
    ]
)

CommonShuffleMode: TypeAlias = Literal["down", "up"]

ScalarShuffleMode: TypeAlias = Literal["offset", "rotate"]


class WritableArrayLike(Protocol):
    """Describe a native array's shape and writable indexing.

    This structural annotation does not establish its memory address space.
    The CuTe compiler validates shared-memory provenance and layout.
    """

    @property
    def shape(self) -> object: ...

    def __getitem__(self, index: Any, /) -> Any: ...

    def __setitem__(self, index: Any, value: Any, /) -> None: ...


class CompilerScalarLike(Protocol):
    """Describe a compiler scalar without depending on a concrete scalar
    subclass."""

    @property
    def width(self) -> int:
        """Return this scalar's bit width."""

    @property
    def dtype(self) -> object:
        """Return this value's compiler dtype."""

    def ir_value(self) -> object:
        """Return this scalar's compiler IR value."""


class CompilerIntegerLike(CompilerScalarLike, Protocol):
    """Compiler scalar carrying the signedness metadata of an integer."""

    @property
    def signed(self) -> bool:
        """Return whether this integer type is signed."""


if not TYPE_CHECKING and numpy is None:
    CommonNumericScalar: TypeAlias = int | float | CompilerScalarLike
    IntegerValue: TypeAlias = int | CompilerIntegerLike
    SignedIntegerScalar: TypeAlias = int | CompilerIntegerLike
    IntegralScalar: TypeAlias = SignedIntegerScalar
else:
    CommonNumericScalar: TypeAlias = (
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

    if TYPE_CHECKING:
        IntegerValue: TypeAlias = int | numpy.integer[Any] | CompilerIntegerLike
    else:
        IntegerValue: TypeAlias = int | numpy.integer | CompilerIntegerLike

    SignedIntegerScalar: TypeAlias = (
        int | numpy.signedinteger[Any] | CompilerIntegerLike
    )

    IntegralScalar: TypeAlias = SignedIntegerScalar | numpy.unsignedinteger[Any]

ValidItems: TypeAlias = IntegerValue

_CommonNumericT = TypeVar("_CommonNumericT", bound=CommonNumericScalar)

CommonThreadDataLike = _ReadableThreadDataLike

__all__ = [
    "CommonThreadDataLike",
    "TempStorageLike",
    "ThreadDataLike",
    "WritableArrayLike",
    "_CommonNumericT",
]


MemoryGroup = ThreadGroup
BlockGroup = ThreadGroup
WarpGroup = ThreadGroup
