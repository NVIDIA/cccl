# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared static contracts for compiler-selected ``cuda.coop`` values."""

from typing import Any, Literal, Protocol, TypeAlias, TypeVar

import numpy

ItemT = TypeVar("ItemT")

ThreadLevel: TypeAlias = Literal[
    "thread",
    "gpu_thread",
    "warp",
    "block",
    "cluster",
    "grid",
]
ThreadGroupKind: TypeAlias = Literal[
    "thread",
    "warp",
    "block",
    "cluster",
    "grid",
    "threads_within_warp",
    "warps_within_block",
]
SynchronizableGroupKind: TypeAlias = Literal[
    "thread",
    "warp",
    "block",
    "cluster",
    "threads_within_warp",
    "warps_within_block",
]
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
LoadStoreAlgorithm: TypeAlias = BlockLoadStoreAlgorithm | WarpLoadStoreAlgorithm
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
ScanMode: TypeAlias = Literal["exclusive", "inclusive"]
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
WarpExchangeMode: TypeAlias = ExchangeMode
PortableShuffleMode: TypeAlias = Literal["down", "up"]
ScalarShuffleMode: TypeAlias = Literal["offset", "rotate"]
ShuffleMode: TypeAlias = PortableShuffleMode | ScalarShuffleMode
TempStorageSharing: TypeAlias = Literal["shared", "exclusive"]

class CompilerScalarLike(Protocol):
    """Backend-optional structural view of one compiler numeric scalar."""

    width: int

    @property
    def dtype(self) -> object:
        """Return this value's compiler dtype."""
    def ir_value(self) -> object:
        """Return this scalar's compiler IR value."""

class CompilerIntegerLike(CompilerScalarLike, Protocol):
    """Compiler scalar carrying the signedness metadata of an integer."""

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
_ReadableItemT = TypeVar("_ReadableItemT", bound=PortableNumericScalar, covariant=True)
ScalarValue: TypeAlias = (
    bool | int | float | complex | numpy.number | CompilerScalarLike
)
IntegerValue: TypeAlias = int | numpy.integer[Any] | CompilerIntegerLike
SignedIntegerScalar: TypeAlias = int | numpy.signedinteger[Any] | CompilerIntegerLike
IntegralScalar: TypeAlias = int | numpy.integer[Any] | CompilerIntegerLike
TraceInteger: TypeAlias = int | numpy.integer[Any]
ValidItems: TypeAlias = IntegerValue

class ThreadDataLike(Protocol[ItemT]):
    """Portable mutable, indexable per-thread payload contract.

    Concrete compiler backends may attach additional helpers and metadata, but
    common operations rely only on this payload shape and item access contract.
    Structural type compatibility does not register arbitrary user classes with
    a compiler; kernels must use payloads that their active backend recognizes.
    """

    items_per_thread: int
    dtype: object | None

    def __len__(self) -> int:
        """Return the number of logical items owned by this thread."""

    def __getitem__(self, index: int, /) -> ItemT:
        """Return one thread-local item."""

    def __setitem__(self, index: int, value: ItemT, /) -> None:
        """Replace one thread-local item."""

class PortableThreadDataLike(Protocol[_ReadableItemT]):
    """Thread payload whose readable item type is in the portable closure."""

    items_per_thread: int
    dtype: object | None

    def __len__(self) -> int:
        """Return the number of items owned by this thread."""

    def __getitem__(self, index: int, /) -> _ReadableItemT:
        """Return one portable numeric register value."""

class TempStorageLike(Protocol):
    """Portable explicit scratch-storage descriptor contract."""

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool
    sharing: TempStorageSharing

__all__ = [
    "BlockExchangeMode",
    "BlockLoadStoreAlgorithm",
    "ExchangeMode",
    "LoadStoreAlgorithm",
    "PortableShuffleMode",
    "ReduceAlgorithm",
    "ReduceOperator",
    "ScanAlgorithm",
    "ScanMode",
    "ScanOperator",
    "SumScanOperator",
    "NonSumScanOperator",
    "ScalarShuffleMode",
    "ShuffleMode",
    "TempStorageLike",
    "TempStorageSharing",
    "ThreadDataLike",
    "ThreadGroupKind",
    "ThreadLevel",
    "WarpLoadStoreAlgorithm",
    "WarpExchangeMode",
]
