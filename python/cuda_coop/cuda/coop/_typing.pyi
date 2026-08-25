# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared static contracts for compiler-selected ``cuda.coop`` values."""

from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeAlias, TypeVar

import numpy as np

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
LoadStoreAlgorithm: TypeAlias = BlockLoadStoreAlgorithm
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
AdjacentDifferenceDirection: TypeAlias = Literal["left", "right"]
DiscontinuityMode: TypeAlias = Literal["heads", "tails"]
PortableShuffleMode: TypeAlias = Literal["down", "up"]
ShuffleMode: TypeAlias = Literal["down", "up", "offset", "rotate"]
HistogramAlgorithm: TypeAlias = Literal["atomic", "sort"]
TempStorageSharing: TypeAlias = Literal["shared", "exclusive"]
BinaryFunction: TypeAlias = Callable[[ItemT, ItemT], ItemT]

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
    | np.uint8
    | np.int32
    | np.uint32
    | np.int64
    | np.uint64
    | np.float32
    | np.float64
    | CompilerScalarLike
)
ScalarValue: TypeAlias = bool | int | float | complex | np.number | CompilerScalarLike
IntegerValue: TypeAlias = int | np.integer[Any] | CompilerIntegerLike
TraceInteger: TypeAlias = int | np.integer[Any]
PortableIntegerKey: TypeAlias = (
    int | np.int32 | np.uint32 | np.int64 | np.uint64 | CompilerIntegerLike
)
PortableIntegerValue: TypeAlias = PortableIntegerKey | np.uint8
PortableRunValue: TypeAlias = PortableIntegerValue
PortableRunLength: TypeAlias = PortableIntegerKey
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

class PortableThreadDataLike(Protocol):
    """Thread payload whose readable item type is in the portable closure."""

    items_per_thread: int
    dtype: object | None

    def __len__(self) -> int:
        """Return the number of items owned by this thread."""

    def __getitem__(self, index: int, /) -> PortableNumericScalar:
        """Return one portable numeric register value."""

class TempStorageLike(Protocol):
    """Portable explicit scratch-storage descriptor contract."""

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool | None
    sharing: TempStorageSharing

__all__ = [
    "AdjacentDifferenceDirection",
    "BinaryFunction",
    "DiscontinuityMode",
    "ExchangeMode",
    "HistogramAlgorithm",
    "LoadStoreAlgorithm",
    "PortableShuffleMode",
    "ReduceAlgorithm",
    "ReduceOperator",
    "ScanAlgorithm",
    "ScanMode",
    "ScanOperator",
    "ShuffleMode",
    "TempStorageLike",
    "TempStorageSharing",
    "ThreadDataLike",
    "ThreadGroupKind",
    "ThreadLevel",
]
