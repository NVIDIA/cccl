# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared static contracts for compiler-selected ``cuda.coop`` values."""

from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeAlias, TypeVar

from numpy import float32 as _NumpyFloat32
from numpy import float64 as _NumpyFloat64
from numpy import int32 as _NumpyInt32
from numpy import int64 as _NumpyInt64
from numpy import integer as _NumpyInteger
from numpy import number as _NumpyNumber
from numpy import uint8 as _NumpyUint8
from numpy import uint32 as _NumpyUint32
from numpy import uint64 as _NumpyUint64

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
_SynchronizableGroupKind: TypeAlias = Literal[
    "thread",
    "warp",
    "block",
    "cluster",
    "threads_within_warp",
    "warps_within_block",
]
_BlockLoadStoreAlgorithm: TypeAlias = Literal[
    "direct",
    "striped",
    "vectorize",
    "transpose",
    "warp_transpose",
    "warp_transpose_timesliced",
]
_WarpLoadStoreAlgorithm: TypeAlias = Literal[
    "direct",
    "striped",
    "vectorize",
    "transpose",
]
LoadStoreAlgorithm: TypeAlias = _BlockLoadStoreAlgorithm
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
_SumScanOperator: TypeAlias = Literal["+", "sum", "add", "plus"]
_NonSumScanOperator: TypeAlias = Literal[
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
ScanOperator: TypeAlias = _SumScanOperator | _NonSumScanOperator
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

class _CompilerScalarLike(Protocol):
    """Backend-optional structural view of one compiler numeric scalar."""

    width: int

    @property
    def dtype(self) -> object:
        """Return this value's compiler dtype."""
    def ir_value(self) -> object:
        """Return this scalar's compiler IR value."""

class _CompilerIntegerLike(_CompilerScalarLike, Protocol):
    """Compiler scalar carrying the signedness metadata of an integer."""

    signed: bool

_PortableNumericScalar: TypeAlias = (
    int
    | float
    | _NumpyUint8
    | _NumpyInt32
    | _NumpyUint32
    | _NumpyInt64
    | _NumpyUint64
    | _NumpyFloat32
    | _NumpyFloat64
    | _CompilerScalarLike
)
_ScalarValue: TypeAlias = (
    bool | int | float | complex | _NumpyNumber | _CompilerScalarLike
)
_IntegerValue: TypeAlias = int | _NumpyInteger[Any] | _CompilerIntegerLike
_TraceInteger: TypeAlias = int | _NumpyInteger[Any]
_PortableIntegerKey: TypeAlias = (
    int | _NumpyInt32 | _NumpyUint32 | _NumpyInt64 | _NumpyUint64 | _CompilerIntegerLike
)
_PortableIntegerValue: TypeAlias = _PortableIntegerKey | _NumpyUint8
_PortableRunValue: TypeAlias = _PortableIntegerValue
_PortableRunLength: TypeAlias = _PortableIntegerKey
_ValidItems: TypeAlias = _IntegerValue

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

class _PortableThreadDataLike(Protocol):
    """Thread payload whose readable item type is in the portable closure."""

    items_per_thread: int
    dtype: object | None

    def __len__(self) -> int:
        """Return the number of items owned by this thread."""

    def __getitem__(self, index: int, /) -> _PortableNumericScalar:
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
