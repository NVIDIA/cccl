# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe common API types without requiring a compiler package.

Protocols let type checkers recognize the attributes supplied by each backend.
They describe compatible values; they do not make a new Python class usable in
a GPU kernel. The compiler must still know how to lower each concrete value.
"""

from typing import Any, Literal, Protocol, TypeAlias, TypeVar

import numpy

_ItemT = TypeVar("_ItemT")

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
CommonShuffleMode: TypeAlias = Literal["down", "up"]
ScalarShuffleMode: TypeAlias = Literal["offset", "rotate"]
ShuffleMode: TypeAlias = CommonShuffleMode | ScalarShuffleMode
TempStorageSharing: TypeAlias = Literal["shared", "exclusive"]

class CompilerScalarLike(Protocol):
    """Describe a compiler scalar without importing its concrete type."""

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

_CommonNumericT = TypeVar("_CommonNumericT", bound=CommonNumericScalar)

class _ExactScalar(Protocol[_ItemT]):
    """Match a seed's exact scalar type without widening the input type.

    The writable ``__class__`` member makes the type parameter invariant, so
    a seed cannot widen the input type. Invariance also keeps NumPy float64,
    a float subclass, out of the Python ``float`` arm. The separate ``int``
    and ``float`` arms let ``ContextualInitialValue`` accept ordinary Python
    literals. The compiler still checks literal values and runtime dtypes.
    """

    __class__: type[_ItemT]  # type: ignore[assignment]

ContextualInitialValue: TypeAlias = (
    _ExactScalar[_ItemT] | _ExactScalar[int] | _ExactScalar[float]
)
_ReadableItemT_co = TypeVar(
    "_ReadableItemT_co", bound=CommonNumericScalar, covariant=True
)
ScalarValue: TypeAlias = (
    bool | int | float | complex | numpy.number | CompilerScalarLike
)
IntegerValue: TypeAlias = int | numpy.integer[Any] | CompilerIntegerLike
SignedIntegerScalar: TypeAlias = (
    int | numpy.signedinteger[Any] | CompilerIntegerLike
)
IntegralScalar: TypeAlias = SignedIntegerScalar | numpy.unsignedinteger[Any]
ThreadGroupQueryScalar: TypeAlias = (
    numpy.int8
    | numpy.uint8
    | numpy.int16
    | numpy.uint16
    | numpy.int32
    | numpy.uint32
    | numpy.int64
    | numpy.uint64
    | CompilerIntegerLike
)
TraceInteger: TypeAlias = int | numpy.integer[Any]
ValidItems: TypeAlias = IntegerValue

class ThreadDataLike(Protocol[_ItemT]):
    """Common mutable, indexable per-thread payload contract.

    Concrete compiler backends may attach additional helpers and metadata, but
    common operations rely only on this payload shape and item access
    contract. Structural type compatibility does not register arbitrary user
    classes with a compiler; kernels must use payloads that their active
    backend recognizes.
    """

    items_per_thread: int
    dtype: object | None

    def __len__(self) -> int:
        """Return the number of logical items owned by this thread."""

    def __getitem__(self, index: int, /) -> _ItemT:
        """Return one thread-local item."""

    def __setitem__(self, index: int, value: _ItemT, /) -> None:
        """Replace one thread-local item."""

class CommonThreadDataLike(Protocol[_ReadableItemT_co]):
    """Describe readable per-thread items with common numeric types.

    Operations that only read a payload use this protocol. Mutable operations
    use ``ThreadDataLike``, which also requires item assignment.
    """

    items_per_thread: int
    dtype: object | None

    def __len__(self) -> int:
        """Return the number of items owned by this thread."""

    def __getitem__(self, index: int, /) -> _ReadableItemT_co:
        """Return one numeric register value supported by the common API."""

class TempStorageLike(Protocol):
    """Describe the attributes of an explicit scratch descriptor.

    ``size_in_bytes`` and ``alignment`` may be ``None`` so the compiler
    derives them from the calls that use the descriptor. ``auto_sync``
    selects automatic reuse barriers. ``sharing`` selects one shared region
    or separate slices per call site. The descriptor holds no buffer. A
    supported compiler allocates the memory and rejects a size that is too
    small for its uses.
    """

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool
    sharing: TempStorageSharing

    def reserve(
        self, num_elems: int, dtype: object, *, alignment: int | None = None
    ) -> Any:
        """Reserve a compiler shared array with manual synchronization."""

__all__ = [
    "BlockExchangeMode",
    "BlockLoadStoreAlgorithm",
    "CommonShuffleMode",
    "ContextualInitialValue",
    "ExchangeMode",
    "LoadStoreAlgorithm",
    "NonSumScanOperator",
    "ReduceAlgorithm",
    "ReduceOperator",
    "ScalarShuffleMode",
    "ScanAlgorithm",
    "ScanMode",
    "ScanOperator",
    "ShuffleMode",
    "SumScanOperator",
    "SynchronizableGroupKind",
    "TempStorageLike",
    "TempStorageSharing",
    "ThreadDataLike",
    "ThreadGroupKind",
    "ThreadLevel",
    "WarpExchangeMode",
    "WarpLoadStoreAlgorithm",
    "_CommonNumericT",
]
