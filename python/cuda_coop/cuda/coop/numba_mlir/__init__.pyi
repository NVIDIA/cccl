# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR-qualified group-first cooperative primitives."""

from enum import IntEnum
from typing import Any, Generic, Literal, Protocol, TypeAlias, overload

from numpy import bool_ as _NumpyBool
from numpy import float16 as _NumpyFloat16
from numpy import float32 as _NumpyFloat32
from numpy import float64 as _NumpyFloat64
from numpy import generic as _NumpyScalar
from numpy import int8 as _NumpyInt8
from numpy import int16 as _NumpyInt16
from numpy import int32 as _NumpyInt32
from numpy import int64 as _NumpyInt64
from numpy import uint8 as _NumpyUint8
from numpy import uint16 as _NumpyUint16
from numpy import uint32 as _NumpyUint32
from numpy import uint64 as _NumpyUint64
from typing_extensions import TypeVar

from .. import ThreadGroup as _CommonThreadGroup
from .. import ThreadHierarchy as ThreadHierarchy
from .._typing import ReduceAlgorithm as _ReduceAlgorithm
from .._typing import ReduceOperator as _ReduceOperator
from .._typing import TempStorageSharing as _TempStorageSharing
from .._typing import ThreadDataLike as _ThreadDataLike
from .._typing import ThreadGroupKind as _ThreadGroupKind
from .._typing import ThreadLevel as _ThreadLevel
from .._typing import _BlockLoadStoreAlgorithm as _BlockLoadStoreAlgorithm
from .._typing import _CompilerScalarLike as _CompilerScalarLike
from .._typing import _IntegerValue as _IntegerValue
from .._typing import _NonSumScanOperator as _NonSumScanOperator
from .._typing import _PortableIntegerKey as _PortableIntegerKey
from .._typing import _PortableRunLength as _PortableRunLength
from .._typing import _PortableRunValue as _PortableRunValue
from .._typing import _ScalarValue as _ScalarValue
from .._typing import _SumScanOperator as _SumScanOperator
from .._typing import _SynchronizableGroupKind as _SynchronizableGroupKind
from .._typing import _TraceInteger as _TraceInteger
from .._typing import _ValidItems as _ValidItems
from .._typing import _WarpLoadStoreAlgorithm as _WarpLoadStoreAlgorithm
from . import _block as _block
from . import _warp as _warp

_ItemT = TypeVar("_ItemT")
_OpT = TypeVar("_OpT")
_PayloadT = TypeVar("_PayloadT", bound=_ThreadDataLike[Any])
_BlockExchangeMode: TypeAlias = Literal[
    "striped_to_blocked",
    "blocked_to_striped",
    "warp_striped_to_blocked",
    "blocked_to_warp_striped",
    "scatter_to_blocked",
    "scatter_to_striped",
    "scatter_to_striped_guarded",
    "scatter_to_striped_flagged",
]
_WarpExchangeMode: TypeAlias = Literal[
    "striped_to_blocked",
    "blocked_to_striped",
    "scatter_to_striped",
]
_ScalarT = TypeVar(
    "_ScalarT",
    bound=bool | int | float | complex | _NumpyScalar,
)
_ValueT = TypeVar("_ValueT")
_CounterT = TypeVar(
    "_CounterT",
    int,
    _NumpyInt32,
    _NumpyUint32,
    _NumpyInt64,
    _NumpyUint64,
)
_DataclassT = TypeVar("_DataclassT")
_GroupKindT_co = TypeVar(
    "_GroupKindT_co",
    bound=_ThreadGroupKind,
    covariant=True,
    default=_ThreadGroupKind,
)
_ArrayShape = int | tuple[int, ...]
_ScalarValueT = TypeVar("_ScalarValueT", bound=_ScalarValue)
_NumbaOrderedItem: TypeAlias = (
    _PortableIntegerKey
    | bool
    | float
    | _NumpyBool
    | _NumpyInt8
    | _NumpyUint8
    | _NumpyInt16
    | _NumpyUint16
    | _NumpyFloat16
    | _NumpyFloat32
    | _NumpyFloat64
    | _CompilerScalarLike
)
_NumbaMergeSortKeyT = TypeVar(
    "_NumbaMergeSortKeyT",
    bound=_NumbaOrderedItem,
)
_NumbaTopKKeyT = TypeVar("_NumbaTopKKeyT", bound=_NumbaOrderedItem)
_NumbaTopKValueT = TypeVar("_NumbaTopKValueT", bound=_NumbaOrderedItem)
_NumbaPairValue: TypeAlias = (
    bool
    | int
    | float
    | _NumpyBool
    | _NumpyInt8
    | _NumpyUint8
    | _NumpyInt16
    | _NumpyUint16
    | _NumpyInt32
    | _NumpyUint32
    | _NumpyInt64
    | _NumpyUint64
    | _NumpyFloat16
    | _NumpyFloat32
    | _NumpyFloat64
)
_NumbaPairValueT = TypeVar("_NumbaPairValueT", bound=_NumbaPairValue)
_IntegerKeyT = TypeVar("_IntegerKeyT", bound=_PortableIntegerKey)
_RunValueT = TypeVar("_RunValueT", bound=_PortableRunValue)
_RunLengthT = TypeVar("_RunLengthT", bound=_PortableRunLength)
_NumbaRunValue: TypeAlias = (
    bool
    | int
    | float
    | _NumpyBool
    | _NumpyInt8
    | _NumpyUint8
    | _NumpyInt16
    | _NumpyUint16
    | _NumpyInt32
    | _NumpyUint32
    | _NumpyInt64
    | _NumpyUint64
    | _NumpyFloat16
    | _NumpyFloat32
    | _NumpyFloat64
    | _CompilerScalarLike
)
_NumbaRunLength: TypeAlias = (
    _PortableRunLength | _NumpyInt8 | _NumpyUint8 | _NumpyInt16 | _NumpyUint16
)
_NumbaRunValueT = TypeVar("_NumbaRunValueT", bound=_NumbaRunValue)
_NumbaRunLengthT = TypeVar("_NumbaRunLengthT", bound=_NumbaRunLength)

Hierarchy = ThreadHierarchy

class NoAlgorithm(IntEnum):
    """Placeholder selector for primitives without an algorithm knob."""

    NO_ALGORITHM = 1

class BlockLoadAlgorithm(IntEnum):
    """CUB block-load algorithm choices."""

    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3
    WARP_TRANSPOSE = 4
    WARP_TRANSPOSE_TIMESLICED = 5

class BlockStoreAlgorithm(IntEnum):
    """CUB block-store algorithm choices."""

    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3
    WARP_TRANSPOSE = 4
    WARP_TRANSPOSE_TIMESLICED = 5

class WarpLoadAlgorithm(IntEnum):
    """CUB warp-load algorithm choices."""

    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3

class WarpStoreAlgorithm(IntEnum):
    """CUB warp-store algorithm choices."""

    DIRECT = 0
    STRIPED = 1
    VECTORIZE = 2
    TRANSPOSE = 3

class StatefulFunction(Generic[_OpT]):
    """Device callable paired with explicit state for generated C++ wrappers."""

    op: _OpT
    dtype: object
    name: str | None

    def __init__(
        self,
        op: _OpT,
        dtype: object,
        name: str | None = None,
    ) -> None:
        """Pair ``op`` with its state ``dtype`` and optional generated name."""

class ThreadGroup(
    _CommonThreadGroup[_GroupKindT_co],
    Generic[_GroupKindT_co],
):
    """Compile-time CUDA group descriptor for Numba-CUDA-MLIR."""

    def rank(self, level: _ThreadLevel = "thread") -> _NumpyInt32:
        """Return this group's rank as a NumPy-compatible ``int32`` scalar."""

    def count(self, level: _ThreadLevel = "thread") -> _NumpyInt32:
        """Return this group's count as a NumPy-compatible ``int32`` scalar."""

    @overload
    def rank_as(self, dtype: type[_ItemT], level: _ThreadLevel = "thread") -> _ItemT:
        """Return the group rank converted to an ordinary scalar dtype."""
    @overload
    def rank_as(self, dtype: object = None, level: _ThreadLevel = "thread") -> Any:
        """Return the group rank converted to a Numba compiler dtype token."""

    @overload
    def count_as(self, dtype: type[_ItemT], level: _ThreadLevel = "thread") -> _ItemT:
        """Return the group count converted to an ordinary scalar dtype."""
    @overload
    def count_as(self, dtype: object = None, level: _ThreadLevel = "thread") -> Any:
        """Return the group count converted to a Numba compiler dtype token."""

    def sync(self: ThreadGroup[_SynchronizableGroupKind]) -> None:
        """Synchronize participating members; grid groups are unsupported."""

    def sync_aligned(self: ThreadGroup[_SynchronizableGroupKind]) -> None:
        """Synchronize a converged non-grid group."""

    @overload
    def group_by(
        self: ThreadGroup[Literal["warp"]],
        count: int,
        *,
        exhaustive: bool = True,
    ) -> ThreadGroup[Literal["threads_within_warp"]]:
        """Partition a physical warp into groups of threads."""
    @overload
    def group_by(
        self: ThreadGroup[Literal["block"]],
        count: int,
        *,
        exhaustive: bool = True,
    ) -> ThreadGroup[Literal["warps_within_block"]]:
        """Partition a block into groups of physical warps."""
    def is_member(self) -> _NumpyUint8:
        """Return a NumPy-compatible ``uint8`` membership flag."""

_MemoryGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp", "block"]]
_ReductionGroup: TypeAlias = ThreadGroup[
    Literal["thread", "warp", "threads_within_warp", "block", "cluster"]
]
_BlockGroup: TypeAlias = ThreadGroup[Literal["block"]]
_WarpGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp"]]

class TempStorage:
    """Explicit opaque byte scratch for planned shared-memory collectives.

    This storage is independent of the cooperative payload dtype. The current
    restriction on LLVM-backed extension dtypes in ``shared.array`` therefore
    does not apply to ``TempStorage``.
    """

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool | None
    sharing: _TempStorageSharing

    def __init__(
        self,
        size_in_bytes: int | None = None,
        alignment: int | None = None,
        auto_sync: bool | None = None,
        sharing: _TempStorageSharing = "shared",
    ) -> None:
        """Configure scratch size, alignment, synchronization, and sharing."""

class _LocalMemory(Protocol):
    """Numba-CUDA-MLIR thread-local memory namespace."""

    def array(
        self,
        shape: _ArrayShape,
        dtype: object,
        alignas: int | None = 8,
        *,
        alignment: int | None = None,
    ) -> Any:
        """Allocate thread-local storage, including LLVM-backed extension dtypes.

        CUB-backed collectives additionally require an exact, inspectable ABI
        layout for an extension dtype, with matching CUDA and MLIR models.
        """

class _SharedMemory(Protocol):
    """Numba-CUDA-MLIR shared-memory namespace."""

    def array(
        self,
        shape: _ArrayShape,
        dtype: object,
        alignas: int | None = 8,
        *,
        alignment: int | None = None,
    ) -> Any:
        """Allocate shared storage for compiler-native element dtypes.

        LLVM-backed extension aggregate dtypes are not yet supported here and
        must fail compilation pending separately qualified shared-address-space
        lowering.
        """

local: _LocalMemory
shared: _SharedMemory

@overload
def ThreadData(
    items_per_thread: int,
    dtype: type[_ItemT],
    *,
    alignas: int = 8,
) -> _ThreadDataLike[_ItemT]:
    """Construct typed thread-local storage, including extension dtypes.

    LLVM-backed extension dtypes are supported in this per-thread storage.
    CUB-backed collectives additionally require an exact, inspectable ABI
    layout with matching CUDA and MLIR models; opaque or inconsistent layouts
    fail specialization. Extension dtypes are not supported as
    ``shared.array`` element types.
    """

@overload
def ThreadData(
    items_per_thread: int,
    dtype: object = None,
    *,
    alignas: int = 8,
) -> _ThreadDataLike[Any]:
    """Construct thread-local storage using a compiler dtype token.

    LLVM-backed extension dtype tokens are supported in this per-thread
    storage. CUB-backed collectives additionally require an exact, inspectable
    ABI layout with matching CUDA and MLIR models; opaque or inconsistent
    layouts fail specialization. Extension dtypes are not supported as
    ``shared.array`` element types.
    """

def gpu_dataclass(
    dc: _DataclassT,
    *,
    compute_temp_storage: bool = True,
) -> _DataclassT:
    """Register a dataclass instance for Numba-CUDA-MLIR device use.

    Primitive descriptors specialize compilation. All scalar fields remain
    by-value runtime data.
    """

def this_thread() -> ThreadGroup[Literal["thread"]]:
    """Describe the current thread."""

def this_warp() -> ThreadGroup[Literal["warp"]]:
    """Describe the current complete physical warp."""

def this_block() -> ThreadGroup[Literal["block"]]:
    """Describe the current CUDA thread block."""

def this_cluster() -> ThreadGroup[Literal["cluster"]]:
    """Describe the current cluster where the launch can represent it."""

def this_grid() -> ThreadGroup[Literal["grid"]]:
    """Describe the current grid; grid collectives remain unavailable."""

@overload
def load(
    group: _BlockGroup,
    source: object,
    output: _PayloadT,
    /,
    *,
    algorithm: _BlockLoadStoreAlgorithm | BlockLoadAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    oob_default: None = None,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> _PayloadT:
    """Populate and return ``output`` with a block tile."""

@overload
def load(
    group: _BlockGroup,
    source: object,
    output: _ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: _BlockLoadStoreAlgorithm | BlockLoadAlgorithm = "direct",
    valid_items: _ValidItems,
    oob_default: _ItemT,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Populate a partial block tile and fill invalid items."""

@overload
def load(
    group: _WarpGroup,
    source: object,
    output: _PayloadT,
    /,
    *,
    algorithm: _WarpLoadStoreAlgorithm | WarpLoadAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    oob_default: None = None,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> _PayloadT:
    """Populate and return ``output`` with a physical- or logical-warp tile."""

@overload
def load(
    group: _WarpGroup,
    source: object,
    output: _ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: _WarpLoadStoreAlgorithm | WarpLoadAlgorithm = "direct",
    valid_items: _ValidItems,
    oob_default: _ItemT,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Populate a partial physical- or logical-warp tile and fill invalid items."""

@overload
def store(
    group: _BlockGroup,
    destination: object,
    value: object,
    /,
    *,
    algorithm: _BlockLoadStoreAlgorithm | BlockStoreAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Store a scalar or thread-local payload cooperatively across a block."""

@overload
def store(
    group: _WarpGroup,
    destination: object,
    value: object,
    /,
    *,
    algorithm: _WarpLoadStoreAlgorithm | WarpStoreAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Store a scalar or thread-local payload across a physical or logical warp."""

@overload
def reduce(
    group: _ReductionGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ItemT:
    """Reduce a full-group payload and preserve its element type."""

@overload
def reduce(
    group: _ReductionGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ScalarValueT:
    """Reduce full-group scalar values while preserving their static type."""

@overload
def reduce(
    group: _BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: _ReduceAlgorithm | None = None,
) -> _ScalarValueT:
    """Reduce a scalar through direct CUB BlockReduce at the block root.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

@overload
def reduce(
    group: _BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: _ReduceAlgorithm,
) -> _ScalarValueT:
    """Reduce a scalar with an explicit direct CUB BlockReduce algorithm."""

@overload
def reduce(
    group: _WarpGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: None = None,
) -> _ScalarValueT:
    """Reduce a valid scalar prefix through direct CUB WarpReduce.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

@overload
def sum(
    group: _ReductionGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ItemT:
    """Sum a full-group payload and preserve its element type."""

@overload
def sum(
    group: _ReductionGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ScalarValueT:
    """Sum full-group scalar values while preserving their static type."""

@overload
def sum(
    group: _BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: _ReduceAlgorithm | None = None,
) -> _ScalarValueT:
    """Sum a scalar through direct CUB BlockReduce at the block root.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

@overload
def sum(
    group: _BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: _ReduceAlgorithm,
) -> _ScalarValueT:
    """Sum a scalar with an explicit direct CUB BlockReduce algorithm."""

@overload
def sum(
    group: _WarpGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: None = None,
) -> _ScalarValueT:
    """Sum a valid scalar prefix through direct CUB WarpReduce.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

__all__ = [
    "BlockLoadAlgorithm",
    "BlockStoreAlgorithm",
    "Hierarchy",
    "NoAlgorithm",
    "StatefulFunction",
    "TempStorage",
    "ThreadData",
    "ThreadGroup",
    "ThreadHierarchy",
    "WarpLoadAlgorithm",
    "WarpStoreAlgorithm",
    "gpu_dataclass",
    "load",
    "local",
    "reduce",
    "shared",
    "store",
    "sum",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]
