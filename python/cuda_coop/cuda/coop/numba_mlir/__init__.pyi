# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR-qualified group-first cooperative primitives."""

from collections.abc import Callable
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
from .._typing import HistogramAlgorithm as _HistogramAlgorithm
from .._typing import PortableShuffleMode as _PortableShuffleMode
from .._typing import ReduceAlgorithm as _ReduceAlgorithm
from .._typing import ReduceOperator as _ReduceOperator
from .._typing import ScanAlgorithm as _ScanAlgorithm
from .._typing import ScanOperator as _ScanOperator
from .._typing import ShuffleMode as _ShuffleMode
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

class BlockScanAlgorithm(IntEnum):
    """CUB block-scan algorithm choices."""

    RAKING = 0
    RAKING_MEMOIZE = 1
    WARP_SCANS = 2

class BlockHistogramAlgorithm(IntEnum):
    """CUB block-histogram algorithm choices."""

    SORT = 0
    ATOMIC = 1

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

@overload
def scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SumScanOperator | None = None,
    initial_value: _ItemT | None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _ThreadDataLike[_ItemT]:
    """Return block-exclusive sums without mutating the input payload."""

@overload
def scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumScanOperator | Callable[[_ItemT, _ItemT], _ItemT],
    initial_value: _ItemT,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _ThreadDataLike[_ItemT]:
    """Return non-sum block prefixes from a required initial value."""

@overload
def scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    initial_value: None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _ThreadDataLike[_ItemT]:
    """Return block-inclusive prefixes without mutating the input payload."""

@overload
def scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SumScanOperator | None = None,
    initial_value: _ScalarT | None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Block-exclusive sum a Python or NumPy scalar from an optional initial value.

    External Numba compiler scalar values typed as ``Any`` necessarily return
    ``Any`` in the static contract.
    """

@overload
def scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT],
    initial_value: _ScalarT,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Block-exclusive scan a scalar with a required non-sum initial value."""

@overload
def scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    initial_value: None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Block-inclusive scan a Python or NumPy scalar."""

@overload
def scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SumScanOperator | None = None,
    initial_value: _ScalarT | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Warp-group-exclusive sum a scalar from an optional initial value.

    Physical and logical warp scans have no algorithm selector or caller-owned scratch.
    """

@overload
def scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT],
    initial_value: _ScalarT,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Warp-group-exclusive scan with a required non-sum initial value."""

@overload
def scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    initial_value: None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Warp-group-inclusive scan a scalar without an initial value."""

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: _PayloadT,
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _PayloadT:
    """Return block-exclusive sums with the input payload shape."""

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Preserve a scalar type through block-exclusive sum."""

@overload
def exclusive_sum(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp exclusive sum."""

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: _PayloadT,
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _PayloadT:
    """Return block-inclusive sums with the input payload shape."""

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Preserve a scalar type through block-inclusive sum."""

@overload
def inclusive_sum(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp inclusive sum."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: _SumScanOperator | None = None,
    initial_value: _ItemT | None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _ThreadDataLike[_ItemT]:
    """Return block-exclusive sums with the input payload shape."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: _NonSumScanOperator | Callable[[_ItemT, _ItemT], _ItemT],
    initial_value: _ItemT,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _ThreadDataLike[_ItemT]:
    """Return non-sum block prefixes from a required initial value."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _SumScanOperator | None = None,
    initial_value: _ScalarT | None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Block-exclusive sum a scalar from an optional initial value."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _NonSumScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT],
    initial_value: _ScalarT,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Block-exclusive scan a scalar with a required non-sum initial value."""

@overload
def exclusive_scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _SumScanOperator | None = None,
    initial_value: _ScalarT | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Warp-group-exclusive sum a scalar from an optional initial value."""

@overload
def exclusive_scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _NonSumScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT],
    initial_value: _ScalarT,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Warp-group-exclusive scan with a required non-sum initial value."""

@overload
def inclusive_scan(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    scan_op: _ScanOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _ThreadDataLike[_ItemT]:
    """Return block-inclusive prefixes with the input payload shape."""

@overload
def inclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Preserve a scalar type through block-inclusive Scan."""

@overload
def inclusive_scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _ScanOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: Any = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp inclusive Scan."""

@overload
def exchange(
    group: _BlockGroup,
    value: _PayloadT,
    /,
    *,
    mode: _BlockExchangeMode = "striped_to_blocked",
    ranks: Any = None,
    valid_flags: Any = None,
    warp_time_slicing: bool = False,
) -> _PayloadT:
    """Return a layout-rearranged ``ThreadData`` payload without mutation.

    The overload set accepts complete blocks, physical warps, and logical warps.
    The portable modes are ``"striped_to_blocked"`` and
    ``"blocked_to_striped"``. Blocks additionally support warp-striped and
    scatter modes; warp groups support scatter-to-striped. Scatter modes consume
    ``ranks``, flagged block scatter also consumes ``valid_flags``, and
    ``warp_time_slicing`` is block-only.
    Cross-backend portability is guaranteed for one through five items per
    participant; this qualified adapter also accepts larger fixed-size payloads
    supported by its backend; scalar inputs are not supported. The result
    preserves the input payload's shape and item type.
    """

@overload
def exchange(
    group: _WarpGroup,
    value: _PayloadT,
    /,
    *,
    mode: _WarpExchangeMode = "striped_to_blocked",
    ranks: Any = None,
    valid_flags: None = None,
    warp_time_slicing: Literal[False] = False,
) -> _PayloadT:
    """Exchange a payload across a physical or logical warp without mutation.

    Warp groups support blocked-to-striped, striped-to-blocked, and
    scatter-to-striped layouts. Scatter-to-striped consumes ``ranks``. Warp
    exchange does not accept ``valid_flags`` or ``warp_time_slicing``.
    """

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    direction: Literal["left"] = "left",
    valid_items: _ValidItems | None = None,
    tile_predecessor_item: _ItemT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    difference_op: Callable[[_ItemT, _ItemT], _ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return left Numba differences for ``value`` across ``group``.

    ``direction`` is left, ``valid_items`` may limit the tile,
    ``tile_predecessor_item`` supplies its boundary,
    ``tile_successor_item`` stays ``None``, ``temp_storage`` supplies scratch,
    and ``difference_op`` supplies optional subtraction.
    """

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    direction: Literal["right"],
    valid_items: _ValidItems | None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    difference_op: Callable[[_ItemT, _ItemT], _ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return right Numba differences for ``value`` across ``group``.

    ``direction`` is right, ``valid_items`` may limit the tile, both
    ``tile_predecessor_item`` and ``tile_successor_item`` stay ``None``,
    ``temp_storage`` supplies scratch, and ``difference_op`` supplies optional
    subtraction.
    """

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    direction: Literal["right"],
    valid_items: None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: _ItemT,
    temp_storage: TempStorage | None = None,
    difference_op: Callable[[_ItemT, _ItemT], _ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return full-tile right Numba differences for ``value`` across ``group``.

    ``direction`` is right, ``valid_items`` and ``tile_predecessor_item`` stay
    ``None``, ``tile_successor_item`` supplies the boundary, ``temp_storage``
    supplies scratch, and ``difference_op`` supplies optional subtraction.
    """

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    direction: Literal["left"] = "left",
    valid_items: _ValidItems | None = None,
    tile_predecessor_item: _ScalarT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    difference_op: Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return one left Numba difference for scalar ``value`` across ``group``.

    ``direction`` is left, ``valid_items`` may limit the tile,
    ``tile_predecessor_item`` supplies its boundary,
    ``tile_successor_item`` stays ``None``, ``temp_storage`` supplies scratch,
    and ``difference_op`` supplies optional subtraction.
    """

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    direction: Literal["right"],
    valid_items: _ValidItems | None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    difference_op: Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return one right Numba difference for scalar ``value`` across ``group``.

    ``direction`` is right, ``valid_items`` may limit the tile, both
    ``tile_predecessor_item`` and ``tile_successor_item`` stay ``None``,
    ``temp_storage`` supplies scratch, and ``difference_op`` supplies optional
    subtraction.
    """

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    direction: Literal["right"],
    valid_items: None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: _ScalarT,
    temp_storage: TempStorage | None = None,
    difference_op: Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
) -> _ScalarT:
    """Return a full-tile right Numba difference for ``value`` across ``group``.

    ``direction`` is right, ``valid_items`` and ``tile_predecessor_item`` stay
    ``None``, ``tile_successor_item`` supplies the boundary, ``temp_storage``
    supplies scratch, and ``difference_op`` supplies optional subtraction.
    """

@overload
def discontinuity(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["heads"] = "heads",
    tile_predecessor_item: _ItemT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    flag_op: Callable[[_ItemT, _ItemT], object] | None = None,
) -> _ThreadDataLike[int]:
    """Return one Numba signed 32-bit head-flag payload.

    ``group`` must be a complete physical block and ``value`` must be a
    fixed-size per-thread payload. ``mode`` is ``"heads"`` and the result
    preserves the payload shape without mutating ``value``.
    ``tile_predecessor_item`` supplies a same-typed head boundary;
    ``tile_successor_item`` stays ``None``. ``temp_storage`` supplies optional
    caller-owned scratch. Without a boundary, the first head is set.
    ``flag_op`` may be a value-level Numba-CUDA-MLIR device predicate over two
    payload elements.
    """

@overload
def discontinuity(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["tails"],
    tile_predecessor_item: None = None,
    tile_successor_item: _ItemT | None = None,
    temp_storage: TempStorage | None = None,
    flag_op: Callable[[_ItemT, _ItemT], object] | None = None,
) -> _ThreadDataLike[int]:
    """Return one Numba signed 32-bit tail-flag payload.

    ``group`` must be a complete physical block and ``value`` must be a
    fixed-size per-thread payload. ``mode`` is ``"tails"`` and the result
    preserves the payload shape without mutating ``value``.
    ``tile_predecessor_item`` stays ``None``; ``tile_successor_item`` supplies
    a same-typed tail boundary. ``temp_storage`` supplies optional caller-owned
    scratch. Without a boundary, the final tail is set. ``flag_op`` may be a
    value-level Numba-CUDA-MLIR device predicate over two payload elements.
    """

@overload
def discontinuity(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    mode: Literal["heads_and_tails"],
    tile_predecessor_item: _ItemT | None = None,
    tile_successor_item: _ItemT | None = None,
    temp_storage: TempStorage | None = None,
    flag_op: Callable[[_ItemT, _ItemT], object] | None = None,
) -> tuple[_ThreadDataLike[int], _ThreadDataLike[int]]:
    """Return Numba signed 32-bit head and tail flag payloads.

    ``group`` must be a complete physical block, ``value`` must be a fixed-size
    per-thread payload, and ``mode`` must be ``"heads_and_tails"``. The two
    results preserve the payload shape without mutating ``value``.
    ``tile_predecessor_item`` and ``tile_successor_item`` supply same-typed head
    and tail boundaries. ``temp_storage`` supplies optional caller-owned
    scratch. Without external boundaries, the first head and final tail are
    set. ``flag_op`` may be a value-level Numba-CUDA-MLIR device predicate over
    two payload elements.
    """

@overload
def discontinuity(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["heads"] = "heads",
    tile_predecessor_item: _ScalarT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorage | None = None,
    flag_op: Callable[[_ScalarT, _ScalarT], object] | None = None,
) -> int:
    """Return one Numba signed 32-bit scalar head flag.

    ``group`` must be a complete physical block and ``value`` supplies one
    scalar per thread. ``mode`` is ``"heads"``. ``tile_predecessor_item``
    supplies a same-typed head boundary and ``tile_successor_item`` stays
    ``None``. ``temp_storage`` supplies optional caller-owned scratch.
    ``flag_op`` may be a value-level Numba-CUDA-MLIR device predicate over two
    values of the scalar type.
    """

@overload
def discontinuity(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["tails"],
    tile_predecessor_item: None = None,
    tile_successor_item: _ScalarT | None = None,
    temp_storage: TempStorage | None = None,
    flag_op: Callable[[_ScalarT, _ScalarT], object] | None = None,
) -> int:
    """Return one Numba signed 32-bit scalar tail flag.

    ``group`` must be a complete physical block and ``value`` supplies one
    scalar per thread. ``mode`` is ``"tails"``. ``tile_predecessor_item`` stays
    ``None`` and ``tile_successor_item`` supplies a same-typed tail boundary.
    ``temp_storage`` supplies optional caller-owned scratch. ``flag_op`` may be
    a value-level Numba-CUDA-MLIR device predicate over two values of the scalar
    type.
    """

@overload
def discontinuity(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["heads_and_tails"],
    tile_predecessor_item: _ScalarT | None = None,
    tile_successor_item: _ScalarT | None = None,
    temp_storage: TempStorage | None = None,
    flag_op: Callable[[_ScalarT, _ScalarT], object] | None = None,
) -> tuple[int, int]:
    """Return Numba signed 32-bit scalar head and tail flags.

    ``group`` must be a complete physical block, ``value`` supplies one scalar
    per thread, and ``mode`` must be ``"heads_and_tails"``.
    ``tile_predecessor_item`` and ``tile_successor_item`` supply same-typed head
    and tail boundaries, while ``temp_storage`` supplies optional caller-owned
    scratch. ``flag_op`` may be a value-level Numba-CUDA-MLIR device predicate
    over two values of the scalar type.
    """

@overload
def shuffle(
    group: _BlockGroup,
    value: _PayloadT,
    /,
    *,
    mode: _PortableShuffleMode = "down",
    distance: Literal[1] = 1,
    block_prefix: None = None,
    block_suffix: None = None,
) -> _PayloadT:
    """Return a unit-shifted Numba payload without mutation.

    ``group`` must be a complete physical block and ``value`` a fixed-size
    per-thread payload. ``mode`` set to ``"up"`` leaves the first flattened
    result item undefined; ``mode="down"`` leaves the final result item
    undefined.
    ``distance`` must be ``1``. The Numba group-first projection does not
    expose ``block_prefix`` or ``block_suffix`` outputs.
    """

@overload
def shuffle(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: _ShuffleMode = "down",
    distance: int = 1,
    block_prefix: None = None,
    block_suffix: None = None,
) -> _ScalarT:
    """Return one Numba scalar selected by a qualified Shuffle mode.

    ``group`` must be a complete physical block and ``value`` one scalar.
    ``mode`` accepts ``"up"``, ``"down"``, ``"offset"``, or ``"rotate"``;
    scalar Up and Down lower through signed Offset. ``distance`` is a
    compile-time integer. Scalar calls do not accept ``block_prefix`` or
    ``block_suffix`` outputs.
    """

@overload
def merge_sort_keys(
    group: _MemoryGroup,
    keys: _ThreadDataLike[_NumbaMergeSortKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[[_NumbaMergeSortKeyT, _NumbaMergeSortKeyT], bool]
    | None = None,
) -> _ThreadDataLike[_NumbaMergeSortKeyT]:
    """Return fully merge-sorted numeric keys without mutating ``keys``.

    Complete physical blocks, physical warps, and logical warps are supported;
    a block must contain a power-of-two number of threads. The returned local
    payload preserves the input item type and item count. ``compare_op`` is a
    Numba-CUDA-MLIR device predicate; omit it for the built-in order selected by
    ``descending``.
    """

@overload
def merge_sort_keys(
    group: _MemoryGroup,
    keys: _ThreadDataLike[_NumbaMergeSortKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _NumbaMergeSortKeyT,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[[_NumbaMergeSortKeyT, _NumbaMergeSortKeyT], bool]
    | None = None,
) -> _ThreadDataLike[_NumbaMergeSortKeyT]:
    """Return a partial-tile merge-sorted numeric payload.

    ``valid_items`` and the key-typed ``oob_default`` are required together.
    A block must contain a power-of-two number of threads. The sentinel must
    sort after every valid key under the selected comparator: greater for the
    built-in ascending order and less for descending. Only the valid sorted
    prefix is defined; the output preserves the input item type and item count.
    """

@overload
def merge_sort_keys(
    group: _MemoryGroup,
    keys: _NumbaMergeSortKeyT,
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[[_NumbaMergeSortKeyT, _NumbaMergeSortKeyT], bool]
    | None = None,
) -> _NumbaMergeSortKeyT:
    """Return one fully merge-sorted Numba scalar key per group member.

    The qualified whole-function rewrite boxes scalar block, physical-warp, and
    logical-warp operands into a one-item local payload and projects the scalar
    result.
    """

@overload
def merge_sort_keys(
    group: _MemoryGroup,
    keys: _NumbaMergeSortKeyT,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _NumbaMergeSortKeyT,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[[_NumbaMergeSortKeyT, _NumbaMergeSortKeyT], bool]
    | None = None,
) -> _NumbaMergeSortKeyT:
    """Return one partial-tile merge-sorted Numba scalar key per member.

    ``oob_default`` must sort after every valid key under the selected
    comparator: greater for the built-in ascending order and less for
    descending. Block thread counts must be powers of two; only the valid
    sorted prefix is defined.
    """

@overload
def merge_sort_pairs(
    group: _MemoryGroup,
    keys: _ThreadDataLike[_NumbaMergeSortKeyT],
    values: _ThreadDataLike[_NumbaPairValueT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[[_NumbaMergeSortKeyT, _NumbaMergeSortKeyT], bool]
    | None = None,
) -> tuple[
    _ThreadDataLike[_NumbaMergeSortKeyT],
    _ThreadDataLike[_NumbaPairValueT],
]:
    """Return fully merge-sorted Numba key/value payloads."""

@overload
def merge_sort_pairs(
    group: _MemoryGroup,
    keys: _ThreadDataLike[_NumbaMergeSortKeyT],
    values: _ThreadDataLike[_NumbaPairValueT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _NumbaMergeSortKeyT,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[[_NumbaMergeSortKeyT, _NumbaMergeSortKeyT], bool]
    | None = None,
) -> tuple[
    _ThreadDataLike[_NumbaMergeSortKeyT],
    _ThreadDataLike[_NumbaPairValueT],
]:
    """Return partial-tile merge-sorted Numba key/value payloads.

    For a partial tile, provide ``valid_items`` and ``oob_default`` together;
    the sentinel must sort after every valid key under the selected comparator.
    Block thread counts must be powers of two.
    """

@overload
def merge_sort_pairs(
    group: _MemoryGroup,
    keys: _NumbaMergeSortKeyT,
    values: _NumbaPairValueT,
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[[_NumbaMergeSortKeyT, _NumbaMergeSortKeyT], bool]
    | None = None,
) -> tuple[_NumbaMergeSortKeyT, _NumbaPairValueT]:
    """Return one fully merge-sorted Numba key/value pair per member."""

@overload
def merge_sort_pairs(
    group: _MemoryGroup,
    keys: _NumbaMergeSortKeyT,
    values: _NumbaPairValueT,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _NumbaMergeSortKeyT,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[[_NumbaMergeSortKeyT, _NumbaMergeSortKeyT], bool]
    | None = None,
) -> tuple[_NumbaMergeSortKeyT, _NumbaPairValueT]:
    """Return one partial-tile merge-sorted Numba pair per member."""

@overload
def radix_sort_keys(
    group: _BlockGroup,
    keys: _ThreadDataLike[_IntegerKeyT],
    /,
    *,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> _ThreadDataLike[_IntegerKeyT]:
    """Return a full-tile radix-sorted Numba block payload.

    ``group`` must be a complete physical block. ``keys`` supplies signed or
    unsigned 32- or 64-bit per-thread keys. ``begin_bit`` and ``end_bit``
    select a half-open interval in CUB's bit-ordered key representation;
    ``end_bit`` defaults to the key width, including when only ``begin_bit`` is
    supplied.
    ``descending`` selects descending order. ``temp_storage`` supplies optional
    caller-owned scratch. ``blocked_to_striped`` selects striped output instead
    of blocked output. The returned payload preserves the input item type and
    item count without mutating ``keys``.
    """

@overload
def radix_sort_keys(
    group: _BlockGroup,
    keys: _IntegerKeyT,
    /,
    *,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> _IntegerKeyT:
    """Return one radix-sorted scalar key per Numba block thread.

    ``group`` must be a complete physical block. ``keys`` supplies one signed
    or unsigned 32- or 64-bit key. ``begin_bit`` selects the least significant
    participating bit; ``end_bit`` is exclusive and defaults to the key width.
    ``descending`` selects descending order. ``temp_storage`` supplies optional
    caller-owned scratch. ``blocked_to_striped`` selects striped output instead
    of blocked output. The scalar result preserves the key type.
    """

@overload
def radix_sort_pairs(
    group: _BlockGroup,
    keys: _ThreadDataLike[_IntegerKeyT],
    values: _ThreadDataLike[_NumbaPairValueT],
    /,
    *,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> tuple[_ThreadDataLike[_IntegerKeyT], _ThreadDataLike[_NumbaPairValueT]]:
    """Return radix-sorted Numba key/value payloads without mutation."""

@overload
def radix_sort_pairs(
    group: _BlockGroup,
    keys: _IntegerKeyT,
    values: _NumbaPairValueT,
    /,
    *,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> tuple[_IntegerKeyT, _NumbaPairValueT]:
    """Return one radix-sorted Numba key/value pair per block member."""

@overload
def radix_rank(
    group: _BlockGroup,
    keys: _ThreadDataLike[_IntegerKeyT],
    /,
    *,
    begin_bit: _TraceInteger = 0,
    end_bit: _TraceInteger | None = None,
    radix_bits: _TraceInteger | None = None,
    descending: bool = False,
    exclusive_digit_prefix: _ThreadDataLike[int]
    | _ThreadDataLike[_NumpyInt32]
    | None = None,
) -> _ThreadDataLike[int]:
    """Return signed 32-bit ranks for a Numba thread-data key payload.

    The complete block ranks one trace-static CUB bit-ordered digit without
    mutating ``keys``. ``begin_bit``, ``end_bit``, and ``radix_bits`` are
    trace-time Python or NumPy integers; the selected interval defaults to
    four bits and may contain at most eight bits. ``exclusive_digit_prefix``
    optionally receives signed 32-bit per-digit prefix counters.
    """

@overload
def radix_rank(
    group: _BlockGroup,
    keys: _IntegerKeyT,
    /,
    *,
    begin_bit: _TraceInteger = 0,
    end_bit: _TraceInteger | None = None,
    radix_bits: _TraceInteger | None = None,
    descending: bool = False,
    exclusive_digit_prefix: _ThreadDataLike[int]
    | _ThreadDataLike[_NumpyInt32]
    | None = None,
) -> int:
    """Return one signed 32-bit rank for a Numba scalar integer key.

    The specialization controls and optional prefix output have the same
    meaning as for the thread-data overload. The whole-function rewrite boxes
    the scalar into a one-item local payload and projects the scalar result.
    """

@overload
def histogram(
    group: _BlockGroup,
    samples: _ThreadDataLike[Any],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[_CounterT],
    algorithm: _HistogramAlgorithm | BlockHistogramAlgorithm = "atomic",
) -> _ThreadDataLike[_CounterT]:
    """Return striped counters typed by a portable dtype class.

    The complete block leaves compiler-produced ``samples`` unchanged.
    Positive static capacity covers every bin; excess striped slots are zero.
    Every sample must satisfy ``0 <= sample < bins``; violating this CUB
    precondition is undefined behavior.
    """

@overload
def histogram(
    group: _BlockGroup,
    samples: _ThreadDataLike[Any],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: None = None,
    algorithm: _HistogramAlgorithm | BlockHistogramAlgorithm = "atomic",
) -> _ThreadDataLike[int]:
    """Return default signed-integer striped block histogram counters.

    The complete block leaves compiler-produced ``samples`` unchanged.
    Positive static capacity covers every bin; excess striped slots are zero.
    Every sample must satisfy ``0 <= sample < bins``; violating this CUB
    precondition is undefined behavior.
    """

@overload
def histogram(
    group: _BlockGroup,
    samples: _ThreadDataLike[Any],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: object,
    algorithm: _HistogramAlgorithm | BlockHistogramAlgorithm = "atomic",
) -> _ThreadDataLike[Any]:
    """Return counters using an external Numba compiler dtype token.

    The complete block leaves compiler-produced ``samples`` unchanged.
    Positive static capacity covers every bin; excess striped slots are zero.
    Every sample must satisfy ``0 <= sample < bins``; violating this CUB
    precondition is undefined behavior.
    """

@overload
def run_length_decode(
    group: _BlockGroup,
    run_values: _ThreadDataLike[_RunValueT],
    run_lengths: _ThreadDataLike[_RunLengthT],
    /,
    *,
    decoded_items_per_thread: _TraceInteger,
    decoded_window_offset: _IntegerValue = 0,
    relative_offsets: _ThreadDataLike[_RunLengthT] | None = None,
    total_decoded_size: _ThreadDataLike[_RunLengthT] | None = None,
    decoded_offset_dtype: object = None,
) -> _ThreadDataLike[_RunValueT]:
    """Decode one blockwise run-length window with Numba side outputs.

    ``group`` is the complete block. ``run_values`` and ``run_lengths`` are
    matching positive-size compiler payloads. ``decoded_items_per_thread``
    fixes the positive result extent, while uniform
    ``decoded_window_offset`` selects a nonnegative stream position
    representable in the run-length dtype; dynamic callers guarantee its range.
    ``relative_offsets`` receives offsets within each run and
    ``total_decoded_size`` receives the block-wide stream size; both use the
    run-length dtype. Out-of-range relative offsets are the length-typed
    all-ones value (the unsigned maximum or signed ``-1``).
    ``decoded_offset_dtype`` may spell that compiler dtype explicitly. Actual
    runs have positive lengths; zeros are allowed only as one trailing padding
    suffix, and the block-wide sum is positive and representable in the
    run-length dtype. Out-of-range decoded values are zero and the inputs remain
    unchanged.
    """

@overload
def run_length_decode(
    group: _BlockGroup,
    run_values: _ThreadDataLike[_NumbaRunValueT],
    run_lengths: _ThreadDataLike[_NumbaRunLengthT],
    /,
    *,
    decoded_items_per_thread: _TraceInteger,
    decoded_window_offset: _IntegerValue = 0,
    relative_offsets: _ThreadDataLike[_NumbaRunLengthT] | None = None,
    total_decoded_size: _ThreadDataLike[_NumbaRunLengthT] | None = None,
    decoded_offset_dtype: object = None,
) -> _ThreadDataLike[_NumbaRunValueT]:
    """Decode values using the broader Numba-CUDA-MLIR dtype surface.

    ``group`` is the complete block. ``run_values`` may use Numba compiler
    integer, floating, or Boolean scalar dtypes; ``run_lengths`` may use its
    8-, 16-, 32-, or 64-bit integer dtypes. Both payloads have matching
    positive extents. ``decoded_items_per_thread`` fixes the positive result
    extent, while uniform ``decoded_window_offset`` selects a nonnegative
    stream position representable in the run-length dtype; dynamic callers
    guarantee its range. ``relative_offsets`` receives offsets within each run
    and ``total_decoded_size`` receives the block-wide stream size; both use
    the run-length dtype. Out-of-range relative offsets are the length-typed
    all-ones value (the unsigned maximum or signed ``-1``).
    ``decoded_offset_dtype`` may spell that compiler dtype explicitly. Actual
    runs have positive lengths; zeros are allowed only as one trailing padding
    suffix, and the block-wide sum is positive and representable in the
    run-length dtype. Out-of-range decoded values are zero and inputs are
    unchanged.
    """

def topk_max_keys(
    group: _BlockGroup,
    keys: _ThreadDataLike[_NumbaTopKKeyT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> _ThreadDataLike[_NumbaTopKKeyT]:
    """Select the largest keys from a Numba ``ThreadData`` block tile.

    ``group`` must be a complete one-dimensional physical block. ``keys`` is
    a fixed-size payload using a built-in Numba TopK key dtype. Uniform ``k``
    and ``valid_items`` satisfy ``1 <= k <= valid_items``; omitting
    ``valid_items`` selects the full tile. Uniform ``begin_bit`` and
    ``end_bit`` select a nonempty half-open interval, with ``end_bit=None``
    selecting the key width. Only the first ``k`` flattened blocked positions
    are defined. That prefix is unordered, ties do not expand it, and the tail
    is undefined. The new payload preserves the input item type and count
    without mutating ``keys``. ``temp_storage`` may name one reusable allocation
    shared with other block primitives.
    """

def topk_min_keys(
    group: _BlockGroup,
    keys: _ThreadDataLike[_NumbaTopKKeyT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> _ThreadDataLike[_NumbaTopKKeyT]:
    """Select the smallest keys from a Numba ``ThreadData`` block tile.

    ``group`` must be a complete one-dimensional physical block. ``keys`` is
    a fixed-size payload using a built-in Numba TopK key dtype. Uniform ``k``
    and ``valid_items`` satisfy ``1 <= k <= valid_items``; omitting
    ``valid_items`` selects the full tile. Uniform ``begin_bit`` and
    ``end_bit`` select a nonempty half-open interval, with ``end_bit=None``
    selecting the key width. Only the first ``k`` flattened blocked positions
    are defined. That prefix is unordered, ties do not expand it, and the tail
    is undefined. The new payload preserves the input item type and count
    without mutating ``keys``. ``temp_storage`` may name one reusable allocation
    shared with other block primitives.
    """

def topk_max_pairs(
    group: _BlockGroup,
    keys: _ThreadDataLike[_NumbaTopKKeyT],
    values: _ThreadDataLike[_NumbaTopKValueT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[
    _ThreadDataLike[_NumbaTopKKeyT],
    _ThreadDataLike[_NumbaTopKValueT],
]:
    """Select largest-key Numba ``ThreadData`` pairs without mutation.

    ``group`` is a complete one-dimensional physical block. ``keys`` and
    ``values`` are matching fixed-size payloads using built-in Numba TopK
    dtypes. Uniform ``k``, ``valid_items``, ``begin_bit``, and ``end_bit``
    follow the qualified keys-only contract. Only the first ``k`` unordered
    pairs are defined, and each value remains attached to its key. Both result
    item types are preserved. ``temp_storage`` may name one reusable allocation
    shared with other block primitives.
    """

def topk_min_pairs(
    group: _BlockGroup,
    keys: _ThreadDataLike[_NumbaTopKKeyT],
    values: _ThreadDataLike[_NumbaTopKValueT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[
    _ThreadDataLike[_NumbaTopKKeyT],
    _ThreadDataLike[_NumbaTopKValueT],
]:
    """Select smallest-key Numba ``ThreadData`` pairs without mutation.

    ``group`` is a complete one-dimensional physical block. ``keys`` and
    ``values`` are matching fixed-size payloads using built-in Numba TopK
    dtypes. Uniform ``k``, ``valid_items``, ``begin_bit``, and ``end_bit``
    follow the qualified keys-only contract. Only the first ``k`` unordered
    pairs are defined, and each value remains attached to its key. Both result
    item types are preserved. ``temp_storage`` may name one reusable allocation
    shared with other block primitives.
    """

__all__ = [
    "BlockHistogramAlgorithm",
    "BlockLoadAlgorithm",
    "BlockScanAlgorithm",
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
    "adjacent_difference",
    "discontinuity",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "gpu_dataclass",
    "histogram",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "local",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_pairs",
    "reduce",
    "run_length_decode",
    "scan",
    "shared",
    "shuffle",
    "store",
    "sum",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
]
