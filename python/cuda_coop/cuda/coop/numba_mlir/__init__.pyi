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
from .._typing import ReduceAlgorithm as _ReduceAlgorithm
from .._typing import ReduceOperator as _ReduceOperator
from .._typing import ScanAlgorithm as _ScanAlgorithm
from .._typing import ScanOperator as _ScanOperator
from .._typing import TempStorageSharing as _TempStorageSharing
from .._typing import ThreadDataLike as _ThreadDataLike
from .._typing import ThreadGroupKind as _ThreadGroupKind
from .._typing import ThreadLevel as _ThreadLevel
from .._typing import _CompilerScalarLike as _CompilerScalarLike
from .._typing import _IntegerValue as _IntegerValue
from .._typing import _NonSumScanOperator as _NonSumScanOperator
from .._typing import _PortableIntegerKey as _PortableIntegerKey
from .._typing import _PortableNumericScalar as _PortableNumericScalar
from .._typing import _PortableRunLength as _PortableRunLength
from .._typing import _PortableRunValue as _PortableRunValue
from .._typing import _ScalarValue as _ScalarValue
from .._typing import _SumScanOperator as _SumScanOperator
from .._typing import _SynchronizableGroupKind as _SynchronizableGroupKind
from .._typing import _TraceInteger as _TraceInteger
from .._typing import _ValidItems as _ValidItems

_ItemT = TypeVar("_ItemT")
_ScalarT = TypeVar("_ScalarT", bound=_ScalarValue)
_IntegerKeyT = TypeVar("_IntegerKeyT", bound=_PortableIntegerKey)
_RadixValueT = TypeVar("_RadixValueT", bound=_PortableNumericScalar)
_OpT = TypeVar("_OpT")
_DataclassT = TypeVar("_DataclassT")
_GroupKindT_co = TypeVar(
    "_GroupKindT_co",
    bound=_ThreadGroupKind,
    covariant=True,
    default=_ThreadGroupKind,
)
_ArrayShape: TypeAlias = int | tuple[int, ...]
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
_NumbaMergeSortKeyT = TypeVar("_NumbaMergeSortKeyT", bound=_NumbaOrderedItem)
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
_TopKKeyT = TypeVar("_TopKKeyT", bound=_NumbaOrderedItem)
_TopKValueT = TypeVar("_TopKValueT", bound=_NumbaPairValue)
_CounterT = TypeVar(
    "_CounterT",
    int,
    _NumpyInt32,
    _NumpyUint32,
    _NumpyInt64,
    _NumpyUint64,
)
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

class BlockLoadAlgorithm(IntEnum):
    DIRECT: int
    STRIPED: int
    VECTORIZE: int
    TRANSPOSE: int
    WARP_TRANSPOSE: int
    WARP_TRANSPOSE_TIMESLICED: int

class BlockStoreAlgorithm(IntEnum):
    DIRECT: int
    STRIPED: int
    VECTORIZE: int
    TRANSPOSE: int
    WARP_TRANSPOSE: int
    WARP_TRANSPOSE_TIMESLICED: int

class BlockHistogramAlgorithm(IntEnum):
    SORT: int
    ATOMIC: int

class BlockScanAlgorithm(IntEnum):
    RAKING: int
    RAKING_MEMOIZE: int
    WARP_SCANS: int

class WarpLoadAlgorithm(IntEnum):
    DIRECT: int
    STRIPED: int
    VECTORIZE: int
    TRANSPOSE: int

class WarpStoreAlgorithm(IntEnum):
    DIRECT: int
    STRIPED: int
    VECTORIZE: int
    TRANSPOSE: int

class StatefulFunction(Generic[_OpT]):
    """Device callable paired with explicit state for generated wrappers."""

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
        """Return the group rank converted to a compiler dtype token."""

    @overload
    def count_as(
        self,
        dtype: type[_ItemT],
        level: _ThreadLevel = "thread",
    ) -> _ItemT:
        """Return the group count converted to an ordinary scalar dtype."""

    @overload
    def count_as(self, dtype: object = None, level: _ThreadLevel = "thread") -> Any:
        """Return the group count converted to a compiler dtype token."""

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

_ReductionGroup: TypeAlias = ThreadGroup[
    Literal[
        "thread",
        "warp",
        "threads_within_warp",
        "warps_within_block",
        "block",
        "cluster",
    ]
]
_BlockGroup: TypeAlias = ThreadGroup[Literal["block"]]
_WarpGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp"]]

class TempStorage:
    """Explicit opaque byte scratch for planned shared-memory operations."""

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool
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
        *,
        alignment: int | None = 8,
    ) -> Any:
        """Allocate thread-local compiler storage."""

class _SharedMemory(Protocol):
    """Numba-CUDA-MLIR shared-memory namespace."""

    def array(
        self,
        shape: _ArrayShape,
        dtype: object,
        *,
        alignment: int | None = 8,
    ) -> Any:
        """Allocate shared compiler storage."""

local: _LocalMemory
shared: _SharedMemory

class _GpuDataclassArgumentHandler(Protocol):
    """Launch-time marshalling extension for registered GPU dataclasses."""

    def prepare_args(
        self,
        ty: Any,
        val: Any,
        stream: Any = None,
        retr: list[Any] | None = None,
    ) -> tuple[Any, Any]:
        """Flatten a registered dataclass and preserve its compiler type."""

gpu_dataclass_argument_handler: _GpuDataclassArgumentHandler

@overload
def ThreadData(
    items_per_thread: int,
    dtype: type[_ItemT],
    *,
    alignas: int = 8,
    alignment: int | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Construct typed thread-local storage."""

@overload
def ThreadData(
    items_per_thread: int,
    dtype: object = None,
    *,
    alignas: int = 8,
    alignment: int | None = None,
) -> _ThreadDataLike[Any]:
    """Construct storage using a compiler dtype token or inferred dtype."""

def gpu_dataclass(
    dc: _DataclassT,
    *,
    compute_temp_storage: bool = True,
) -> _DataclassT:
    """Register a dataclass instance for Numba-CUDA-MLIR device use."""

@overload
def load(
    group: ThreadGroup[Literal["block"]],
    source: Any,
    output: _ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: str | int | BlockLoadAlgorithm = "direct",
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    temp_storage: TempStorage | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Load a per-thread tile through a block group."""

@overload
def load(
    group: ThreadGroup[Literal["warp", "threads_within_warp"]],
    source: Any,
    output: _ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: str | int | WarpLoadAlgorithm = "direct",
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    temp_storage: None = None,
) -> _ThreadDataLike[_ItemT]:
    """Load a per-thread tile through a physical or logical warp."""

@overload
def store(
    group: ThreadGroup[Literal["block"]],
    destination: Any,
    value: Any,
    /,
    *,
    algorithm: str | int | BlockStoreAlgorithm = "direct",
    valid_items: Any = None,
    offset: Any = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Store a per-thread tile through a block group."""

@overload
def store(
    group: ThreadGroup[Literal["warp", "threads_within_warp"]],
    destination: Any,
    value: Any,
    /,
    *,
    algorithm: str | int | WarpStoreAlgorithm = "direct",
    valid_items: Any = None,
    offset: Any = None,
    temp_storage: None = None,
) -> None:
    """Store a per-thread tile through a physical or logical warp."""

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
    """Reduce a full-group payload through the CUDAX group provider."""

@overload
def reduce(
    group: _ReductionGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ScalarT:
    """Reduce full-group scalar values through the CUDAX group provider."""

@overload
def reduce(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    binary_op: _ReduceOperator | Callable[[_ItemT, _ItemT], _ItemT] | None = None,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: _ReduceAlgorithm,
) -> _ItemT:
    """Reduce ThreadData through explicitly selected CUB BlockReduce."""

@overload
def reduce(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: _ReduceOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: _ReduceAlgorithm | None = None,
) -> _ScalarT:
    """Reduce a valid scalar prefix through direct CUB BlockReduce."""

@overload
def reduce(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: _ReduceOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: _ReduceAlgorithm,
) -> _ScalarT:
    """Reduce a scalar through explicitly selected CUB BlockReduce."""

@overload
def reduce(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    binary_op: _ReduceOperator | Callable[[_ScalarT, _ScalarT], _ScalarT] | None = None,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: None = None,
) -> _ScalarT:
    """Reduce a valid scalar prefix through direct CUB WarpReduce."""

@overload
def reduce(
    group: _BlockGroup,
    value: _ItemT,
    /,
    *,
    binary_op: Callable[[_ItemT, _ItemT], _ItemT],
    broadcast: Literal[False],
    valid_items: _ValidItems | None = None,
    algorithm: _ReduceAlgorithm | None = None,
) -> _ItemT:
    """Reduce scalar values with a qualified BlockReduce callback."""

@overload
def reduce(
    group: _WarpGroup,
    value: _ItemT,
    /,
    *,
    binary_op: Callable[[_ItemT, _ItemT], _ItemT],
    broadcast: Literal[False],
    valid_items: _ValidItems | None = None,
    algorithm: None = None,
) -> _ItemT:
    """Reduce scalar values with a qualified WarpReduce callback."""

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
    """Sum a full-group payload through the CUDAX group provider."""

@overload
def sum(
    group: _ReductionGroup,
    value: _ScalarT,
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ScalarT:
    """Sum full-group scalar values through the CUDAX group provider."""

@overload
def sum(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: _ReduceAlgorithm,
) -> _ItemT:
    """Sum ThreadData through explicitly selected CUB BlockReduce."""

@overload
def sum(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: _ReduceAlgorithm | None = None,
) -> _ScalarT:
    """Sum a valid scalar prefix through direct CUB BlockReduce."""

@overload
def sum(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: _ReduceAlgorithm,
) -> _ScalarT:
    """Sum a scalar through explicitly selected CUB BlockReduce."""

@overload
def sum(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: None = None,
) -> _ScalarT:
    """Sum a valid scalar prefix through direct CUB WarpReduce."""

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
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return out-of-place block-exclusive sums."""

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
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return a non-sum block-exclusive scan from an explicit initial value."""

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
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return an out-of-place block-inclusive scan."""

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
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-exclusive scalar sum."""

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
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a non-sum block-exclusive scalar scan."""

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
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-inclusive scalar scan."""

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
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-exclusive scalar sum."""

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
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a non-sum warp-exclusive scalar scan."""

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
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-inclusive scalar scan."""

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return out-of-place block-exclusive sums with the input shape."""

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-exclusive scalar sum."""

@overload
def exclusive_sum(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-exclusive scalar sum."""

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return out-of-place block-inclusive sums with the input shape."""

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: _ScanAlgorithm | BlockScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-inclusive scalar sum."""

@overload
def inclusive_sum(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
    valid_items: _ValidItems | None = None,
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-inclusive scalar sum."""

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
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return an out-of-place block-exclusive sum."""

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
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return a non-sum block-exclusive scan."""

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
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-exclusive scalar sum."""

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
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a non-sum block-exclusive scalar scan."""

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
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-exclusive scalar sum."""

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
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a non-sum warp-exclusive scalar scan."""

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
    aggregate_output: _ThreadDataLike[_ItemT] | None = None,
) -> _ThreadDataLike[_ItemT]:
    """Return an out-of-place block-inclusive scan."""

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
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a block-inclusive scalar scan."""

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
    aggregate_output: _ThreadDataLike[_ScalarT] | None = None,
) -> _ScalarT:
    """Return a physical- or logical-warp-inclusive scalar scan."""

@overload
def merge_sort_keys(
    group: _BlockGroup,
    keys: _ThreadDataLike[_NumbaMergeSortKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> _ThreadDataLike[_NumbaMergeSortKeyT]:
    """Return fresh block-wide merge-sorted keys."""

@overload
def merge_sort_keys(
    group: _BlockGroup,
    keys: _NumbaMergeSortKeyT,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> _NumbaMergeSortKeyT:
    """Return one fresh merge-sorted key per block member."""

@overload
def merge_sort_keys(
    group: _WarpGroup,
    keys: _ThreadDataLike[_NumbaMergeSortKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> _ThreadDataLike[_NumbaMergeSortKeyT]:
    """Return fresh physical- or logical-warp merge-sorted keys."""

@overload
def merge_sort_keys(
    group: _WarpGroup,
    keys: _NumbaMergeSortKeyT,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> _NumbaMergeSortKeyT:
    """Return one fresh merge-sorted key per warp member."""

@overload
def merge_sort_pairs(
    group: _BlockGroup,
    keys: _ThreadDataLike[_NumbaMergeSortKeyT],
    values: _ThreadDataLike[_NumbaPairValueT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> tuple[
    _ThreadDataLike[_NumbaMergeSortKeyT],
    _ThreadDataLike[_NumbaPairValueT],
]:
    """Return fresh block-wide merge-sorted key/value payloads."""

@overload
def merge_sort_pairs(
    group: _BlockGroup,
    keys: _NumbaMergeSortKeyT,
    values: _NumbaPairValueT,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: TempStorage | None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> tuple[_NumbaMergeSortKeyT, _NumbaPairValueT]:
    """Return one fresh merge-sorted key/value pair per block member."""

@overload
def merge_sort_pairs(
    group: _WarpGroup,
    keys: _ThreadDataLike[_NumbaMergeSortKeyT],
    values: _ThreadDataLike[_NumbaPairValueT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> tuple[
    _ThreadDataLike[_NumbaMergeSortKeyT],
    _ThreadDataLike[_NumbaPairValueT],
]:
    """Return fresh physical- or logical-warp sorted key/value payloads."""

@overload
def merge_sort_pairs(
    group: _WarpGroup,
    keys: _NumbaMergeSortKeyT,
    values: _NumbaPairValueT,
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems | None = None,
    oob_default: _NumbaMergeSortKeyT | None = None,
    temp_storage: None = None,
    compare_op: Callable[
        [_NumbaMergeSortKeyT, _NumbaMergeSortKeyT],
        bool,
    ]
    | None = None,
) -> tuple[_NumbaMergeSortKeyT, _NumbaPairValueT]:
    """Return one fresh merge-sorted key/value pair per warp member."""

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

    The complete block leaves the fixed-size ``samples`` payload unchanged.
    Positive static capacity covers every bin; excess striped slots are zero.
    Every sample satisfies CUB's ``0 <= sample < bins`` precondition.
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
    """Return default signed-integer striped counters."""

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
    """Return counters using a Numba-CUDA-MLIR dtype token."""

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
    """Decode a blockwise window with optional side outputs.

    Inputs have matching fixed extents and use blocked run ownership. The
    decoded extent is positive and static. The uniform window offset is
    nonnegative and representable in the run-length dtype; dynamic callers
    guarantee its range. Side outputs use that same dtype. Actual runs have
    positive lengths followed only by an optional trailing zero-padding
    suffix, and their positive total is representable in the length dtype.
    The decoded result is fresh, inputs are unchanged, and positions past the
    total decode to zero.
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
    """Decode using the broader Numba-CUDA-MLIR scalar dtype surface."""

def exchange(
    group: ThreadGroup[Any],
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    mode: str = "striped_to_blocked",
    ranks: Any = None,
    valid_flags: Any = None,
    warp_time_slicing: bool = False,
) -> _ThreadDataLike[_ItemT]:
    """Rearrange a fixed-size per-thread tile within a group."""

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
    """Return a fresh radix-sorted block payload."""

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
    """Return one fresh radix-sorted scalar key per block thread."""

@overload
def radix_sort_pairs(
    group: _BlockGroup,
    keys: _ThreadDataLike[_IntegerKeyT],
    values: _ThreadDataLike[_RadixValueT],
    /,
    *,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> tuple[_ThreadDataLike[_IntegerKeyT], _ThreadDataLike[_RadixValueT]]:
    """Return fresh radix-sorted key/value payloads."""

@overload
def radix_sort_pairs(
    group: _BlockGroup,
    keys: _IntegerKeyT,
    values: _RadixValueT,
    /,
    *,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> tuple[_IntegerKeyT, _RadixValueT]:
    """Return one fresh radix-sorted scalar pair per block thread."""

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
    """Return fresh signed 32-bit ranks for one radix digit."""

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
    """Return one signed 32-bit radix rank per block thread."""

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
    """Return left differences in a fresh per-thread payload."""

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
    """Return right differences for a full or partial tile."""

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
    """Return right differences with a full-tile successor boundary."""

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
    """Return one left scalar difference per thread."""

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
    """Return one right scalar difference per thread."""

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
    """Return one right scalar difference with a successor boundary."""

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
    """Return fresh signed 32-bit head flags."""

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
    """Return fresh signed 32-bit tail flags."""

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
    """Return fresh signed 32-bit head and tail flag payloads."""

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
    """Return one signed 32-bit scalar head flag per thread."""

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
    """Return one signed 32-bit scalar tail flag per thread."""

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
    """Return signed 32-bit scalar head and tail flags per thread."""

@overload
def shuffle(
    group: ThreadGroup[Literal["block"]],
    value: _ThreadDataLike[_ItemT],
    /,
    *,
    mode: str = "down",
    distance: int = 1,
    block_prefix: None = None,
    block_suffix: None = None,
) -> _ThreadDataLike[_ItemT]:
    """Shuffle a tile without exposing private boundary outputs."""

@overload
def shuffle(
    group: ThreadGroup[Literal["block"]],
    value: _ItemT,
    /,
    *,
    mode: str = "down",
    distance: int = 1,
    block_prefix: None = None,
    block_suffix: None = None,
) -> _ItemT:
    """Shuffle a scalar value without boundary outputs."""

def topk_max_keys(
    group: _BlockGroup,
    keys: _ThreadDataLike[_TopKKeyT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> _ThreadDataLike[_TopKKeyT]:
    """Select the largest keys into a fresh fixed-size block payload."""

def topk_min_keys(
    group: _BlockGroup,
    keys: _ThreadDataLike[_TopKKeyT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> _ThreadDataLike[_TopKKeyT]:
    """Select the smallest keys into a fresh fixed-size block payload."""

def topk_max_pairs(
    group: _BlockGroup,
    keys: _ThreadDataLike[_TopKKeyT],
    values: _ThreadDataLike[_TopKValueT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[_ThreadDataLike[_TopKKeyT], _ThreadDataLike[_TopKValueT]]:
    """Select largest-key pairs into fresh matching block payloads."""

def topk_min_pairs(
    group: _BlockGroup,
    keys: _ThreadDataLike[_TopKKeyT],
    values: _ThreadDataLike[_TopKValueT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[_ThreadDataLike[_TopKKeyT], _ThreadDataLike[_TopKValueT]]:
    """Select smallest-key pairs into fresh matching block payloads."""

def this_thread() -> ThreadGroup[Literal["thread"]]:
    """Describe the current thread."""

def this_warp() -> ThreadGroup[Literal["warp"]]:
    """Describe the current complete physical warp."""

def this_block() -> ThreadGroup[Literal["block"]]:
    """Describe the current CUDA thread block."""

def this_cluster() -> ThreadGroup[Literal["cluster"]]:
    """Describe the current cluster where the launch can represent it."""

def this_grid() -> ThreadGroup[Literal["grid"]]:
    """Describe the current grid."""

__all__ = [
    "BlockHistogramAlgorithm",
    "BlockLoadAlgorithm",
    "BlockScanAlgorithm",
    "BlockStoreAlgorithm",
    "Hierarchy",
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
    "gpu_dataclass_argument_handler",
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
