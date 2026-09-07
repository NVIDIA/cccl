# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared typing declarations for the qualified CUTLASS surface."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from enum import Enum
from typing import Any, Generic, Literal, Protocol, TypeAlias, overload

from numpy import float32 as _NumpyFloat32
from numpy import float64 as _NumpyFloat64
from numpy import int32 as _NumpyInt32
from numpy import int64 as _NumpyInt64
from numpy import uint8 as _NumpyUint8
from numpy import uint32 as _NumpyUint32
from numpy import uint64 as _NumpyUint64
from typing_extensions import TypeVar

from .. import ThreadGroup as _CommonThreadGroup
from .. import ThreadHierarchy as ThreadHierarchy
from .._typing import TempStorageSharing as _TempStorageSharing
from .._typing import ThreadDataLike as _ThreadDataLike
from .._typing import ThreadGroupKind as _ThreadGroupKind
from .._typing import ThreadLevel as _ThreadLevel
from .._typing import _CompilerIntegerLike as _CompilerIntegerLike
from .._typing import _PortableIntegerKey as _PortableIntegerKey
from .._typing import _PortableIntegerValue as _PortableIntegerValue
from .._typing import _PortableNumericScalar as _PortableNumericScalar
from .._typing import _PortableRunLength as _PortableRunLength
from .._typing import _PortableRunValue as _PortableRunValue

_ItemT = TypeVar("_ItemT", default=Any)
_SourceT_co = TypeVar("_SourceT_co", covariant=True)
_ScalarT = TypeVar("_ScalarT", bound=_PortableNumericScalar)
_ValueT = TypeVar("_ValueT")
_OrdinaryNumericScalar: TypeAlias = (
    int
    | float
    | _NumpyUint8
    | _NumpyInt32
    | _NumpyUint32
    | _NumpyInt64
    | _NumpyUint64
    | _NumpyFloat32
    | _NumpyFloat64
)
_DtypeT = TypeVar("_DtypeT", bound=_OrdinaryNumericScalar)
_CutlassNumericT = TypeVar("_CutlassNumericT", bound=_PortableNumericScalar)
_CounterT = TypeVar("_CounterT", bound=_PortableIntegerKey)
_HistogramSampleT = TypeVar("_HistogramSampleT", bound=_PortableIntegerValue)
_GroupKindT_co = TypeVar(
    "_GroupKindT_co",
    bound=_ThreadGroupKind,
    covariant=True,
    default=_ThreadGroupKind,
)
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
_DifferenceOperator: TypeAlias = Literal["-", "sub", "subtract"]
_FlagOperator: TypeAlias = Literal["!=", "ne", "not_equal"]
_CompareOperator: TypeAlias = Literal[
    "<",
    "lt",
    "less",
    "ascending",
    "asc",
    ">",
    "gt",
    "greater",
    "descending",
    "desc",
]

class _CutlassNumericDtype(Protocol):
    """Structural metadata carried by a CUTLASS ``Numeric`` dtype class."""

    width: int

class _CutlassTensorSample(Protocol):
    """Structural view of one CUTLASS register-memory Tensor."""

    @property
    def element_type(self) -> object: ...
    @property
    def shape(self) -> object: ...
    @property
    def memspace(self) -> object: ...
    def load(self) -> object:
        """Load this register-memory tensor as an immutable value."""

class _CutlassTensorSSASample(Protocol):
    """Structural view of one CUTLASS immutable register TensorSSA."""

    @property
    def dtype(self) -> object: ...
    @property
    def shape(self) -> object: ...
    def ir_value(self) -> object:
        """Return this immutable register tensor's compiler IR value."""

class _ThreadDataVectorSource(Protocol[_SourceT_co]):
    """Integer-indexable register payload accepted by ``ThreadData``."""

    def __getitem__(self, index: int, /) -> _SourceT_co:
        """Return one register value."""

class _ThreadDataSizedVectorSource(
    _ThreadDataVectorSource[_SourceT_co],
    Protocol[_SourceT_co],
):
    """Indexable register payload with a statically inferable shape."""

    @property
    def shape(self) -> object:
        """Return static payload-shape metadata."""

class _ThreadDataNumelVectorSource(
    _ThreadDataVectorSource[_SourceT_co],
    Protocol[_SourceT_co],
):
    """Indexable register payload with a statically inferable element count."""

    def numel(self) -> object:
        """Return the trace-static payload item count."""

_ThreadDataStaticallySizedVectorSource: TypeAlias = (
    _ThreadDataSizedVectorSource[_SourceT_co]
    | _ThreadDataNumelVectorSource[_SourceT_co]
)

class _ThreadDataRegisterTensorSource(
    _ThreadDataVectorSource[_SourceT_co],
    Protocol[_SourceT_co],
):
    """Integer-indexable register-memory tensor accepted by ``ThreadData``."""

    @property
    def memspace(self) -> object:
        """Return the tensor memory-space token."""

class _ThreadDataSizedRegisterTensorSource(
    _ThreadDataRegisterTensorSource[_SourceT_co],
    _ThreadDataSizedVectorSource[_SourceT_co],
    Protocol[_SourceT_co],
):
    """Register-memory tensor with a statically inferable shape."""

_ScalarValueT = TypeVar("_ScalarValueT", bound=_PortableNumericScalar)
_IntegerKeyT = TypeVar("_IntegerKeyT", bound=_PortableIntegerKey)
_RunValueT = TypeVar("_RunValueT", bound=_PortableRunValue)
_RunLengthT = TypeVar("_RunLengthT", bound=_PortableRunLength)
_CutlassOrderedItem: TypeAlias = _PortableNumericScalar
_WarpMergeSortKeyT = TypeVar("_WarpMergeSortKeyT", bound=_CutlassOrderedItem)
_CutlassTopKKeyT = TypeVar("_CutlassTopKKeyT", bound=_CutlassOrderedItem)
_CutlassTopKValueT = TypeVar("_CutlassTopKValueT", bound=_CutlassOrderedItem)
_CutlassPairValueT = TypeVar("_CutlassPairValueT", bound=_CutlassOrderedItem)

Hierarchy = ThreadHierarchy

class ThreadDataLoadSource(Protocol[_SourceT_co]):
    """Producer of a payload whose static extent can be inferred."""

    def __cuda_coop_thread_data_load__(
        self,
    ) -> ThreadData[_SourceT_co] | _ThreadDataStaticallySizedVectorSource[_SourceT_co]:
        """Return one producer-owned register payload exactly once."""

class _ThreadDataIndexableLoadSource(Protocol[_SourceT_co]):
    """Producer of an indexable payload requiring an explicit item count."""

    def __cuda_coop_thread_data_load__(
        self,
    ) -> ThreadData[_SourceT_co] | _ThreadDataVectorSource[_SourceT_co]:
        """Return one producer-owned register payload exactly once."""

class ThreadDataTensorMetadata(Protocol):
    """Static shape and dtype used to reconstruct a CUTLASS register tensor."""

    @property
    def shape(self) -> object:
        """Return the static register-payload shape."""

    @property
    def dtype(self) -> object:
        """Return the CUTLASS element-type token."""

class ThreadDataSource(
    ThreadDataLoadSource[_SourceT_co],
    ThreadDataTensorMetadata,
    Protocol[_SourceT_co],
):
    """Producer supporting both register loading and tensor reconstruction."""

class ThreadData(_ThreadDataLike[_ItemT], Generic[_ItemT]):
    """Generic CUTLASS per-thread register payload.

    Python, NumPy, compiler, and user-defined register values preserve their
    item type through construction and indexing. Unknown external dtype tokens
    can be annotated as ``Any`` when no more precise type is available.

    The container is broader than the group-first collectives: each operation
    separately restricts payloads to the numeric, ordered-key, or integer-key
    family implemented by its CUTLASS provider.

    Python's type system treats ``bool`` as a subtype of ``int``, so static
    checking cannot exclude it from these overloads. CUTLASS rejects Boolean
    payload dtypes and values when it validates a traced primitive.
    """

    items_per_thread: int
    dtype: object | None

    @overload
    def __init__(
        self,
        items_per_thread: int,
        dtype: type[_ItemT],
        *,
        values: tuple[_ItemT, ...] | list[_ItemT] | None = None,
    ) -> None:
        """Construct typed register storage with optional initial values."""
    @overload
    def __init__(
        self,
        items_per_thread: int,
        dtype: None = None,
        *,
        values: tuple[_ItemT, ...] | list[_ItemT] | None = None,
    ) -> None:
        """Construct storage whose item type is inferred from initial values."""

    @classmethod
    def from_values(
        cls,
        first: _ValueT,
        *rest: _ValueT,
        dtype: type[Any] | None = None,
    ) -> ThreadData[_ValueT]:
        """Preserve the value type; ``dtype`` records metadata without casting."""

    @overload
    @classmethod
    def from_fn(
        cls,
        items_per_thread: int,
        fn: Callable[[int], object],
        *,
        dtype: type[_DtypeT],
    ) -> ThreadData[_DtypeT]:
        """Cast results to ``dtype``; an opaque ``Any`` dtype yields ``Any``."""
    @overload
    @classmethod
    def from_fn(
        cls,
        items_per_thread: int,
        fn: Callable[[int], _ValueT],
        *,
        dtype: None = None,
    ) -> ThreadData[_ValueT]:
        """Infer the item type from ``fn`` when no dtype cast is requested."""
    @overload
    @classmethod
    def from_fn(
        cls,
        items_per_thread: int,
        fn: Callable[[int], object],
        *,
        dtype: object,
    ) -> ThreadData[Any]:
        """Accept opaque dtype metadata whose resulting item type is unknown."""

    @overload
    @classmethod
    def from_register_tensor(
        cls,
        fragment: _ThreadDataSizedRegisterTensorSource[Any],
        *,
        items_per_thread: int | None = None,
        dtype: type[_DtypeT],
    ) -> ThreadData[_DtypeT]:
        """Cast a shaped register tensor to the explicit CUTLASS dtype."""
    @overload
    @classmethod
    def from_register_tensor(
        cls,
        fragment: _ThreadDataRegisterTensorSource[Any],
        *,
        items_per_thread: int,
        dtype: type[_DtypeT],
    ) -> ThreadData[_DtypeT]:
        """Cast an explicitly sized register tensor to ``dtype``."""
    @overload
    @classmethod
    def from_register_tensor(
        cls,
        fragment: _ThreadDataSizedRegisterTensorSource[_ValueT],
        *,
        items_per_thread: int | None = None,
        dtype: None = None,
    ) -> ThreadData[_ValueT]:
        """Adapt a shaped register tensor and preserve its item type."""
    @overload
    @classmethod
    def from_register_tensor(
        cls,
        fragment: _ThreadDataRegisterTensorSource[_ValueT],
        *,
        items_per_thread: int,
        dtype: None = None,
    ) -> ThreadData[_ValueT]:
        """Adapt an indexable register tensor using an explicit item count."""
    @overload
    @classmethod
    def from_register_tensor(
        cls,
        fragment: _ThreadDataSizedRegisterTensorSource[Any],
        *,
        items_per_thread: int | None = None,
        dtype: object,
    ) -> ThreadData[Any]:
        """Adapt a shaped register tensor with opaque dtype metadata."""
    @overload
    @classmethod
    def from_register_tensor(
        cls,
        fragment: _ThreadDataRegisterTensorSource[Any],
        *,
        items_per_thread: int,
        dtype: object,
    ) -> ThreadData[Any]:
        """Adapt an explicitly sized tensor with opaque dtype metadata."""

    @overload
    @classmethod
    def from_vector(
        cls,
        vector: _ThreadDataStaticallySizedVectorSource[Any],
        *,
        items_per_thread: int | None = None,
        dtype: type[_DtypeT],
    ) -> ThreadData[_DtypeT]:
        """Cast a statically sized register vector to the explicit dtype."""
    @overload
    @classmethod
    def from_vector(
        cls,
        vector: _ThreadDataVectorSource[Any],
        *,
        items_per_thread: int,
        dtype: type[_DtypeT],
    ) -> ThreadData[_DtypeT]:
        """Cast an explicitly sized register vector to ``dtype``."""
    @overload
    @classmethod
    def from_vector(
        cls,
        vector: _ThreadDataStaticallySizedVectorSource[_ValueT],
        *,
        items_per_thread: int | None = None,
        dtype: None = None,
    ) -> ThreadData[_ValueT]:
        """Adapt a statically sized register vector and preserve its item type."""
    @overload
    @classmethod
    def from_vector(
        cls,
        vector: _ThreadDataVectorSource[_ValueT],
        *,
        items_per_thread: int,
        dtype: None = None,
    ) -> ThreadData[_ValueT]:
        """Adapt an indexable register vector using an explicit item count."""
    @overload
    @classmethod
    def from_vector(
        cls,
        vector: _ThreadDataStaticallySizedVectorSource[Any],
        *,
        items_per_thread: int | None = None,
        dtype: object,
    ) -> ThreadData[Any]:
        """Adapt a statically sized vector with opaque dtype metadata."""
    @overload
    @classmethod
    def from_vector(
        cls,
        vector: _ThreadDataVectorSource[Any],
        *,
        items_per_thread: int,
        dtype: object,
    ) -> ThreadData[Any]:
        """Adapt an explicitly sized vector with opaque dtype metadata."""

    @overload
    @classmethod
    def from_payload(
        cls,
        payload: ThreadData[Any],
        *,
        items_per_thread: int | None = None,
        dtype: type[_DtypeT],
    ) -> ThreadData[_DtypeT]:
        """Constrain or cast an existing payload to the explicit dtype."""
    @overload
    @classmethod
    def from_payload(
        cls,
        payload: _ThreadDataStaticallySizedVectorSource[Any],
        *,
        items_per_thread: int | None = None,
        dtype: type[_DtypeT],
    ) -> ThreadData[_DtypeT]:
        """Cast a statically sized backend payload to the explicit dtype."""
    @overload
    @classmethod
    def from_payload(
        cls,
        payload: _ThreadDataVectorSource[Any],
        *,
        items_per_thread: int,
        dtype: type[_DtypeT],
    ) -> ThreadData[_DtypeT]:
        """Cast an explicitly sized backend payload to ``dtype``."""
    @overload
    @classmethod
    def from_payload(
        cls,
        payload: ThreadData[_ValueT],
        *,
        items_per_thread: int | None = None,
        dtype: None = None,
    ) -> ThreadData[_ValueT]:
        """Preserve the item type of an existing CUTLASS payload."""
    @overload
    @classmethod
    def from_payload(
        cls,
        payload: _ThreadDataStaticallySizedVectorSource[_ValueT],
        *,
        items_per_thread: int | None = None,
        dtype: None = None,
    ) -> ThreadData[_ValueT]:
        """Adapt a statically sized backend payload and preserve its item type."""
    @overload
    @classmethod
    def from_payload(
        cls,
        payload: _ThreadDataVectorSource[_ValueT],
        *,
        items_per_thread: int,
        dtype: None = None,
    ) -> ThreadData[_ValueT]:
        """Adapt an indexable payload using an explicit item count."""
    @overload
    @classmethod
    def from_payload(
        cls,
        payload: ThreadData[Any],
        *,
        items_per_thread: int | None = None,
        dtype: object,
    ) -> ThreadData[Any]:
        """Constrain an existing payload with opaque dtype metadata."""
    @overload
    @classmethod
    def from_payload(
        cls,
        payload: _ThreadDataStaticallySizedVectorSource[Any],
        *,
        items_per_thread: int | None = None,
        dtype: object,
    ) -> ThreadData[Any]:
        """Adapt a statically sized payload with opaque dtype metadata."""
    @overload
    @classmethod
    def from_payload(
        cls,
        payload: _ThreadDataVectorSource[Any],
        *,
        items_per_thread: int,
        dtype: object,
    ) -> ThreadData[Any]:
        """Adapt an explicitly sized payload with opaque dtype metadata."""

    @overload
    @classmethod
    def load(
        cls,
        source: ThreadDataLoadSource[Any],
        *,
        items_per_thread: int | None = None,
        dtype: type[_DtypeT],
    ) -> ThreadData[_DtypeT]:
        """Load and cast a statically sized producer payload to ``dtype``."""
    @overload
    @classmethod
    def load(
        cls,
        source: _ThreadDataIndexableLoadSource[Any],
        *,
        items_per_thread: int,
        dtype: type[_DtypeT],
    ) -> ThreadData[_DtypeT]:
        """Load and cast an explicitly sized producer payload to ``dtype``."""
    @overload
    @classmethod
    def load(
        cls,
        source: ThreadDataLoadSource[_ValueT],
        *,
        items_per_thread: int | None = None,
        dtype: None = None,
    ) -> ThreadData[_ValueT]:
        """Load registers and preserve the producer's declared item type."""
    @overload
    @classmethod
    def load(
        cls,
        source: _ThreadDataIndexableLoadSource[_ValueT],
        *,
        items_per_thread: int,
        dtype: None = None,
    ) -> ThreadData[_ValueT]:
        """Load an indexable producer payload using an explicit item count."""
    @overload
    @classmethod
    def load(
        cls,
        source: ThreadDataLoadSource[Any],
        *,
        items_per_thread: int | None = None,
        dtype: object,
    ) -> ThreadData[Any]:
        """Load a statically sized producer using opaque dtype metadata."""
    @overload
    @classmethod
    def load(
        cls,
        source: _ThreadDataIndexableLoadSource[Any],
        *,
        items_per_thread: int,
        dtype: object,
    ) -> ThreadData[Any]:
        """Load an explicitly sized producer using opaque dtype metadata."""
    def to_tensor_ssa(
        self,
        *,
        dtype: type[_CutlassNumericDtype] | None = None,
        shape: object = None,
        like: ThreadDataTensorMetadata | None = None,
    ) -> object:
        """Materialize this payload as a register-only CUTLASS TensorSSA."""

    def to_register_tensor(
        self,
        *,
        dtype: type[_CutlassNumericDtype] | None = None,
        shape: object = None,
    ) -> object:
        """Materialize this payload as a fresh mutable register tensor."""

    def values(self, primitive_name: str) -> tuple[_ItemT, ...]:
        """Return initialized register values for one primitive."""

    def __len__(self) -> int:
        """Return ``items_per_thread``."""

    def __iter__(self) -> Iterator[_ItemT]:
        """Iterate initialized register values."""

    def __getitem__(self, index: int, /) -> _ItemT:
        """Return one register value."""

    def __setitem__(self, index: int, value: _ItemT, /) -> None:
        """Replace one register value."""

_CutlassHistogramOpaqueSamples: TypeAlias = (
    _CutlassTensorSample | _CutlassTensorSSASample
)
_CutlassRunTensor: TypeAlias = _CutlassTensorSample | _CutlassTensorSSASample

class TempStorage:
    """Explicit CUTLASS shared-memory scratch planner."""

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
        """Configure scratch capacity, alignment, synchronization, and sharing."""

    @property
    def required_size_in_bytes(self) -> int:
        """Return scratch bytes required by recorded collective uses."""

    @property
    def capacity_size_in_bytes(self) -> int | None:
        """Return the explicit or planned scratch capacity."""

    @property
    def required_alignment(self) -> int:
        """Return the strongest alignment required by recorded uses."""

    def sync(self) -> None:
        """Synchronize threads that may reuse this scratch allocation."""

class ThreadGroup(
    _CommonThreadGroup[_GroupKindT_co],
    Generic[_GroupKindT_co],
):
    """Common CUDA group descriptor with CUTLASS lowering methods."""

    def rank(self, level: _ThreadLevel = "thread") -> _CompilerIntegerLike:
        """Return this group's rank as a CUTLASS ``Int32`` scalar."""

    def count(self, level: _ThreadLevel = "thread") -> _CompilerIntegerLike:
        """Return this group's count as a CUTLASS ``Int32`` scalar."""

    @overload
    def rank_as(
        self, dtype: type[_ScalarT], level: _ThreadLevel = "thread"
    ) -> _ScalarT:
        """Convert rank to a portable or structural CUTLASS numeric dtype."""
    @overload
    def rank_as(self, dtype: None = None, level: _ThreadLevel = "thread") -> Any:
        """Omit dtype or use an ``Any``-typed external CUTLASS dtype token."""

    @overload
    def count_as(
        self, dtype: type[_ScalarT], level: _ThreadLevel = "thread"
    ) -> _ScalarT:
        """Convert count to a portable or structural CUTLASS numeric dtype."""
    @overload
    def count_as(self, dtype: None = None, level: _ThreadLevel = "thread") -> Any:
        """Omit dtype or use an ``Any``-typed external CUTLASS dtype token."""

    def sync(self) -> None:
        """Synchronize participating members.

        Grid synchronization requires a compiler-verified cooperative launch.
        """

    def sync_aligned(self) -> None:
        """Synchronize an aligned group in converged control flow.

        Grid synchronization requires a compiler-verified cooperative launch.
        """

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
    def is_member(self) -> _CompilerIntegerLike:
        """Return a CUTLASS ``Uint8`` membership flag for the current thread."""

_MemoryGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp", "block"]]
_MergeSortWarpGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp"]]
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

class Payload(str, Enum):
    """Explicit CUTLASS load/store payload selector."""

    PRIMS = "prims"

def this_thread() -> ThreadGroup[Literal["thread"]]:
    """Describe the current thread."""

def this_warp() -> ThreadGroup[Literal["warp"]]:
    """Describe the current complete physical warp."""

def this_block() -> ThreadGroup[Literal["block"]]:
    """Describe the current CUDA thread block."""

def this_cluster() -> ThreadGroup[Literal["cluster"]]:
    """Describe the current thread-block cluster."""

def this_grid() -> ThreadGroup[Literal["grid"]]:
    """Describe the current grid."""

__all__ = [
    "Hierarchy",
    "Payload",
    "TempStorage",
    "ThreadData",
    "ThreadDataLoadSource",
    "ThreadDataSource",
    "ThreadDataTensorMetadata",
    "ThreadGroup",
    "ThreadHierarchy",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]
