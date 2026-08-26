# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for CUTLASS per-thread register payloads."""

from collections.abc import Callable, Iterator
from typing import Any, Generic, Protocol, TypeAlias, overload

import numpy as np
from typing_extensions import TypeVar

from .._typing import ThreadDataLike

_ItemT = TypeVar("_ItemT", default=Any)
_SourceT_co = TypeVar("_SourceT_co", covariant=True)
_ValueT = TypeVar("_ValueT")
_OrdinaryNumericScalar: TypeAlias = (
    int
    | float
    | np.uint8
    | np.int32
    | np.uint32
    | np.int64
    | np.uint64
    | np.float32
    | np.float64
)
_DtypeT = TypeVar("_DtypeT", bound=_OrdinaryNumericScalar)

class _CutlassNumericDtype(Protocol):
    """Structural metadata carried by a CUTLASS ``Numeric`` dtype class."""

    width: int

class CutlassTensorSample(Protocol):
    """Structural view of one CUTLASS register-memory Tensor."""

    @property
    def element_type(self) -> object: ...
    @property
    def shape(self) -> object: ...
    @property
    def memspace(self) -> object: ...
    def __getitem__(self, index: int, /) -> Any:
        """Return one register value."""
    def load(self) -> object:
        """Load this register-memory tensor as an immutable value."""

class CutlassTensorSSASample(Protocol):
    """Structural view of one CUTLASS immutable register TensorSSA."""

    @property
    def dtype(self) -> object: ...
    @property
    def shape(self) -> object: ...
    def __getitem__(self, index: int, /) -> Any:
        """Return one register value."""
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

class _ThreadDataReadOnlyView(
    _ThreadDataVectorSource[_SourceT_co],
    Protocol[_SourceT_co],
):
    """Read-only payload view returned by covariant load sources."""

    @property
    def items_per_thread(self) -> int:
        """Return the static number of register values."""

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

class ThreadDataLoadSource(Protocol[_SourceT_co]):
    """Producer of a payload whose static extent can be inferred."""

    def __cuda_coop_thread_data_load__(
        self,
    ) -> (
        _ThreadDataReadOnlyView[_SourceT_co]
        | _ThreadDataStaticallySizedVectorSource[_SourceT_co]
    ):
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

class ThreadData(ThreadDataLike[_ItemT], Generic[_ItemT]):
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
    ) -> CutlassTensorSSASample:
        """Materialize this payload as a register-only CUTLASS TensorSSA."""

    def to_register_tensor(
        self,
        *,
        dtype: type[_CutlassNumericDtype] | None = None,
        shape: object = None,
    ) -> CutlassTensorSample:
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

__all__ = [
    "ThreadData",
    "ThreadDataLoadSource",
    "ThreadDataSource",
    "ThreadDataTensorMetadata",
]
