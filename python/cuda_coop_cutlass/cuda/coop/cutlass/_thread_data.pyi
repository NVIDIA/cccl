# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type per-thread payloads and their qualified register conversions.

Structural samples describe the CuTe attributes used by conversion methods.
Matching a protocol does not establish register address space, static extent,
or compiler support; runtime tracing checks those requirements.
"""

from collections.abc import Callable, Iterator
from typing import Any, Generic, Protocol, TypeAlias, overload

from typing_extensions import TypeVar

from ._typing import ThreadDataLike

_ItemT = TypeVar("_ItemT", default=Any)
_ValueT = TypeVar("_ValueT")
_SourceT_co = TypeVar("_SourceT_co", covariant=True)
_ProducerT = TypeVar("_ProducerT")

class CutlassTensorSample(Protocol):
    """Describe the attributes read from a mutable CuTe tensor.

    Each operation checks whether register or global memory is required.
    """

    @property
    def element_type(self) -> object: ...
    @property
    def shape(self) -> object: ...
    @property
    def memspace(self) -> object: ...
    def __getitem__(self, index: int, /) -> Any: ...
    def load(self) -> object: ...

class CutlassTensorSSASample(Protocol):
    """Describe an immutable register value for payload conversion."""

    @property
    def dtype(self) -> object: ...
    @property
    def shape(self) -> object: ...
    def __getitem__(self, index: int, /) -> Any: ...
    def ir_value(self) -> object: ...

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

class ThreadDataLoadSource(Protocol[_ProducerT]):
    """Producer of a payload whose static extent can be inferred."""

    def __cuda_coop_thread_data_load__(
        self,
    ) -> (
        ThreadData[_ProducerT]
        | _ThreadDataStaticallySizedVectorSource[_ProducerT]
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
    ThreadDataLoadSource[_ProducerT],
    ThreadDataTensorMetadata,
    Protocol[_ProducerT],
):
    """Producer supporting both register loading and tensor reconstruction."""

class ThreadData(ThreadDataLike[_ItemT], Generic[_ItemT]):
    """Fixed-size per-thread payload with qualified CuTe conversions."""

    items_per_thread: int
    dtype: object | None
    alignment: int | None

    def __init__(
        self,
        items_per_thread: int,
        dtype: type[_ItemT] | None = None,
        *,
        values: tuple[_ItemT, ...] | list[_ItemT] | None = None,
        alignment: int | None = None,
    ) -> None: ...
    @classmethod
    def from_values(
        cls,
        first: _ValueT,
        *rest: _ValueT,
        dtype: type[Any] | None = None,
    ) -> ThreadData[_ValueT]: ...
    @overload
    @classmethod
    def from_fn(
        cls,
        items_per_thread: int,
        fn: Callable[[int], object],
        *,
        dtype: type[_ValueT],
    ) -> ThreadData[_ValueT]: ...
    @overload
    @classmethod
    def from_fn(
        cls,
        items_per_thread: int,
        fn: Callable[[int], _ValueT],
        *,
        dtype: None = None,
    ) -> ThreadData[_ValueT]: ...
    @overload
    @classmethod
    def from_register_tensor(
        cls,
        fragment: CutlassTensorSample,
        *,
        items_per_thread: int | None = None,
        dtype: type[_ValueT],
    ) -> ThreadData[_ValueT]: ...
    @overload
    @classmethod
    def from_register_tensor(
        cls,
        fragment: CutlassTensorSample,
        *,
        items_per_thread: int | None = None,
        dtype: None = None,
    ) -> ThreadData[Any]: ...
    @overload
    @classmethod
    def from_vector(
        cls,
        vector: object,
        *,
        items_per_thread: int | None = None,
        dtype: type[_ValueT],
    ) -> ThreadData[_ValueT]: ...
    @overload
    @classmethod
    def from_vector(
        cls,
        vector: object,
        *,
        items_per_thread: int | None = None,
        dtype: None = None,
    ) -> ThreadData[Any]: ...
    @overload
    @classmethod
    def from_payload(
        cls,
        payload: ThreadData[_ValueT],
        *,
        items_per_thread: int | None = None,
        dtype: None = None,
    ) -> ThreadData[_ValueT]: ...
    @overload
    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        items_per_thread: int | None = None,
        dtype: type[_ValueT],
    ) -> ThreadData[_ValueT]: ...
    @overload
    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        items_per_thread: int | None = None,
        dtype: None = None,
    ) -> ThreadData[Any]: ...
    @overload
    @classmethod
    def load(
        cls,
        source: ThreadDataLoadSource[Any],
        *,
        items_per_thread: int | None = None,
        dtype: type[_ValueT],
    ) -> ThreadData[_ValueT]:
        """Load and cast a statically sized producer payload to ``dtype``."""

    @overload
    @classmethod
    def load(
        cls,
        source: _ThreadDataIndexableLoadSource[Any],
        *,
        items_per_thread: int,
        dtype: type[_ValueT],
    ) -> ThreadData[_ValueT]:
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
        dtype: object | None = None,
        shape: object | None = None,
        like: ThreadDataTensorMetadata | None = None,
    ) -> CutlassTensorSSASample: ...
    def to_register_tensor(
        self,
        *,
        dtype: object | None = None,
        shape: object | None = None,
    ) -> CutlassTensorSample: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> _ItemT: ...
    def __setitem__(self, index: int, value: _ItemT) -> None: ...
    def __iter__(self) -> Iterator[_ItemT]: ...
    def __copy__(self) -> ThreadData[_ItemT]: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> ThreadData[_ItemT]: ...
    def values(self, primitive_name: str) -> tuple[_ItemT, ...]: ...
