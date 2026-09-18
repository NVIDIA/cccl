# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for CUTLASS per-thread register payloads."""

from collections.abc import Callable, Iterator
from typing import Any, Generic, Protocol, overload

from typing_extensions import TypeVar

from .._typing import ThreadDataLike

_ItemT = TypeVar("_ItemT", default=Any)
_ValueT = TypeVar("_ValueT")

class CutlassTensorSample(Protocol):
    """Structural view of a CuTe register-memory tensor."""

    @property
    def element_type(self) -> object: ...
    @property
    def shape(self) -> object: ...
    @property
    def memspace(self) -> object: ...
    def __getitem__(self, index: int, /) -> Any: ...
    def load(self) -> object: ...

class CutlassTensorSSASample(Protocol):
    """Structural view of an immutable CuTe register tensor."""

    @property
    def dtype(self) -> object: ...
    @property
    def shape(self) -> object: ...
    def __getitem__(self, index: int, /) -> Any: ...
    def ir_value(self) -> object: ...

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
    def to_tensor_ssa(
        self,
        *,
        dtype: object | None = None,
        shape: object | None = None,
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
