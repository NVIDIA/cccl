# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed physical-warp Numba-CUDA-MLIR cooperative primitives."""

from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, overload

from ..._typing import LoadStoreAlgorithm, ThreadDataLike
from ..._typing import _ValidItems as _ValidItems
from .. import TempStorage, WarpLoadAlgorithm, WarpStoreAlgorithm

_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")
_Methods = Mapping[str, object]

class _CompilerDTypeToken(Protocol):
    """Structural Numba-CUDA-MLIR dtype token used by factory overloads."""

    name: str

_DTypeToken = type[object] | _CompilerDTypeToken

class _Invocable(Protocol):
    """Metadata shared by generated Numba-CUDA-MLIR device callables."""

    temp_storage_bytes: int
    temp_storage_alignment: int
    files: list[str]

class _LoadInvocable(_Invocable, Protocol[_T]):
    """Generated physical-warp load callable."""

    @overload
    def __call__(
        self,
        source: object,
        output: ThreadDataLike[_T],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Load one physical-warp tile into ``output``."""
    @overload
    def __call__(
        self,
        source: object,
        output: ThreadDataLike[_T],
        offset_or_valid_items: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Load an offset or partial physical-warp tile."""
    @overload
    def __call__(
        self,
        source: object,
        output: ThreadDataLike[_T],
        first: object,
        second: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Load with a valid count plus an OOB value or offset."""
    @overload
    def __call__(
        self,
        source: object,
        output: ThreadDataLike[_T],
        valid_items: object,
        oob_default: _T,
        offset: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Load a partial offset tile and fill invalid items."""

class _StoreInvocable(_Invocable, Protocol[_T]):
    """Generated physical-warp store callable."""

    @overload
    def __call__(
        self,
        destination: object,
        value: ThreadDataLike[_T],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Store one physical-warp tile from ``value``."""
    @overload
    def __call__(
        self,
        destination: object,
        value: ThreadDataLike[_T],
        offset_or_valid_items: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Store an offset or partial physical-warp tile."""
    @overload
    def __call__(
        self,
        destination: object,
        value: ThreadDataLike[_T],
        valid_items: object,
        offset: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Store a partial physical-warp tile at ``offset``."""

class _ReduceInvocable(_Invocable, Protocol[_T]):
    """Generated physical-warp reduction callable."""

    @overload
    def __call__(
        self,
        value: _T,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> _T:
        """Reduce one lane value."""
    @overload
    def __call__(
        self,
        value: _T,
        valid_items: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> _T:
        """Reduce the valid prefix of physical-warp lanes."""

class _ScanInvocable(_Invocable, Protocol[_T]):
    """Generated physical-warp scan callable."""

    @overload
    def __call__(
        self,
        value: _T,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> _T:
        """Return one lane's prefix value."""
    @overload
    def __call__(
        self,
        value: _T,
        valid_items_or_aggregate: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> _T:
        """Return a prefix with one enabled runtime operand."""
    @overload
    def __call__(
        self,
        value: _T,
        valid_items: object,
        warp_aggregate: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> _T:
        """Return a partial prefix and write the physical-warp aggregate."""

class _TransformInvocable(_Invocable, Protocol[_T]):
    """Generated one-payload physical-warp transform."""

    def __call__(
        self,
        value: ThreadDataLike[_T],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Transform ``value`` in place."""

class _OutTransformInvocable(_Invocable, Protocol[_T]):
    """Generated out-of-place physical-warp transform."""

    def __call__(
        self,
        value: ThreadDataLike[_T],
        output: ThreadDataLike[_T],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Transform ``value`` into ``output``."""

class _PairTransformInvocable(_Invocable, Protocol[_K, _V]):
    """Generated physical-warp key-value transform."""

    def __call__(
        self,
        keys: ThreadDataLike[_K],
        values: ThreadDataLike[_V],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Transform key-value payloads in place."""

class _ExchangeInvocable(_Invocable, Protocol[_T]):
    """Generated physical-warp exchange callable."""

    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        output_or_ranks: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Exchange into ``output`` or scatter in place by rank."""
    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        output: ThreadDataLike[_T],
        ranks: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Scatter ``value`` into ``output`` according to ``ranks``."""

class _MergeSortInvocable(_Invocable, Protocol[_K, _V]):
    """Generated keys-only or key-value physical-warp merge sort."""

    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_K],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Sort keys in place."""
    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_K],
        valid_items: object,
        oob_default: _K,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Sort a partial key tile in place."""
    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_K],
        values: ThreadDataLike[_V],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Sort keys and associated values in place."""
    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_K],
        values: ThreadDataLike[_V],
        valid_items: object,
        oob_default: _K,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Sort a partial key-value tile in place."""

@overload
def load(
    dtype: type[_T],
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: LoadStoreAlgorithm | WarpLoadAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: _T | None = None,
    methods: _Methods | None = None,
    threads_per_block: int | tuple[int, ...] | None = None,
) -> _LoadInvocable[_T]:
    """Build a dtype-preserving physical-warp load callable."""

@overload
def load(
    source: object,
    output: ThreadDataLike[_T],
    /,
    *,
    algorithm: LoadStoreAlgorithm | WarpLoadAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    offset: object = None,
    items_per_thread: int | None = None,
    dtype: object = None,
    threads_in_warp: int = 32,
    threads_per_block: int | tuple[int, ...] | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Load a physical-warp tile into an existing payload."""

@overload
def load(
    source: object,
    output: ThreadDataLike[_T],
    offset_or_valid_items: object,
    /,
    *,
    algorithm: LoadStoreAlgorithm | WarpLoadAlgorithm = "direct",
    dtype: object = None,
    items_per_thread: int | None = None,
    threads_in_warp: int = 32,
    threads_per_block: int | tuple[int, ...] | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Load an offset or partial physical-warp tile."""

@overload
def load(
    source: object,
    output: ThreadDataLike[_T],
    valid_items: object,
    oob_default: object,
    /,
    *,
    algorithm: LoadStoreAlgorithm | WarpLoadAlgorithm = "direct",
    dtype: object = None,
    items_per_thread: int | None = None,
    threads_in_warp: int = 32,
    threads_per_block: int | tuple[int, ...] | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Load a partial physical-warp tile and fill invalid items."""

@overload
def load(
    dtype: object,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: LoadStoreAlgorithm | WarpLoadAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    methods: _Methods | None = None,
    threads_per_block: int | tuple[int, ...] | None = None,
) -> _LoadInvocable[Any]:
    """Build a physical-warp load callable from a compiler dtype token."""

@overload
def store(
    dtype: type[_T],
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: LoadStoreAlgorithm | WarpStoreAlgorithm = "direct",
    num_valid_items: object = None,
    methods: _Methods | None = None,
    threads_per_block: int | tuple[int, ...] | None = None,
) -> _StoreInvocable[_T]:
    """Build a dtype-preserving physical-warp store callable."""

@overload
def store(
    destination: object,
    value: ThreadDataLike[_T],
    /,
    *,
    algorithm: LoadStoreAlgorithm | WarpStoreAlgorithm = "direct",
    num_valid_items: object = None,
    offset: object = None,
    items_per_thread: int | None = None,
    dtype: object = None,
    threads_in_warp: int = 32,
    threads_per_block: int | tuple[int, ...] | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Store an existing payload to a physical-warp tile."""

@overload
def store(
    destination: object,
    value: ThreadDataLike[_T],
    offset_or_valid_items: object,
    /,
    *,
    algorithm: LoadStoreAlgorithm | WarpStoreAlgorithm = "direct",
    dtype: object = None,
    items_per_thread: int | None = None,
    threads_in_warp: int = 32,
    threads_per_block: int | tuple[int, ...] | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Store an offset or partial physical-warp tile."""

@overload
def store(
    dtype: object,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: LoadStoreAlgorithm | WarpStoreAlgorithm = "direct",
    num_valid_items: object = None,
    methods: _Methods | None = None,
    threads_per_block: int | tuple[int, ...] | None = None,
) -> _StoreInvocable[Any]:
    """Build a physical-warp store callable from a compiler dtype token."""

@overload
def warp_load(
    dtype: type[_T],
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: LoadStoreAlgorithm | WarpLoadAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    methods: _Methods | None = None,
    threads_per_block: int | tuple[int, ...] | None = None,
) -> _LoadInvocable[_T]:
    """Build a dtype-preserving physical-warp load callable."""

@overload
def warp_load(
    dtype: object,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: LoadStoreAlgorithm | WarpLoadAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    methods: _Methods | None = None,
    threads_per_block: int | tuple[int, ...] | None = None,
) -> _LoadInvocable[Any]:
    """Build a physical-warp load callable from a compiler dtype token."""

@overload
def make_load(
    dtype: type[_T],
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: LoadStoreAlgorithm | WarpLoadAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    methods: _Methods | None = None,
    threads_per_block: int | tuple[int, ...] | None = None,
) -> _LoadInvocable[_T]:
    """Build a dtype-preserving physical-warp load callable."""

@overload
def make_load(
    dtype: object,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: LoadStoreAlgorithm | WarpLoadAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    methods: _Methods | None = None,
    threads_per_block: int | tuple[int, ...] | None = None,
) -> _LoadInvocable[Any]:
    """Build a physical-warp load callable from a compiler dtype token."""

@overload
def warp_store(
    dtype: type[_T],
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: LoadStoreAlgorithm | WarpStoreAlgorithm = "direct",
    num_valid_items: object = None,
    methods: _Methods | None = None,
    threads_per_block: int | tuple[int, ...] | None = None,
) -> _StoreInvocable[_T]:
    """Build a dtype-preserving physical-warp store callable."""

@overload
def warp_store(
    dtype: object,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: LoadStoreAlgorithm | WarpStoreAlgorithm = "direct",
    num_valid_items: object = None,
    methods: _Methods | None = None,
    threads_per_block: int | tuple[int, ...] | None = None,
) -> _StoreInvocable[Any]:
    """Build a physical-warp store callable from a compiler dtype token."""

@overload
def make_store(
    dtype: type[_T],
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: LoadStoreAlgorithm | WarpStoreAlgorithm = "direct",
    num_valid_items: object = None,
    methods: _Methods | None = None,
    threads_per_block: int | tuple[int, ...] | None = None,
) -> _StoreInvocable[_T]:
    """Build a dtype-preserving physical-warp store callable."""

@overload
def make_store(
    dtype: object,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: LoadStoreAlgorithm | WarpStoreAlgorithm = "direct",
    num_valid_items: object = None,
    methods: _Methods | None = None,
    threads_per_block: int | tuple[int, ...] | None = None,
) -> _StoreInvocable[Any]:
    """Build a physical-warp store callable from a compiler dtype token."""

__all__ = [
    "load",
    "make_load",
    "make_store",
    "store",
    "warp_load",
    "warp_store",
]
