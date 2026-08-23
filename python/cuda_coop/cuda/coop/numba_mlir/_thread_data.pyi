# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fixed-size per-thread data and backend memory namespace declarations."""

from typing import Any, Protocol, TypeAlias, overload

from typing_extensions import TypeVar

from .._typing import ThreadDataLike as _ThreadDataLike

_ItemT = TypeVar("_ItemT")

_ArrayShape: TypeAlias = int | tuple[int, ...]

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

__all__ = ["ThreadData", "local", "shared"]
