# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

_ItemT = TypeVar("_ItemT")


@runtime_checkable
class _ReadableThreadDataLike(Protocol[_ItemT]):
    """Readable fixed-size per-thread payload understood by supported
    backends.
    """

    items_per_thread: int
    dtype: object | None

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _ItemT: ...


@runtime_checkable
class ThreadDataLike(_ReadableThreadDataLike[_ItemT], Protocol[_ItemT]):
    """Mutable fixed-size per-thread payload understood by supported backends.

    See :ref:`per-thread payloads <coop-thread-data>` for construction, item
    access, and dtype requirements.
    """

    def __setitem__(self, index: int, value: _ItemT) -> None: ...


@runtime_checkable
class TempStorageLike(Protocol):
    """Explicit cooperative scratch descriptor understood by supported backends.

    See :ref:`temporary storage <coop-temp-storage>` for construction,
    allocation sharing, and synchronization.
    """

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool
    sharing: str


__all__ = ["TempStorageLike", "ThreadDataLike"]
