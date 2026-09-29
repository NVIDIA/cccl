# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Per-thread payload and scratch-storage interfaces shared by compilers."""

from __future__ import annotations

import operator
from typing import Protocol, SupportsIndex, TypeVar, runtime_checkable

_ItemT = TypeVar("_ItemT")


@runtime_checkable
class _ReadableThreadDataLike(Protocol[_ItemT]):
    """Readable fixed-size per-thread payload understood by supported backends."""

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


def _normalize_alignment(alignment: SupportsIndex | None) -> int | None:
    if alignment is None:
        return None
    if isinstance(alignment, bool):
        raise TypeError("alignment must be an integer or None")
    try:
        alignment = operator.index(alignment)
    except TypeError as exc:
        raise TypeError("alignment must be an integer or None") from exc
    if alignment <= 0:
        raise ValueError("alignment must be a positive integer")
    if alignment & (alignment - 1):
        raise ValueError("alignment must be a power of 2")
    return alignment


__all__ = ["TempStorageLike", "ThreadDataLike"]
