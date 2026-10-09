# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Define the payload and scratch interfaces used by common API annotations.

Each compiler backend supplies concrete values with these attributes and
operations. The backend must also recognize those values during compilation.
This lets the common API describe inputs without importing compiler types.
"""

from __future__ import annotations

import operator
from typing import Any, Protocol, SupportsIndex, TypeVar, runtime_checkable

_ItemT = TypeVar("_ItemT")


@runtime_checkable
class _ReadableThreadDataLike(Protocol[_ItemT]):
    """Describe a fixed number of readable values owned by one thread.

    ``items_per_thread`` is the fixed extent. ``dtype`` can remain unknown
    until a supported producer establishes the element type. The compiler
    backend supplies the storage and indexed access.
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
    """Explicit scratch descriptor understood by supported backends.

    See :ref:`temporary storage <coop-temp-storage>` for construction,
    allocation sharing, and synchronization.
    """

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool
    sharing: str

    def reserve(
        self, num_elems: int, dtype: object, *, alignment: int | None = None
    ) -> Any:
        """Return a typed array that follows the descriptor's sharing policy."""
        ...


def _normalize_alignment(alignment: SupportsIndex | None) -> int | None:
    """Normalize a payload or scratch alignment request to bytes.

    Preserve ``None`` so the compiler can choose the alignment. Explicit
    requests must be positive powers of two. Accept integer-like values
    through ``__index__``, but reject booleans as accidental requests.
    """

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


__all__ = [
    "TempStorageLike",
    "ThreadDataLike",
    "_ReadableThreadDataLike",
    "_normalize_alignment",
]
