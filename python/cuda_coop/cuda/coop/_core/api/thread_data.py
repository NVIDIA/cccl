# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from ..thread_group import CoopCompilerContextRequiredError
from ._payload import ThreadDataLike


def ThreadData(
    items_per_thread: int,
    dtype: object = None,
    *,
    alignment: int | None = None,
) -> ThreadDataLike[Any]:
    """Construct a fixed-size payload owned by the calling thread.

    Each thread has its own slots. See :ref:`per-thread payloads
    <coop-thread-data>` for their relationship to a group tile and the
    :ref:`blocked and striped layouts <coop-data-layouts>`.

    Parameters
    ----------
    items_per_thread : int
        Positive compile-time number of items owned by each thread.
        This extent is fixed for the lifetime of the payload and is available
        inside the kernel as ``items.items_per_thread``.
    dtype : dtype-like, optional
        Numeric element dtype, for example ``numpy.int32``. ``None`` lets
        the compiler infer it from a supported producer such as
        :func:`cuda.coop.load`. All items have the same dtype.
    alignment : int, optional
        Compile-time minimum storage alignment in bytes, expressed as a
        positive power of two. ``None`` lets the compiler choose. The request
        applies when payload storage is materialized; it does not assert
        alignment of a Load source or Store destination.

    Returns
    -------
    cuda.coop.ThreadDataLike
        Writable per-thread payload with indexed reads and writes. Its
        contents are uninitialized; write every item before reading it.
        Construction does not synchronize threads.
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.ThreadData must be called from a supported GPU kernel."
    )


__all__ = ["ThreadData", "ThreadDataLike"]
