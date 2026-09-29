# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declare shared-memory requirements for the kernel compiler."""

from __future__ import annotations

from ..thread_group import CoopCompilerContextRequiredError
from ._payload import TempStorageLike


def TempStorage(
    size_in_bytes: int | None = None,
    *,
    alignment: int | None = None,
    auto_sync: bool | None = False,
    sharing: str = "shared",
) -> TempStorageLike:
    """Describe shared scratch for supported cooperative block operations.

    Construct the descriptor inside the kernel and pass it as
    ``temp_storage`` to operations that accept explicit block scratch.
    See :ref:`temporary storage <coop-temp-storage>` for supported operations,
    allocation lifetime, and synchronization requirements.

    Parameters
    ----------
    size_in_bytes : int, optional
        Positive compile-time capacity in bytes. ``None`` lets the compiler
        determine the capacity from all uses. An explicit capacity must be
        large enough for those operations; undersized storage is rejected.
    alignment : int, optional
        Compile-time minimum alignment in bytes, expressed as a positive
        power of two. ``None`` lets the compiler choose. The allocation
        satisfies both this request and the operations' alignment needs.
    auto_sync : bool, optional
        Whether to insert a trailing barrier after each scratch-using call.
        Defaults to ``False``; ``None`` also disables automatic reuse
        synchronization. The caller must synchronize before reusing the
        scratch, including on the next iteration of a loop. Pass ``True``
        to request automatic reuse barriers.
    sharing : {"shared", "exclusive"}, optional
        Compile-time allocation policy, default ``"shared"``. Shared call
        sites can reuse one scratch slice. ``"exclusive"`` gives distinct
        call sites separate slices, which may consume more shared memory.
        Synchronization is independent: repeated executions of a single
        call site still reuse its slice.

    Returns
    -------
    cuda.coop.TempStorageLike
        Compiler-recognized scratch descriptor. The storage contents are
        opaque; keep application data in :func:`cuda.coop.ThreadData` or
        application-owned arrays.
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.TempStorage must be called from a supported GPU kernel."
    )


__all__ = ["TempStorage", "TempStorageLike"]
