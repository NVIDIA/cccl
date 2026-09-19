# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common explicit temporary-storage construction.

This frontend delegates caller-selected size, alignment, synchronization, and
sharing controls to the active backend. Allocation layout and reuse barriers
remain backend compiler responsibilities.
"""

from __future__ import annotations

from typing import Any

from ._dispatch import _backend_member
from ._payload import TempStorageLike, _normalize_alignment


def TempStorage(
    size_in_bytes: Any = None,
    *,
    alignment: int | None = None,
    auto_sync: Any = None,
    sharing: str = "shared",
) -> TempStorageLike:
    """Describe shared scratch for supported cooperative block operations.

    Construct the descriptor inside the kernel and pass it as
    ``temp_storage`` to operations that accept explicit block scratch.
    See :ref:`temporary storage <coop-temp-storage>` for supported operations,
    allocation lifetime, and launch-time shared-memory requirements.

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
        ``None`` and ``True`` enable automatic reuse synchronization.
        ``False`` requires the caller to synchronize before the scratch is
        reused, including on the next iteration of a loop.
    sharing : {"shared", "exclusive"}, optional
        Compile-time allocation policy, default ``"shared"``. Shared call
        sites can reuse one scratch slice. ``"exclusive"`` gives distinct
        call sites separate slices, which may consume more shared memory.
        It does not disable automatic synchronization: repeated executions
        of a single call site still reuse its slice.

    Returns
    -------
    cuda.coop.TempStorageLike
        Compiler-recognized scratch descriptor. The storage contents are
        opaque; keep application data in :func:`cuda.coop.ThreadData` or
        application-owned arrays.

    Examples
    --------
    Reuse one descriptor for transpose Load, Scan, and transpose Store.
    The loop processes two independent tiles. Automatic barriers protect
    reuse between operations and between iterations.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_storage_examples.py
        :language: python
        :start-after: # temp-storage-example-begin
        :end-before: # temp-storage-example-end
        :dedent: 4
    """

    alignment = _normalize_alignment(alignment)

    return _backend_member("TempStorage")(
        size_in_bytes=size_in_bytes,
        alignment=alignment,
        auto_sync=auto_sync,
        sharing=sharing,
    )


__all__ = ["TempStorage", "TempStorageLike"]
