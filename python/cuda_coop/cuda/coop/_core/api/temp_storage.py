# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe explicit scratch storage for compiler-supported block operations.

The constructor describes a scratch request inside a GPU kernel. A tracing
compiler, such as CUTLASS, runs this body, which returns the active backend's
descriptor. Other compilers recognize the call directly. Without an active
compiler environment, the call raises. It never allocates storage; the
compiler chooses shared-memory space and any requested reuse barriers from
the descriptor's uses.
"""

from __future__ import annotations

from ._dispatch import _backend_member
from ._payload import TempStorageLike, _normalize_alignment


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
    ``descriptor.reserve(num_elems, dtype, alignment=...)`` also obtains typed
    shared arrays or tensors for application or library data.
    See :ref:`temporary storage <coop-common-storage>` for supported operations,
    allocation lifetime, and launch-time shared-memory requirements.

    Parameters
    ----------
    size_in_bytes : int, optional
        Positive compile-time capacity in bytes. ``None`` lets the compiler
        determine the capacity from all uses. An explicit capacity must be
        large enough for those operations and any typed reservations;
        undersized storage is rejected.
    alignment : int, optional
        Compile-time minimum alignment in bytes, expressed as a positive
        power of two. ``None`` lets the compiler choose. The allocation
        satisfies both this request and the operations' alignment needs.
    auto_sync : bool, optional
        Whether to insert a trailing barrier after each scratch-using call.
        Defaults to ``False``; ``None`` also disables automatic reuse
        synchronization. The caller must synchronize before reusing the
        scratch, including on the next iteration of a loop. Pass ``True``
        to request automatic reuse barriers. Descriptors used with
        ``reserve()`` cannot enable automatic synchronization.
    sharing : {"shared", "exclusive"}, optional
        Compile-time allocation policy, default ``"shared"``. Calls using
        the same descriptor reuse one scratch slice, including reservations.
        ``"exclusive"`` gives distinct primitive and reservation call sites
        separate slices, using more shared memory
        to avoid barriers needed solely for cross-call scratch reuse when
        ``auto_sync=False``. Repeated execution of one call site still reuses
        its slice and requires synchronization before reuse. Omitting
        ``temp_storage`` leaves layout and reuse barriers to the compiler.
        Different descriptors always have separate storage.

    Returns
    -------
    cuda.coop.TempStorageLike
        Compiler-recognized scratch descriptor. Primitive scratch is opaque;
        ``reserve()`` returns typed shared views that follow the descriptor's
        sharing policy. Use separate descriptors or ``sharing="exclusive"``
        for buffers whose contents must remain live simultaneously. The caller
        synchronizes their accesses and completes asynchronous operations
        before reusing shared regions.

    Examples
    --------
    Reuse one descriptor for transpose Load, Scan, and transpose Store.
    The loop processes two independent tiles. Explicit ``auto_sync=True``
    enables barriers between operations and between iterations.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_storage_examples.py
        :language: python
        :start-after: # temp-storage-example-begin
        :end-before: # temp-storage-example-end
        :dedent: 4

    The :ref:`CUTLASS storage example <coop-cutlass-storage>` demonstrates
    the same shared/exclusive policies and automatic or explicit reuse
    synchronization in a CuTe kernel.
    """

    alignment = _normalize_alignment(alignment)

    return _backend_member("TempStorage")(
        size_in_bytes=size_in_bytes,
        alignment=alignment,
        auto_sync=auto_sync,
        sharing=sharing,
    )


__all__ = ["TempStorage", "TempStorageLike"]
