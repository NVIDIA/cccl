# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common per-thread payload construction.

ThreadData is a compiler-owned fixed-size value container; this frontend only
forwards its static extent, optional dtype, and alignment to the active backend.
Primitive payload validation lives in the family frontends and shared helpers.
"""

from __future__ import annotations

from typing import Any

from ._dispatch import _backend_member, _common_root_operation_scope
from ._payload import ThreadDataLike, _normalize_alignment


def ThreadData(
    items_per_thread: int,
    dtype: Any = None,
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

    Examples
    --------
    Construct two items per thread, fill them with squared indices, and
    store the resulting blocked tiles:

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_storage_examples.py
        :language: python
        :start-after: # thread-data-example-begin
        :end-before: # thread-data-example-end
        :dedent: 4

    For per-thread payloads in CuTe kernels, see
    :ref:`CUTLASS Load and Store <coop-cutlass-load-store>`. The qualified
    :class:`cuda.coop.cutlass.ThreadData` also converts CuTe register values.
    """

    alignment = _normalize_alignment(alignment)

    with _common_root_operation_scope("ThreadData"):
        return _backend_member("ThreadData")(
            items_per_thread, dtype=dtype, alignment=alignment
        )


__all__ = ["ThreadData", "ThreadDataLike"]
