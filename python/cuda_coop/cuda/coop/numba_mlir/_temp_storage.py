# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe scratch allocation and reuse for the Numba-CUDA-MLIR planner.

The descriptor validates host-side options and retains them for compilation.
Constructing it does not allocate shared memory or synchronize threads.
"""

import operator
from enum import Enum

from .._core.api._payload import _normalize_alignment


class TempStorage:
    """Shared-memory requirements for cooperative operations in one kernel.

    Parameters, defaults, synchronization rules, and the executable reuse
    example follow :func:`cuda.coop.TempStorage`. This qualified descriptor
    exposes ``size_in_bytes``, ``alignment``, ``auto_sync``, and ``sharing``
    for the Numba-CUDA-MLIR planner. ``auto_sync=None`` becomes ``False``.
    Set ``auto_sync=True`` to request automatic reuse barriers.

    Only supported block algorithms accept an explicit descriptor. The
    planner determines capacity and alignment from its uses. ``release()``
    hands the complete region to user code after its final cooperative use.
    See :ref:`temporary storage <coop-temp-storage>`
    for shared versus exclusive slices and manual reuse synchronization.
    """

    def __init__(
        self,
        size_in_bytes=None,
        *,
        alignment=None,
        auto_sync=False,
        sharing="shared",
    ):
        if size_in_bytes is not None:
            if isinstance(size_in_bytes, bool):
                raise TypeError(
                    "TempStorage size_in_bytes must be an integer or None."
                )
            try:
                size_in_bytes = operator.index(size_in_bytes)
            except TypeError as exc:
                raise TypeError(
                    "TempStorage size_in_bytes must be an integer or None."
                ) from exc
            if size_in_bytes <= 0:
                raise ValueError(
                    "TempStorage size_in_bytes must be a positive integer."
                )

        alignment = _normalize_alignment(alignment)

        if not isinstance(sharing, str) or isinstance(sharing, Enum):
            raise TypeError(
                "TempStorage sharing must be a string: 'shared' or 'exclusive'."
            )
        sharing_value = sharing.strip().lower()
        if sharing_value not in {"shared", "exclusive"}:
            raise ValueError(
                "TempStorage sharing must be 'shared' or 'exclusive'."
            )

        if auto_sync is not None and not isinstance(auto_sync, bool):
            raise TypeError("TempStorage auto_sync must be None/True/False.")

        self.size_in_bytes = size_in_bytes
        self.alignment = alignment
        self.sharing = sharing_value
        # Sharing selects the slice layout; synchronization is independent.
        # The caller synchronizes before reuse unless auto_sync=True.
        self.auto_sync = False if auto_sync is None else auto_sync

    def release(self, dtype=None):
        """Consume this descriptor and return its complete shared-memory region.

        ``dtype=None`` returns a one-dimensional byte array. A supplied
        compile-time numeric dtype must divide the region's byte capacity and
        satisfy its existing alignment. Release does not initialize the returned
        memory, allocate memory, or synchronize.
        Finish previous accesses and required asynchronous operations before
        using it. Automatic barriers on preceding primitive calls are retained.

        Every alias of this descriptor is consumed. A subsequent primitive or
        release on any reachable control-flow path is a compile error,
        including reuse through a loop backedge. A new construction in
        a loop creates a new instance but reuses the same physical region, so
        the caller must finish accesses and synchronize before that reuse.
        """
        from .._core.thread_group import CoopCompilerContextRequiredError

        raise CoopCompilerContextRequiredError(
            "TempStorage.release must be called from a supported GPU kernel."
        )


__all__ = ["TempStorage"]
