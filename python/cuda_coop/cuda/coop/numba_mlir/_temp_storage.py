# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe explicit shared-memory storage for cooperative operations."""

from enum import Enum

from .._core.api._payload import _normalize_alignment


class TempStorage:
    """Shared-memory requirements for cooperative operations in one kernel.

    Parameters, defaults, synchronization rules, and the executable reuse
    example follow :func:`cuda.coop.TempStorage`. This qualified descriptor
    exposes ``size_in_bytes``, ``alignment``, ``auto_sync``, and ``sharing``
    for the Numba-CUDA-MLIR planner. ``auto_sync=None`` becomes ``True``.

    Only supported block algorithms accept an explicit descriptor. The
    planner determines capacity and alignment from its uses; its contents
    are opaque to user code. See :ref:`temporary storage <coop-temp-storage>`
    for shared versus exclusive slices and manual reuse synchronization.
    """

    def __init__(
        self,
        size_in_bytes=None,
        *,
        alignment=None,
        auto_sync=None,
        sharing="shared",
    ):
        if size_in_bytes is not None:
            if not isinstance(size_in_bytes, int) or isinstance(size_in_bytes, bool):
                raise TypeError("TempStorage size_in_bytes must be an integer or None.")
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
            raise ValueError("TempStorage sharing must be 'shared' or 'exclusive'.")

        if auto_sync is not None and not isinstance(auto_sync, bool):
            raise TypeError("TempStorage auto_sync must be None/True/False.")

        self.size_in_bytes = size_in_bytes
        self.alignment = alignment
        self.sharing = sharing_value
        # Sharing selects the slice layout; synchronization is independent.
        # A call site inside a loop reuses its slice under either layout, so
        # the trailing reuse barrier stays on unless the caller opts out.
        self.auto_sync = True if auto_sync is None else auto_sync


__all__ = ["TempStorage"]
