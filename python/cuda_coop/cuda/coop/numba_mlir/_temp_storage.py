# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe scratch allocation and reuse for the Numba-CUDA-MLIR planner.

The descriptor validates host-side options and retains them for compilation.
Constructing it does not allocate shared memory or synchronize threads.
"""

from enum import Enum

from .._core.api._payload import _normalize_alignment
from .._core.thread_group import CoopCompilerContextRequiredError


class TempStorage:
    """Shared-memory requirements for cooperative operations in one kernel.

    Parameters, defaults, synchronization rules, and the executable reuse
    example follow :func:`cuda.coop.TempStorage`. This qualified descriptor
    exposes ``size_in_bytes``, ``alignment``, ``auto_sync``, and ``sharing``
    for the Numba-CUDA-MLIR planner. ``auto_sync=None`` becomes ``False``.
    Set ``auto_sync=True`` to request automatic reuse barriers.

    Supported block algorithms accept the descriptor as ``temp_storage``.
    :meth:`reserve` obtains typed shared arrays for application or library
    data. The planner includes both uses when determining capacity and
    alignment. See :ref:`temporary storage <coop-temp-storage>` for shared
    versus exclusive slices and manual reuse synchronization.
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
            if not isinstance(size_in_bytes, int) or isinstance(
                size_in_bytes, bool
            ):
                raise TypeError(
                    "TempStorage size_in_bytes must be an integer or None."
                )
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

    def reserve(self, num_elems, dtype, *, alignment=None):
        """Reserve a contiguous one-dimensional shared array in the kernel.

        Parameters
        ----------
        num_elems : int
            Positive compile-time number of elements.
        dtype
            Compile-time fixed-size integer, floating-point, or complex
            scalar dtype.
        alignment : int, optional
            Minimum byte alignment, expressed as a positive power of two.
            The planner also satisfies the element type's alignment.

        Returns
        -------
        shared array
            An ordinary compiler array with the requested dtype and extent.
            Its elements are uninitialized and shared by the block's threads.

        Notes
        -----
        Each reservation call site receives a disjoint region that remains
        allocated for the kernel's execution on that block. It does not
        overlap another reservation or the descriptor's primitive scratch,
        regardless of ``sharing``. Repeated execution of a call site returns
        the same region; it does not allocate again.

        A descriptor used for reservations must have ``auto_sync=False``
        (or ``None``). The caller supplies synchronization and any completion
        or release operations required by consumers. Reserving storage does
        not infer when a foreign library has finished using it.
        """
        raise CoopCompilerContextRequiredError(
            "TempStorage.reserve must be called from a supported GPU kernel."
        )


__all__ = ["TempStorage"]
