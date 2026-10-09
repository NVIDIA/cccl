# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe scratch reuse while CuTe traces a kernel.

The descriptor holds compile-time policy and identity, not a device pointer.
Finalization allocates storage after C++ layout probes establish its size.
The MLIR hooks expose no values and rebuild to the same object, so traced
loops keep the descriptor identity used to group storage requests.
"""

from enum import Enum

from .._core.api._payload import _normalize_alignment


class TempStorage:
    """Describe shared-memory requirements for operations in one kernel.

    Parameters, defaults, synchronization rules, and the executable reuse
    example follow :func:`cuda.coop.TempStorage`. Only supported block
    algorithms accept an explicit descriptor. ``reserve`` exposes typed arrays
    for application and library code. See
    :ref:`temporary storage <coop-common-storage>` for allocation lifetime
    and :ref:`the CUTLASS example <coop-cutlass-storage>` for reuse.

    Attributes
    ----------
    size_in_bytes : int or None
        Explicit positive capacity, or ``None`` to infer it from all uses,
        including reserved arrays.
        An explicit capacity must cover the resolved allocation.
    alignment : int or None
        Optional minimum byte alignment, expressed as a power of two. The
        planner also satisfies the C++ primitive's alignment requirement.
    auto_sync : bool
        Whether each scratch-using call emits a trailing block barrier.
        Defaults to ``False``; ``None`` is normalized to ``False``. Without
        automatic barriers, the caller must synchronize before scratch reuse.
    sharing : {"shared", "exclusive"}
        Shared calls and reservations alias one slice sized for the largest
        requirement. Exclusive calls and reservations get separate slices
        in trace order. Repeated
        execution of one call site reuses its slice, including in a loop.

    Equal settings do not combine separate descriptors. Reuse the same object
    to share an allocation across calls in a kernel. The descriptor carries
    no runtime MLIR values. Finalization supplies the storage operands.
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

    def __extract_mlir_values__(self):
        # The descriptor carries compile-time identity, including through loops.
        # Its storage pointer is supplied by finalization at each call site.
        return []

    def __new_from_mlir_values__(self, values):
        if values:
            raise ValueError("TempStorage has no runtime MLIR values")
        return self

    @property
    def capacity_size_in_bytes(self):
        """Return explicit capacity without resolving an inferred size."""
        return self.size_in_bytes

    @property
    def is_deferred(self):
        """Indicate that layout resolution waits for trace finalization."""
        return True

    def sync(self):
        """Synchronize the block before manually reusing temporary storage."""
        from cutlass.cute.arch import sync_threads

        sync_threads()

    def reserve(self, num_elems, dtype, *, alignment=None):
        """Return a one-dimensional CuTe shared tensor with a fixed extent.

        ``num_elems``, ``dtype``, and optional byte ``alignment`` must be
        compile-time constants. Supported dtypes are 8/16/32/64-bit signed
        and unsigned integers, Float16, BFloat16, Float32, and Float64.
        Ordinary NumPy scalar types and dtypes are accepted where available.

        The descriptor's ``sharing`` policy applies to both reservations
        and primitive scratch. Shared views alias the same bytes; exclusive
        views occupy separate slices. Each traced call site has one view,
        including when it executes repeatedly in a runtime loop. Separate
        descriptors always own separate allocations.

        Reservations require ``auto_sync=False``. The caller must finish
        all accesses and asynchronous library operations before reusing
        overlapping bytes, and synchronize participating threads as needed.
        ``sync()`` is a block barrier, not an asynchronous completion wait.
        """

        from ._compiler._storage import reserve_temp_storage_tensor

        return reserve_temp_storage_tensor(
            self, num_elems, dtype, alignment=alignment
        )


__all__ = ["TempStorage"]
