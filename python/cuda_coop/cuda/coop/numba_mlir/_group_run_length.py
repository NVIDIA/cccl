# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block Run Length Decode operations."""

from __future__ import annotations

from typing import Any

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation(
    "run_length_decode",
    family_module="cuda.coop.numba_mlir._compiler._group_run_length",
)
def run_length_decode(
    group: ThreadGroup,
    run_values: Any,
    run_lengths: Any,
    /,
    *,
    decoded_items_per_thread: int,
    decoded_window_offset: Any = 0,
    total_decoded_size: Any = None,
    relative_offsets: Any = None,
    decoded_offset_dtype: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Decode one window with optional per-thread totals and run offsets.

    Shared parameters, participation, ordering, zero-filled tail, input
    preservation, and scratch behavior follow
    :func:`cuda.coop.run_length_decode`. This qualified overload also accepts
    fixed-size local arrays for the run inputs and auxiliary outputs.

    Additional parameters
    ---------------------
    total_decoded_size : ThreadDataLike or local array, optional
        Mutable extent-one per-thread output. Every member receives the full
        stream's total, including when its requested window is empty. Its
        dtype must match ``decoded_offset_dtype``; an untyped ThreadData
        output adopts that dtype.
    relative_offsets : ThreadDataLike or local array, optional
        Mutable per-thread output with ``decoded_items_per_thread`` items and
        the selected offset dtype. Each valid slot receives its zero-based
        offset within its run. Invalid slots contain the dtype maximum.
    decoded_offset_dtype : dtype, optional
        uint32 by default, or uint64. Both auxiliary outputs and the decoded
        total use this dtype. The total must fit it. For uint64, the total
        must also be at most ``UINT64_MAX - block_threads *
        decoded_items_per_thread`` so CUB's internal tail indices cannot
        overflow. Validation occurs before converting wide input lengths.

    Returns
    -------
    ThreadDataLike
        The fresh value payload described by the common operation; optional
        auxiliary outputs are updated in place. Outputs must not overlap one
        another or either run input.
    """
    return group_primitive_marker(
        "run_length_decode",
        group,
        run_values,
        run_lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        decoded_window_offset=decoded_window_offset,
        total_decoded_size=total_decoded_size,
        relative_offsets=relative_offsets,
        decoded_offset_dtype=decoded_offset_dtype,
        temp_storage=temp_storage,
    )


@group_operation(
    "run_length_decode_into",
    family_module="cuda.coop.numba_mlir._compiler._group_run_length",
)
def run_length_decode_into(
    group: ThreadGroup,
    run_values: Any,
    run_lengths: Any,
    destination: Any,
    /,
    *,
    decoded_items_per_thread: int,
    destination_offset: Any = 0,
    relative_offsets: Any = None,
    decoded_offset_dtype: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Decode a full stream with optional global relative offsets.

    Shared parameters, input preservation, destination capacity checks,
    block participation, and prepared-table lifetime follow
    :func:`cuda.coop.run_length_decode_into`. Fixed-size local arrays are also
    accepted for the run inputs.

    Additional parameters
    ---------------------
    relative_offsets : array, optional
        Writable contiguous one-dimensional global array of the selected
        offset dtype. It receives the zero-based position within each run,
        using the same ``destination_offset`` as ``destination``. Every
        member must supply the same array. Both capacities are checked before
        either output is written. The output arrays must not overlap each
        other or either run input. Elements outside the decoded interval are
        preserved.
    decoded_offset_dtype : dtype, optional
        uint32 by default, or uint64. Selects the returned total and relative
        offset dtype, with the total limits documented by
        :func:`cuda.coop.numba_mlir.run_length_decode`.

    Returns
    -------
    uint32 or uint64
        The full decoded size in the selected dtype, available to every block
        member. Empty input returns zero and writes neither output array.
    """
    return group_primitive_marker(
        "run_length_decode_into",
        group,
        run_values,
        run_lengths,
        destination,
        decoded_items_per_thread=decoded_items_per_thread,
        destination_offset=destination_offset,
        relative_offsets=relative_offsets,
        decoded_offset_dtype=decoded_offset_dtype,
        temp_storage=temp_storage,
    )


__all__ = ["run_length_decode", "run_length_decode_into"]
