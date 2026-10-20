# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose decode markers with optional totals and relative run offsets.

When the compiler first sees one of these calls, it imports the module that
registers their rewrite rules. The qualified window API adds per-thread
total and relative-offset buffers; the bulk API adds an optional global
relative-offset array. Both can select uint32 or uint64 totals and offsets.
These functions describe compiled operations and do not expand runs in Python.
"""

from __future__ import annotations

from typing import TypeVar

import numpy

from ..._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    IntegralScalar,
    TempStorageLike,
    ThreadDataLike,
)
from .._compiler._operations import group_operation
from .._thread_group import BlockGroup
from ._marker import group_primitive_marker

_LengthT = TypeVar("_LengthT", bound=IntegralScalar)

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)


@group_operation(
    "run_length_decode",
    family_module="cuda.coop.numba_mlir._compiler._group_run_length",
)
def run_length_decode(
    group: BlockGroup,
    run_values: CommonThreadDataLike[_ItemT] | numpy.ndarray,
    run_lengths: CommonThreadDataLike[_LengthT] | numpy.ndarray,
    /,
    *,
    decoded_items_per_thread: int,
    decoded_window_offset: IntegralScalar = 0,
    total_decoded_size: ThreadDataLike[numpy.uint32]
    | ThreadDataLike[numpy.uint64]
    | numpy.ndarray
    | None = None,
    relative_offsets: ThreadDataLike[numpy.uint32]
    | ThreadDataLike[numpy.uint64]
    | numpy.ndarray
    | None = None,
    decoded_offset_dtype: object = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_ItemT]:
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

    Examples
    --------
    Decode a shifted window with per-run offsets and the full decoded size.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_run_length_examples.py
        :language: python
        :start-after: # run-length-window-example-begin
        :end-before: # run-length-window-example-end
        :dedent: 4
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
    group: BlockGroup,
    run_values: CommonThreadDataLike[_ItemT] | numpy.ndarray,
    run_lengths: CommonThreadDataLike[_LengthT] | numpy.ndarray,
    destination: object,
    /,
    *,
    decoded_items_per_thread: int,
    destination_offset: IntegralScalar = 0,
    relative_offsets: object = None,
    decoded_offset_dtype: object = None,
    temp_storage: TempStorageLike | None = None,
) -> numpy.uint32 | numpy.uint64:
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

    Examples
    --------
    Decode all runs with uint64 relative offsets and totals. Both output
    arrays preserve their margins around the decoded interval.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_run_length_examples.py
        :language: python
        :start-after: # qualified-run-length-bulk-example-begin
        :end-before: # qualified-run-length-bulk-example-end
        :dedent: 4
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
