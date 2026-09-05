# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable fused run-length-decode entry point.

This frontend enforces the shared payload extents, integer dtypes, and decode
window controls before delegation. Result contracts and the CUB driver
specialization live in the portable planner and backend lowering.
"""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _group_primitive_marker,
    _validate_portable_operation_group,
)
from ._payload import (
    _common_thread_data_extent,
    _validate_common_run_length_decode_controls,
    _validate_common_run_length_decode_dtype,
    _validate_common_thread_data_payload,
)


def run_length_decode(
    group: ThreadGroup,
    run_values: Any,
    run_lengths: Any,
    /,
    *,
    decoded_items_per_thread: Any,
    decoded_window_offset: Any = 0,
) -> Any:
    """Decode run-length values through the compiler-selected backend.

    ``group`` must be a complete physical block. ``run_values`` and
    ``run_lengths`` are matching, positive-size ``ThreadData`` payloads whose
    runs are owned in blocked order. ``decoded_items_per_thread`` is a positive
    trace-static integer. Member ``rank`` receives decoded positions
    ``decoded_window_offset + rank * decoded_items_per_thread + i``. The
    uniform window offset must be nonnegative and representable in the
    run-length dtype; callers providing a dynamic offset guarantee that range.
    Positions beyond the decoded stream are zero.

    The result is a new ``ThreadData`` payload with
    ``decoded_items_per_thread`` values per member and the run-value dtype.
    Neither input is mutated. Actual runs must have positive lengths; zero
    lengths are permitted only as one trailing padding suffix, and the
    block-wide sum must be positive and representable in the run-length dtype.
    Use the qualified
    ``cuda.coop.<backend>`` API for relative offsets, total decoded size, or
    backend-specific payloads.
    """

    if _backend_module_name() is not None:
        _validate_portable_operation_group("run_length_decode", group)
        _validate_common_thread_data_payload(
            "run_length_decode", "run_values", run_values
        )
        _validate_common_thread_data_payload(
            "run_length_decode", "run_lengths", run_lengths
        )
        value_extent = _common_thread_data_extent(
            "run_length_decode", "run_values", run_values
        )
        length_extent = _common_thread_data_extent(
            "run_length_decode", "run_lengths", run_lengths
        )
        if value_extent != length_extent:
            raise ValueError(
                "cuda.coop.run_length_decode run_values and run_lengths must "
                "have matching items_per_thread"
            )
        _validate_common_run_length_decode_dtype(
            "run_values", run_values, allow_uint8=True
        )
        run_length_width, run_length_signed = _validate_common_run_length_decode_dtype(
            "run_lengths", run_lengths, allow_uint8=False
        )
        _validate_common_run_length_decode_controls(
            decoded_items_per_thread=decoded_items_per_thread,
            decoded_window_offset=decoded_window_offset,
            run_length_width=run_length_width,
            run_length_signed=run_length_signed,
        )

    return _group_primitive_marker(
        "run_length_decode",
        group,
        run_values,
        run_lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        decoded_window_offset=decoded_window_offset,
    )


__all__ = ["run_length_decode"]
