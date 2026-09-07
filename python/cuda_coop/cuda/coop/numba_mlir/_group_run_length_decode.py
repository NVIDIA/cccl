# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Run Length Decode marker for Numba-CUDA-MLIR.

This module owns the decoded-window signature.  Compiler planning resolves
payload dtypes, output capacity, and optional offset bookkeeping.
"""

from __future__ import annotations

from typing import Any

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation("run_length_decode")
def run_length_decode(
    group: ThreadGroup,
    run_values: Any,
    run_lengths: Any,
    /,
    *,
    decoded_items_per_thread: Any,
    decoded_window_offset: Any = 0,
    relative_offsets: Any = None,
    total_decoded_size: Any = None,
    decoded_offset_dtype: Any = None,
) -> Any:
    """Decode one blockwise run-length window into a fresh payload."""

    return group_primitive_marker(
        "run_length_decode",
        group,
        run_values,
        run_lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        decoded_window_offset=decoded_window_offset,
        relative_offsets=relative_offsets,
        total_decoded_size=total_decoded_size,
        decoded_offset_dtype=decoded_offset_dtype,
    )


__all__ = ["run_length_decode"]
