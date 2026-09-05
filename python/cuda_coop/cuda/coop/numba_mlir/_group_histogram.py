# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first histogram marker for Numba-CUDA-MLIR.

This marker owns the portable counter API.  Histogram dtype and capacity
validation occur during lowering once payload provenance is known.
"""

from __future__ import annotations

from typing import Any

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation("histogram")
def histogram(
    group: ThreadGroup,
    samples: Any,
    /,
    *,
    bins: Any,
    bins_per_thread: Any = 1,
    counter_dtype: Any = None,
    algorithm: Any = "atomic",
) -> Any:
    """Return striped block-histogram counters without mutating samples."""

    return group_primitive_marker(
        "histogram",
        group,
        samples,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=counter_dtype,
        algorithm=algorithm,
    )


__all__ = ["histogram"]
