# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fresh block histogram counters."""

from __future__ import annotations

from typing import Any

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation(
    "histogram", family_module="cuda.coop.numba_mlir._compiler._group_histogram"
)
def histogram(
    group: ThreadGroup,
    samples: Any,
    /,
    *,
    bins: Any,
    bins_per_thread: Any = 1,
    counter_dtype: Any = None,
    algorithm: str = "atomic",
    temp_storage: Any = None,
) -> Any:
    """Count bins from ThreadData, local-array, or scalar samples.

    Shared parameters, participation, supported dtypes, algorithms, striped
    ownership, and scratch behavior follow :func:`cuda.coop.histogram`.

    Additional parameters
    ---------------------
    samples : ThreadDataLike, local array, or scalar
        Fixed-size local arrays and one scalar sample per thread are accepted
        in addition to ThreadData. The sample dtype and extent are inferred.

    Returns
    -------
    ThreadDataLike
        A fresh payload with ``bins_per_thread`` counters per member, even
        for scalar input. Result ownership and zero output padding follow
        the common operation.
    """

    return group_primitive_marker(
        "histogram",
        group,
        samples,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=counter_dtype,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


__all__ = ["histogram"]
