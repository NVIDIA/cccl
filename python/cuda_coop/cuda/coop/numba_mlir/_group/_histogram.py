# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose the Histogram marker for array and scalar sample inputs.

Compiler recognition loads the histogram planning family on demand. The
marker describes a fresh counter result whose dtype and extent can differ
from the samples; calling this body does not compute a Python histogram.
"""

from __future__ import annotations

from typing import Literal, TypeVar

import numpy

from cuda.coop._typing import (
    CommonThreadDataLike,
    CompilerIntegerLike,
    TempStorageLike,
    ThreadDataLike,
)

from .._compiler._operations import group_operation
from .._thread_group import BlockGroup
from ._marker import group_primitive_marker

_Counter = TypeVar(
    "_Counter", numpy.int32, numpy.uint32, numpy.int64, numpy.uint64
)


@group_operation(
    "histogram", family_module="cuda.coop.numba_mlir._compiler._group_histogram"
)
def histogram(
    group: BlockGroup,
    samples: CommonThreadDataLike[
        int
        | numpy.uint8
        | numpy.int32
        | numpy.uint32
        | numpy.int64
        | numpy.uint64
        | CompilerIntegerLike
    ]
    | int
    | numpy.uint8
    | numpy.int32
    | numpy.uint32
    | numpy.int64
    | numpy.uint64
    | CompilerIntegerLike
    | numpy.ndarray,
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[int | _Counter] | numpy.dtype | None = None,
    algorithm: Literal["atomic", "sort"] = "atomic",
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[numpy.int32] | ThreadDataLike[_Counter]:
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

    Examples
    --------
    Count local-array samples with both atomic and sort algorithms, then
    store bins in striped order.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_histogram_examples.py
        :language: python
        :start-after: # qualified-histogram-example-begin
        :end-before: # qualified-histogram-example-end
        :dedent: 4
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
