# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Histogram signatures for block groups."""

from typing import Any, overload

import numpy as np
from typing_extensions import TypeVar

from .._typing import HistogramAlgorithm, ThreadDataLike
from ._enums import BlockHistogramAlgorithm
from ._thread_group import BlockGroup

_CounterT = TypeVar(
    "_CounterT",
    int,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
)

@overload
def histogram(
    group: BlockGroup,
    samples: ThreadDataLike[Any],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[_CounterT],
    algorithm: HistogramAlgorithm | BlockHistogramAlgorithm = "atomic",
) -> ThreadDataLike[_CounterT]:
    """Return striped counters typed by a portable dtype class.

    The complete block leaves the fixed-size ``samples`` payload unchanged.
    Positive static capacity covers every bin; excess striped slots are zero.
    Every sample satisfies CUB's ``0 <= sample < bins`` precondition.
    """

@overload
def histogram(
    group: BlockGroup,
    samples: ThreadDataLike[Any],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: None = None,
    algorithm: HistogramAlgorithm | BlockHistogramAlgorithm = "atomic",
) -> ThreadDataLike[int]:
    """Return default signed-integer striped counters."""

@overload
def histogram(
    group: BlockGroup,
    samples: ThreadDataLike[Any],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: object,
    algorithm: HistogramAlgorithm | BlockHistogramAlgorithm = "atomic",
) -> ThreadDataLike[Any]:
    """Return counters using a Numba-CUDA-MLIR dtype token."""
