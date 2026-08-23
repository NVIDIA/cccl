# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Histogram signatures for block groups."""

from typing import Any, overload

from numpy import int32 as _NumpyInt32
from numpy import int64 as _NumpyInt64
from numpy import uint32 as _NumpyUint32
from numpy import uint64 as _NumpyUint64
from typing_extensions import TypeVar

from .._typing import HistogramAlgorithm as _HistogramAlgorithm
from .._typing import ThreadDataLike as _ThreadDataLike
from ._enums import BlockHistogramAlgorithm
from ._thread_group import _BlockGroup

_CounterT = TypeVar(
    "_CounterT",
    int,
    _NumpyInt32,
    _NumpyUint32,
    _NumpyInt64,
    _NumpyUint64,
)

@overload
def histogram(
    group: _BlockGroup,
    samples: _ThreadDataLike[Any],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[_CounterT],
    algorithm: _HistogramAlgorithm | BlockHistogramAlgorithm = "atomic",
) -> _ThreadDataLike[_CounterT]:
    """Return striped counters typed by a portable dtype class.

    The complete block leaves the fixed-size ``samples`` payload unchanged.
    Positive static capacity covers every bin; excess striped slots are zero.
    Every sample satisfies CUB's ``0 <= sample < bins`` precondition.
    """

@overload
def histogram(
    group: _BlockGroup,
    samples: _ThreadDataLike[Any],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: None = None,
    algorithm: _HistogramAlgorithm | BlockHistogramAlgorithm = "atomic",
) -> _ThreadDataLike[int]:
    """Return default signed-integer striped counters."""

@overload
def histogram(
    group: _BlockGroup,
    samples: _ThreadDataLike[Any],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: object,
    algorithm: _HistogramAlgorithm | BlockHistogramAlgorithm = "atomic",
) -> _ThreadDataLike[Any]:
    """Return counters using a Numba-CUDA-MLIR dtype token."""
