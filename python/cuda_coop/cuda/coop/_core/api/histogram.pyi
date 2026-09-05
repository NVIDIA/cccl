# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for the portable histogram family."""

from typing import overload

from typing_extensions import TypeVar

from cuda.coop._typing import (
    HistogramAlgorithm,
    PortableIntegerKey,
    PortableIntegerValue,
    ThreadDataLike,
)

from .thread_group import BlockGroup

_CounterT = TypeVar("_CounterT", bound=PortableIntegerKey)
_HistogramSampleT = TypeVar("_HistogramSampleT", bound=PortableIntegerValue)

@overload
def histogram(
    group: BlockGroup,
    samples: ThreadDataLike[_HistogramSampleT],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[_CounterT],
    algorithm: HistogramAlgorithm = "atomic",
) -> ThreadDataLike[_CounterT]:
    """Return striped counters typed by a portable dtype class.

    The complete block leaves compiler-produced ``samples`` unchanged.
    Positive static dimensions must cover every bin; excess slots are zero.
    Every sample must satisfy ``0 <= sample < bins``; violating this CUB
    precondition is undefined behavior.
    """

@overload
def histogram(
    group: BlockGroup,
    samples: ThreadDataLike[_HistogramSampleT],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: None = None,
    algorithm: HistogramAlgorithm = "atomic",
) -> ThreadDataLike[int]:
    """Return default signed-integer striped counters.

    The complete block leaves compiler-produced ``samples`` unchanged.
    Positive static dimensions must cover every bin; excess slots are zero.
    Every sample must satisfy ``0 <= sample < bins``; violating this CUB
    precondition is undefined behavior.
    """
