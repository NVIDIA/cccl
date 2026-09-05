# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS histograms."""

from __future__ import annotations

from typing import overload

from typing_extensions import TypeVar

from .._typing import HistogramAlgorithm, PortableIntegerKey, PortableIntegerValue
from ._thread_data import (
    CutlassHistogramOpaqueSamples,
    ThreadData,
)
from ._thread_group import BlockGroup

_CounterT = TypeVar("_CounterT", bound=PortableIntegerKey)
_HistogramSampleT = TypeVar("_HistogramSampleT", bound=PortableIntegerValue)

@overload
def histogram(
    group: BlockGroup,
    samples: (
        ThreadData[_HistogramSampleT]
        | _HistogramSampleT
        | CutlassHistogramOpaqueSamples
    ),
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[_CounterT],
    algorithm: HistogramAlgorithm = "atomic",
) -> ThreadData[_CounterT]:
    """Histogram a CUTLASS register payload with a typed counter dtype.

    Supported integral scalars, ``ThreadData``, rmem ``Tensor``, and
    ``TensorSSA`` are accepted; tensor address space and element dtype are
    checked while tracing. The complete block leaves samples unchanged,
    requires positive static capacity, and zero-fills excess striped slots.
    Every sample must satisfy ``0 <= sample < bins``; violating this CUB
    precondition is undefined behavior.
    """

@overload
def histogram(
    group: BlockGroup,
    samples: (
        ThreadData[_HistogramSampleT]
        | _HistogramSampleT
        | CutlassHistogramOpaqueSamples
    ),
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: None = None,
    algorithm: HistogramAlgorithm = "atomic",
) -> ThreadData[int]:
    """Histogram a CUTLASS register payload into signed-integer counters.

    Supported integral scalars, ``ThreadData``, rmem ``Tensor``, and
    ``TensorSSA`` are accepted; tensor address space and element dtype are
    checked while tracing. The complete block leaves samples unchanged,
    requires positive static capacity, and zero-fills excess striped slots.
    Every sample must satisfy ``0 <= sample < bins``; violating this CUB
    precondition is undefined behavior.
    """

__all__ = [
    "histogram",
]
