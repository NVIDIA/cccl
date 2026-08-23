# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS histograms."""

from __future__ import annotations

from typing import overload

from typing_extensions import TypeVar

from .._typing import HistogramAlgorithm as _HistogramAlgorithm
from .._typing import _PortableIntegerKey as _PortableIntegerKey
from .._typing import _PortableIntegerValue as _PortableIntegerValue
from ._thread_data import ThreadData
from ._thread_data import (
    _CutlassHistogramOpaqueSamples as _CutlassHistogramOpaqueSamples,
)
from ._thread_group import _BlockGroup as _BlockGroup

_CounterT = TypeVar("_CounterT", bound=_PortableIntegerKey)
_HistogramSampleT = TypeVar("_HistogramSampleT", bound=_PortableIntegerValue)

@overload
def histogram(
    group: _BlockGroup,
    samples: (
        ThreadData[_HistogramSampleT]
        | _HistogramSampleT
        | _CutlassHistogramOpaqueSamples
    ),
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[_CounterT],
    algorithm: _HistogramAlgorithm = "atomic",
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
    group: _BlockGroup,
    samples: (
        ThreadData[_HistogramSampleT]
        | _HistogramSampleT
        | _CutlassHistogramOpaqueSamples
    ),
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: None = None,
    algorithm: _HistogramAlgorithm = "atomic",
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
