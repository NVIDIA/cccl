# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for portable run-length decode."""

from typing_extensions import TypeVar

from cuda.coop._typing import (
    IntegerValue,
    PortableRunLength,
    PortableRunValue,
    ThreadDataLike,
    TraceInteger,
)

from .thread_group import BlockGroup

_RunValueT = TypeVar("_RunValueT", bound=PortableRunValue)
_RunLengthT = TypeVar("_RunLengthT", bound=PortableRunLength)

def run_length_decode(
    group: BlockGroup,
    run_values: ThreadDataLike[_RunValueT],
    run_lengths: ThreadDataLike[_RunLengthT],
    /,
    *,
    decoded_items_per_thread: TraceInteger,
    decoded_window_offset: IntegerValue = 0,
) -> ThreadDataLike[_RunValueT]:
    """Return one fixed-size window from a blockwise run-length stream.

    ``group`` is the complete physical block. ``run_values`` and
    ``run_lengths`` are matching positive-size payloads whose flattened runs
    use blocked ownership. ``decoded_items_per_thread`` is a positive
    trace-static count. ``decoded_window_offset`` is a uniform nonnegative
    integer representable in the run-length dtype; callers providing a dynamic
    offset guarantee that range. Out-of-range decoded positions are zero. The
    new result preserves the run-value dtype and leaves both inputs unchanged.
    Actual runs have positive lengths; zero lengths are allowed only as one
    trailing padding suffix, and the block-wide sum is positive and
    representable in the run-length dtype. Relative offsets and total decoded
    size are available only from a qualified backend import.
    """
