# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type common decoding inputs and distinguish payload from size results.

A window keeps the run-value type in a new per-thread payload. Whole-stream
decoding instead returns a uint32 total and writes a caller's destination.
Run lengths use an integer item type, and offsets accept integer scalars.

The compiler checks the block shape, matching run extents, and destination
layout. Generated device code checks destination capacity when the kernel
runs and traps if it is too small. These signatures cannot express the
positive-length prefix or the rule that buffers must not overlap.
"""

import numpy
from typing_extensions import TypeVar

from ..._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    CompilerIntegerLike,
    IntegralScalar,
    TempStorageLike,
    ThreadDataLike,
)
from .thread_group import BlockGroup

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)
_LengthT = TypeVar("_LengthT", bound=IntegralScalar)

def run_length_decode(
    group: BlockGroup,
    run_values: CommonThreadDataLike[_ItemT],
    run_lengths: CommonThreadDataLike[_LengthT],
    /,
    *,
    decoded_items_per_thread: int,
    decoded_window_offset: IntegralScalar = 0,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_ItemT]: ...
def run_length_decode_into(
    group: BlockGroup,
    run_values: CommonThreadDataLike[_ItemT],
    run_lengths: CommonThreadDataLike[_LengthT],
    destination: object,
    /,
    *,
    decoded_items_per_thread: int,
    destination_offset: IntegralScalar = 0,
    temp_storage: TempStorageLike | None = None,
) -> numpy.uint32 | CompilerIntegerLike: ...
