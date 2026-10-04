# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type common decoding inputs and distinguish payload from size results.

A window keeps the run-value type in a new per-thread payload. Whole-stream
decoding instead returns a uint32 total and writes a caller's destination.
Run lengths use an integer item type, and offsets accept integer scalars.
The compiler checks block shape, matching run extents and destination
layout; the driver checks capacity at runtime. These signatures do not
express the positive-length prefix or buffer non-overlap.
"""

import numpy
from typing_extensions import TypeVar

from ..._typing import (
    IntegralScalar,
    PortableNumericScalar,
    PortableThreadDataLike,
    TempStorageLike,
    ThreadDataLike,
)
from .thread_group import BlockGroup

_ItemT = TypeVar("_ItemT", bound=PortableNumericScalar)
_LengthT = TypeVar("_LengthT", bound=IntegralScalar)

def run_length_decode(
    group: BlockGroup,
    run_values: PortableThreadDataLike[_ItemT],
    run_lengths: PortableThreadDataLike[_LengthT],
    /,
    *,
    decoded_items_per_thread: int,
    decoded_window_offset: IntegralScalar = 0,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_ItemT]: ...
def run_length_decode_into(
    group: BlockGroup,
    run_values: PortableThreadDataLike[_ItemT],
    run_lengths: PortableThreadDataLike[_LengthT],
    destination: object,
    /,
    *,
    decoded_items_per_thread: int,
    destination_offset: IntegralScalar = 0,
    temp_storage: TempStorageLike | None = None,
) -> numpy.uint32: ...
