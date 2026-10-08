# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type decoded values separately from lengths and offset outputs.

A window retains the run-value item type in a new payload. Bulk decoding
returns an unsigned total while updating the destination. The compiler checks
matching run extents and auxiliary dtypes. Generated device code checks
destination capacity at runtime and traps if it is too small. These
signatures do not express the positive-length prefix or buffer non-overlap.
"""

import numpy
from typing_extensions import TypeVar

from ..._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    IntegralScalar,
    TempStorageLike,
    ThreadDataLike,
)
from .._thread_group import BlockGroup

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
    total_decoded_size: ThreadDataLike[numpy.uint32]
    | ThreadDataLike[numpy.uint64]
    | None = None,
    relative_offsets: ThreadDataLike[numpy.uint32]
    | ThreadDataLike[numpy.uint64]
    | None = None,
    decoded_offset_dtype: object = None,
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
    relative_offsets: object = None,
    decoded_offset_dtype: object = None,
    temp_storage: TempStorageLike | None = None,
) -> numpy.uint32 | numpy.uint64: ...
