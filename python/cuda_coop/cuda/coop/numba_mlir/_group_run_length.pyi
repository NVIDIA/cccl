# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed block Run Length Decode operations."""

import numpy
from typing_extensions import TypeVar

from .._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    IntegralScalar,
    TempStorageLike,
    ThreadDataLike,
)
from ._thread_group import BlockGroup

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
