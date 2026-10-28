# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer checks for run length result types."""

from __future__ import annotations

from typing import Any

from cutlass import (
    Int32,
    Int64,
    Uint32,
    Uint64,
)
from typing_extensions import assert_type

import cuda.coop.cutlass as coop
from cuda.coop.cutlass._thread_data import CutlassTensorSample


def check_cutlass_run_length_surface(destination: CutlassTensorSample) -> None:
    """Distinguish a value-typed decoded window from the bulk Uint32 total.

    Run lengths and offsets have independent integer types. Bulk decoding
    writes the destination tensor and returns a count; it does not return a
    payload. Register run values give a ThreadData[Any] window result;
    register run lengths do not change either result type.
    """
    block = coop.this_block()
    values = coop.ThreadData(items_per_thread=2, dtype=Int32)
    lengths = coop.ThreadData(items_per_thread=2, dtype=Uint64)
    storage = coop.TempStorage()
    assert_type(
        coop.run_length_decode(
            block,
            values,
            lengths,
            decoded_items_per_thread=3,
            decoded_window_offset=Uint64(1),
            temp_storage=storage,
        ),
        coop.ThreadData[Int32],
    )
    assert_type(
        coop.run_length_decode(
            block, values.to_tensor_ssa(), lengths, decoded_items_per_thread=3
        ),
        coop.ThreadData[Any],
    )
    assert_type(
        coop.run_length_decode_into(
            block,
            values,
            lengths.to_register_tensor(),
            destination,
            decoded_items_per_thread=3,
            destination_offset=Int64(0),
        ),
        Uint32,
    )
