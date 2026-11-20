# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer checks for neighbors result types."""

from __future__ import annotations

from typing import Any

import numpy as np
from cutlass import (
    Int32,
)
from typing_extensions import assert_type

import cuda.coop.cutlass as coop


def check_cutlass_neighbors() -> None:
    """Separate value-typed differences from signed Int32 boundary flags.

    Discontinuity's literal mode selects one payload or a heads/tails pair.
    Its flag dtype stays known even when a register input has no static
    element type, unlike the value-preserving Adjacent Difference result.
    """
    block = coop.this_block()
    values = coop.ThreadData(items_per_thread=3, dtype=np.int32)
    storage = coop.TempStorage(alignment=16)
    assert_type(
        coop.adjacent_difference(
            block, values, valid_items=17, temp_storage=storage
        ),
        coop.ThreadData[np.int32],
    )
    assert_type(
        coop.adjacent_difference(block, values.to_register_tensor()),
        coop.ThreadData[Any],
    )
    assert_type(coop.discontinuity(block, values), coop.ThreadData[Int32])
    assert_type(
        coop.discontinuity(block, values, mode="heads_and_tails"),
        tuple[coop.ThreadData[Int32], coop.ThreadData[Int32]],
    )
    assert_type(
        coop.discontinuity(block, values.to_tensor_ssa(), mode="tails"),
        coop.ThreadData[Int32],
    )
