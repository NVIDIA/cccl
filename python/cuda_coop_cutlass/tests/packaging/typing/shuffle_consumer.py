# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer checks for shuffle result types."""

from __future__ import annotations

from typing import Any

import numpy as np
from cutlass import (
    Int16,
    Uint32,
)
from typing_extensions import assert_type

import cuda.coop.cutlass as coop


def check_cutlass_shuffle_surface(scalar: Uint32) -> None:
    """Keep payload and scalar Shuffle result types distinct."""
    block = coop.this_block()
    values = coop.ThreadData(items_per_thread=3, dtype=np.int32)
    assert_type(coop.shuffle(block, values), coop.ThreadData[np.int32])
    assert_type(
        coop.shuffle(block, values, mode="up"), coop.ThreadData[np.int32]
    )
    assert_type(coop.shuffle(block, scalar, mode="offset", distance=-2), Uint32)
    assert_type(
        coop.shuffle(block, scalar, mode="rotate", distance=Int16(2)), Uint32
    )
    assert_type(
        coop.shuffle(block, scalar, mode="rotate", distance=np.uint32(2)),
        Uint32,
    )
    assert_type(
        coop.shuffle(block, values.to_register_tensor()), coop.ThreadData[Any]
    )
    assert_type(
        coop.shuffle(block, values.to_tensor_ssa(), mode="up"),
        coop.ThreadData[Any],
    )
