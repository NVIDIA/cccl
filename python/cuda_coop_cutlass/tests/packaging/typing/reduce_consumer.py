# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer checks for reduce result types."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
from cutlass import (
    Uint32,
)
from typing_extensions import assert_type

import cuda.coop.cutlass as coop


def check_cutlass_reduce_surface(scalar: Uint32) -> None:
    """Preserve scalar result types for supported Reduce payloads."""
    block = coop.this_block()
    values = coop.ThreadData(items_per_thread=2, dtype=np.int32)
    assert_type(coop.reduce(block, values), np.int32)
    assert_type(coop.sum(block, values), np.int32)
    warp = coop.this_warp()
    logical_warp = warp.group_by(8)
    assert_type(coop.sum(warp, values), np.int32)
    assert_type(coop.reduce(logical_warp, values), np.int32)
    fragment = values.to_register_tensor()
    vector = values.to_tensor_ssa()
    assert_type(coop.sum(block, fragment), Any)
    assert_type(coop.reduce(block, vector), Any)
    assert_type(coop.sum(warp, vector), Any)
    assert_type(coop.reduce(logical_warp, fragment), Any)
    assert_type(coop.reduce(block, scalar, binary_op="max"), Uint32)
    assert_type(coop.sum(block, scalar), Uint32)
    assert_type(coop.reduce(block, scalar, binary_op=operator.add), Uint32)
    assert_type(coop.reduce(block, values, binary_op=np.maximum), np.int32)
    assert_type(coop.sum(block, scalar, valid_items=17), Uint32)
    assert_type(coop.reduce(block, values, algorithm="raking"), np.int32)
    assert_type(
        coop.sum(coop.this_warp().group_by(8), scalar, valid_items=7), Uint32
    )
    scratch = coop.TempStorage(auto_sync=True)
    assert_type(coop.sum(block, values, temp_storage=scratch), np.int32)
    assert_type(
        coop.sum(block, scalar, valid_items=17, temp_storage=scratch), Uint32
    )
