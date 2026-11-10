# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported reduce calls through stubs."""

from __future__ import annotations

import numpy as np
from cutlass import Int32

import cuda.coop.cutlass as coop

block = coop.this_block()
warp = coop.this_warp()
logical = warp.group_by(8)
mapped = block.group_by(2)
scalar = Int32(1)
values = coop.ThreadData(items_per_thread=2, dtype=np.int32)


def callback(left: Int32, right: Int32) -> Int32:
    """Supply a callable for rejected operators, comparators, and prefixes."""

    del right
    return left


coop.reduce(
    block,
    scalar,
    binary_op=callback,  # expected-error: [arg-type]
)
coop.sum(  # expected-error: [call-overload]
    block, scalar, broadcast=False
)
coop.sum(  # expected-error: [call-overload]
    block, values, valid_items=7
)
coop.sum(  # expected-error: [call-overload]
    warp, scalar, algorithm="raking"
)
coop.sum(coop.this_grid(), scalar)  # expected-error: [arg-type]
coop.sum(
    coop.this_thread(),  # expected-error: [arg-type]
    scalar,
)
coop.sum(
    coop.this_cluster(),  # expected-error: [arg-type]
    scalar,
)
coop.sum(mapped, scalar)  # expected-error: [arg-type]
coop.sum(  # expected-error: [call-overload]
    warp, values, valid_items=7
)
coop.sum(
    warp,  # expected-error: [arg-type]
    scalar,
    temp_storage=coop.TempStorage(),
)
