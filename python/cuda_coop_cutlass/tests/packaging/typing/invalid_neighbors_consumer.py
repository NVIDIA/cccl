# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported neighbors calls through stubs."""

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


coop.adjacent_difference(warp, values)  # expected-error: [arg-type]
coop.adjacent_difference(  # expected-error: [call-overload]
    block, scalar
)
coop.discontinuity(  # expected-error: [call-overload]
    block, values, mode="up"
)
coop.discontinuity(  # expected-error: [call-overload]
    block, values, flag_op=callback
)
