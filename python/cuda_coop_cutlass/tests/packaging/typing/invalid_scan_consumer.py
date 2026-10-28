# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported scan calls through stubs."""

from __future__ import annotations

import numpy as np
from cutlass import Float32, Int32, Int64, Uint32

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


coop.inclusive_sum(warp, values)  # expected-error: [arg-type]
coop.exclusive_sum(
    block,  # expected-error: [arg-type]
    scalar,
    valid_items=7,
)
coop.scan(  # expected-error: [call-overload]
    block, scalar, scan_op="max"
)
coop.exclusive_scan(
    block,
    scalar,
    scan_op=np.multiply,  # expected-error: [arg-type]
)
coop.scan(  # expected-error: [call-overload]
    block, scalar, mode="inclusive", initial_value=0
)
coop.inclusive_scan(
    block,
    scalar,
    scan_op=callback,  # expected-error: [arg-type]
)
coop.scan(  # expected-error: [call-overload]
    block, scalar, prefix_op=callback
)
coop.exclusive_sum(  # expected-error: [call-overload]
    block, scalar, values
)
coop.scan(
    warp,  # expected-error: [arg-type]
    scalar,
    temp_storage=coop.TempStorage(),
)
coop.scan(  # expected-error: [call-overload]
    warp, scalar, algorithm="raking"
)
coop.scan(  # expected-error: [call-overload]
    block, scalar, aggregate_output=scalar
)
coop.inclusive_sum(
    coop.this_cluster(),  # expected-error: [arg-type]
    scalar,
)
coop.exclusive_scan(  # expected-error: [misc]
    block, scalar, initial_value=Float32(0)
)
coop.scan(  # expected-error: [call-overload]
    block, scalar, mode="exclusive", initial_value=Int64(0)
)
coop.exclusive_scan(  # expected-error: [misc]
    block, values, initial_value=Uint32(0)
)
coop.scan(  # expected-error: [call-overload]
    warp, Float32(1), mode="exclusive", initial_value=np.float64(0)
)
