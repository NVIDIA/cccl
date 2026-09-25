# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unsupported qualified hierarchy queries and reduction controls."""

from __future__ import annotations

import numpy as np
from cutlass import Float32, Int32

import cuda.coop.cutlass as cutlass_coop
from cuda import coop as common

block = cutlass_coop.this_block()
warp = cutlass_coop.this_warp()
logical = warp.group_by(8)
mapped = block.group_by(2)
block.rank_as(Float32)  # expected-error: [arg-type]
block.count_as(np.bool_)  # expected-error: [arg-type]
block.rank_as(bool)  # expected-error: [arg-type]
logical.count("block")  # expected-error: [call-overload]
mapped.rank("grid")  # expected-error: [call-overload]
mapped.sync()  # expected-error: [misc]
cutlass_coop.this_grid().sync_aligned()  # expected-error: [misc]


def callback(left: Int32, right: Int32) -> Int32:
    del right
    return left


scalar = Int32(1)
values = cutlass_coop.ThreadData(2, np.int32)
cutlass_coop.reduce(block, scalar, binary_op=callback)  # expected-error: [arg-type]
cutlass_coop.sum(block, scalar, valid_items=7)  # expected-error: [call-overload]
cutlass_coop.sum(  # expected-error: [call-overload]
    block, values, broadcast=False, valid_items=7
)
cutlass_coop.sum(  # expected-error: [call-overload]
    warp, scalar, broadcast=False, algorithm="raking"
)
cutlass_coop.sum(cutlass_coop.this_grid(), scalar)  # expected-error: [arg-type]
cutlass_coop.sum(  # expected-error: [call-overload]
    block, scalar, temp_storage=cutlass_coop.TempStorage()
)

cutlass_coop.inclusive_sum(warp, values)  # expected-error: [arg-type]
cutlass_coop.exclusive_sum(block, scalar, valid_items=7)  # expected-error: [arg-type]
cutlass_coop.scan(block, scalar, scan_op="max")  # expected-error: [call-overload]
cutlass_coop.exclusive_scan(
    block,
    scalar,
    scan_op=np.multiply,  # expected-error: [arg-type]
)
cutlass_coop.scan(  # expected-error: [call-overload]
    block, scalar, mode="inclusive", initial_value=0
)
cutlass_coop.inclusive_scan(
    block,
    scalar,
    scan_op=callback,  # expected-error: [arg-type]
)
cutlass_coop.scan(block, scalar, prefix_op=callback)  # expected-error: [call-overload]
cutlass_coop.exclusive_sum(block, scalar, values)  # expected-error: [call-overload]
cutlass_coop.scan(
    warp,  # expected-error: [arg-type]
    scalar,
    temp_storage=cutlass_coop.TempStorage(),
)
cutlass_coop.scan(warp, scalar, algorithm="raking")  # expected-error: [call-overload]
cutlass_coop.scan(  # expected-error: [call-overload]
    block, scalar, aggregate_output=scalar
)
cutlass_coop.inclusive_sum(
    cutlass_coop.this_cluster(),  # expected-error: [arg-type]
    scalar,
)
common.exclusive_sum(warp, scalar, valid_items=7)  # expected-error: [call-overload]
common.scan(  # expected-error: [call-overload]
    block, scalar, aggregate_output=cutlass_coop.ThreadData(1, Int32)
)

wrong_seed: Int32 = cutlass_coop.exclusive_scan(  # expected-error: [assignment]
    block, scalar, initial_value=Float32(0)
)
