# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import numpy as np
from cutlass import Float32, Int32

import cuda.coop.cutlass as cutlass_coop

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
values = cutlass_coop.ThreadData(items_per_thread=2, dtype=np.int32)
cutlass_coop.load(
    cutlass_coop.this_grid(),  # expected-error: [arg-type]
    object(),
    values,
)
cutlass_coop.store(
    cutlass_coop.this_thread(),  # expected-error: [arg-type]
    object(),
    values,
)
cutlass_coop.load(  # expected-error: [call-overload]
    block, object(), values, algorithm="unknown"
)
cutlass_coop.store(  # expected-error: [call-overload]
    block, object(), values, valid_items=1.5
)
cutlass_coop.store(  # expected-error: [call-overload]
    block, object(), values, offset=1.5
)
cutlass_coop.load(  # expected-error: [call-overload]
    block, object(), values, oob_default=0
)
cutlass_coop.load(
    warp,  # expected-error: [arg-type]
    object(),
    values,
    temp_storage=cutlass_coop.TempStorage(),
)
cutlass_coop.store(  # expected-error: [call-overload]
    warp, object(), values, algorithm="warp_transpose"
)
cutlass_coop.store(  # expected-error: [call-overload]
    block, object(), values.to_register_tensor()
)
cutlass_coop.reduce(
    block,
    scalar,
    binary_op=callback,  # expected-error: [arg-type]
)
cutlass_coop.sum(  # expected-error: [call-overload]
    block, scalar, valid_items=7
)
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
