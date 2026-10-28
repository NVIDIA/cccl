# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported merge sort calls through stubs."""

from __future__ import annotations

import numpy as np
from cutlass import Float32, Int32

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


coop.merge_sort_keys(mapped, values)  # expected-error: [arg-type]
coop.merge_sort_pairs(
    coop.this_cluster(),  # expected-error: [arg-type]
    values,
    values,
)
coop.merge_sort_keys(block, scalar)  # expected-error: [call-overload]
coop.merge_sort_pairs(  # expected-error: [call-overload]
    block, values, scalar
)
coop.merge_sort_keys(  # expected-error: [call-overload]
    block, values, True
)
coop.merge_sort_keys(  # expected-error: [call-overload]
    group=block, keys=values
)
coop.merge_sort_keys(  # expected-error: [call-overload]
    block, values, compare_op=callback
)
coop.merge_sort_keys(  # expected-error: [call-overload]
    block, values, algorithm="merge"
)
coop.merge_sort_keys(  # expected-error: [call-overload]
    block, values, valid_items=3
)
coop.merge_sort_pairs(  # expected-error: [call-overload]
    block, values, values, oob_default=1000
)
coop.merge_sort_keys(
    warp,  # expected-error: [arg-type]
    values,
    temp_storage=coop.TempStorage(),
)
coop.merge_sort_pairs(
    logical,  # expected-error: [arg-type]
    values,
    values,
    temp_storage=coop.TempStorage(),
)
coop.merge_sort_keys(
    block,
    values,
    valid_items=np.uint64(3),  # expected-error: [arg-type]
    oob_default=1000,
)
coop.merge_sort_keys(  # expected-error: [call-overload]
    block, values, valid_items=Float32(3), oob_default=1000
)
