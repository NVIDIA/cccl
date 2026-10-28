# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported topk calls through stubs."""

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


coop.topk_min_keys(warp, values, k=1)  # expected-error: [arg-type]
coop.topk_max_pairs(
    logical,  # expected-error: [arg-type]
    values,
    values,
    k=1,
)
coop.topk_min_keys(block, values)  # expected-error: [call-overload]
coop.topk_max_keys(  # expected-error: [call-overload]
    block, scalar, k=1
)
coop.topk_min_pairs(  # expected-error: [call-overload]
    block, values, scalar, k=1
)
coop.topk_max_pairs(  # expected-error: [call-overload]
    block, scalar, values, k=1
)
coop.topk_min_keys(block, values, 1)  # expected-error: [call-overload]
coop.topk_max_keys(  # expected-error: [call-overload]
    group=block, keys=values, k=1
)
coop.topk_min_keys(
    block,
    values,
    k=np.uint64(1),  # expected-error: [arg-type]
)
coop.topk_max_keys(  # expected-error: [call-overload]
    block, values, k=Float32(1)
)
coop.topk_min_pairs(
    block,
    values,
    values,
    k=1,
    valid_items=np.uint64(3),  # expected-error: [arg-type]
)
coop.topk_max_pairs(  # expected-error: [call-overload]
    block, values, values, k=1, valid_items=1.5
)
coop.topk_min_keys(  # expected-error: [call-overload]
    block, values, k=1, algorithm="radix"
)
coop.topk_max_keys(  # expected-error: [call-overload]
    block, values, k=1, descending=True
)
coop.topk_min_pairs(  # expected-error: [call-overload]
    block, values, values, k=1, compare_op=callback
)
