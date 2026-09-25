# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unsupported qualified groups, payloads, and primitive controls."""

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

ranks = cutlass_coop.ThreadData(2, np.int32)
flags = cutlass_coop.ThreadData(2, np.uint8)
cutlass_coop.exchange(block, scalar)  # expected-error: [call-overload]
cutlass_coop.exchange(  # expected-error: [call-overload]
    block, values, mode="scatter_to_blocked"
)
cutlass_coop.exchange(block, values, ranks=ranks)  # expected-error: [call-overload]
cutlass_coop.exchange(  # expected-error: [call-overload]
    warp, values, mode="scatter_to_blocked", ranks=ranks
)
cutlass_coop.exchange(  # expected-error: [call-overload]
    logical, values, mode="blocked_to_warp_striped"
)
cutlass_coop.exchange(
    warp,  # expected-error: [arg-type]
    values,
    warp_time_slicing=True,
)
cutlass_coop.exchange(  # expected-error: [call-overload]
    block,
    values,
    mode="scatter_to_striped_guarded",
    ranks=ranks,
    warp_time_slicing=True,
)
cutlass_coop.exchange(  # expected-error: [call-overload]
    block, values, mode="scatter_to_striped_flagged", ranks=ranks
)
cutlass_coop.exchange(  # expected-error: [call-overload]
    block, values, mode="scatter_to_blocked", ranks=flags
)
cutlass_coop.exchange(  # expected-error: [call-overload]
    block,
    values,
    mode="scatter_to_striped_flagged",
    ranks=ranks,
    valid_flags=cutlass_coop.ThreadData(2, np.bool_),
)
cutlass_coop.exchange(  # expected-error: [call-overload]
    block, values, temp_storage=cutlass_coop.TempStorage()
)
common.exchange(block, values, ranks=ranks)  # expected-error: [call-arg]
cutlass_coop.shuffle(warp, values)  # expected-error: [arg-type]
cutlass_coop.shuffle(block, values, distance=2)  # expected-error: [call-overload]
cutlass_coop.shuffle(block, scalar)  # expected-error: [call-overload]
cutlass_coop.shuffle(block, values, mode="rotate")  # expected-error: [call-overload]
cutlass_coop.shuffle(  # expected-error: [call-overload]
    block, scalar, mode="rotate", distance=np.uint64(1)
)
cutlass_coop.shuffle(block, values, prefix=scalar)  # expected-error: [call-overload]
cutlass_coop.shuffle(block, values, suffix=scalar)  # expected-error: [call-overload]
cutlass_coop.shuffle(  # expected-error: [call-overload]
    block, values, temp_storage=cutlass_coop.TempStorage()
)
common.shuffle(block, scalar, mode="rotate")  # expected-error: [arg-type]

cutlass_coop.merge_sort_keys(mapped, values)  # expected-error: [arg-type]
cutlass_coop.merge_sort_pairs(
    cutlass_coop.this_cluster(),  # expected-error: [arg-type]
    values,
    values,
)
cutlass_coop.merge_sort_keys(block, scalar)  # expected-error: [call-overload]
cutlass_coop.merge_sort_pairs(block, values, scalar)  # expected-error: [call-overload]
cutlass_coop.merge_sort_keys(block, values, True)  # expected-error: [call-overload]
cutlass_coop.merge_sort_keys(  # expected-error: [call-overload]
    group=block, keys=values
)
cutlass_coop.merge_sort_keys(  # expected-error: [call-overload]
    block, values, compare_op=callback
)
cutlass_coop.merge_sort_keys(  # expected-error: [call-overload]
    block, values, algorithm="merge"
)
cutlass_coop.merge_sort_keys(  # expected-error: [call-overload]
    block, values, valid_items=3
)
cutlass_coop.merge_sort_pairs(  # expected-error: [call-overload]
    block, values, values, oob_default=1000
)
cutlass_coop.merge_sort_keys(
    warp,  # expected-error: [arg-type]
    values,
    temp_storage=cutlass_coop.TempStorage(),
)
cutlass_coop.merge_sort_pairs(
    logical,  # expected-error: [arg-type]
    values,
    values,
    temp_storage=cutlass_coop.TempStorage(),
)
cutlass_coop.merge_sort_keys(
    block,
    values,
    valid_items=np.uint64(3),  # expected-error: [arg-type]
    oob_default=1000,
)
cutlass_coop.merge_sort_keys(  # expected-error: [call-overload]
    block, values, valid_items=Float32(3), oob_default=1000
)
common.merge_sort_keys(  # expected-error: [call-overload]
    block, values.to_register_tensor()
)
common.merge_sort_pairs(  # expected-error: [call-overload]
    block, values, values.to_tensor_ssa()
)

cutlass_coop.radix_sort_keys(warp, values)  # expected-error: [arg-type]
cutlass_coop.radix_rank(logical, values)  # expected-error: [arg-type]
cutlass_coop.radix_sort_pairs(  # expected-error: [call-overload]
    block, scalar, values
)
cutlass_coop.radix_sort_pairs(  # expected-error: [call-overload]
    block, values, scalar
)
cutlass_coop.radix_sort_keys(  # expected-error: [type-var]
    block, cutlass_coop.ThreadData(2, np.int16)
)
cutlass_coop.radix_sort_keys(  # expected-error: [type-var]
    block, cutlass_coop.ThreadData(2, np.complex64)
)
cutlass_coop.radix_rank(  # expected-error: [type-var]
    block, cutlass_coop.ThreadData(2, np.float32)
)
cutlass_coop.radix_rank(block, Float32(1))  # expected-error: [call-overload]
cutlass_coop.radix_sort_keys(  # expected-error: [call-overload]
    block, values, valid_items=3
)
cutlass_coop.radix_sort_pairs(  # expected-error: [call-overload]
    block, values, values, algorithm="radix"
)
cutlass_coop.radix_sort_keys(
    block,
    values,
    begin_bit=np.uint64(0),  # expected-error: [arg-type]
)
cutlass_coop.radix_rank(  # expected-error: [call-overload]
    block, values, begin_bit=Int32(1)
)
cutlass_coop.radix_rank(  # expected-error: [call-overload]
    block, values, temp_storage=cutlass_coop.TempStorage()
)
cutlass_coop.radix_rank(  # expected-error: [call-overload]
    block, values, blocked_to_striped=True
)
cutlass_coop.radix_rank(
    block,
    values,
    exclusive_digit_prefix=cutlass_coop.ThreadData(  # expected-error: [arg-type]
        1, np.uint32
    ),
)
cutlass_coop.radix_rank(  # expected-error: [call-overload]
    block, values, exclusive_digit_prefix=scalar
)
cutlass_coop.radix_sort_keys(  # expected-error: [call-overload]
    group=block, keys=values
)
cutlass_coop.radix_rank(block, values, 0, 4)  # expected-error: [call-overload]
common.radix_sort_keys(  # expected-error: [call-arg]
    block, values, blocked_to_striped=True
)
common.radix_rank(  # expected-error: [call-arg]
    block, values, exclusive_digit_prefix=values
)
common.radix_sort_keys(block, scalar)  # expected-error: [arg-type]


class _ReadOnlyPrefix:
    items_per_thread: int = 1
    dtype: object | None = Int32

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int, /) -> Int32:
        return Int32(index)


cutlass_coop.radix_rank(  # expected-error: [call-overload]
    block, values, exclusive_digit_prefix=_ReadOnlyPrefix()
)
common.radix_sort_keys(  # expected-error: [type-var]
    block, cutlass_coop.ThreadData(2, np.float32)
)

cutlass_coop.topk_min_keys(warp, values, k=1)  # expected-error: [arg-type]
cutlass_coop.topk_max_pairs(logical, values, values, k=1)  # expected-error: [arg-type]
cutlass_coop.topk_min_keys(block, values)  # expected-error: [call-overload]
cutlass_coop.topk_max_keys(block, scalar, k=1)  # expected-error: [call-overload]
cutlass_coop.topk_min_pairs(  # expected-error: [call-overload]
    block, values, scalar, k=1
)
cutlass_coop.topk_max_pairs(  # expected-error: [call-overload]
    block, scalar, values, k=1
)
cutlass_coop.topk_min_keys(block, values, 1)  # expected-error: [call-overload]
cutlass_coop.topk_max_keys(  # expected-error: [call-overload]
    group=block, keys=values, k=1
)
cutlass_coop.topk_min_keys(
    block,
    values,
    k=np.uint64(1),  # expected-error: [arg-type]
)
cutlass_coop.topk_max_keys(  # expected-error: [call-overload]
    block, values, k=Float32(1)
)
cutlass_coop.topk_min_pairs(
    block,
    values,
    values,
    k=1,
    valid_items=np.uint64(3),  # expected-error: [arg-type]
)
cutlass_coop.topk_max_pairs(  # expected-error: [call-overload]
    block, values, values, k=1, valid_items=1.5
)
cutlass_coop.topk_min_keys(  # expected-error: [call-overload]
    block, values, k=1, algorithm="radix"
)
cutlass_coop.topk_max_keys(  # expected-error: [call-overload]
    block, values, k=1, descending=True
)
cutlass_coop.topk_min_pairs(  # expected-error: [call-overload]
    block, values, values, k=1, compare_op=callback
)
common.topk_min_keys(
    block,
    values.to_register_tensor(),  # expected-error: [arg-type]
    k=1,
)
common.topk_max_pairs(
    block,
    values,
    values.to_tensor_ssa(),  # expected-error: [arg-type]
    k=1,
)

cutlass_coop.adjacent_difference(warp, values)  # expected-error: [arg-type]
cutlass_coop.adjacent_difference(block, scalar)  # expected-error: [call-overload]
cutlass_coop.discontinuity(block, values, mode="up")  # expected-error: [call-overload]
cutlass_coop.discontinuity(  # expected-error: [call-overload]
    block, values, flag_op=callback
)

cutlass_coop.histogram(warp, values, bins=32)  # expected-error: [arg-type]
cutlass_coop.histogram(block, scalar, bins=32)  # expected-error: [call-overload]
cutlass_coop.histogram(
    block,
    values,
    bins=32,
    counter_dtype=np.float32,  # expected-error: [arg-type]
)
cutlass_coop.histogram(  # expected-error: [call-overload]
    block, values, bins=32, algorithm="other"
)
