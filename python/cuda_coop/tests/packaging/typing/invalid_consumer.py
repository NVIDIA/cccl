# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deliberately invalid calls proving the public stubs reject misuse."""

from __future__ import annotations

import operator
from typing import cast

import numpy as np

import cuda.coop as common
import cuda.coop.numba_mlir as numba_coop

common.register("numba")  # expected-error: [arg-type]

numba_coop.TempStorage(64, 16)  # expected-error: [call-arg]
common.TempStorage(64, 16)  # expected-error: [call-arg]
values = numba_coop.ThreadData(2, np.int32)
common_values = common.ThreadData(2, np.int32)
common_block = common.this_block()
common_block.rank_as(np.float32)  # expected-error: [arg-type]
common_block.count_as(np.bool_)  # expected-error: [arg-type]
common_block.rank_as(bool)  # expected-error: [arg-type]
common.this_grid().sync()  # expected-error: [misc]
common_block.group_by(2).sync()  # expected-error: [misc]
common_block.group_by(2).rank("grid")  # expected-error: [call-overload]
common.this_warp().group_by(8).count("block")  # expected-error: [call-overload]
common.StatefulFunction  # expected-error: [attr-defined]
qualified_block = numba_coop.this_block()
qualified_block.rank_as(np.float32)  # expected-error: [arg-type]
qualified_block.count_as(np.bool_)  # expected-error: [arg-type]
qualified_block.count_as(bool)  # expected-error: [arg-type]
numba_coop.this_grid().sync_aligned()  # expected-error: [misc]
qualified_block.group_by(2).sync_aligned()  # expected-error: [misc]
qualified_block.group_by(2).count("grid")  # expected-error: [call-overload]
numba_coop.this_warp().group_by(8).rank("cluster")  # expected-error: [call-overload]
common.load(  # expected-error: [call-overload]
    common.this_block(),
    object(),
    common_values,
    algorithm="stripd",
)
common.load(  # expected-error: [call-overload]
    common.this_warp(),
    object(),
    common_values,
    algorithm="warp_transpose",
)
common.load(
    common.this_warp(),  # expected-error: [arg-type]
    object(),
    common_values,
    temp_storage=common.TempStorage(),
)
common.load(
    common.this_warp().group_by(8),  # expected-error: [arg-type]
    object(),
    common_values,
    temp_storage=common.TempStorage(),
)
common.store(  # expected-error: [call-overload]
    common.this_warp(),
    object(),
    common_values,
    algorithm="warp_transpose",
)
numba_coop.load(  # expected-error: [call-overload]
    numba_coop.this_warp(),
    object(),
    values,
    algorithm="warp_transpose",
)
numba_coop.load(
    numba_coop.this_warp(),  # expected-error: [arg-type]
    object(),
    values,
    temp_storage=numba_coop.TempStorage(),
)
numba_coop.load(
    numba_coop.this_warp().group_by(8),  # expected-error: [arg-type]
    object(),
    values,
    temp_storage=numba_coop.TempStorage(),
)
numba_coop.load(  # expected-error: [call-overload]
    numba_coop.this_warp(),
    object(),
    values,
    algorithm=0,
)
numba_coop.store(  # expected-error: [call-overload]
    numba_coop.this_warp(),
    object(),
    values,
    algorithm=True,
)
numba_coop.load(  # expected-error: [call-overload]
    numba_coop.this_block(),
    object(),
    values,
    algorithm=0,
)
numba_coop.store(  # expected-error: [call-overload]
    numba_coop.this_block(),
    object(),
    values,
    algorithm=True,
)
numba_coop.load(  # expected-error: [call-overload]
    numba_coop.this_block(),
    object(),
    values,
    algorithm="stripd",
)
numba_coop.load(  # expected-error: [call-overload]
    numba_coop.this_block(),
    object(),
    values,
    valid_items=1.5,
)
numba_coop.load(  # expected-error: [call-overload]
    numba_coop.this_block(),
    object(),
    values,
    oob_default=0,
)
numba_coop.store(  # expected-error: [call-overload]
    numba_coop.this_block(),
    object(),
    values,
    offset="1",
)
numba_coop.BlockLoadAlgorithm  # expected-error: [attr-defined]
numba_coop.BlockStoreAlgorithm  # expected-error: [attr-defined]
numba_coop.WarpLoadAlgorithm  # expected-error: [attr-defined]
numba_coop.WarpStoreAlgorithm  # expected-error: [attr-defined]
common.exchange(
    common.this_block(),
    common_values,
    mode="scatter_to_striped",  # expected-error: [arg-type]
)
common.shuffle(
    common.this_block(),
    common_values,
    distance=2,  # expected-error: [arg-type]
)
numba_coop.exchange(  # expected-error: [call-overload]
    numba_coop.this_block(),
    values,
    mode="scatter_to_blocked",
)
numba_coop.exchange(  # expected-error: [call-overload]
    numba_coop.this_warp(),
    values,
    mode="warp_striped_to_blocked",
)
numba_coop.exchange(  # expected-error: [call-overload]
    numba_coop.this_warp(),
    values,
    mode="scatter_to_striped",
    ranks=numba_coop.ThreadData(2, np.int32),
)
numba_coop.shuffle(  # expected-error: [call-overload]
    numba_coop.this_block(),
    values,
    mode="offset",
)
numba_coop.shuffle(  # expected-error: [call-overload]
    numba_coop.this_block(),
    np.int32(1),
    mode="up",
)
floating_ranks = numba_coop.ThreadData(2, np.float32)
floating_flags = numba_coop.ThreadData(2, np.float32)
numba_coop.exchange(  # expected-error: [call-overload]
    numba_coop.this_block(),
    values,
    mode="scatter_to_blocked",
    ranks=floating_ranks,
)
numba_coop.exchange(  # expected-error: [call-overload]
    numba_coop.this_block(),
    values,
    mode="scatter_to_striped_flagged",
    ranks=np.int32(0),
    valid_flags=floating_flags,
)


def select_left(left: np.int32, right: np.int32) -> np.int32:
    del right
    return left


def prefix_from_aggregate(block_aggregate: np.int32) -> np.int32:
    return block_aggregate


def carry_prefix(
    state: common.ThreadDataLike[np.int32],
    block_aggregate: np.int32,
) -> np.int32:
    previous = state[0]
    state[0] = block_aggregate
    return previous


def float32_prefix(block_aggregate: np.float32) -> np.float32:
    return block_aggregate


def float32_return_prefix(block_aggregate: np.int32) -> np.float32:
    return np.float32(block_aggregate)


def unary_stateful_prefix(block_aggregate: np.int32) -> np.int32:
    return block_aggregate


def carry_int64_state(
    state: common.ThreadDataLike[np.int64],
    block_aggregate: np.int32,
) -> np.int32:
    return block_aggregate + np.int32(state[0])


def carry_float32_value(
    state: common.ThreadDataLike[np.int32],
    block_aggregate: np.float32,
) -> np.float32:
    return block_aggregate + np.float32(state[0])


def carry_wrong_return(
    state: common.ThreadDataLike[np.int32],
    block_aggregate: np.int32,
) -> np.float32:
    return np.float32(block_aggregate + state[0])


class Float32PrefixFunctor:
    def __call__(self, block_aggregate: np.float32) -> np.float32:
        return block_aggregate


class BinaryPrefixFunctor:
    def __call__(self, left: np.int32, right: np.int32) -> np.int32:
        return left + right


prefix_state = numba_coop.ThreadData(1, np.int32)
stateful_prefix = numba_coop.StatefulFunction(carry_prefix, np.int32)
stateful_int64_state = numba_coop.StatefulFunction(carry_int64_state, np.int64)
stateful_float32_value = numba_coop.StatefulFunction(carry_float32_value, np.int32)
numba_coop.StatefulFunction(
    unary_stateful_prefix,  # expected-error: [arg-type]
    np.int32,
)
numba_coop.StatefulFunction(42, np.int32)  # expected-error: [arg-type]
numba_coop.StatefulFunction(  # expected-error: [misc]
    carry_wrong_return,
    np.int32,
)
bad_functor: numba_coop.StatefulFunction[np.int64, np.int32] = (
    numba_coop.StatefulFunction(
        Float32PrefixFunctor,  # expected-error: [arg-type]
        np.int64,
    )
)
bad_binary_functor: numba_coop.StatefulFunction[np.int64, np.int32] = (
    numba_coop.StatefulFunction(
        BinaryPrefixFunctor,  # expected-error: [arg-type]
        np.int64,
    )
)


common.reduce(  # expected-error: [call-overload]
    common_block,
    np.int32(1),
    binary_op=select_left,
    broadcast=False,
)
common.reduce(  # expected-error: [call-overload]
    common_block,
    np.int32(1),
    binary_op=0,
)
common.sum(  # expected-error: [call-overload]
    common_block,
    np.int32(1),
    broadcast=False,
    algorithm=0,
)
common.sum(  # expected-error: [call-overload]
    common_block,
    np.complex64(1),
)
numba_coop.sum(  # expected-error: [call-overload]
    qualified_block,
    np.bool_(True),
)
numba_coop.reduce(  # expected-error: [call-overload]
    qualified_block,
    np.complex64(1),
    binary_op="sum",
)
numba_coop.reduce(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    binary_op=0,
)
numba_coop.sum(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    broadcast=False,
    algorithm=0,
)
complex_values = cast(common.ThreadDataLike[np.complex64], object())
numba_coop.sum(  # expected-error: [type-var]
    qualified_block,
    complex_values,
)
numba_coop.sum(  # expected-error: [call-overload]
    qualified_block,
    values,
    broadcast=False,
    valid_items=2,
)
numba_coop.sum(  # expected-error: [call-overload]
    numba_coop.this_warp(),
    np.int32(1),
    broadcast=False,
    algorithm="raking",
)
numba_coop.reduce(
    qualified_block,
    np.int32(1),
    binary_op=select_left,  # expected-error: [arg-type]
    broadcast=True,
)
numba_coop.reduce(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    binary_op=select_left,
    broadcast=False,
    algorithm="raking_commutative_only",
)
numba_coop.BlockScanAlgorithm  # expected-error: [attr-defined]
common.scan(  # expected-error: [call-overload]
    common_block,
    np.int32(1),
    mode=object(),
)
numba_coop.scan(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    algorithm=object(),
)
numba_coop.inclusive_scan(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    scan_op=object(),
)
bad_common_scan: np.int32 = common.exclusive_scan(
    common_block,
    np.int32(1),
    initial_value=np.float32(0),  # expected-error: [arg-type]
)
bad_qualified_scan: common.ThreadDataLike[np.int32] = numba_coop.exclusive_scan(
    qualified_block,
    values,
    initial_value=np.float32(0),  # expected-error: [arg-type]
)
common.scan(  # expected-error: [call-overload]
    common_block,
    np.int32(1),
    valid_items=1,
)
common.inclusive_sum(  # expected-error: [call-overload]
    common_block,
    np.int32(1),
    aggregate_output=common.ThreadData(1, np.int32),
)
common.inclusive_scan(  # expected-error: [call-overload]
    common_block,
    np.int32(1),
    scan_op=select_left,
)
common.exclusive_scan(  # expected-error: [call-overload]
    common_block,
    np.int32(1),
    scan_op="max",
)
common.scan(  # expected-error: [call-overload]
    common_block,
    np.int32(1),
    mode="inclusive",
    initial_value=np.int32(0),
)
numba_coop.exclusive_scan(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    scan_op="max",
)
numba_coop.scan(
    qualified_block,
    values,
    scan_op=np.multiply,  # expected-error: [arg-type]
)
numba_coop.exclusive_scan(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    scan_op=operator.mul,
)
numba_coop.exclusive_scan(  # expected-error: [call-overload]
    numba_coop.this_warp(),
    np.int32(1),
    scan_op=select_left,
)
numba_coop.scan(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    mode="inclusive",
    initial_value=np.int32(0),
)
numba_coop.inclusive_sum(
    numba_coop.this_warp(),  # expected-error: [arg-type]
    values,
)
numba_coop.inclusive_sum(  # expected-error: [call-overload]
    numba_coop.this_warp(),
    np.int32(1),
    algorithm="raking",
)
numba_coop.inclusive_sum(
    qualified_block,  # expected-error: [arg-type]
    np.int32(1),
    valid_items=1,
)
numba_coop.inclusive_sum(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    aggregate_output=np.int32(0),
)
numba_coop.inclusive_sum(  # expected-error: [call-overload]
    qualified_block,
    np.bool_(True),
)
numba_coop.inclusive_scan(  # expected-error: [call-overload]
    qualified_block,
    np.complex64(1),
    scan_op="max",
)
numba_coop.scan(
    qualified_block,
    np.int32(1),
    prefix_op=select_left,  # expected-error: [arg-type]
)
common.inclusive_sum(  # expected-error: [call-overload]
    common_block,
    np.int32(1),
    prefix_op=prefix_from_aggregate,
)
numba_coop.inclusive_sum(
    numba_coop.this_warp(),  # expected-error: [arg-type]
    np.int32(1),
    prefix_op=prefix_from_aggregate,
)
numba_coop.exclusive_sum(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    prefix_state,
    prefix_op=prefix_from_aggregate,
)
numba_coop.exclusive_sum(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    prefix_op=stateful_prefix,
)
numba_coop.exclusive_sum(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    prefix_state,
)
numba_coop.exclusive_sum(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    prefix_state=prefix_state,
    prefix_op=stateful_prefix,
)
numba_coop.inclusive_sum(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    block_prefix_callback_op=prefix_from_aggregate,
)
numba_coop.inclusive_sum(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    aggregate_output=numba_coop.ThreadData(1, np.int32),
    prefix_op=prefix_from_aggregate,
)
numba_coop.exclusive_scan(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    scan_op=select_left,
    initial_value=np.int32(0),
    prefix_op=prefix_from_aggregate,
)
numba_coop.inclusive_sum(
    qualified_block,
    np.int32(1),
    prefix_op=float32_prefix,  # expected-error: [arg-type]
)
numba_coop.inclusive_sum(
    qualified_block,
    np.int32(1),
    prefix_op=float32_return_prefix,  # expected-error: [arg-type]
)
numba_coop.inclusive_sum(
    qualified_block,
    values,
    prefix_op=float32_prefix,  # expected-error: [arg-type]
)
bool_prefix_values = cast(common.ThreadDataLike[np.bool_], object())
numba_coop.inclusive_sum(  # expected-error: [type-var]
    qualified_block,
    bool_prefix_values,
    prefix_op=lambda aggregate: aggregate,
)
complex_prefix_values = cast(common.ThreadDataLike[np.complex64], object())
numba_coop.inclusive_sum(  # expected-error: [type-var]
    qualified_block,
    complex_prefix_values,
    prefix_op=lambda aggregate: aggregate,
)
numba_coop.exclusive_sum(  # expected-error: [misc]
    qualified_block,
    np.int32(1),
    prefix_state,
    prefix_op=stateful_int64_state,
)
numba_coop.exclusive_sum(  # expected-error: [misc]
    qualified_block,
    np.int32(1),
    prefix_state,
    prefix_op=stateful_float32_value,
)

common.merge_sort_keys(  # expected-error: [call-overload]
    common_block, common_values, compare_op=lambda a, b: a < b
)
common.merge_sort_keys(
    common.this_warp(),  # expected-error: [arg-type]
    common_values,
    temp_storage=common.TempStorage(),
)
common.merge_sort_keys(  # expected-error: [call-overload]
    common_block, common_values, valid_items=4
)
common.merge_sort_keys(
    common.this_grid(),  # expected-error: [arg-type]
    common_values,
)
numba_coop.merge_sort_keys(  # expected-error: [call-overload]
    qualified_block, values, descending=True, compare_op=lambda a, b: a < b
)
numba_coop.merge_sort_keys(  # expected-error: [call-overload]
    qualified_block, values, compare_op="less"
)
numba_coop.merge_sort_pairs(  # expected-error: [call-overload]
    qualified_block, values, values, oob_default=99
)


radix_keys = common.ThreadData(2, np.int32)
radix_float = common.ThreadData(2, np.float32)
common.radix_sort_keys(common.this_warp(), radix_keys)  # expected-error: [arg-type]
common.radix_sort_keys(  # expected-error: [type-var]
    common.this_block(), radix_float
)
common.radix_sort_keys(  # expected-error: [call-arg]
    common.this_block(), radix_keys, blocked_to_striped=True
)
common.radix_rank(  # expected-error: [call-arg]
    common.this_block(), radix_keys, exclusive_digit_prefix=radix_keys
)
common.radix_sort_keys(
    common.this_block(),
    radix_keys,
    descending="yes",  # expected-error: [arg-type]
)


common.topk_min_keys(
    common.this_warp(),  # expected-error: [arg-type]
    common_values,
    k=3,
)
common.topk_max_pairs(
    common_block,
    common_values,
    common_values,
    k="3",  # expected-error: [arg-type]
)
numba_coop.topk_min_pairs(
    numba_coop.this_grid(),  # expected-error: [arg-type]
    values,
    values,
    k=3,
)
numba_coop.topk_max_keys(
    qualified_block,
    values,
    k=3,
    valid_items=1.5,  # expected-error: [arg-type]
)


common.adjacent_difference(
    common.this_warp(),  # expected-error: [arg-type]
    common_values,
)
common.adjacent_difference(
    common_block,
    common_values,
    valid_items=1.5,  # expected-error: [arg-type]
)
common.adjacent_difference(  # expected-error: [call-arg]
    common_block,
    common_values,
    difference_op=lambda a, b: a - b,
)
common.discontinuity(  # expected-error: [call-overload]
    common_block,
    common_values,
    valid_items=4,
)
numba_coop.discontinuity(  # expected-error: [call-overload]
    qualified_block,
    values,
    mode="unknown",
)
numba_coop.adjacent_difference(
    qualified_block,
    values,
    difference_op="minus",  # expected-error: [arg-type]
)


common.histogram(
    common.this_warp(),  # expected-error: [arg-type]
    common.ThreadData(2, np.int32),
    bins=32,
)
histogram_floats = common.ThreadData(2, np.float32)
common.histogram(
    common_block,
    histogram_floats,  # expected-error: [arg-type]
    bins=32,
)
common.histogram(
    common_block,
    common.ThreadData(2, np.int32),
    bins=32,
    counter_dtype=np.float32,  # expected-error: [arg-type]
)
numba_coop.histogram(  # expected-error: [call-overload]
    qualified_block,
    numba_coop.ThreadData(2, np.int32),
    bins=32,
    algorithm="other",
)
common.histogram(  # expected-error: [call-overload]
    common_block,
    3,
    bins=32,
)


common.reduce_batched(
    common.this_block(),  # expected-error: [arg-type]
    common_values,
)
common.reduce_batched(
    common.this_warp(),
    common_values,
    output_layout="broadcast",  # expected-error: [arg-type]
)
