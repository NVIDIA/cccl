# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Differential GPU coverage for the common V1 profile under CUTLASS."""

from __future__ import annotations

import inspect

import pytest

import cuda.coop.cutlass as cutlass_coop
from cuda import coop

from ..support.runtime import (
    Int32,
    Uint8,
    Uint32,
    cute,
    cutlass,
    from_dlpack,
    runtime_pytestmark,
    torch,
)

pytestmark = runtime_pytestmark

_BLOCK_THREADS = 32
_ITEMS_PER_THREAD = 3
_TWO_WARP_THREADS = 64
_WARP_ITEMS_PER_THREAD = 2
_TOTAL_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
_VALID_ITEMS = _TOTAL_ITEMS - 13
_TOPK_K = 9


def _populate_items(api, values_in: cute.Tensor, dtype, items_per_thread: int):
    tidx, _, _ = cute.arch.thread_idx()
    items = api.ThreadData(items_per_thread, dtype=dtype)
    for item_idx in range(items_per_thread):
        items[item_idx] = values_in[tidx * items_per_thread + item_idx]
    return items


def _store_items(values_out: cute.Tensor, items, items_per_thread: int) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    for item_idx in range(items_per_thread):
        values_out[tidx * items_per_thread + item_idx] = items[item_idx]


def _pair_sort_body(
    api,
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    original_keys_out: cute.Tensor,
    original_values_out: cute.Tensor,
    merge_keys_out: cute.Tensor,
    merge_values_out: cute.Tensor,
    radix_keys_out: cute.Tensor,
    radix_values_out: cute.Tensor,
):
    keys = _populate_items(api, keys_in, Int32, _ITEMS_PER_THREAD)
    values = _populate_items(api, values_in, Int32, _ITEMS_PER_THREAD)
    group = api.this_block()
    merge_keys, merge_values = api.merge_sort_pairs(
        group, keys, values, descending=True
    )
    radix_keys, radix_values = api.radix_sort_pairs(group, keys, values)
    _store_items(original_keys_out, keys, _ITEMS_PER_THREAD)
    _store_items(original_values_out, values, _ITEMS_PER_THREAD)
    _store_items(merge_keys_out, merge_keys, _ITEMS_PER_THREAD)
    _store_items(merge_values_out, merge_values, _ITEMS_PER_THREAD)
    _store_items(radix_keys_out, radix_keys, _ITEMS_PER_THREAD)
    _store_items(radix_values_out, radix_values, _ITEMS_PER_THREAD)


@cute.kernel
def _common_pair_sort_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    original_keys_out: cute.Tensor,
    original_values_out: cute.Tensor,
    merge_keys_out: cute.Tensor,
    merge_values_out: cute.Tensor,
    radix_keys_out: cute.Tensor,
    radix_values_out: cute.Tensor,
):
    _pair_sort_body(
        coop,
        keys_in,
        values_in,
        original_keys_out,
        original_values_out,
        merge_keys_out,
        merge_values_out,
        radix_keys_out,
        radix_values_out,
    )


@cute.kernel
def _qualified_pair_sort_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    original_keys_out: cute.Tensor,
    original_values_out: cute.Tensor,
    merge_keys_out: cute.Tensor,
    merge_values_out: cute.Tensor,
    radix_keys_out: cute.Tensor,
    radix_values_out: cute.Tensor,
):
    _pair_sort_body(
        cutlass_coop,
        keys_in,
        values_in,
        original_keys_out,
        original_values_out,
        merge_keys_out,
        merge_values_out,
        radix_keys_out,
        radix_values_out,
    )


@cute.jit
def _run_common_pair_sort(keys_in: cute.Tensor, values_in: cute.Tensor, *outputs):
    _common_pair_sort_kernel(keys_in, values_in, *outputs).launch(
        grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1)
    )


@cute.jit
def _run_qualified_pair_sort(keys_in: cute.Tensor, values_in: cute.Tensor, *outputs):
    _qualified_pair_sort_kernel(keys_in, values_in, *outputs).launch(
        grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1)
    )


@cute.kernel
def _common_two_physical_warps_kernel(
    values_in: cute.Tensor,
    exchange_out: cute.Tensor,
    default_scan_out: cute.Tensor,
    exclusive_scan_out: cute.Tensor,
    inclusive_scan_out: cute.Tensor,
    total_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp = coop.this_warp()
    striped = coop.ThreadData(_WARP_ITEMS_PER_THREAD, dtype=Int32)
    coop.load(warp, values_in, striped, algorithm="striped")
    blocked = coop.exchange(warp, striped)
    _store_items(exchange_out, blocked, _WARP_ITEMS_PER_THREAD)

    value = values_in[tidx]
    default_scan_out[tidx] = coop.scan(warp, value)
    exclusive_scan_out[tidx] = coop.exclusive_scan(warp, value)
    inclusive_scan_out[tidx] = coop.inclusive_scan(warp, value)
    total_out[tidx] = coop.sum(warp, value)


@cute.kernel
def _qualified_two_physical_warps_kernel(
    values_in: cute.Tensor,
    exchange_out: cute.Tensor,
    default_scan_out: cute.Tensor,
    exclusive_scan_out: cute.Tensor,
    inclusive_scan_out: cute.Tensor,
    total_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp = cutlass_coop.this_warp()
    striped = cutlass_coop.ThreadData(_WARP_ITEMS_PER_THREAD, dtype=Int32)
    cutlass_coop.load(warp, values_in, striped, algorithm="striped")
    blocked = cutlass_coop.exchange(warp, striped)
    _store_items(exchange_out, blocked, _WARP_ITEMS_PER_THREAD)

    value = values_in[tidx]
    default_scan_out[tidx] = cutlass_coop.scan(warp, value)
    exclusive_scan_out[tidx] = cutlass_coop.exclusive_scan(warp, value)
    inclusive_scan_out[tidx] = cutlass_coop.inclusive_scan(warp, value)
    total_out[tidx] = cutlass_coop.sum(warp, value)


@cute.kernel
def _common_memory_reduce_scan_kernel(
    values_in: cute.Tensor,
    original_out: cute.Tensor,
    round_trip_out: cute.Tensor,
    scan_out: cute.Tensor,
    exclusive_sum_out: cute.Tensor,
    inclusive_sum_out: cute.Tensor,
    exclusive_max_out: cute.Tensor,
    inclusive_max_out: cute.Tensor,
    exchange_out: cute.Tensor,
    warp_round_trip_out: cute.Tensor,
    warp_exchange_out: cute.Tensor,
    sum_out: cute.Tensor,
    warp_scan_out: cute.Tensor,
    warp_default_scan_out: cute.Tensor,
    warp_exclusive_scan_out: cute.Tensor,
    warp_inclusive_scan_out: cute.Tensor,
    thread_reduce_out: cute.Tensor,
    thread_sum_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    block = coop.this_block()
    storage = coop.TempStorage()
    items = coop.ThreadData(2, dtype=Int32)
    loaded = coop.load(block, values_in, items, temp_storage=storage)
    coop.store(block, original_out, items, temp_storage=storage)
    coop.store(block, round_trip_out, loaded, temp_storage=storage)

    scan_items = coop.scan(block, items, temp_storage=storage)
    exclusive_items = coop.exclusive_sum(block, items, temp_storage=storage)
    inclusive_items = coop.inclusive_sum(block, items, temp_storage=storage)
    exclusive_max_items = coop.exclusive_scan(
        block,
        items,
        scan_op="max",
        initial_value=-2_147_483_648,
        temp_storage=storage,
    )
    inclusive_max_items = coop.inclusive_scan(
        block,
        items,
        scan_op="max",
        temp_storage=storage,
    )
    exchanged = coop.exchange(block, items, mode="blocked_to_striped")
    total = coop.sum(block, items)
    warp = coop.this_warp()
    warp_items = coop.ThreadData(2, dtype=Int32)
    coop.load(warp, values_in, warp_items, algorithm="striped")
    coop.store(warp, warp_round_trip_out, warp_items, algorithm="striped")
    warp_exchanged = coop.exchange(warp, warp_items)

    coop.store(block, scan_out, scan_items, temp_storage=storage)
    coop.store(block, exclusive_sum_out, exclusive_items, temp_storage=storage)
    coop.store(block, inclusive_sum_out, inclusive_items, temp_storage=storage)
    coop.store(block, exclusive_max_out, exclusive_max_items, temp_storage=storage)
    coop.store(block, inclusive_max_out, inclusive_max_items, temp_storage=storage)
    coop.store(block, exchange_out, exchanged, temp_storage=storage)
    _store_items(warp_exchange_out, warp_exchanged, 2)
    sum_out[tidx] = total
    warp_scan_out[tidx] = coop.inclusive_sum(warp, values_in[tidx])
    warp_default_scan_out[tidx] = coop.scan(warp, values_in[tidx])
    warp_exclusive_scan_out[tidx] = coop.exclusive_scan(warp, values_in[tidx])
    warp_inclusive_scan_out[tidx] = coop.inclusive_scan(warp, values_in[tidx])
    thread = coop.this_thread()
    thread_reduce_out[tidx] = coop.reduce(thread, values_in[tidx])
    thread_sum_out[tidx] = coop.sum(thread, values_in[tidx])


@cute.kernel
def _qualified_memory_reduce_scan_kernel(
    values_in: cute.Tensor,
    original_out: cute.Tensor,
    round_trip_out: cute.Tensor,
    scan_out: cute.Tensor,
    exclusive_sum_out: cute.Tensor,
    inclusive_sum_out: cute.Tensor,
    exclusive_max_out: cute.Tensor,
    inclusive_max_out: cute.Tensor,
    exchange_out: cute.Tensor,
    warp_round_trip_out: cute.Tensor,
    warp_exchange_out: cute.Tensor,
    sum_out: cute.Tensor,
    warp_scan_out: cute.Tensor,
    warp_default_scan_out: cute.Tensor,
    warp_exclusive_scan_out: cute.Tensor,
    warp_inclusive_scan_out: cute.Tensor,
    thread_reduce_out: cute.Tensor,
    thread_sum_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    block = cutlass_coop.this_block()
    storage = cutlass_coop.TempStorage()
    items = cutlass_coop.ThreadData(2, dtype=Int32)
    loaded = cutlass_coop.load(block, values_in, items, temp_storage=storage)
    cutlass_coop.store(block, original_out, items, temp_storage=storage)
    cutlass_coop.store(block, round_trip_out, loaded, temp_storage=storage)

    scan_items = cutlass_coop.scan(block, items, temp_storage=storage)
    exclusive_items = cutlass_coop.exclusive_sum(block, items, temp_storage=storage)
    inclusive_items = cutlass_coop.inclusive_sum(block, items, temp_storage=storage)
    exclusive_max_items = cutlass_coop.exclusive_scan(
        block,
        items,
        scan_op="max",
        initial_value=-2_147_483_648,
        temp_storage=storage,
    )
    inclusive_max_items = cutlass_coop.inclusive_scan(
        block,
        items,
        scan_op="max",
        temp_storage=storage,
    )
    exchanged = cutlass_coop.exchange(block, items, mode="blocked_to_striped")
    total = cutlass_coop.sum(block, items)
    warp = cutlass_coop.this_warp()
    warp_items = cutlass_coop.ThreadData(2, dtype=Int32)
    cutlass_coop.load(warp, values_in, warp_items, algorithm="striped")
    cutlass_coop.store(warp, warp_round_trip_out, warp_items, algorithm="striped")
    warp_exchanged = cutlass_coop.exchange(warp, warp_items)

    cutlass_coop.store(block, scan_out, scan_items, temp_storage=storage)
    cutlass_coop.store(block, exclusive_sum_out, exclusive_items, temp_storage=storage)
    cutlass_coop.store(block, inclusive_sum_out, inclusive_items, temp_storage=storage)
    cutlass_coop.store(
        block, exclusive_max_out, exclusive_max_items, temp_storage=storage
    )
    cutlass_coop.store(
        block, inclusive_max_out, inclusive_max_items, temp_storage=storage
    )
    cutlass_coop.store(block, exchange_out, exchanged, temp_storage=storage)
    _store_items(warp_exchange_out, warp_exchanged, 2)
    sum_out[tidx] = total
    warp_scan_out[tidx] = cutlass_coop.inclusive_sum(warp, values_in[tidx])
    warp_default_scan_out[tidx] = cutlass_coop.scan(warp, values_in[tidx])
    warp_exclusive_scan_out[tidx] = cutlass_coop.exclusive_scan(warp, values_in[tidx])
    warp_inclusive_scan_out[tidx] = cutlass_coop.inclusive_scan(warp, values_in[tidx])
    thread = cutlass_coop.this_thread()
    thread_reduce_out[tidx] = cutlass_coop.reduce(thread, values_in[tidx])
    thread_sum_out[tidx] = cutlass_coop.sum(thread, values_in[tidx])


@cute.jit
def _run_common_memory_reduce_scan(values_in: cute.Tensor, *outputs):
    _common_memory_reduce_scan_kernel(values_in, *outputs).launch(
        grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1)
    )


@cute.jit
def _run_qualified_memory_reduce_scan(values_in: cute.Tensor, *outputs):
    _qualified_memory_reduce_scan_kernel(values_in, *outputs).launch(
        grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1)
    )


@cute.jit
def _run_common_two_physical_warps(values_in: cute.Tensor, *outputs):
    _common_two_physical_warps_kernel(values_in, *outputs).launch(
        grid=(1, 1, 1), block=(_TWO_WARP_THREADS, 1, 1)
    )


@cute.jit
def _run_qualified_two_physical_warps(values_in: cute.Tensor, *outputs):
    _qualified_two_physical_warps_kernel(values_in, *outputs).launch(
        grid=(1, 1, 1), block=(_TWO_WARP_THREADS, 1, 1)
    )


@cute.kernel
def _common_difference_shuffle_kernel(
    values_in: cute.Tensor,
    original_out: cute.Tensor,
    difference_out: cute.Tensor,
    flags_out: cute.Tensor,
    shuffle_out: cute.Tensor,
):
    group = coop.this_block()
    items = _populate_items(coop, values_in, Int32, 2)
    storage = coop.TempStorage()
    original = coop.ThreadData(2, dtype=Int32)
    original[0] = items[0]
    original[1] = items[1]
    difference = coop.adjacent_difference(
        group,
        items,
        valid_items=61,
        tile_predecessor_item=-7,
        temp_storage=storage,
    )
    flags = coop.discontinuity(
        group,
        items,
        mode="heads",
        tile_predecessor_item=-7,
        temp_storage=storage,
    )
    shuffled = coop.shuffle(group, items, mode="up")
    _store_items(original_out, original, 2)
    _store_items(difference_out, difference, 2)
    _store_items(flags_out, flags, 2)
    _store_items(shuffle_out, shuffled, 2)


@cute.kernel
def _qualified_difference_shuffle_kernel(
    values_in: cute.Tensor,
    original_out: cute.Tensor,
    difference_out: cute.Tensor,
    flags_out: cute.Tensor,
    shuffle_out: cute.Tensor,
):
    group = cutlass_coop.this_block()
    items = _populate_items(cutlass_coop, values_in, Int32, 2)
    storage = cutlass_coop.TempStorage()
    original = cutlass_coop.ThreadData(2, dtype=Int32)
    original[0] = items[0]
    original[1] = items[1]
    difference = cutlass_coop.adjacent_difference(
        group,
        items,
        valid_items=61,
        tile_predecessor_item=-7,
        temp_storage=storage,
    )
    flags = cutlass_coop.discontinuity(
        group,
        items,
        mode="heads",
        tile_predecessor_item=-7,
        temp_storage=storage,
    )
    shuffled = cutlass_coop.shuffle(group, items, mode="up")
    _store_items(original_out, original, 2)
    _store_items(difference_out, difference, 2)
    _store_items(flags_out, flags, 2)
    _store_items(shuffle_out, shuffled, 2)


@cute.jit
def _run_common_difference_shuffle(values_in: cute.Tensor, *outputs):
    _common_difference_shuffle_kernel(values_in, *outputs).launch(
        grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1)
    )


@cute.jit
def _run_qualified_difference_shuffle(values_in: cute.Tensor, *outputs):
    _qualified_difference_shuffle_kernel(values_in, *outputs).launch(
        grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1)
    )


def _sort_rank_topk_body(api, values_in: cute.Tensor, *outputs) -> None:
    (
        original_out,
        merge_out,
        radix_out,
        rank_out,
        topk_max_out,
        topk_min_out,
    ) = outputs
    group = api.this_block()
    items = _populate_items(api, values_in, Int32, _ITEMS_PER_THREAD)
    storage = api.TempStorage()
    merged = api.merge_sort_keys(
        group,
        items,
        valid_items=_VALID_ITEMS,
        oob_default=2_147_483_647,
        temp_storage=storage,
    )
    radix = api.radix_sort_keys(
        group,
        items,
        begin_bit=0,
        end_bit=8,
        descending=True,
        temp_storage=storage,
    )
    ranks = api.radix_rank(
        group,
        items,
        begin_bit=0,
        end_bit=4,
        radix_bits=4,
    )
    topk_max = api.topk_max_keys(
        group,
        items,
        _TOPK_K,
        valid_items=_VALID_ITEMS,
        begin_bit=0,
        end_bit=8,
    )
    topk_min = api.topk_min_keys(
        group,
        items,
        _TOPK_K,
        valid_items=_VALID_ITEMS,
        begin_bit=0,
        end_bit=8,
    )
    _store_items(original_out, items, _ITEMS_PER_THREAD)
    _store_items(merge_out, merged, _ITEMS_PER_THREAD)
    _store_items(radix_out, radix, _ITEMS_PER_THREAD)
    _store_items(rank_out, ranks, _ITEMS_PER_THREAD)
    _store_items(topk_max_out, topk_max, _ITEMS_PER_THREAD)
    _store_items(topk_min_out, topk_min, _ITEMS_PER_THREAD)


@cute.kernel
def _common_sort_rank_topk_kernel(
    values_in: cute.Tensor,
    original_out: cute.Tensor,
    merge_out: cute.Tensor,
    radix_out: cute.Tensor,
    rank_out: cute.Tensor,
    topk_max_out: cute.Tensor,
    topk_min_out: cute.Tensor,
):
    _sort_rank_topk_body(
        coop,
        values_in,
        original_out,
        merge_out,
        radix_out,
        rank_out,
        topk_max_out,
        topk_min_out,
    )


@cute.kernel
def _qualified_sort_rank_topk_kernel(
    values_in: cute.Tensor,
    original_out: cute.Tensor,
    merge_out: cute.Tensor,
    radix_out: cute.Tensor,
    rank_out: cute.Tensor,
    topk_max_out: cute.Tensor,
    topk_min_out: cute.Tensor,
):
    _sort_rank_topk_body(
        cutlass_coop,
        values_in,
        original_out,
        merge_out,
        radix_out,
        rank_out,
        topk_max_out,
        topk_min_out,
    )


@cute.jit
def _run_common_sort_rank_topk(values_in: cute.Tensor, *outputs):
    _common_sort_rank_topk_kernel(values_in, *outputs).launch(
        grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1)
    )


@cute.jit
def _run_qualified_sort_rank_topk(values_in: cute.Tensor, *outputs):
    _qualified_sort_rank_topk_kernel(values_in, *outputs).launch(
        grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1)
    )


@cute.kernel
def _warp_merge_sort_kernel(
    values_in: cute.Tensor,
    original_out: cute.Tensor,
    common_out: cute.Tensor,
    qualified_out: cute.Tensor,
):
    common_items = _populate_items(coop, values_in, Int32, 2)
    qualified_items = _populate_items(cutlass_coop, values_in, Int32, 2)
    common_sorted = coop.merge_sort_keys(
        coop.this_warp(), common_items, descending=True
    )
    qualified_sorted = cutlass_coop.merge_sort_keys(
        cutlass_coop.this_warp(), qualified_items, descending=True
    )
    _store_items(original_out, common_items, 2)
    _store_items(common_out, common_sorted, 2)
    _store_items(qualified_out, qualified_sorted, 2)


@cute.jit
def _run_warp_merge_sort(
    values_in: cute.Tensor,
    original_out: cute.Tensor,
    common_out: cute.Tensor,
    qualified_out: cute.Tensor,
):
    _warp_merge_sort_kernel(values_in, original_out, common_out, qualified_out).launch(
        grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1)
    )


def _pair_topk_body(
    api,
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    original_keys_out: cute.Tensor,
    original_values_out: cute.Tensor,
    max_keys_out: cute.Tensor,
    max_values_out: cute.Tensor,
    min_keys_out: cute.Tensor,
    min_values_out: cute.Tensor,
):
    keys = _populate_items(api, keys_in, Int32, _ITEMS_PER_THREAD)
    values = _populate_items(api, values_in, Int32, _ITEMS_PER_THREAD)
    group = api.this_block()
    max_keys, max_values = api.topk_max_pairs(
        group,
        keys,
        values,
        _TOPK_K,
        valid_items=_VALID_ITEMS,
        begin_bit=0,
        end_bit=8,
    )
    min_keys, min_values = api.topk_min_pairs(
        group,
        keys,
        values,
        _TOPK_K,
        valid_items=_VALID_ITEMS,
        begin_bit=0,
        end_bit=8,
    )
    _store_items(original_keys_out, keys, _ITEMS_PER_THREAD)
    _store_items(original_values_out, values, _ITEMS_PER_THREAD)
    _store_items(max_keys_out, max_keys, _ITEMS_PER_THREAD)
    _store_items(max_values_out, max_values, _ITEMS_PER_THREAD)
    _store_items(min_keys_out, min_keys, _ITEMS_PER_THREAD)
    _store_items(min_values_out, min_values, _ITEMS_PER_THREAD)


@cute.kernel
def _common_pair_topk_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    original_keys_out: cute.Tensor,
    original_values_out: cute.Tensor,
    max_keys_out: cute.Tensor,
    max_values_out: cute.Tensor,
    min_keys_out: cute.Tensor,
    min_values_out: cute.Tensor,
):
    _pair_topk_body(
        coop,
        keys_in,
        values_in,
        original_keys_out,
        original_values_out,
        max_keys_out,
        max_values_out,
        min_keys_out,
        min_values_out,
    )


@cute.kernel
def _qualified_pair_topk_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    original_keys_out: cute.Tensor,
    original_values_out: cute.Tensor,
    max_keys_out: cute.Tensor,
    max_values_out: cute.Tensor,
    min_keys_out: cute.Tensor,
    min_values_out: cute.Tensor,
):
    _pair_topk_body(
        cutlass_coop,
        keys_in,
        values_in,
        original_keys_out,
        original_values_out,
        max_keys_out,
        max_values_out,
        min_keys_out,
        min_values_out,
    )


@cute.jit
def _run_common_pair_topk(keys_in: cute.Tensor, values_in: cute.Tensor, *outputs):
    _common_pair_topk_kernel(keys_in, values_in, *outputs).launch(
        grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1)
    )


@cute.jit
def _run_qualified_pair_topk(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    original_keys_out: cute.Tensor,
    original_values_out: cute.Tensor,
    max_keys_out: cute.Tensor,
    max_values_out: cute.Tensor,
    min_keys_out: cute.Tensor,
    min_values_out: cute.Tensor,
):
    _qualified_pair_topk_kernel(
        keys_in,
        values_in,
        original_keys_out,
        original_values_out,
        max_keys_out,
        max_values_out,
        min_keys_out,
        min_values_out,
    ).launch(grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1))


_CLUSTER_BLOCK_THREADS = 64
_CLUSTER_BLOCKS = 2
_CLUSTER_THREADS = _CLUSTER_BLOCK_THREADS * _CLUSTER_BLOCKS


@cute.kernel
def _cluster_reduce_kernel(
    values_in: cute.Tensor,
    common_reduce_out: cute.Tensor,
    common_sum_out: cute.Tensor,
    qualified_reduce_out: cute.Tensor,
    qualified_sum_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    index = bidx * _CLUSTER_BLOCK_THREADS + tidx
    value = values_in[index]
    common_group = coop.this_cluster()
    qualified_group = cutlass_coop.this_cluster()
    common_reduce_out[index] = coop.reduce(common_group, value)
    common_sum_out[index] = coop.sum(common_group, value)
    qualified_reduce_out[index] = cutlass_coop.reduce(qualified_group, value)
    qualified_sum_out[index] = cutlass_coop.sum(qualified_group, value)


@cute.jit
def _run_cluster_reduce(
    values_in: cute.Tensor,
    common_reduce_out: cute.Tensor,
    common_sum_out: cute.Tensor,
    qualified_reduce_out: cute.Tensor,
    qualified_sum_out: cute.Tensor,
):
    _cluster_reduce_kernel(
        values_in,
        common_reduce_out,
        common_sum_out,
        qualified_reduce_out,
        qualified_sum_out,
    ).launch(
        grid=(_CLUSTER_BLOCKS, 1, 1),
        block=(_CLUSTER_BLOCK_THREADS, 1, 1),
        cluster=(_CLUSTER_BLOCKS, 1, 1),
    )


def _histogram_decode_body(
    api,
    samples_in: cute.Tensor,
    run_values_in: cute.Tensor,
    histogram_out: cute.Tensor,
    decoded_out: cute.Tensor,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    samples = _populate_items(api, samples_in, Uint8, 2)
    counts = api.histogram(
        api.this_block(),
        samples,
        bins=32,
        bins_per_thread=1,
        counter_dtype=Int32,
    )
    run_values = api.ThreadData(1, dtype=Uint32)
    run_lengths = api.ThreadData(1, dtype=Uint32)
    run_values[0] = run_values_in[tidx]
    run_lengths[0] = Uint32(2)
    decoded = api.run_length_decode(
        api.this_block(),
        run_values,
        run_lengths,
        decoded_items_per_thread=2,
    )
    histogram_out[tidx] = counts[0]
    decoded_out[tidx * 2 + 0] = decoded[0]
    decoded_out[tidx * 2 + 1] = decoded[1]


@cute.kernel
def _common_histogram_decode_kernel(
    samples_in: cute.Tensor,
    run_values_in: cute.Tensor,
    histogram_out: cute.Tensor,
    decoded_out: cute.Tensor,
):
    _histogram_decode_body(coop, samples_in, run_values_in, histogram_out, decoded_out)


@cute.kernel
def _qualified_histogram_decode_kernel(
    samples_in: cute.Tensor,
    run_values_in: cute.Tensor,
    histogram_out: cute.Tensor,
    decoded_out: cute.Tensor,
):
    _histogram_decode_body(
        cutlass_coop, samples_in, run_values_in, histogram_out, decoded_out
    )


@cute.jit
def _run_common_histogram_decode(
    samples_in: cute.Tensor,
    run_values_in: cute.Tensor,
    histogram_out: cute.Tensor,
    decoded_out: cute.Tensor,
):
    _common_histogram_decode_kernel(
        samples_in, run_values_in, histogram_out, decoded_out
    ).launch(grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1))


@cute.jit
def _run_qualified_histogram_decode(
    samples_in: cute.Tensor,
    run_values_in: cute.Tensor,
    histogram_out: cute.Tensor,
    decoded_out: cute.Tensor,
):
    _qualified_histogram_decode_kernel(
        samples_in, run_values_in, histogram_out, decoded_out
    ).launch(grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1))


def _device_outputs(count: int, size: int, dtype):
    return [torch.zeros((size,), dtype=dtype, device="cuda") for _ in range(count)]


def _launch_pair(common_runner, qualified_runner, values_in, outputs) -> None:
    midpoint = len(outputs) // 2
    common_runner(
        from_dlpack(values_in),
        *(from_dlpack(output) for output in outputs[:midpoint]),
    )
    qualified_runner(
        from_dlpack(values_in),
        *(from_dlpack(output) for output in outputs[midpoint:]),
    )
    torch.cuda.synchronize()


def test_common_memory_reduce_scan_matches_qualified_cutlass() -> None:
    cutlass.cuda.initialize_cuda_context()
    values_host = torch.arange(1, 65, dtype=torch.int32)
    values_in = values_host.cuda()
    common_outputs = [
        *_device_outputs(10, 64, torch.int32),
        *_device_outputs(7, 32, torch.int32),
    ]
    qualified_outputs = [torch.zeros_like(output) for output in common_outputs]
    outputs = [*common_outputs, *qualified_outputs]
    _launch_pair(
        _run_common_memory_reduce_scan,
        _run_qualified_memory_reduce_scan,
        values_in,
        outputs,
    )

    for common_output, qualified_output in zip(
        common_outputs, qualified_outputs, strict=True
    ):
        torch.testing.assert_close(common_output, qualified_output, atol=0, rtol=0)

    expected_inclusive = torch.cumsum(values_host.to(torch.int64), dim=0).to(
        torch.int32
    )
    expected_exclusive = expected_inclusive - values_host
    expected_exclusive_max = torch.cat(
        (torch.tensor([-2_147_483_648], dtype=torch.int32), values_host[:-1])
    )
    torch.testing.assert_close(common_outputs[0].cpu(), values_host, atol=0, rtol=0)
    torch.testing.assert_close(common_outputs[1].cpu(), values_host, atol=0, rtol=0)
    torch.testing.assert_close(
        common_outputs[2].cpu(), expected_exclusive, atol=0, rtol=0
    )
    torch.testing.assert_close(
        common_outputs[3].cpu(), expected_exclusive, atol=0, rtol=0
    )
    torch.testing.assert_close(
        common_outputs[4].cpu(), expected_inclusive, atol=0, rtol=0
    )
    torch.testing.assert_close(
        common_outputs[5].cpu(), expected_exclusive_max, atol=0, rtol=0
    )
    torch.testing.assert_close(common_outputs[6].cpu(), values_host, atol=0, rtol=0)
    expected_striped = torch.stack((values_host[:32], values_host[32:]), dim=1).reshape(
        -1
    )
    torch.testing.assert_close(
        common_outputs[7].cpu(), expected_striped, atol=0, rtol=0
    )
    torch.testing.assert_close(common_outputs[8].cpu(), values_host, atol=0, rtol=0)
    torch.testing.assert_close(common_outputs[9].cpu(), values_host, atol=0, rtol=0)
    torch.testing.assert_close(
        common_outputs[10].cpu(),
        torch.full((32,), int(values_host.sum()), dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        common_outputs[11].cpu(),
        torch.cumsum(values_host[:32], dim=0).to(torch.int32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        common_outputs[12].cpu(), expected_exclusive[:32], atol=0, rtol=0
    )
    torch.testing.assert_close(
        common_outputs[13].cpu(), expected_exclusive[:32], atol=0, rtol=0
    )
    torch.testing.assert_close(
        common_outputs[14].cpu(), expected_inclusive[:32], atol=0, rtol=0
    )
    torch.testing.assert_close(
        common_outputs[15].cpu(), values_host[:32], atol=0, rtol=0
    )
    torch.testing.assert_close(
        common_outputs[16].cpu(), values_host[:32], atol=0, rtol=0
    )


def test_common_collectives_isolate_two_physical_warps() -> None:
    cutlass.cuda.initialize_cuda_context()
    item_count = _TWO_WARP_THREADS * _WARP_ITEMS_PER_THREAD
    values_host = torch.arange(1, item_count + 1, dtype=torch.int32)
    values_in = values_host.cuda()
    common_outputs = [
        *_device_outputs(1, item_count, torch.int32),
        *_device_outputs(4, _TWO_WARP_THREADS, torch.int32),
    ]
    qualified_outputs = [torch.zeros_like(output) for output in common_outputs]
    _launch_pair(
        _run_common_two_physical_warps,
        _run_qualified_two_physical_warps,
        values_in,
        [*common_outputs, *qualified_outputs],
    )

    for common_output, qualified_output in zip(
        common_outputs, qualified_outputs, strict=True
    ):
        torch.testing.assert_close(common_output, qualified_output, atol=0, rtol=0)

    scalar_warps = values_host[:_TWO_WARP_THREADS].reshape(2, 32)
    expected_inclusive = torch.cumsum(scalar_warps, dim=1).to(torch.int32).reshape(-1)
    expected_exclusive = (
        torch.cumsum(scalar_warps, dim=1).to(torch.int32) - scalar_warps
    ).reshape(-1)
    expected_totals = scalar_warps.sum(dim=1).to(torch.int32).repeat_interleave(32)

    torch.testing.assert_close(common_outputs[0].cpu(), values_host, atol=0, rtol=0)
    torch.testing.assert_close(
        common_outputs[1].cpu(), expected_exclusive, atol=0, rtol=0
    )
    torch.testing.assert_close(
        common_outputs[2].cpu(), expected_exclusive, atol=0, rtol=0
    )
    torch.testing.assert_close(
        common_outputs[3].cpu(), expected_inclusive, atol=0, rtol=0
    )
    torch.testing.assert_close(common_outputs[4].cpu(), expected_totals, atol=0, rtol=0)


def test_common_difference_discontinuity_shuffle_matches_qualified_cutlass() -> None:
    cutlass.cuda.initialize_cuda_context()
    values_host = torch.tensor([idx // 3 - 7 for idx in range(64)], dtype=torch.int32)
    values_in = values_host.cuda()
    common_outputs = _device_outputs(4, 64, torch.int32)
    qualified_outputs = [torch.zeros_like(output) for output in common_outputs]
    _launch_pair(
        _run_common_difference_shuffle,
        _run_qualified_difference_shuffle,
        values_in,
        [*common_outputs, *qualified_outputs],
    )

    for common_output, qualified_output in zip(
        common_outputs[:3], qualified_outputs[:3], strict=True
    ):
        torch.testing.assert_close(common_output, qualified_output, atol=0, rtol=0)
    # BlockShuffle::Up leaves the first flattened tile position undefined when
    # the portable call does not provide a qualified-only block prefix.
    torch.testing.assert_close(
        common_outputs[3][1:], qualified_outputs[3][1:], atol=0, rtol=0
    )
    torch.testing.assert_close(common_outputs[0].cpu(), values_host, atol=0, rtol=0)
    assert common_outputs[2].dtype == torch.int32


def test_common_sort_rank_topk_matches_qualified_cutlass_without_mutation() -> None:
    cutlass.cuda.initialize_cuda_context()
    values_host = torch.tensor(
        [((idx * 29 + 17) % 251) for idx in range(_TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    values_in = values_host.cuda()
    common_outputs = _device_outputs(6, _TOTAL_ITEMS, torch.int32)
    qualified_outputs = [torch.zeros_like(output) for output in common_outputs]
    _launch_pair(
        _run_common_sort_rank_topk,
        _run_qualified_sort_rank_topk,
        values_in,
        [*common_outputs, *qualified_outputs],
    )

    for index in range(4):
        torch.testing.assert_close(
            common_outputs[index], qualified_outputs[index], atol=0, rtol=0
        )
    torch.testing.assert_close(common_outputs[0].cpu(), values_host, atol=0, rtol=0)
    torch.testing.assert_close(
        common_outputs[1][:_VALID_ITEMS].cpu(),
        torch.sort(values_host[:_VALID_ITEMS]).values,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        common_outputs[2].cpu(),
        torch.sort(values_host, descending=True).values,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        torch.sort(common_outputs[3].cpu()).values,
        torch.arange(_TOTAL_ITEMS, dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    for index, descending in ((4, True), (5, False)):
        common_window = torch.sort(common_outputs[index][:_TOPK_K].cpu()).values
        qualified_window = torch.sort(qualified_outputs[index][:_TOPK_K].cpu()).values
        expected = torch.sort(values_host[:_VALID_ITEMS], descending=descending).values[
            :_TOPK_K
        ]
        torch.testing.assert_close(common_window, qualified_window, atol=0, rtol=0)
        torch.testing.assert_close(
            common_window, torch.sort(expected).values, atol=0, rtol=0
        )


def test_common_physical_warp_merge_sort_matches_qualified_cutlass() -> None:
    cutlass.cuda.initialize_cuda_context()
    values_host = torch.tensor(
        [((idx * 11 + 7) % 67) - 31 for idx in range(64)],
        dtype=torch.int32,
    )
    values_in = values_host.cuda()
    original_out = torch.zeros_like(values_in)
    common_out = torch.zeros_like(values_in)
    qualified_out = torch.zeros_like(values_in)

    _run_warp_merge_sort(
        from_dlpack(values_in),
        from_dlpack(original_out),
        from_dlpack(common_out),
        from_dlpack(qualified_out),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(original_out.cpu(), values_host, atol=0, rtol=0)
    torch.testing.assert_close(common_out, qualified_out, atol=0, rtol=0)
    torch.testing.assert_close(
        common_out.cpu(),
        torch.sort(values_host, descending=True).values,
        atol=0,
        rtol=0,
    )


@pytest.mark.evidence_for(
    "group.merge_sort_pairs", backend="cutlass", evidence="runtime"
)
@pytest.mark.evidence_for(
    "group.radix_sort_pairs", backend="cutlass", evidence="runtime"
)
def test_common_pair_sorts_match_qualified_oracles_without_mutation() -> None:
    cutlass.cuda.initialize_cuda_context()
    keys_host = torch.tensor(
        [((idx * 29 + 17) % 31) - 15 for idx in range(_TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    values_host = torch.arange(_TOTAL_ITEMS, dtype=torch.int32) * 7 + 3
    keys_in = keys_host.cuda()
    values_in = values_host.cuda()
    common_outputs = _device_outputs(6, _TOTAL_ITEMS, torch.int32)
    qualified_outputs = _device_outputs(6, _TOTAL_ITEMS, torch.int32)
    for runner, outputs in (
        (_run_common_pair_sort, common_outputs),
        (_run_qualified_pair_sort, qualified_outputs),
    ):
        runner(
            from_dlpack(keys_in),
            from_dlpack(values_in),
            *(from_dlpack(output) for output in outputs),
        )
    torch.cuda.synchronize()

    source_pairs = set(zip(keys_host.tolist(), values_host.tolist(), strict=True))
    for outputs in (common_outputs, qualified_outputs):
        torch.testing.assert_close(outputs[0].cpu(), keys_host, atol=0, rtol=0)
        torch.testing.assert_close(outputs[1].cpu(), values_host, atol=0, rtol=0)
        assert (
            set(zip(outputs[2].cpu().tolist(), outputs[3].cpu().tolist(), strict=True))
            == source_pairs
        )
        assert (
            set(zip(outputs[4].cpu().tolist(), outputs[5].cpu().tolist(), strict=True))
            == source_pairs
        )
        torch.testing.assert_close(
            outputs[2].cpu(),
            torch.sort(keys_host, descending=True).values,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            outputs[4].cpu(), torch.sort(keys_host).values, atol=0, rtol=0
        )
    torch.testing.assert_close(common_outputs[4], qualified_outputs[4], atol=0, rtol=0)
    torch.testing.assert_close(common_outputs[5], qualified_outputs[5], atol=0, rtol=0)


@pytest.mark.evidence_for("group.topk_max_pairs", backend="cutlass", evidence="runtime")
@pytest.mark.evidence_for("group.topk_min_pairs", backend="cutlass", evidence="runtime")
def test_common_pair_topk_matches_qualified_and_preserves_association() -> None:
    cutlass.cuda.initialize_cuda_context()
    keys_host = torch.tensor(
        [((idx * 29 + 17) % 251) for idx in range(_TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    values_host = torch.arange(_TOTAL_ITEMS, dtype=torch.int32) * 7 + 3
    keys_in = keys_host.cuda()
    values_in = values_host.cuda()
    common_outputs = _device_outputs(6, _TOTAL_ITEMS, torch.int32)
    qualified_outputs = _device_outputs(6, _TOTAL_ITEMS, torch.int32)

    _run_common_pair_topk(
        from_dlpack(keys_in),
        from_dlpack(values_in),
        *(from_dlpack(output) for output in common_outputs),
    )
    _run_qualified_pair_topk(
        from_dlpack(keys_in),
        from_dlpack(values_in),
        *(from_dlpack(output) for output in qualified_outputs),
    )
    torch.cuda.synchronize()

    for outputs in (common_outputs, qualified_outputs):
        torch.testing.assert_close(outputs[0].cpu(), keys_host, atol=0, rtol=0)
        torch.testing.assert_close(outputs[1].cpu(), values_host, atol=0, rtol=0)
    source_pairs = list(
        zip(
            keys_host[:_VALID_ITEMS].tolist(),
            values_host[:_VALID_ITEMS].tolist(),
            strict=True,
        )
    )
    for common_key, common_value, qualified_key, qualified_value, descending in (
        (*common_outputs[2:4], *qualified_outputs[2:4], True),
        (*common_outputs[4:6], *qualified_outputs[4:6], False),
    ):
        actual = sorted(
            zip(
                common_key[:_TOPK_K].cpu().tolist(),
                common_value[:_TOPK_K].cpu().tolist(),
                strict=True,
            )
        )
        qualified = sorted(
            zip(
                qualified_key[:_TOPK_K].cpu().tolist(),
                qualified_value[:_TOPK_K].cpu().tolist(),
                strict=True,
            )
        )
        assert actual == qualified
        expected = sorted(
            sorted(source_pairs, key=lambda pair: pair[0], reverse=descending)[:_TOPK_K]
        )
        assert actual == expected


@pytest.mark.evidence_for("group.sum", backend="cutlass", evidence="runtime")
def test_common_cluster_reduce_and_sum_match_qualified_cutlass(
    cutlass_cluster_runtime_available: None,
) -> None:
    del cutlass_cluster_runtime_available
    cutlass.cuda.initialize_cuda_context()
    values_host = torch.arange(1, _CLUSTER_THREADS + 1, dtype=torch.int32)
    values_in = values_host.cuda()
    outputs = _device_outputs(4, _CLUSTER_THREADS, torch.int32)

    _run_cluster_reduce(
        from_dlpack(values_in),
        *(from_dlpack(output) for output in outputs),
    )
    torch.cuda.synchronize()

    expected = torch.full_like(values_host, int(values_host.sum()))
    for output in outputs:
        torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


def test_common_histogram_and_decode_match_qualified_cutlass_layouts() -> None:
    cutlass.cuda.initialize_cuda_context()
    samples_host = torch.tensor(
        [((idx * 7 + idx // 3) % 32) for idx in range(64)],
        dtype=torch.uint8,
    )
    run_values_host = torch.tensor(list(range(_BLOCK_THREADS)), dtype=torch.uint32)
    samples_in = samples_host.cuda()
    run_values_in = run_values_host.cuda()
    common_histogram = torch.zeros((32,), dtype=torch.int32, device="cuda")
    common_decoded = torch.zeros((64,), dtype=torch.uint32, device="cuda")
    qualified_histogram = torch.zeros_like(common_histogram)
    qualified_decoded = torch.zeros_like(common_decoded)

    _run_common_histogram_decode(
        from_dlpack(samples_in),
        from_dlpack(run_values_in),
        from_dlpack(common_histogram),
        from_dlpack(common_decoded),
    )
    _run_qualified_histogram_decode(
        from_dlpack(samples_in),
        from_dlpack(run_values_in),
        from_dlpack(qualified_histogram),
        from_dlpack(qualified_decoded),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(common_histogram, qualified_histogram, atol=0, rtol=0)
    torch.testing.assert_close(common_decoded, qualified_decoded, atol=0, rtol=0)
    torch.testing.assert_close(
        common_histogram.cpu(),
        torch.bincount(samples_host.to(torch.int64), minlength=32).to(torch.int32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        common_decoded.cpu(),
        torch.repeat_interleave(run_values_host, 2),
        atol=0,
        rtol=0,
    )


def test_common_profile_excludes_qualified_only_controls() -> None:
    for name in (
        "merge_sort_pairs",
        "radix_sort_pairs",
        "topk_max_pairs",
        "topk_min_pairs",
    ):
        assert hasattr(coop, name)
        assert hasattr(cutlass_coop, name)
    common_parameters = inspect.signature(coop.adjacent_difference).parameters
    qualified_parameters = inspect.signature(
        cutlass_coop.adjacent_difference
    ).parameters
    assert "difference_op" not in common_parameters
    assert "difference_op" in qualified_parameters
    common_parameters = inspect.signature(coop.discontinuity).parameters
    qualified_parameters = inspect.signature(cutlass_coop.discontinuity).parameters
    assert "flag_op" not in common_parameters
    assert "flag_op" in qualified_parameters
