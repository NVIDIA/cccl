# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict positive typing fixture for dual-use Numba scoped primitives."""

# pyright: strict, reportPrivateUsage=none, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

from typing_extensions import assert_type


def _add(left: int, right: int) -> int:
    return left + right


def _subtract(left: int, right: int) -> int:
    return left - right


def _different(left: int, right: int) -> bool:
    return left != right


def _less(left: int, right: int) -> bool:
    return left < right


if TYPE_CHECKING:
    import cuda.coop.numba_mlir as coop

    storage = coop.TempStorage()
    values = coop.ThreadData(2, int)
    output = coop.ThreadData(2, int)
    flags = coop.ThreadData(2, int)
    ranks = coop.ThreadData(2, int)

    # Unprefixed names are host factories outside compilation. Their legacy
    # scoped parameter names and positional conventions remain available.
    block_load = coop._block.load(int, 128, 2, "direct", 64, 0)
    block_store = coop._block.store(int, 128, 2, "direct", 64)
    block_reduce = coop._block.reduce(int, 128, _add, 1, "warp_reductions", 64)
    block_sum = coop._block.sum(int, 128, 1, "warp_reductions", 64)
    block_scan = coop._block.scan(int, 128, 2)
    block_exclusive_sum = coop._block.exclusive_sum(int, 128, 2)
    block_inclusive_sum = coop._block.inclusive_sum(int, 128, 2)
    block_exclusive_scan = coop._block.exclusive_scan(int, 128, _add, 2)
    block_inclusive_scan = coop._block.inclusive_scan(int, 128, _add, 2)
    block_exchange = coop._block.exchange(
        coop._block.BlockExchangeType.StripedToBlocked,
        int,
        128,
        2,
    )
    block_adjacent = coop._block.adjacent_difference(
        coop._block.BlockAdjacentDifferenceType.SubtractLeft,
        int,
        128,
        2,
        _subtract,
    )
    block_discontinuity = coop._block.discontinuity(int, 128, 2, _different, int)
    block_shuffle = coop._block.shuffle(
        coop._block.BlockShuffleType.Up,
        int,
        128,
        2,
    )
    block_merge_keys = coop._block.merge_sort_keys(int, 128, 2, _less)
    block_merge_pairs = coop._block.merge_sort_pairs(int, int, 128, 2, _less)
    block_radix_keys = coop._block.radix_sort_keys(int, 128, 2)
    block_radix_pairs = coop._block.radix_sort_pairs(int, int, 128, 2)
    block_radix_rank = coop._block.radix_rank(int, 128, 2, 0, 32)
    block_topk_keys = coop._block.topk_max_keys(int, 128, 2)
    block_topk_pairs = coop._block.topk_min_pairs(int, int, 128, 2)

    assert_type(block_load.temp_storage_bytes, int)
    assert_type(block_load.temp_storage_alignment, int)
    assert_type(block_load.files, list[str])
    block_load(object(), values, 64, 0, temp_storage=storage)
    block_store(object(), values, 64, temp_storage=storage)
    assert_type(block_reduce(1, 64, temp_storage=storage), int)
    assert_type(block_sum(1, 64, temp_storage=storage), int)
    block_scan(values, output, temp_storage=storage)
    block_scan(values, output, flags, temp_storage=storage)
    block_exclusive_sum(values, output, temp_storage=storage)
    block_inclusive_sum(values, output, temp_storage=storage)
    block_exclusive_scan(values, output, temp_storage=storage)
    block_inclusive_scan(values, output, temp_storage=storage)
    block_exchange(values, output, temp_storage=storage)
    block_exchange(values, output, ranks, flags, temp_storage=storage)
    block_adjacent(values, output, temp_storage=storage)
    block_adjacent(values, output, 64, 0, temp_storage=storage)
    block_discontinuity(values, flags, temp_storage=storage)
    block_discontinuity(values, flags, output, 0, 0, temp_storage=storage)
    block_shuffle(values, output, temp_storage=storage)
    block_shuffle(values, output, flags, temp_storage=storage)
    block_merge_keys(values, temp_storage=storage)
    block_merge_keys(values, 64, 0, temp_storage=storage)
    block_merge_pairs(values, output, temp_storage=storage)
    block_merge_pairs(values, output, 64, 0, temp_storage=storage)
    block_radix_keys(values, temp_storage=storage)
    block_radix_keys(values, 0, 32, temp_storage=storage)
    block_radix_pairs(values, output, temp_storage=storage)
    block_radix_pairs(values, output, 0, 32, temp_storage=storage)
    block_radix_rank(values, ranks, temp_storage=storage)
    block_radix_rank(values, ranks, flags, temp_storage=storage)
    block_topk_keys(values, 1, temp_storage=storage)
    block_topk_keys(values, 1, 64, 0, 32, temp_storage=storage)
    block_topk_pairs(values, output, 1, temp_storage=storage)
    block_topk_pairs(values, output, 1, 64, 0, 32, temp_storage=storage)

    histogram_factory = coop._block.histogram(
        item_dtype=int,
        counter_dtype=int,
        threads_per_block=128,
        items_per_thread=2,
    )
    histogram_factory.init(flags)
    histogram_factory.composite(values, flags)
    run_length_factory = coop._block.run_length(
        item_dtype=int,
        threads_per_block=128,
        runs_per_thread=2,
        decoded_items_per_thread=2,
    )
    run_length_parent = run_length_factory(values, values)
    run_length_parent.decode(output)
    assert_type(run_length_factory.temp_storage_bytes, int)

    # ``make_*`` is the unambiguous spelling for the same host factories.
    coop._block.make_load(int, threads_per_block=128, num_valid_items=64)
    coop._block.make_store(int, threads_per_block=128, num_valid_items=64)
    coop._block.make_reduce(int, 128, _add, num_valid=64)
    coop._block.make_sum(int, 128, num_valid=64)
    coop._block.make_scan(int, 128, 2)
    coop._block.make_exclusive_sum(int, 128, 2)
    coop._block.make_inclusive_sum(int, 128, 2)
    coop._block.make_exclusive_scan(int, 128, _add, 2)
    coop._block.make_inclusive_scan(int, 128, _add, 2)
    coop._block.make_exchange(int, threads_per_block=128, items_per_thread=2)
    coop._block.make_adjacent_difference(
        int,
        threads_per_block=128,
        items_per_thread=2,
        difference_op=_subtract,
    )
    coop._block.make_discontinuity(int, 128, 2, _different, int)
    coop._block.make_shuffle(int, threads_per_block=128, items_per_thread=2)
    coop._block.make_merge_sort_keys(int, 128, 2, _less)
    coop._block.make_merge_sort_pairs(int, int, 128, 2, _less)
    coop._block.make_radix_sort_keys(int, 128, 2)
    coop._block.make_radix_sort_pairs(int, int, 128, 2)
    coop._block.make_radix_rank(int, 128, 2, 0, 32)
    coop._block.make_histogram(int, int, threads_per_block=128)
    coop._block.make_run_length(int, threads_per_block=128)
    coop._block.make_topk_max_keys(int, 128, 2)
    coop._block.make_topk_min_pairs(int, int, 128, 2)

    warp_load = coop._warp.load(int, 2, 32, "direct", 16, 0)
    warp_store = coop._warp.store(int, 2, 32, "direct", 16)
    warp_reduce = coop._warp.reduce(int, _add, 32, 16)
    warp_sum = coop._warp.sum(int, 32, 16)
    warp_max = coop._warp.max(int, 32, 16)
    warp_min = coop._warp.min(int, 32, 16)
    warp_exclusive_sum = coop._warp.exclusive_sum(int, 32, output)
    warp_inclusive_sum = coop._warp.inclusive_sum(int, 32, output)
    warp_exclusive_scan = coop._warp.exclusive_scan(int, _add, 0, 32, 16, output)
    warp_inclusive_scan = coop._warp.inclusive_scan(int, _add, 0, 32, 16, output)
    warp_exchange = coop._warp.exchange(int, 2, 32)
    warp_merge_keys = coop._warp.merge_sort_keys(int, 2, _less)
    warp_merge_pairs = coop._warp.merge_sort_pairs(int, int, 2, _less)

    warp_load(object(), values, 16, 0, temp_storage=storage)
    warp_store(object(), values, 16, temp_storage=storage)
    assert_type(warp_reduce(1, 16, temp_storage=storage), int)
    assert_type(warp_sum(1, 16, temp_storage=storage), int)
    assert_type(warp_max(1, 16, temp_storage=storage), int)
    assert_type(warp_min(1, 16, temp_storage=storage), int)
    assert_type(warp_exclusive_sum(1, output, temp_storage=storage), int)
    assert_type(warp_inclusive_sum(1, output, temp_storage=storage), int)
    assert_type(warp_exclusive_scan(1, 16, output, temp_storage=storage), int)
    assert_type(warp_inclusive_scan(1, 16, output, temp_storage=storage), int)
    warp_exchange(values, output, temp_storage=storage)
    warp_merge_keys(values, temp_storage=storage)
    warp_merge_pairs(values, output, temp_storage=storage)

    coop._warp.make_load(int, 2, 32, "direct", 16, 0)
    coop._warp.make_store(int, 2, 32, "direct", 16)
    coop._warp.make_reduce(int, _add, 32, 16)
    coop._warp.make_sum(int, 32, 16)
    coop._warp.make_max(int, 32, 16)
    coop._warp.make_min(int, 32, 16)
    coop._warp.make_exclusive_sum(int, 32, output)
    coop._warp.make_inclusive_sum(int, 32, output)
    coop._warp.make_exclusive_scan(int, _add, 0, 32, 16, output)
    coop._warp.make_inclusive_scan(int, _add, 0, 32, 16, output)
    coop._warp.make_exchange(int, 2, 32)
    coop._warp.make_merge_sort_keys(int, 2, _less)
    coop._warp.make_merge_sort_pairs(int, int, 2, _less)

    # In a compiled kernel, the same unprefixed objects are rewritten as
    # device operations rather than invoked as host factories.
    coop._block.load(object(), values)
    coop._block.load(object(), values, 64, 0)
    coop._block.store(object(), values)
    coop._block.store(object(), values, 64)
    assert_type(coop._block.reduce(1, binary_op=_add), int)
    assert_type(coop._block.reduce(1, 64, binary_op=_add), int)
    assert_type(coop._block.sum(1), int)
    assert_type(coop._block.sum(1, 64), int)
    coop._block.scan(values, output)
    coop._block.scan(values, output, flags)
    coop._block.exclusive_sum(values, output)
    coop._block.inclusive_sum(values, output)
    coop._block.exclusive_scan(values, output, scan_op=_add)
    coop._block.inclusive_scan(values, output, scan_op=_add)
    coop._block.exchange(values, output)
    coop._block.exchange(values, output, ranks, flags)
    coop._block.adjacent_difference(values, output, difference_op=_subtract)
    coop._block.adjacent_difference(values, output, 64, 0, difference_op=_subtract)
    coop._block.discontinuity(values, flags, flag_op=_different)
    coop._block.discontinuity(values, flags, output, 0, 0, flag_op=_different)
    coop._block.shuffle(values, output)
    coop._block.shuffle(values, output, flags)
    coop._block.merge_sort_keys(values, compare_op=_less)
    coop._block.merge_sort_keys(values, 64, 0, compare_op=_less)
    coop._block.merge_sort_pairs(values, output, compare_op=_less)
    coop._block.merge_sort_pairs(values, output, 64, 0, compare_op=_less)
    coop._block.radix_sort_keys(values)
    coop._block.radix_sort_keys(values, 0, 32)
    coop._block.radix_sort_pairs(values, output)
    coop._block.radix_sort_pairs(values, output, 0, 32)
    coop._block.radix_rank(values, ranks)
    coop._block.radix_rank(values, ranks, flags)
    coop._block.histogram(values, flags)
    coop._block.run_length(values, values).decode(output)
    coop._block.topk_max_keys(values, 1)
    coop._block.topk_max_keys(values, 1, 64, 0, 32)
    coop._block.topk_min_pairs(values, output, 1)
    coop._block.topk_min_pairs(values, output, 1, 64, 0, 32)

    coop._warp.load(object(), values)
    coop._warp.store(object(), values)
    assert_type(coop._warp.reduce(1, binary_op=_add), int)
    assert_type(coop._warp.sum(1), int)
    assert_type(coop._warp.max(1), int)
    assert_type(coop._warp.min(1), int)
    assert_type(coop._warp.exclusive_sum(1), int)
    assert_type(coop._warp.inclusive_sum(1), int)
    assert_type(coop._warp.exclusive_scan(1, scan_op=_add), int)
    assert_type(coop._warp.inclusive_scan(1, scan_op=_add), int)
    coop._warp.exchange(values, output)
    coop._warp.merge_sort_keys(values, compare_op=_less)
    coop._warp.merge_sort_keys(
        values,
        compare_op=_less,
        valid_items=53,
        oob_default=0,
    )
    coop._warp.merge_sort_pairs(values, output, compare_op=_less)
    coop._warp.merge_sort_pairs(
        values,
        output,
        compare_op=_less,
        valid_items=53,
        oob_default=0,
    )
