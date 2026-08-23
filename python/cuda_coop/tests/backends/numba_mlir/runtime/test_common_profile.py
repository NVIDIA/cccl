# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Runtime conformance probes for the backend-neutral V1 profile."""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

THREADS = 32
ITEMS_PER_THREAD = 2
VALID_ITEMS = THREADS * ITEMS_PER_THREAD - 5
INT32_MAX = np.int32(np.iinfo(np.int32).max)
NUMERIC_DTYPE_SEGMENTS = 10
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _common_numeric_dtype_kernel(source, values_out, flags_out, reduce_out):
    tid = cuda.threadIdx.x
    item_count = THREADS * ITEMS_PER_THREAD
    common_group = coop.this_block()
    qualified_group = numba_coop.this_block()

    common_loaded = coop.ThreadData(ITEMS_PER_THREAD, dtype=source.dtype)
    qualified_loaded = numba_coop.ThreadData(ITEMS_PER_THREAD, dtype=source.dtype)
    coop.load(common_group, source, common_loaded)
    numba_coop.load(qualified_group, source, qualified_loaded)
    coop.store(common_group, values_out, common_loaded)
    numba_coop.store(
        qualified_group,
        values_out,
        qualified_loaded,
        offset=item_count,
    )

    common_exchange = coop.exchange(
        common_group,
        common_loaded,
        mode="blocked_to_striped",
    )
    qualified_exchange = numba_coop.exchange(
        qualified_group,
        qualified_loaded,
        mode="blocked_to_striped",
    )
    common_shuffle = coop.shuffle(common_group, common_loaded)
    qualified_shuffle = numba_coop.shuffle(qualified_group, qualified_loaded)
    boundary = source[0]
    common_adjacent = coop.adjacent_difference(
        common_group,
        common_loaded,
        tile_predecessor_item=boundary,
    )
    qualified_adjacent = numba_coop.adjacent_difference(
        qualified_group,
        qualified_loaded,
        tile_predecessor_item=boundary,
    )
    common_scan = coop.inclusive_scan(
        common_group,
        common_loaded,
        scan_op="max",
    )
    qualified_scan = numba_coop.inclusive_scan(
        qualified_group,
        qualified_loaded,
        scan_op="max",
    )
    common_flags = coop.discontinuity(
        common_group,
        common_loaded,
        tile_predecessor_item=boundary,
    )
    qualified_flags = numba_coop.discontinuity(
        qualified_group,
        qualified_loaded,
        tile_predecessor_item=boundary,
    )
    common_max = coop.reduce(common_group, common_loaded, binary_op="max")
    qualified_max = numba_coop.reduce(
        qualified_group,
        qualified_loaded,
        binary_op="max",
    )

    for item in range(ITEMS_PER_THREAD):
        index = tid * ITEMS_PER_THREAD + item
        values_out[2 * item_count + index] = common_exchange[item]
        values_out[3 * item_count + index] = qualified_exchange[item]
        if index + 1 < item_count:
            values_out[4 * item_count + index] = common_shuffle[item]
            values_out[5 * item_count + index] = qualified_shuffle[item]
        values_out[6 * item_count + index] = common_adjacent[item]
        values_out[7 * item_count + index] = qualified_adjacent[item]
        values_out[8 * item_count + index] = common_scan[item]
        values_out[9 * item_count + index] = qualified_scan[item]
        flags_out[index] = common_flags[item]
        flags_out[item_count + index] = qualified_flags[item]
    reduce_out[tid] = common_max
    reduce_out[THREADS + tid] = qualified_max


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64],
)
def test_common_numeric_dtype_closure_matches_qualified_numba_and_oracles(dtype):
    item_count = THREADS * ITEMS_PER_THREAD
    values = np.arange(1, item_count + 1, dtype=dtype)
    original = values.copy()
    outputs = np.zeros(NUMERIC_DTYPE_SEGMENTS * item_count, dtype=dtype)
    flags = np.full(2 * item_count, -1, dtype=np.int32)
    reductions = np.zeros(2 * THREADS, dtype=dtype)

    _common_numeric_dtype_kernel[1, THREADS](values, outputs, flags, reductions)
    cuda.synchronize()

    np.testing.assert_array_equal(values, original)
    segments = outputs.reshape(NUMERIC_DTYPE_SEGMENTS, item_count)
    np.testing.assert_array_equal(segments[0], values)
    np.testing.assert_array_equal(segments[1], values)

    expected_exchange = np.stack(
        (values[:THREADS], values[THREADS:]),
        axis=1,
    ).reshape(-1)
    np.testing.assert_array_equal(segments[2], expected_exchange)
    np.testing.assert_array_equal(segments[3], expected_exchange)

    expected_shuffle = np.zeros_like(values)
    expected_shuffle[:-1] = values[1:]
    np.testing.assert_array_equal(segments[4], expected_shuffle)
    np.testing.assert_array_equal(segments[5], expected_shuffle)

    expected_adjacent = np.ones_like(values)
    expected_adjacent[0] = 0
    np.testing.assert_array_equal(segments[6], expected_adjacent)
    np.testing.assert_array_equal(segments[7], expected_adjacent)
    np.testing.assert_array_equal(segments[8], values)
    np.testing.assert_array_equal(segments[9], values)

    expected_flags = np.ones(item_count, dtype=np.int32)
    expected_flags[0] = 0
    np.testing.assert_array_equal(flags[:item_count], expected_flags)
    np.testing.assert_array_equal(flags[item_count:], expected_flags)
    np.testing.assert_array_equal(reductions, np.full_like(reductions, values[-1]))


@cuda.jit
def _common_transform_kernel(
    d_input,
    d_exchange,
    d_exclusive,
    d_inclusive,
    d_adjacent,
    d_flags,
    d_original,
):
    tid = cuda.threadIdx.x
    group = coop.this_block()
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(ITEMS_PER_THREAD):
        items[item] = d_input[tid + item * THREADS]

    exchanged = coop.exchange(group, items)
    adjacent = coop.adjacent_difference(group, items)
    flags = coop.discontinuity(group, items)
    d_exclusive[tid] = coop.exclusive_sum(group, d_input[tid])
    d_inclusive[tid] = coop.inclusive_sum(group, d_input[tid])
    for item in range(ITEMS_PER_THREAD):
        blocked_index = tid * ITEMS_PER_THREAD + item
        d_exchange[blocked_index] = exchanged[item]
        d_adjacent[blocked_index] = adjacent[item]
        d_flags[blocked_index] = flags[item]
        d_original[tid + item * THREADS] = items[item]


@cuda.jit
def _qualified_transform_kernel(
    d_input,
    d_exchange,
    d_exclusive,
    d_inclusive,
    d_adjacent,
    d_flags,
    d_original,
):
    tid = cuda.threadIdx.x
    group = numba_coop.this_block()
    items = numba_coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(ITEMS_PER_THREAD):
        items[item] = d_input[tid + item * THREADS]

    exchanged = numba_coop.exchange(group, items)
    adjacent = numba_coop.adjacent_difference(group, items)
    flags = numba_coop.discontinuity(group, items)
    d_exclusive[tid] = numba_coop.exclusive_sum(group, d_input[tid])
    d_inclusive[tid] = numba_coop.inclusive_sum(group, d_input[tid])
    for item in range(ITEMS_PER_THREAD):
        blocked_index = tid * ITEMS_PER_THREAD + item
        d_exchange[blocked_index] = exchanged[item]
        d_adjacent[blocked_index] = adjacent[item]
        d_flags[blocked_index] = flags[item]
        d_original[tid + item * THREADS] = items[item]


def test_common_transforms_are_functional_and_preserve_inputs():
    values = np.arange(1, THREADS * ITEMS_PER_THREAD + 1, dtype=np.int32)
    exchange = np.zeros_like(values)
    exclusive = np.zeros(THREADS, dtype=np.int32)
    inclusive = np.zeros(THREADS, dtype=np.int32)
    adjacent = np.zeros_like(values)
    flags = np.zeros_like(values)
    original = np.zeros_like(values)
    qualified_outputs = [
        np.zeros_like(exchange),
        np.zeros_like(exclusive),
        np.zeros_like(inclusive),
        np.zeros_like(adjacent),
        np.zeros_like(flags),
        np.zeros_like(original),
    ]

    _common_transform_kernel[1, THREADS](
        values,
        exchange,
        exclusive,
        inclusive,
        adjacent,
        flags,
        original,
    )
    _qualified_transform_kernel[1, THREADS](values, *qualified_outputs)

    for common_output, qualified_output in zip(
        (exchange, exclusive, inclusive, flags, original),
        (
            qualified_outputs[0],
            qualified_outputs[1],
            qualified_outputs[2],
            qualified_outputs[4],
            qualified_outputs[5],
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(common_output, qualified_output)
    np.testing.assert_array_equal(adjacent[1:], qualified_outputs[3][1:])

    np.testing.assert_array_equal(exchange, values)
    np.testing.assert_array_equal(
        exclusive,
        np.concatenate((np.asarray([0], dtype=np.int32), np.cumsum(values[:31]))),
    )
    np.testing.assert_array_equal(inclusive, np.cumsum(values[:THREADS]))
    np.testing.assert_array_equal(original, values)
    blocked = values.reshape(ITEMS_PER_THREAD, THREADS).T.reshape(-1)
    np.testing.assert_array_equal(adjacent[1:], np.diff(blocked))
    np.testing.assert_array_equal(flags, np.ones_like(flags))


@cuda.jit
def _common_sort_rank_topk_kernel(
    d_input,
    d_merge,
    d_radix,
    d_ranks,
    d_topk_min,
    d_topk_max,
    d_original,
    k,
):
    tid = cuda.threadIdx.x
    group = coop.this_block()
    keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(ITEMS_PER_THREAD):
        keys[item] = d_input[tid * ITEMS_PER_THREAD + item]

    merged = coop.merge_sort_keys(
        group,
        keys,
        valid_items=VALID_ITEMS,
        oob_default=INT32_MAX,
    )
    radix = coop.radix_sort_keys(group, keys)
    ranks = coop.radix_rank(group, keys, begin_bit=0, end_bit=6)
    topk_min = coop.topk_min_keys(group, keys, k, valid_items=VALID_ITEMS)
    topk_max = coop.topk_max_keys(group, keys, k, valid_items=VALID_ITEMS)
    for item in range(ITEMS_PER_THREAD):
        index = tid * ITEMS_PER_THREAD + item
        d_merge[index] = merged[item]
        d_radix[index] = radix[item]
        d_ranks[index] = ranks[item]
        if index < k:
            d_topk_min[index] = topk_min[item]
            d_topk_max[index] = topk_max[item]
        d_original[index] = keys[item]


@cuda.jit
def _qualified_sort_rank_topk_kernel(
    d_input,
    d_merge,
    d_radix,
    d_ranks,
    d_topk_min,
    d_topk_max,
    d_original,
    k,
):
    tid = cuda.threadIdx.x
    group = numba_coop.this_block()
    keys = numba_coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(ITEMS_PER_THREAD):
        keys[item] = d_input[tid * ITEMS_PER_THREAD + item]

    merged = numba_coop.merge_sort_keys(
        group,
        keys,
        valid_items=VALID_ITEMS,
        oob_default=INT32_MAX,
    )
    radix = numba_coop.radix_sort_keys(group, keys)
    ranks = numba_coop.radix_rank(group, keys, begin_bit=0, end_bit=6)
    topk_min = numba_coop.topk_min_keys(
        group,
        keys,
        k,
        valid_items=VALID_ITEMS,
    )
    topk_max = numba_coop.topk_max_keys(
        group,
        keys,
        k,
        valid_items=VALID_ITEMS,
    )
    for item in range(ITEMS_PER_THREAD):
        index = tid * ITEMS_PER_THREAD + item
        d_merge[index] = merged[item]
        d_radix[index] = radix[item]
        d_ranks[index] = ranks[item]
        if index < k:
            d_topk_min[index] = topk_min[item]
            d_topk_max[index] = topk_max[item]
        d_original[index] = keys[item]


def test_common_sort_rank_and_topk_preserve_input_payload():
    values = np.arange(THREADS * ITEMS_PER_THREAD, 0, -1, dtype=np.int32)
    merge = np.zeros_like(values)
    radix = np.zeros_like(values)
    ranks = np.zeros_like(values)
    topk_min = np.full_like(values, -1)
    topk_max = np.full_like(values, -1)
    original = np.zeros_like(values)
    k = np.int32(7)
    qualified_outputs = [np.zeros_like(values) for _ in range(6)]
    qualified_outputs[3].fill(-1)
    qualified_outputs[4].fill(-1)

    _common_sort_rank_topk_kernel[1, THREADS](
        values,
        merge,
        radix,
        ranks,
        topk_min,
        topk_max,
        original,
        k,
    )
    _qualified_sort_rank_topk_kernel[1, THREADS](
        values,
        *qualified_outputs,
        k,
    )

    for common_output, qualified_output in zip(
        (radix, ranks, original),
        (
            qualified_outputs[1],
            qualified_outputs[2],
            qualified_outputs[5],
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(common_output, qualified_output)

    expected_valid = np.sort(values[:VALID_ITEMS])
    np.testing.assert_array_equal(
        merge[:VALID_ITEMS],
        qualified_outputs[0][:VALID_ITEMS],
    )
    np.testing.assert_array_equal(merge[:VALID_ITEMS], expected_valid)
    np.testing.assert_array_equal(radix, np.sort(values))
    np.testing.assert_array_equal(np.sort(ranks), np.arange(values.size))
    np.testing.assert_array_equal(
        np.sort(topk_min[:k]),
        np.sort(qualified_outputs[3][:k]),
    )
    np.testing.assert_array_equal(np.sort(topk_min[:k]), expected_valid[:k])
    np.testing.assert_array_equal(
        np.sort(topk_max[:k]),
        np.sort(qualified_outputs[4][:k]),
    )
    np.testing.assert_array_equal(
        np.sort(topk_max[:k]),
        np.sort(expected_valid[-int(k) :]),
    )
    np.testing.assert_array_equal(
        topk_min[int(k) :],
        np.full_like(topk_min[int(k) :], -1),
    )
    np.testing.assert_array_equal(
        topk_max[int(k) :],
        np.full_like(topk_max[int(k) :], -1),
    )
    np.testing.assert_array_equal(original, values)


@cuda.jit
def _common_histogram_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    group = coop.this_block()
    samples = coop.ThreadData(1, dtype=types.int32)
    samples[0] = d_input[tid]
    counters = coop.histogram(group, samples, bins=8)
    d_output[tid] = counters[0]


@cuda.jit
def _qualified_histogram_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    group = numba_coop.this_block()
    samples = numba_coop.ThreadData(1, dtype=types.int32)
    samples[0] = d_input[tid]
    counters = numba_coop.histogram(group, samples, bins=8)
    d_output[tid] = counters[0]


def test_common_histogram_returns_striped_int32_counters():
    values = np.arange(THREADS, dtype=np.int32) % np.int32(8)
    counters = np.zeros(THREADS, dtype=np.int32)
    qualified_counters = np.zeros_like(counters)

    _common_histogram_kernel[1, THREADS](values, counters)
    _qualified_histogram_kernel[1, THREADS](values, qualified_counters)

    expected = np.zeros_like(counters)
    expected[:8] = np.bincount(values, minlength=8)
    np.testing.assert_array_equal(counters, qualified_counters)
    np.testing.assert_array_equal(counters, expected)


@cuda.jit
def _common_run_length_kernel(d_values, d_lengths, d_output):
    tid = cuda.threadIdx.x
    group = coop.this_block()
    run_values = coop.ThreadData(1, dtype=types.uint32)
    run_lengths = coop.ThreadData(1, dtype=types.uint32)
    run_values[0] = d_values[tid]
    run_lengths[0] = d_lengths[tid]
    decoded = coop.run_length_decode(
        group,
        run_values,
        run_lengths,
        decoded_items_per_thread=2,
    )
    for item in range(2):
        d_output[tid * 2 + item] = decoded[item]


@cuda.jit
def _qualified_run_length_kernel(d_values, d_lengths, d_output):
    tid = cuda.threadIdx.x
    group = numba_coop.this_block()
    run_values = numba_coop.ThreadData(1, dtype=types.uint32)
    run_lengths = numba_coop.ThreadData(1, dtype=types.uint32)
    run_values[0] = d_values[tid]
    run_lengths[0] = d_lengths[tid]
    decoded = numba_coop.run_length_decode(
        group,
        run_values,
        run_lengths,
        decoded_items_per_thread=2,
    )
    for item in range(2):
        d_output[tid * 2 + item] = decoded[item]


def test_common_run_length_decode_returns_decoded_values_only():
    run_values = np.arange(THREADS, dtype=np.uint32)
    run_lengths = np.full(THREADS, 2, dtype=np.uint32)
    decoded = np.zeros(THREADS * 2, dtype=np.uint32)
    qualified_decoded = np.zeros_like(decoded)

    _common_run_length_kernel[1, THREADS](run_values, run_lengths, decoded)
    _qualified_run_length_kernel[1, THREADS](
        run_values,
        run_lengths,
        qualified_decoded,
    )

    np.testing.assert_array_equal(decoded, qualified_decoded)
    np.testing.assert_array_equal(decoded, np.repeat(run_values, 2))


@cuda.jit
def _common_physical_warp_kernel(
    d_input,
    d_output_valid,
    d_exclusive,
    d_inclusive,
    d_total,
    d_default_scan,
    d_exclusive_scan,
    d_inclusive_scan,
    d_exchange,
    offset,
    valid_items,
):
    tid = cuda.threadIdx.x
    group = coop.this_warp()
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    loaded = coop.load(
        group,
        d_input,
        items,
        algorithm="transpose",
        valid_items=valid_items,
        oob_default=np.int32(-7),
        offset=offset,
    )
    coop.store(
        group,
        d_output_valid,
        loaded,
        algorithm="transpose",
        valid_items=valid_items,
        offset=offset,
    )

    value = d_input[offset + tid]
    d_exclusive[tid] = coop.exclusive_sum(group, value)
    d_inclusive[tid] = coop.inclusive_sum(group, value)
    d_total[tid] = coop.sum(group, value)
    d_default_scan[tid] = coop.scan(group, value)
    d_exclusive_scan[tid] = coop.exclusive_scan(group, value)
    d_inclusive_scan[tid] = coop.inclusive_scan(group, value)

    lane = tid % THREADS
    warp = tid // THREADS
    striped = coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(ITEMS_PER_THREAD):
        striped[item] = d_input[
            offset + warp * THREADS * ITEMS_PER_THREAD + lane + item * THREADS
        ]
    blocked = coop.exchange(group, striped)
    for item in range(ITEMS_PER_THREAD):
        d_exchange[
            warp * THREADS * ITEMS_PER_THREAD + lane * ITEMS_PER_THREAD + item
        ] = blocked[item]


@cuda.jit
def _qualified_physical_warp_kernel(
    d_input,
    d_output_valid,
    d_exclusive,
    d_inclusive,
    d_total,
    d_default_scan,
    d_exclusive_scan,
    d_inclusive_scan,
    d_exchange,
    offset,
    valid_items,
):
    tid = cuda.threadIdx.x
    group = numba_coop.this_warp()
    items = numba_coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    loaded = numba_coop.load(
        group,
        d_input,
        items,
        algorithm="transpose",
        valid_items=valid_items,
        oob_default=np.int32(-7),
        offset=offset,
    )
    numba_coop.store(
        group,
        d_output_valid,
        loaded,
        algorithm="transpose",
        valid_items=valid_items,
        offset=offset,
    )

    value = d_input[offset + tid]
    d_exclusive[tid] = numba_coop.exclusive_sum(group, value)
    d_inclusive[tid] = numba_coop.inclusive_sum(group, value)
    d_total[tid] = numba_coop.sum(group, value)
    d_default_scan[tid] = numba_coop.scan(group, value)
    d_exclusive_scan[tid] = numba_coop.exclusive_scan(group, value)
    d_inclusive_scan[tid] = numba_coop.inclusive_scan(group, value)

    lane = tid % THREADS
    warp = tid // THREADS
    striped = numba_coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(ITEMS_PER_THREAD):
        striped[item] = d_input[
            offset + warp * THREADS * ITEMS_PER_THREAD + lane + item * THREADS
        ]
    blocked = numba_coop.exchange(group, striped)
    for item in range(ITEMS_PER_THREAD):
        d_exchange[
            warp * THREADS * ITEMS_PER_THREAD + lane * ITEMS_PER_THREAD + item
        ] = blocked[item]


def test_common_physical_warps_own_distinct_tiles_and_collectives():
    warps = 2
    threads = warps * THREADS
    tile_items = THREADS * ITEMS_PER_THREAD
    offset = np.int32(3)
    valid_items = np.int32(tile_items - 5)
    values = np.arange(offset + warps * tile_items, dtype=np.int32)
    output_valid = np.full_like(values, -99)
    exclusive = np.zeros(threads, dtype=np.int32)
    inclusive = np.zeros(threads, dtype=np.int32)
    totals = np.zeros(threads, dtype=np.int32)
    default_scan = np.zeros_like(exclusive)
    exclusive_scan = np.zeros_like(exclusive)
    inclusive_scan = np.zeros_like(inclusive)
    exchange = np.zeros(warps * tile_items, dtype=np.int32)
    qualified_output_valid = np.full_like(values, -99)
    qualified_exclusive = np.zeros_like(exclusive)
    qualified_inclusive = np.zeros_like(inclusive)
    qualified_totals = np.zeros_like(totals)
    qualified_default_scan = np.zeros_like(default_scan)
    qualified_exclusive_scan = np.zeros_like(exclusive_scan)
    qualified_inclusive_scan = np.zeros_like(inclusive_scan)
    qualified_exchange = np.zeros_like(exchange)

    _common_physical_warp_kernel[1, threads](
        values,
        output_valid,
        exclusive,
        inclusive,
        totals,
        default_scan,
        exclusive_scan,
        inclusive_scan,
        exchange,
        offset,
        valid_items,
    )
    _qualified_physical_warp_kernel[1, threads](
        values,
        qualified_output_valid,
        qualified_exclusive,
        qualified_inclusive,
        qualified_totals,
        qualified_default_scan,
        qualified_exclusive_scan,
        qualified_inclusive_scan,
        qualified_exchange,
        offset,
        valid_items,
    )

    for common_output, qualified_output in zip(
        (
            output_valid,
            exclusive,
            inclusive,
            totals,
            default_scan,
            exclusive_scan,
            inclusive_scan,
            exchange,
        ),
        (
            qualified_output_valid,
            qualified_exclusive,
            qualified_inclusive,
            qualified_totals,
            qualified_default_scan,
            qualified_exclusive_scan,
            qualified_inclusive_scan,
            qualified_exchange,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(common_output, qualified_output)

    expected_valid = np.full_like(values, -99)
    for warp in range(warps):
        tile_begin = int(offset) + warp * tile_items
        valid_end = tile_begin + int(valid_items)
        expected_valid[tile_begin:valid_end] = values[tile_begin:valid_end]
    np.testing.assert_array_equal(output_valid, expected_valid)

    scan_input = values[int(offset) : int(offset) + threads]
    expected_exclusive = np.empty_like(scan_input)
    expected_inclusive = np.empty_like(scan_input)
    expected_totals = np.empty_like(scan_input)
    for warp in range(warps):
        begin = warp * THREADS
        end = begin + THREADS
        warp_values = scan_input[begin:end]
        expected_exclusive[begin] = 0
        expected_exclusive[begin + 1 : end] = np.cumsum(warp_values[:-1])
        expected_inclusive[begin:end] = np.cumsum(warp_values)
        expected_totals[begin:end] = np.sum(warp_values)
    np.testing.assert_array_equal(exclusive, expected_exclusive)
    np.testing.assert_array_equal(inclusive, expected_inclusive)
    np.testing.assert_array_equal(totals, expected_totals)
    np.testing.assert_array_equal(default_scan, expected_exclusive)
    np.testing.assert_array_equal(exclusive_scan, expected_exclusive)
    np.testing.assert_array_equal(inclusive_scan, expected_inclusive)
    np.testing.assert_array_equal(
        exchange,
        values[int(offset) : int(offset) + warps * tile_items],
    )


@cuda.jit
def _common_mapped_reduction_kernel(
    d_input,
    d_common_lanes,
    d_qualified_lanes,
):
    tid = cuda.threadIdx.x
    common_lanes = coop.this_warp().group_by(8)
    qualified_lanes = numba_coop.this_warp().group_by(8)
    value = d_input[tid]
    d_common_lanes[tid] = coop.sum(common_lanes, value)
    d_qualified_lanes[tid] = numba_coop.sum(qualified_lanes, value)


@pytest.mark.evidence_for("group.sum", backend="numba_mlir", evidence="runtime")
def test_common_static_mapped_lane_reduction_matches_qualified_numba():
    threads = 2 * THREADS
    values = np.arange(1, threads + 1, dtype=np.int32)
    common_lanes = np.zeros_like(values)
    qualified_lanes = np.zeros_like(values)

    _common_mapped_reduction_kernel[1, threads](
        values,
        common_lanes,
        qualified_lanes,
    )

    np.testing.assert_array_equal(common_lanes, qualified_lanes)
    np.testing.assert_array_equal(
        common_lanes,
        np.repeat(values.reshape(-1, 8).sum(axis=1), 8),
    )


@cuda.jit
def _common_physical_warp_merge_sort_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    warp = tid // THREADS
    lane = tid % THREADS
    group = coop.this_warp()
    keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(ITEMS_PER_THREAD):
        index = warp * THREADS * ITEMS_PER_THREAD + lane * ITEMS_PER_THREAD + item
        keys[item] = d_input[index]
    result = coop.merge_sort_keys(group, keys, descending=True)
    for item in range(ITEMS_PER_THREAD):
        index = warp * THREADS * ITEMS_PER_THREAD + lane * ITEMS_PER_THREAD + item
        d_output[index] = result[item]


@cuda.jit
def _qualified_physical_warp_merge_sort_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    warp = tid // THREADS
    lane = tid % THREADS
    group = numba_coop.this_warp()
    keys = numba_coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(ITEMS_PER_THREAD):
        index = warp * THREADS * ITEMS_PER_THREAD + lane * ITEMS_PER_THREAD + item
        keys[item] = d_input[index]
    result = numba_coop.merge_sort_keys(group, keys, descending=True)
    for item in range(ITEMS_PER_THREAD):
        index = warp * THREADS * ITEMS_PER_THREAD + lane * ITEMS_PER_THREAD + item
        d_output[index] = result[item]


def test_common_physical_warp_merge_sort_matches_qualified_numba():
    warps = 2
    items_per_warp = THREADS * ITEMS_PER_THREAD
    total_items = warps * items_per_warp
    values = np.asarray(
        [((index * 11 + 7) % 131) - 65 for index in range(total_items)],
        dtype=np.int32,
    )
    common_output = np.zeros_like(values)
    qualified_output = np.zeros_like(values)

    _common_physical_warp_merge_sort_kernel[1, warps * THREADS](
        values,
        common_output,
    )
    _qualified_physical_warp_merge_sort_kernel[1, warps * THREADS](
        values,
        qualified_output,
    )

    expected = np.empty_like(values)
    for warp in range(warps):
        begin = warp * items_per_warp
        end = begin + items_per_warp
        expected[begin:end] = np.sort(values[begin:end])[::-1]
    np.testing.assert_array_equal(common_output, qualified_output)
    np.testing.assert_array_equal(common_output, expected)


@cuda.jit
def _common_explicit_temp_storage_kernel(d_input, d_output):
    group = coop.this_block()
    temp_storage = coop.TempStorage()
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    coop.load(
        group,
        d_input,
        items,
        algorithm="transpose",
        temp_storage=temp_storage,
    )
    coop.store(
        group,
        d_output,
        items,
        algorithm="transpose",
        temp_storage=temp_storage,
    )


def test_common_load_store_accept_explicit_temp_storage():
    values = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    output = np.zeros_like(values)

    _common_explicit_temp_storage_kernel[1, THREADS](values, output)

    np.testing.assert_array_equal(output, values)


def _custom_scan_op(lhs, rhs):
    return lhs + rhs


def _custom_reduce_op(lhs, rhs):
    return lhs + rhs


@cuda.jit
def _common_custom_scan_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    d_output[tid] = coop.inclusive_scan(
        coop.this_block(),
        d_input[tid],
        scan_op=_custom_scan_op,
    )


@cuda.jit
def _common_heads_and_tails_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    items = coop.ThreadData(1, dtype=types.int32)
    items[0] = d_input[tid]
    flags = coop.discontinuity(
        coop.this_block(),
        items,
        mode="heads_and_tails",
    )
    d_output[tid] = flags[0]


@cuda.jit
def _common_custom_reduce_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    d_output[tid] = coop.reduce(
        coop.this_block(),
        d_input[tid],
        binary_op=_custom_reduce_op,
    )


@cuda.jit
def _common_backend_reduce_algorithm_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    d_output[tid] = coop.sum(
        coop.this_block(),
        d_input[tid],
        algorithm="warp_reductions_nondeterministic",
    )


@cuda.jit
def _common_grid_sync_kernel(d_input, d_output):
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    group = coop.this_grid()
    group.sync()
    d_output[index] = d_input[index]


@cuda.jit
def _common_grid_reduce_kernel(d_input, d_output):
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    d_output[index] = coop.reduce(coop.this_grid(), d_input[index])


@cuda.jit
def _common_grid_sum_kernel(d_input, d_output):
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    d_output[index] = coop.sum(coop.this_grid(), d_input[index])


@cuda.jit
def _common_logical_warp_merge_sort_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    group = coop.this_warp().group_by(8)
    keys = coop.ThreadData(1, dtype=types.int32)
    keys[0] = d_input[tid]
    result = coop.merge_sort_keys(group, keys)
    d_output[tid] = result[0]


@cuda.jit
def _common_mapped_warps_reduce_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    group = coop.this_block().group_by(1)
    d_output[tid] = coop.sum(group, d_input[tid])


@cuda.jit
def _qualified_mapped_warps_reduce_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    group = numba_coop.this_block().group_by(1)
    d_output[tid] = numba_coop.sum(group, d_input[tid])


def test_common_profile_rejects_qualified_only_controls():
    values = np.arange(THREADS, dtype=np.int32)
    output = np.zeros_like(values)

    with pytest.raises(ValueError, match="built-in operators only"):
        _common_custom_scan_kernel[1, THREADS](values, output)
    with pytest.raises(ValueError, match="backend-qualified import"):
        _common_heads_and_tails_kernel[1, THREADS](values, output)
    with pytest.raises(ValueError, match="built-in operators only"):
        _common_custom_reduce_kernel[1, THREADS](values, output)
    with pytest.raises(ValueError, match="backend-qualified import"):
        _common_backend_reduce_algorithm_kernel[1, THREADS](values, output)


@pytest.mark.parametrize(
    "kernel",
    [_common_grid_sync_kernel, _common_grid_reduce_kernel, _common_grid_sum_kernel],
)
def test_common_profile_rejects_grid_collectives(kernel):
    values = np.arange(2 * THREADS, dtype=np.int32)
    output = np.zeros_like(values)

    with pytest.raises(NotImplementedError, match="common V1"):
        kernel[2, THREADS](values, output)


def test_common_profile_sorts_each_logical_warp_independently():
    values = np.arange(THREADS, dtype=np.int32).reshape(-1, 8)[:, ::-1].reshape(-1)
    output = np.zeros_like(values)

    _common_logical_warp_merge_sort_kernel[1, THREADS](values, output)

    expected = np.sort(values.reshape(-1, 8), axis=1).reshape(-1)
    np.testing.assert_array_equal(output, expected)


def test_common_profile_rejects_uncertified_mapped_warp_reduction():
    threads = 2 * THREADS
    values = np.arange(threads, dtype=np.int32)
    output = np.zeros_like(values)
    expected = (
        "cuda.coop.sum does not support group kind 'warps_within_block' in "
        "common V1; supported group kinds: thread, physical_warp, "
        "threads_within_warp, block, cluster; use a backend-qualified import "
        "for backend-specific group support"
    )

    with pytest.raises(NotImplementedError) as error_info:
        _common_mapped_warps_reduce_kernel[1, threads](values, output)

    assert str(error_info.value) == expected


def test_qualified_numba_rejects_incorrect_mapped_warp_reduction():
    threads = 2 * THREADS
    values = np.arange(threads, dtype=np.int32)
    output = np.zeros_like(values)
    expected = (
        "cuda.coop.numba_mlir reduce/sum does not support "
        "warps_within_block groups because the current CUDAX mapping does not "
        "preserve independent mapped-group reduction semantics"
    )

    with pytest.raises(NotImplementedError) as error_info:
        _qualified_mapped_warps_reduce_kernel[1, threads](values, output)

    assert str(error_info.value) == expected


@cuda.jit
def _common_multi_item_scan_kernel(
    d_input,
    d_exclusive,
    d_inclusive,
    d_original,
):
    tid = cuda.threadIdx.x
    group = coop.this_block()
    temp_storage = coop.TempStorage()
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(ITEMS_PER_THREAD):
        items[item] = d_input[tid * ITEMS_PER_THREAD + item]

    exclusive = coop.exclusive_sum(group, items, temp_storage=temp_storage)
    inclusive = coop.inclusive_sum(group, items, temp_storage=temp_storage)
    for item in range(ITEMS_PER_THREAD):
        index = tid * ITEMS_PER_THREAD + item
        d_exclusive[index] = exclusive[item]
        d_inclusive[index] = inclusive[item]
        d_original[index] = items[item]


def test_common_multi_item_scans_preserve_shape_and_input():
    values = np.arange(1, THREADS * ITEMS_PER_THREAD + 1, dtype=np.int32)
    exclusive = np.zeros_like(values)
    inclusive = np.zeros_like(values)
    original = np.zeros_like(values)

    _common_multi_item_scan_kernel[1, THREADS](
        values,
        exclusive,
        inclusive,
        original,
    )

    expected_exclusive = np.concatenate(
        (np.asarray([0], dtype=np.int32), np.cumsum(values[:-1]))
    )
    np.testing.assert_array_equal(exclusive, expected_exclusive)
    np.testing.assert_array_equal(inclusive, np.cumsum(values))
    np.testing.assert_array_equal(original, values)


@cuda.jit
def _common_thread_cluster_reduce_kernel(
    d_input,
    d_thread,
    d_cluster,
    d_qualified_thread,
    d_qualified_cluster,
):
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    thread = coop.this_thread()
    cluster = coop.this_cluster()
    cluster.sync()
    d_thread[index] = coop.sum(thread, d_input[index])
    d_cluster[index] = coop.sum(cluster, d_input[index])
    qualified_thread = numba_coop.this_thread()
    qualified_cluster = numba_coop.this_cluster()
    qualified_cluster.sync()
    d_qualified_thread[index] = numba_coop.sum(
        qualified_thread,
        d_input[index],
    )
    d_qualified_cluster[index] = numba_coop.sum(
        qualified_cluster,
        d_input[index],
    )


@pytest.mark.evidence_for("group.sum", backend="numba_mlir", evidence="runtime")
def test_common_thread_and_cluster_reduce():
    if cuda.get_current_device().compute_capability < (9, 0):
        pytest.skip("thread-block clusters require compute capability 9.0 or newer")

    blocks = 2
    members = blocks * THREADS
    values = np.arange(1, members + 1, dtype=np.int32)
    thread_results = np.zeros_like(values)
    cluster_results = np.zeros_like(values)
    qualified_thread_results = np.zeros_like(values)
    qualified_cluster_results = np.zeros_like(values)

    _common_thread_cluster_reduce_kernel[blocks, THREADS, 0, 0, blocks](
        values,
        thread_results,
        cluster_results,
        qualified_thread_results,
        qualified_cluster_results,
    )

    np.testing.assert_array_equal(thread_results, qualified_thread_results)
    np.testing.assert_array_equal(cluster_results, qualified_cluster_results)
    np.testing.assert_array_equal(thread_results, values)
    np.testing.assert_array_equal(
        cluster_results,
        np.full_like(values, values.sum()),
    )


@cuda.jit
def _common_partial_reduce_kernel(
    d_input,
    d_block_sum,
    d_block_max,
    d_warp_sum,
    valid_block_items,
    valid_warp_items,
):
    tid = cuda.threadIdx.x
    value = d_input[tid]
    block_sum = coop.sum(
        coop.this_block(),
        value,
        broadcast=False,
        valid_items=valid_block_items,
    )
    block_max = coop.reduce(
        coop.this_block(),
        value,
        binary_op="max",
        broadcast=False,
        algorithm="raking",
    )
    warp_sum = coop.sum(
        coop.this_warp(),
        value,
        broadcast=False,
        valid_items=valid_warp_items,
    )
    if tid == 0:
        d_block_sum[0] = block_sum
        d_block_max[0] = block_max
    if tid % THREADS == 0:
        d_warp_sum[tid // THREADS] = warp_sum


def test_common_partial_and_algorithm_reductions_are_root_owned():
    threads = 2 * THREADS
    values = np.arange(1, threads + 1, dtype=np.int32)
    block_sum = np.full(1, -999, dtype=np.int32)
    block_max = np.full(1, -999, dtype=np.int32)
    warp_sum = np.full(2, -999, dtype=np.int32)
    valid_block_items = np.int32(threads - 7)
    valid_warp_items = np.int32(THREADS - 5)

    _common_partial_reduce_kernel[1, threads](
        values,
        block_sum,
        block_max,
        warp_sum,
        valid_block_items,
        valid_warp_items,
    )

    np.testing.assert_array_equal(
        block_sum,
        np.asarray([values[:valid_block_items].sum()], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        block_max,
        np.asarray([values.max()], dtype=np.int32),
    )
    expected_warp = np.empty(2, dtype=np.int32)
    for warp in range(2):
        begin = warp * THREADS
        valid_end = begin + int(valid_warp_items)
        expected_warp[warp] = values[begin:valid_end].sum()
    np.testing.assert_array_equal(warp_sum, expected_warp)


@cuda.jit
def _qualified_pair_composition_kernel(
    d_keys,
    d_values,
    d_sorted_keys,
    d_sorted_values,
    d_topk_min_keys,
    d_topk_min_values,
    d_topk_max_keys,
    d_topk_max_values,
    d_original_keys,
    d_original_values,
    k,
):
    tid = cuda.threadIdx.x
    group = numba_coop.this_block()
    keys = numba_coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    values = numba_coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(ITEMS_PER_THREAD):
        index = tid * ITEMS_PER_THREAD + item
        keys[item] = d_keys[index]
        values[item] = d_values[index]

    merged_keys, merged_values = numba_coop.merge_sort_pairs(group, keys, values)
    sorted_keys, sorted_values = numba_coop.radix_sort_pairs(
        group,
        merged_keys,
        merged_values,
        descending=True,
    )
    topk_min_keys, topk_min_values = numba_coop.topk_min_pairs(
        group,
        keys,
        values,
        k,
        valid_items=VALID_ITEMS,
        begin_bit=4,
    )
    topk_max_keys, topk_max_values = numba_coop.topk_max_pairs(
        group,
        keys,
        values,
        k,
        valid_items=VALID_ITEMS,
        begin_bit=4,
    )
    for item in range(ITEMS_PER_THREAD):
        index = tid * ITEMS_PER_THREAD + item
        d_sorted_keys[index] = sorted_keys[item]
        d_sorted_values[index] = sorted_values[item]
        if index < k:
            d_topk_min_keys[index] = topk_min_keys[item]
            d_topk_min_values[index] = topk_min_values[item]
            d_topk_max_keys[index] = topk_max_keys[item]
            d_topk_max_values[index] = topk_max_values[item]
        d_original_keys[index] = keys[item]
        d_original_values[index] = values[item]


def test_qualified_pair_results_compose_and_preserve_payload_types():
    keys = np.arange(THREADS * ITEMS_PER_THREAD, 0, -1, dtype=np.int32)
    values = keys * np.int32(7)
    sorted_keys = np.zeros_like(keys)
    sorted_values = np.zeros_like(values)
    k = np.int32(7)
    topk_min_keys = np.full_like(keys, -1)
    topk_min_values = np.full_like(values, -1)
    topk_max_keys = np.full_like(keys, -1)
    topk_max_values = np.full_like(values, -1)
    original_keys = np.zeros_like(keys)
    original_values = np.zeros_like(values)

    _qualified_pair_composition_kernel[1, THREADS](
        keys,
        values,
        sorted_keys,
        sorted_values,
        topk_min_keys,
        topk_min_values,
        topk_max_keys,
        topk_max_values,
        original_keys,
        original_values,
        k,
    )

    expected_keys = np.sort(keys)[::-1]
    np.testing.assert_array_equal(sorted_keys, expected_keys)
    np.testing.assert_array_equal(sorted_values, expected_keys * np.int32(7))
    valid_keys = keys[:VALID_ITEMS]
    valid_digits = valid_keys.astype(np.uint32) >> np.uint32(4)
    min_digits = topk_min_keys[:k].astype(np.uint32) >> np.uint32(4)
    max_digits = topk_max_keys[:k].astype(np.uint32) >> np.uint32(4)
    np.testing.assert_array_equal(np.sort(min_digits), np.sort(valid_digits)[:k])
    np.testing.assert_array_equal(
        topk_min_values[:k],
        topk_min_keys[:k] * np.int32(7),
    )
    np.testing.assert_array_equal(
        np.sort(max_digits),
        np.sort(valid_digits)[-int(k) :],
    )
    np.testing.assert_array_equal(
        topk_max_values[:k],
        topk_max_keys[:k] * np.int32(7),
    )
    for undefined_tail in (
        topk_min_keys,
        topk_min_values,
        topk_max_keys,
        topk_max_values,
    ):
        np.testing.assert_array_equal(
            undefined_tail[int(k) :],
            np.full_like(undefined_tail[int(k) :], -1),
        )
    np.testing.assert_array_equal(original_keys, keys)
    np.testing.assert_array_equal(original_values, values)
