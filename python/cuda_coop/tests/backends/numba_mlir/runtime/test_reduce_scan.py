# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_THREADS = 128
_LOGICAL_WARP_THREADS = 8
_VALID_LOGICAL_ITEMS = 5
_ITEMS_PER_THREAD = 2


@cuda.jit(device=True)
def _prefix_with_block_aggregate(block_aggregate):
    return block_aggregate


class _RunningPrefix:
    def __call__(self_ptr, block_aggregate):
        previous = self_ptr[0]
        self_ptr[0] = previous + block_aggregate
        return previous


_RUNNING_PREFIX = coop.StatefulFunction(
    _RunningPrefix,
    types.int32,
    name="cuda_coop_test_running_prefix",
)


class _PrefixBeforeAggregate:
    def __call__(self_ptr, block_aggregate):
        previous = self_ptr[0]
        self_ptr[0] = previous + block_aggregate
        return previous


class _PrefixAfterAggregate:
    def __call__(self_ptr, block_aggregate):
        previous = self_ptr[0]
        self_ptr[0] = previous + block_aggregate
        return previous + block_aggregate


_PREFIX_BEFORE_AGGREGATE = coop.StatefulFunction(
    _PrefixBeforeAggregate,
    types.int32,
    name="cuda_coop_test_shared_prefix_name",
)
_PREFIX_AFTER_AGGREGATE = coop.StatefulFunction(
    _PrefixAfterAggregate,
    types.int32,
    name="cuda_coop_test_shared_prefix_name",
)


@cuda.jit
def _logical_warp_reduce(source, output):
    tid = cuda.threadIdx.x
    logical_warp = coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
    output[tid] = coop.sum(logical_warp, source[tid])


def test_logical_warp_reduce_broadcasts_each_group_result():
    source = np.arange(_THREADS, dtype=np.int32)
    output = np.full_like(source, -1)

    _logical_warp_reduce[1, _THREADS](source, output)

    expected = np.repeat(
        source.reshape(-1, _LOGICAL_WARP_THREADS).sum(axis=1, dtype=np.int32),
        _LOGICAL_WARP_THREADS,
    )
    np.testing.assert_array_equal(output, expected)


@cuda.jit
def _mapped_group_reductions(source, one_warp_output, two_warp_output):
    tid = cuda.threadIdx.x
    block = coop.this_block()
    one_warp = block.group_by(1)
    two_warps = block.group_by(2)
    one_warp_output[tid] = coop.sum(one_warp, source[tid])
    two_warp_output[tid] = coop.sum(two_warps, source[tid])


def test_multiple_mapped_groups_reduce_independently():
    source = np.arange(_THREADS, dtype=np.int32)
    one_warp_output = np.full_like(source, -1)
    two_warp_output = np.full_like(source, -1)

    _mapped_group_reductions[1, _THREADS](
        source,
        one_warp_output,
        two_warp_output,
    )

    one_warp_expected = np.repeat(
        source.reshape(-1, 32).sum(axis=1, dtype=np.int32),
        32,
    )
    two_warp_expected = np.repeat(
        source.reshape(-1, 64).sum(axis=1, dtype=np.int32),
        64,
    )
    np.testing.assert_array_equal(one_warp_output, one_warp_expected)
    np.testing.assert_array_equal(two_warp_output, two_warp_expected)


@cuda.jit
def _logical_warp_valid_prefix(source, output, valid_items):
    tid = cuda.threadIdx.x
    logical_warp = coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
    total = coop.sum(
        logical_warp,
        source[tid],
        broadcast=False,
        valid_items=valid_items,
    )
    if logical_warp.rank() == 0:
        output[tid // _LOGICAL_WARP_THREADS] = total


def test_direct_logical_warp_reduce_accepts_a_runtime_valid_prefix():
    source = np.arange(_THREADS, dtype=np.int32)
    output = np.full(_THREADS // _LOGICAL_WARP_THREADS, -1, dtype=np.int32)

    _logical_warp_valid_prefix[1, _THREADS](
        source,
        output,
        np.int32(_VALID_LOGICAL_ITEMS),
    )

    expected = source.reshape(-1, _LOGICAL_WARP_THREADS)[:, :_VALID_LOGICAL_ITEMS].sum(
        axis=1, dtype=np.int32
    )
    np.testing.assert_array_equal(output, expected)


@cuda.jit
def _logical_warp_scan(source, output, aggregates, valid_items):
    tid = cuda.threadIdx.x
    logical_warp = coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
    aggregate = coop.ThreadData(1, dtype=types.int32)
    output[tid] = coop.exclusive_sum(
        logical_warp,
        source[tid],
        valid_items=valid_items,
        aggregate_output=aggregate,
    )
    aggregates[tid] = aggregate[0]


def test_logical_warp_scan_reports_valid_prefix_and_aggregate():
    source = np.arange(_THREADS, dtype=np.int32)
    output = np.full_like(source, -1)
    aggregates = np.full_like(source, -1)

    _logical_warp_scan[1, _THREADS](
        source,
        output,
        aggregates,
        np.int32(_VALID_LOGICAL_ITEMS),
    )

    for start in range(0, _THREADS, _LOGICAL_WARP_THREADS):
        valid = source[start : start + _VALID_LOGICAL_ITEMS]
        expected = np.empty_like(valid)
        expected[0] = 0
        expected[1:] = np.cumsum(valid[:-1], dtype=np.int32)
        np.testing.assert_array_equal(
            output[start : start + _VALID_LOGICAL_ITEMS],
            expected,
        )
        np.testing.assert_array_equal(
            aggregates[start : start + _VALID_LOGICAL_ITEMS],
            np.full(_VALID_LOGICAL_ITEMS, valid.sum(dtype=np.int32)),
        )


@cuda.jit
def _block_scan_fixed_storage(source, output, unchanged, aggregates):
    tid = cuda.threadIdx.x
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    aggregate = coop.ThreadData(1, dtype=types.int32)
    storage = coop.TempStorage(4096, alignment=16)
    for index in range(_ITEMS_PER_THREAD):
        items[index] = source[tid * _ITEMS_PER_THREAD + index]
    scanned = coop.exclusive_scan(
        coop.this_block(),
        items,
        initial_value=7,
        aggregate_output=aggregate,
        temp_storage=storage,
    )
    for index in range(_ITEMS_PER_THREAD):
        position = tid * _ITEMS_PER_THREAD + index
        output[position] = scanned[index]
        unchanged[position] = items[index]
    aggregates[tid] = aggregate[0]


def test_block_scan_is_nonmutating_and_aggregate_excludes_initial():
    source = np.arange(_THREADS * _ITEMS_PER_THREAD, dtype=np.int32)
    output = np.full_like(source, -1)
    unchanged = np.full_like(source, -1)
    aggregates = np.full(_THREADS, -1, dtype=np.int32)

    _block_scan_fixed_storage[1, _THREADS](
        source,
        output,
        unchanged,
        aggregates,
    )

    expected = np.empty_like(source)
    expected[0] = 7
    expected[1:] = 7 + np.cumsum(source[:-1], dtype=np.int32)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(unchanged, source)
    np.testing.assert_array_equal(
        aggregates,
        np.full(_THREADS, source.sum(dtype=np.int32)),
    )


@cuda.jit
def _block_scan_prefix_callback(source, output):
    tid = cuda.threadIdx.x
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for index in range(_ITEMS_PER_THREAD):
        items[index] = source[tid * _ITEMS_PER_THREAD + index]
    scanned = coop.scan(
        coop.this_block(),
        items,
        prefix_op=_prefix_with_block_aggregate,
    )
    for index in range(_ITEMS_PER_THREAD):
        output[tid * _ITEMS_PER_THREAD + index] = scanned[index]


@cuda.jit
def _block_scan_legacy_prefix_callback(source, output):
    tid = cuda.threadIdx.x
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for index in range(_ITEMS_PER_THREAD):
        items[index] = source[tid * _ITEMS_PER_THREAD + index]
    scanned = coop.scan(
        coop.this_block(),
        items,
        block_prefix_callback_op=_prefix_with_block_aggregate,
    )
    for index in range(_ITEMS_PER_THREAD):
        output[tid * _ITEMS_PER_THREAD + index] = scanned[index]


@pytest.mark.parametrize(
    "kernel",
    [_block_scan_prefix_callback, _block_scan_legacy_prefix_callback],
)
def test_block_scan_prefix_callback_receives_block_aggregate(kernel):
    source = np.ones(_THREADS * _ITEMS_PER_THREAD, dtype=np.int32)
    output = np.full_like(source, -1)

    kernel[1, _THREADS](source, output)

    block_aggregate = source.sum(dtype=np.int32)
    expected = block_aggregate + np.arange(source.size, dtype=np.int32)
    np.testing.assert_array_equal(output, expected)


@cuda.jit
def _block_scan_running_prefix(source, output, final_prefix):
    tid = cuda.threadIdx.x
    prefix_state = coop.ThreadData(1, dtype=types.int32)
    prefix_state[0] = 0
    tile_offset = 0
    while tile_offset < source.size:
        item = coop.ThreadData(1, dtype=types.int32)
        item[0] = source[tile_offset + tid]
        scanned = coop.exclusive_sum(
            coop.this_block(),
            item,
            prefix_state,
            prefix_op=_RUNNING_PREFIX,
        )
        output[tile_offset + tid] = scanned[0]
        tile_offset += _THREADS
        cuda.syncthreads()
    if tid == 0:
        final_prefix[0] = prefix_state[0]


def test_block_scan_stateful_callback_tracks_running_prefix():
    tile_count = 4
    source = np.arange(1, _THREADS * tile_count + 1, dtype=np.int32) % 5
    output = np.full_like(source, -1)
    final_prefix = np.zeros(1, dtype=np.int32)

    _block_scan_running_prefix[1, _THREADS](source, output, final_prefix)

    expected = np.empty_like(source)
    expected[0] = 0
    expected[1:] = np.cumsum(source[:-1], dtype=np.int32)
    np.testing.assert_array_equal(output, expected)
    assert final_prefix[0] == source.sum(dtype=np.int32)


@cuda.jit
def _block_scan_distinct_same_named_callbacks(source, before_output, after_output):
    tid = cuda.threadIdx.x
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    before_state = coop.ThreadData(1, dtype=types.int32)
    after_state = coop.ThreadData(1, dtype=types.int32)
    before_state[0] = 0
    after_state[0] = 0
    for index in range(_ITEMS_PER_THREAD):
        items[index] = source[tid * _ITEMS_PER_THREAD + index]
    before = coop.exclusive_sum(
        coop.this_block(),
        items,
        before_state,
        prefix_op=_PREFIX_BEFORE_AGGREGATE,
    )
    after = coop.exclusive_sum(
        coop.this_block(),
        items,
        after_state,
        prefix_op=_PREFIX_AFTER_AGGREGATE,
    )
    for index in range(_ITEMS_PER_THREAD):
        position = tid * _ITEMS_PER_THREAD + index
        before_output[position] = before[index]
        after_output[position] = after[index]


def test_same_named_stateful_callbacks_link_as_distinct_device_functions():
    source = np.arange(1, _THREADS * _ITEMS_PER_THREAD + 1, dtype=np.int32) % 5
    before_output = np.full_like(source, -1)
    after_output = np.full_like(source, -1)

    _block_scan_distinct_same_named_callbacks[1, _THREADS](
        source,
        before_output,
        after_output,
    )

    expected = np.empty_like(source)
    expected[0] = 0
    expected[1:] = np.cumsum(source[:-1], dtype=np.int32)
    np.testing.assert_array_equal(before_output, expected)
    np.testing.assert_array_equal(
        after_output,
        expected + source.sum(dtype=np.int32),
    )
