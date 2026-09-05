# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import re
import shutil

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as coop

from ..support.runtime import (
    ITEMS_PER_THREAD,
    NUMBA_MLIR_PREFIX_CALLBACK_OP,
    THREADS,
    _add,
    _prefix_with_block_aggregate,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _warp_reduce_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    value = d_in[tid]

    total = coop._warp.sum(value, dtype="int32", threads_in_warp=THREADS)
    reduced = coop._warp.reduce(
        value, binary_op=_add, dtype="int32", threads_in_warp=THREADS
    )
    maximum = coop._warp.max(value, dtype="int32", threads_in_warp=THREADS)
    minimum = coop._warp.min(value, dtype="int32", threads_in_warp=THREADS)

    if tid == 0:
        d_out[0] = total
        d_out[1] = reduced
        d_out[2] = maximum
        d_out[3] = minimum


@cuda.jit
def _warp_reduce_valid_items_kernel(d_in, d_out, valid_items):
    tid = cuda.threadIdx.x
    value = d_in[tid]

    total = coop._warp.sum(value, valid_items, dtype="int32", threads_in_warp=THREADS)
    minimum = coop._warp.min(
        value, valid_items=valid_items, dtype="int32", threads_in_warp=THREADS
    )

    if tid == 0:
        d_out[0] = total
        d_out[1] = minimum


def test_warp_reduce_methods_share_one_kernel():
    h_input = np.arange(1, THREADS + 1, dtype=np.int32)
    h_output = np.zeros(4, dtype=np.int32)

    _warp_reduce_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(
        h_output,
        np.asarray(
            [np.sum(h_input), np.sum(h_input), np.max(h_input), np.min(h_input)],
            dtype=np.int32,
        ),
    )


def test_warp_reduce_valid_items():
    h_input = np.arange(THREADS, 0, -1, dtype=np.int32)
    h_output = np.zeros(2, dtype=np.int32)
    valid_items = np.int32(11)

    _warp_reduce_valid_items_kernel[1, THREADS](h_input, h_output, valid_items)

    valid = h_input[:valid_items]
    np.testing.assert_array_equal(
        h_output,
        np.asarray([np.sum(valid), np.min(valid)], dtype=np.int32),
    )


@cuda.jit
def _warp_scan_kernel(
    d_in, d_exclusive_sum, d_inclusive_sum, d_exclusive_max, d_inclusive_max
):
    tid = cuda.threadIdx.x
    value = d_in[tid]

    d_exclusive_sum[tid] = coop._warp.exclusive_sum(
        value, dtype="int32", threads_in_warp=THREADS
    )
    d_inclusive_sum[tid] = coop._warp.inclusive_sum(
        value, dtype="int32", threads_in_warp=THREADS
    )
    d_exclusive_max[tid] = coop._warp.exclusive_scan(
        value,
        scan_op="max",
        initial_value=0,
        dtype="int32",
        threads_in_warp=THREADS,
    )
    d_inclusive_max[tid] = coop._warp.inclusive_scan(
        value,
        scan_op="max",
        dtype="int32",
        threads_in_warp=THREADS,
    )


@cuda.jit
def _warp_scan_aggregate_kernel(
    d_in,
    d_inclusive_sum,
    d_sum_aggregate,
    d_inclusive_max,
    d_max_aggregate,
    valid_items,
):
    tid = cuda.threadIdx.x
    value = d_in[tid]
    sum_aggregate = cuda.local.array(1, cuda.int32)
    max_aggregate = cuda.local.array(1, cuda.int32)

    d_inclusive_sum[tid] = coop._warp.inclusive_sum(
        value,
        dtype="int32",
        threads_in_warp=THREADS,
        warp_aggregate=sum_aggregate,
    )
    d_sum_aggregate[tid] = sum_aggregate[0]
    d_inclusive_max[tid] = coop._warp.inclusive_scan(
        value,
        scan_op="max",
        dtype="int32",
        threads_in_warp=THREADS,
        valid_items=valid_items,
        warp_aggregate=max_aggregate,
    )
    d_max_aggregate[tid] = max_aggregate[0]


def test_warp_scan_methods_share_one_kernel():
    h_input = np.arange(1, THREADS + 1, dtype=np.int32)
    h_exclusive_sum = np.zeros_like(h_input)
    h_inclusive_sum = np.zeros_like(h_input)
    h_exclusive_max = np.zeros_like(h_input)
    h_inclusive_max = np.zeros_like(h_input)

    _warp_scan_kernel[1, THREADS](
        h_input,
        h_exclusive_sum,
        h_inclusive_sum,
        h_exclusive_max,
        h_inclusive_max,
    )

    np.testing.assert_array_equal(
        h_exclusive_sum,
        np.concatenate(
            [np.asarray([0], dtype=np.int32), np.cumsum(h_input[:-1])]
        ).astype(np.int32),
    )
    np.testing.assert_array_equal(h_inclusive_sum, np.cumsum(h_input).astype(np.int32))
    np.testing.assert_array_equal(
        h_exclusive_max,
        np.concatenate(
            [np.asarray([0], dtype=np.int32), np.maximum.accumulate(h_input[:-1])]
        ).astype(np.int32),
    )
    np.testing.assert_array_equal(
        h_inclusive_max, np.maximum.accumulate(h_input).astype(np.int32)
    )


def test_warp_scan_aggregate_outputs():
    h_input = np.arange(1, THREADS + 1, dtype=np.int32)
    h_inclusive_sum = np.zeros_like(h_input)
    h_sum_aggregate = np.zeros_like(h_input)
    h_inclusive_max = np.zeros_like(h_input)
    h_max_aggregate = np.zeros_like(h_input)
    valid_items = np.int32(13)

    _warp_scan_aggregate_kernel[1, THREADS](
        h_input,
        h_inclusive_sum,
        h_sum_aggregate,
        h_inclusive_max,
        h_max_aggregate,
        valid_items,
    )

    np.testing.assert_array_equal(h_inclusive_sum, np.cumsum(h_input).astype(np.int32))
    np.testing.assert_array_equal(
        h_sum_aggregate,
        np.full_like(h_sum_aggregate, np.sum(h_input, dtype=np.int32)),
    )
    np.testing.assert_array_equal(
        h_inclusive_max[:valid_items],
        np.maximum.accumulate(h_input[:valid_items]).astype(np.int32),
    )
    np.testing.assert_array_equal(
        h_max_aggregate[:valid_items],
        np.full(valid_items, np.max(h_input[:valid_items]), dtype=np.int32),
    )


@cuda.jit
def _block_reduce_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    value = d_in[tid]

    total = coop._block.sum(value, dtype="int32", threads_per_block=THREADS)
    reduced = coop._block.reduce(
        value, binary_op=_add, dtype="int32", threads_per_block=THREADS
    )

    if tid == 0:
        d_out[0] = total
        d_out[1] = reduced


def test_block_reduce_methods_share_one_kernel():
    h_input = np.arange(1, THREADS + 1, dtype=np.int32)
    h_output = np.zeros(2, dtype=np.int32)

    _block_reduce_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(
        h_output, np.asarray([np.sum(h_input), np.sum(h_input)], dtype=np.int32)
    )


@cuda.jit(device=True, inline="always")
def _inlined_block_sum(value):
    return coop._block.sum(value, dtype=types.int32, threads_per_block=THREADS)


@cuda.jit
def _inlined_block_sum_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    total = _inlined_block_sum(d_in[tid])
    if tid == 0:
        d_out[0] = total


def test_block_reduce_rewrites_after_device_function_inlining():
    h_input = np.arange(1, THREADS + 1, dtype=np.int32)
    h_output = np.zeros(1, dtype=np.int32)

    _inlined_block_sum_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(
        h_output, np.asarray([np.sum(h_input)], dtype=np.int32)
    )
    if shutil.which("nvdisasm") is None:
        return
    sass = _inlined_block_sum_kernel.inspect_sass(
        _inlined_block_sum_kernel.signatures[0]
    )
    assert re.search(r"\b(?:CALL|LDL|STL)(?:\.[A-Z0-9_]+)*\b", sass) is None


@cuda.jit
def _block_scan_kernel(
    d_in, d_exclusive_sum, d_inclusive_sum, d_exclusive_max, d_inclusive_max
):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    scanned = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.exclusive_sum(
        items,
        scanned,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )
    for i in range(ITEMS_PER_THREAD):
        d_exclusive_sum[tid * ITEMS_PER_THREAD + i] = scanned[i]

    coop._block.inclusive_sum(
        items,
        scanned,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )
    for i in range(ITEMS_PER_THREAD):
        d_inclusive_sum[tid * ITEMS_PER_THREAD + i] = scanned[i]

    coop._block.exclusive_scan(
        items,
        scanned,
        scan_op="max",
        initial_value=0,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )
    for i in range(ITEMS_PER_THREAD):
        d_exclusive_max[tid * ITEMS_PER_THREAD + i] = scanned[i]

    coop._block.inclusive_scan(
        items,
        scanned,
        scan_op="max",
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )
    for i in range(ITEMS_PER_THREAD):
        d_inclusive_max[tid * ITEMS_PER_THREAD + i] = scanned[i]


def test_block_scan_methods_share_a_kernel():
    h_input = np.arange(1, THREADS * ITEMS_PER_THREAD + 1, dtype=np.int32)
    h_exclusive_sum = np.zeros_like(h_input)
    h_inclusive_sum = np.zeros_like(h_input)
    h_exclusive_max = np.zeros_like(h_input)
    h_inclusive_max = np.zeros_like(h_input)

    _block_scan_kernel[1, THREADS](
        h_input,
        h_exclusive_sum,
        h_inclusive_sum,
        h_exclusive_max,
        h_inclusive_max,
    )

    np.testing.assert_array_equal(
        h_exclusive_sum,
        np.concatenate(
            [np.asarray([0], dtype=np.int32), np.cumsum(h_input[:-1])]
        ).astype(np.int32),
    )
    np.testing.assert_array_equal(h_inclusive_sum, np.cumsum(h_input).astype(np.int32))
    np.testing.assert_array_equal(
        h_exclusive_max,
        np.concatenate(
            [np.asarray([0], dtype=np.int32), np.maximum.accumulate(h_input[:-1])]
        ).astype(np.int32),
    )
    np.testing.assert_array_equal(
        h_inclusive_max, np.maximum.accumulate(h_input).astype(np.int32)
    )


@cuda.jit
def _block_scan_prefix_callback_kernel(d_in, d_exclusive_sum, d_inclusive_sum):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    scanned = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.exclusive_sum(
        items,
        scanned,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        block_prefix_callback_op=_prefix_with_block_aggregate,
    )
    for i in range(ITEMS_PER_THREAD):
        d_exclusive_sum[tid * ITEMS_PER_THREAD + i] = scanned[i]

    coop._block.inclusive_sum(
        items,
        scanned,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        prefix_op=_prefix_with_block_aggregate,
    )
    for i in range(ITEMS_PER_THREAD):
        d_inclusive_sum[tid * ITEMS_PER_THREAD + i] = scanned[i]


def test_block_scan_prefix_callback_op():
    h_input = np.ones(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_exclusive_sum = np.zeros_like(h_input)
    h_inclusive_sum = np.zeros_like(h_input)
    block_aggregate = np.sum(h_input).astype(np.int32)

    _block_scan_prefix_callback_kernel[1, THREADS](
        h_input, h_exclusive_sum, h_inclusive_sum
    )

    expected_exclusive = block_aggregate + np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(h_input[:-1])]
    ).astype(np.int32)
    expected_inclusive = block_aggregate + np.cumsum(h_input).astype(np.int32)
    np.testing.assert_array_equal(h_exclusive_sum, expected_exclusive)
    np.testing.assert_array_equal(h_inclusive_sum, expected_inclusive)


@cuda.jit
def _block_scan_block_aggregate_kernel(d_out, d_aggregates):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    scanned = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    block_aggregate = cuda.local.array(1, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        items[i] = 1

    coop._block.scan(
        items,
        scanned,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        mode="exclusive",
        scan_op="+",
        block_aggregate=block_aggregate,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = scanned[i]
    d_aggregates[tid] = block_aggregate[0]


def test_block_scan_block_aggregate():
    h_output = np.zeros(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_aggregates = np.zeros(THREADS, dtype=np.int32)

    _block_scan_block_aggregate_kernel[1, THREADS](h_output, h_aggregates)

    np.testing.assert_array_equal(
        h_output, np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        h_aggregates,
        np.full(THREADS, THREADS * ITEMS_PER_THREAD, dtype=np.int32),
    )


@cuda.jit
def _block_scan_stateful_prefix_callback_kernel(d_in, d_out, d_final_prefix):
    tid = cuda.threadIdx.x
    prefix_state = cuda.local.array(1, cuda.int32)
    prefix_state[0] = 0

    block_offset = 0
    while block_offset < d_in.size:
        item = cuda.local.array(1, cuda.int32)
        scanned = cuda.local.array(1, cuda.int32)
        item[0] = d_in[block_offset + tid]

        coop._block.exclusive_sum(
            item,
            scanned,
            prefix_state,
            dtype="int32",
            threads_per_block=THREADS,
            items_per_thread=1,
            prefix_op=NUMBA_MLIR_PREFIX_CALLBACK_OP,
        )
        d_out[block_offset + tid] = scanned[0]

        block_offset += THREADS
        cuda.syncthreads()

    if tid == 0:
        d_final_prefix[0] = prefix_state[0]


def test_block_scan_stateful_prefix_callback_op_grid_stride():
    num_tiles = 4
    h_input = np.arange(1, THREADS * num_tiles + 1, dtype=np.int32) % 5
    h_output = np.zeros_like(h_input)
    h_final_prefix = np.zeros(1, dtype=np.int32)

    _block_scan_stateful_prefix_callback_kernel[1, THREADS](
        h_input, h_output, h_final_prefix
    )

    expected = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(h_input[:-1])]
    ).astype(np.int32)
    np.testing.assert_array_equal(h_output, expected)
    assert h_final_prefix[0] == np.sum(h_input)
