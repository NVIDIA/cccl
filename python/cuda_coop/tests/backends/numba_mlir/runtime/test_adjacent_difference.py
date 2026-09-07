# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")


import cuda.coop.numba_mlir as coop
from cuda.coop.numba_mlir._block import (
    BlockAdjacentDifferenceType,
)

from ..support.runtime import (
    ITEMS_PER_THREAD,
    THREADS,
    _subtract,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _block_adjacent_difference_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.adjacent_difference(
        items,
        output,
        block_adjacent_difference_type=BlockAdjacentDifferenceType.SubtractLeft,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        difference_op=_subtract,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = output[i]


@cuda.jit
def _block_adjacent_difference_right_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.adjacent_difference(
        items,
        output,
        block_adjacent_difference_type=BlockAdjacentDifferenceType.SubtractRight,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        difference_op=_subtract,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = output[i]


@cuda.jit
def _block_adjacent_difference_left_tile_kernel(d_in, d_out, tile_predecessor):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.adjacent_difference(
        items,
        output,
        tile_predecessor,
        block_adjacent_difference_type=BlockAdjacentDifferenceType.SubtractLeft,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        difference_op=_subtract,
        tile_predecessor_item=True,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = output[i]


@cuda.jit
def _block_adjacent_difference_left_partial_tile_kernel(
    d_in, d_out, valid_items, tile_predecessor
):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.adjacent_difference(
        items,
        output,
        valid_items,
        tile_predecessor,
        block_adjacent_difference_type=BlockAdjacentDifferenceType.SubtractLeft,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        difference_op=_subtract,
        valid_items=True,
        tile_predecessor_item=True,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = output[i]


@cuda.jit
def _block_adjacent_difference_right_tile_kernel(d_in, d_out, tile_successor):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.adjacent_difference(
        items,
        output,
        tile_successor,
        block_adjacent_difference_type=BlockAdjacentDifferenceType.SubtractRight,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        difference_op=_subtract,
        tile_successor_item=True,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = output[i]


@cuda.jit
def _block_adjacent_difference_thread_data_temp_storage_kernel(d_in, d_out):
    temp_storage = coop.TempStorage()
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)
    output = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)

    coop._block.load(d_in, items)
    coop._block.adjacent_difference[temp_storage](
        items,
        output,
        block_adjacent_difference_type=BlockAdjacentDifferenceType.SubtractLeft,
        difference_op=_subtract,
    )
    coop._block.store(d_out, output)


def test_block_adjacent_difference_subtract_left():
    h_input = (np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32) % 7) * 3
    h_output = np.zeros_like(h_input)

    _block_adjacent_difference_kernel[1, THREADS](h_input, h_output)

    expected = np.empty_like(h_input)
    expected[0] = h_input[0]
    expected[1:] = h_input[1:] - h_input[:-1]
    np.testing.assert_array_equal(h_output, expected)


def test_block_adjacent_difference_subtract_right():
    h_input = (np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32) % 7) * 3
    h_output = np.zeros_like(h_input)

    _block_adjacent_difference_right_kernel[1, THREADS](h_input, h_output)

    expected = np.empty_like(h_input)
    expected[-1] = h_input[-1]
    expected[:-1] = h_input[:-1] - h_input[1:]
    np.testing.assert_array_equal(h_output, expected)


def test_block_adjacent_difference_subtract_left_tile_predecessor():
    h_input = (np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32) % 7) * 3
    h_output = np.zeros_like(h_input)
    tile_predecessor = np.int32(-9)

    _block_adjacent_difference_left_tile_kernel[1, THREADS](
        h_input, h_output, tile_predecessor
    )

    expected = np.empty_like(h_input)
    expected[0] = h_input[0] - tile_predecessor
    expected[1:] = h_input[1:] - h_input[:-1]
    np.testing.assert_array_equal(h_output, expected)


def test_block_adjacent_difference_subtract_left_partial_tile_predecessor():
    h_input = (np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32) % 7) * 3
    h_output = np.zeros_like(h_input)
    valid_items = h_input.size - 3
    tile_predecessor = np.int32(-9)

    _block_adjacent_difference_left_partial_tile_kernel[1, THREADS](
        h_input,
        h_output,
        valid_items,
        tile_predecessor,
    )

    expected = h_input.copy()
    expected[0] = h_input[0] - tile_predecessor
    expected[1:valid_items] = h_input[1:valid_items] - h_input[: valid_items - 1]
    np.testing.assert_array_equal(h_output, expected)


def test_block_adjacent_difference_subtract_right_tile_successor():
    h_input = (np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32) % 7) * 3
    h_output = np.zeros_like(h_input)
    tile_successor = np.int32(41)

    _block_adjacent_difference_right_tile_kernel[1, THREADS](
        h_input,
        h_output,
        tile_successor,
    )

    expected = np.empty_like(h_input)
    expected[:-1] = h_input[:-1] - h_input[1:]
    expected[-1] = h_input[-1] - tile_successor
    np.testing.assert_array_equal(h_output, expected)


def test_block_adjacent_difference_thread_data_temp_storage_getitem_sugar():
    h_input = (np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32) % 7) * 3
    h_output = np.zeros_like(h_input)

    _block_adjacent_difference_thread_data_temp_storage_kernel[1, THREADS](
        h_input, h_output
    )

    expected = np.empty_like(h_input)
    expected[0] = h_input[0]
    expected[1:] = h_input[1:] - h_input[:-1]
    np.testing.assert_array_equal(h_output, expected)
