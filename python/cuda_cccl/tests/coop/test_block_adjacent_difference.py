# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import reduce
from operator import mul

import numba
import numpy as np
from helpers import row_major_tid
from numba import cuda, types

from cuda import coop
from cuda.coop.block import BlockAdjacentDifferenceType

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@cuda.jit(device=True)
def diff_op(lhs, rhs):
    return lhs - rhs


def test_block_adjacent_difference_subtract_left_thread_data_temp_storage():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4

    block_adj = coop.block.adjacent_difference(
        BlockAdjacentDifferenceType.SubtractLeft,
        dtype,
        threads_per_block,
        items_per_thread,
        diff_op,
    )
    temp_storage_bytes = block_adj.temp_storage_bytes
    temp_storage_alignment = block_adj.temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        items = coop.ThreadData(items_per_thread, dtype=d_in.dtype)
        output = coop.ThreadData(items_per_thread, dtype=d_in.dtype)

        coop.block.load(d_in, items)
        coop.block.adjacent_difference(
            items,
            output,
            difference_op=diff_op,
            block_adjacent_difference_type=BlockAdjacentDifferenceType.SubtractLeft,
            temp_storage=temp_storage,
        )
        coop.block.store(d_out, output)

    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )
    num_items = num_threads * items_per_thread

    h_input = np.random.randint(0, 32, num_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(num_items, dtype=dtype)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_reference = np.zeros_like(h_output)
    for idx in range(num_items):
        if idx == 0:
            h_reference[idx] = h_input[idx]
        else:
            h_reference[idx] = h_input[idx] - h_input[idx - 1]

    np.testing.assert_array_equal(h_output, h_reference)


@cuda.jit(device=True)
def diff_op_right(lhs, rhs):
    return lhs - rhs


def test_block_adjacent_difference_subtract_right():
    threads_per_block = 64
    items_per_thread = 2
    dtype = types.int32
    dtype_np = np.int32

    @cuda.jit
    def kernel(d_in, d_out):
        tid = row_major_tid()
        items = cuda.local.array(items_per_thread, dtype=dtype)
        output = cuda.local.array(items_per_thread, dtype=dtype)

        base = tid * items_per_thread
        for i in range(items_per_thread):
            items[i] = d_in[base + i]

        coop.block.adjacent_difference(
            items,
            output,
            items_per_thread=items_per_thread,
            difference_op=diff_op_right,
            block_adjacent_difference_type=BlockAdjacentDifferenceType.SubtractRight,
        )

        for i in range(items_per_thread):
            d_out[base + i] = output[i]

    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )
    num_items = num_threads * items_per_thread

    h_input = np.random.randint(0, 32, num_items, dtype=dtype_np)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(num_items, dtype=dtype_np)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_reference = np.zeros_like(h_output)
    for idx in range(num_items):
        if idx == num_items - 1:
            h_reference[idx] = h_input[idx]
        else:
            h_reference[idx] = h_input[idx] - h_input[idx + 1]

    np.testing.assert_array_equal(h_output, h_reference)


def test_block_adjacent_difference_two_phase_subtract_left():
    dtype = np.int32
    threads_per_block = 64
    items_per_thread = 2

    block_adj = coop.block.adjacent_difference(
        BlockAdjacentDifferenceType.SubtractLeft,
        dtype,
        threads_per_block,
        items_per_thread,
        difference_op=diff_op,
    )
    temp_storage_bytes = block_adj.temp_storage_bytes
    temp_storage_alignment = block_adj.temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        items = coop.ThreadData(items_per_thread, dtype=d_in.dtype)
        output = coop.ThreadData(items_per_thread, dtype=d_in.dtype)

        coop.block.load(d_in, items)
        block_adj(
            items,
            output,
            difference_op=diff_op,
            block_adjacent_difference_type=BlockAdjacentDifferenceType.SubtractLeft,
            temp_storage=temp_storage,
        )
        coop.block.store(d_out, output)

    num_threads = threads_per_block
    num_items = num_threads * items_per_thread

    h_input = np.random.randint(0, 32, num_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(num_items, dtype=dtype)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_reference = np.zeros_like(h_output)
    for idx in range(num_items):
        if idx == 0:
            h_reference[idx] = h_input[idx]
        else:
            h_reference[idx] = h_input[idx] - h_input[idx - 1]

    np.testing.assert_array_equal(h_output, h_reference)
