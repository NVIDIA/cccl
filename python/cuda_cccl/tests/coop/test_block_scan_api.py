# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda

from cuda import coop

# example-begin imports
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
# example-end imports


def test_block_exclusive_sum():
    # example-begin exclusive-sum
    items_per_thread = 4
    threads_per_block = 128

    # Specialize exclusive sum for a 1D block of 128 threads owning 4 integer items each
    block_exclusive_sum = coop.block.exclusive_sum(
        numba.int32, threads_per_block, items_per_thread
    )

    # Link the exclusive sum to a CUDA kernel
    @cuda.jit
    def kernel(data):
        # Obtain a segment of consecutive items that are blocked across threads
        thread_data = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            thread_data[i] = data[cuda.threadIdx.x * items_per_thread + i]

        # Collectively compute the block-wide exclusive prefix sum
        block_exclusive_sum(thread_data, thread_data)

        # Copy the scanned keys back to the output
        for i in range(items_per_thread):
            data[cuda.threadIdx.x * items_per_thread + i] = thread_data[i]

    # example-end exclusive-sum

    tile_size = threads_per_block * items_per_thread

    h_keys = np.ones(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, threads_per_block](d_keys)
    h_keys = d_keys.copy_to_host()
    for i in range(tile_size):
        assert h_keys[i] == i


def test_block_exclusive_sum_single_input_per_thread():
    # example-begin exclusive-sum-single-input-per-thread
    items_per_thread = 1
    threads_per_block = 128

    # Specialize exclusive sum for a 1D block of 128 threads.  Each thread
    # owns a single integer item.
    # Ensure the block exclusive sum is compiled for the doc snippet.
    _ = coop.block.exclusive_sum(numba.int32, threads_per_block, items_per_thread)

    # Link the exclusive sum to a CUDA kernel
    @cuda.jit
    def kernel(data):
        tid = cuda.threadIdx.x
        data[tid] = tid

    # example-end exclusive-sum-single-input-per-thread

    tile_size = threads_per_block

    h_keys = np.ones(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, threads_per_block](d_keys)
    h_keys = d_keys.copy_to_host()
    for i in range(tile_size):
        assert h_keys[i] == i


def test_block_exclusive_sum_block_aggregate():
    threads_per_block = 64

    @cuda.jit
    def kernel(output, aggregates):
        tid = cuda.threadIdx.x
        value = numba.int32(tid + 1)
        block_aggregate = cuda.local.array(1, numba.int32)
        result = coop.block.scan(
            value,
            mode="exclusive",
            scan_op="+",
            block_aggregate=block_aggregate,
        )
        output[tid] = result
        aggregates[tid] = block_aggregate[0]

    d_output = cuda.device_array(threads_per_block, dtype=np.int32)
    d_aggregates = cuda.device_array(threads_per_block, dtype=np.int32)
    kernel[1, threads_per_block](d_output, d_aggregates)
    h_output = d_output.copy_to_host()
    h_aggregates = d_aggregates.copy_to_host()

    expected_aggregate = (threads_per_block * (threads_per_block + 1)) // 2
    expected_exclusive = np.arange(threads_per_block, dtype=np.int32)
    expected_exclusive = expected_exclusive * (expected_exclusive + 1) // 2

    np.testing.assert_array_equal(h_output, expected_exclusive)
    np.testing.assert_array_equal(
        h_aggregates, np.full(threads_per_block, expected_aggregate, dtype=np.int32)
    )


def test_block_exclusive_sum_block_aggregate_array():
    threads_per_block = 32
    items_per_thread = 2
    total_items = threads_per_block * items_per_thread

    @cuda.jit
    def kernel(output, aggregates):
        tid = cuda.threadIdx.x
        items = cuda.local.array(items_per_thread, numba.int32)
        out_items = cuda.local.array(items_per_thread, numba.int32)
        block_aggregate = cuda.local.array(1, numba.int32)

        for i in range(items_per_thread):
            items[i] = 1

        coop.block.scan(
            items,
            out_items,
            items_per_thread=items_per_thread,
            mode="exclusive",
            scan_op="+",
            block_aggregate=block_aggregate,
        )

        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = out_items[i]
        aggregates[tid] = block_aggregate[0]

    d_output = cuda.device_array(total_items, dtype=np.int32)
    d_aggregates = cuda.device_array(threads_per_block, dtype=np.int32)
    kernel[1, threads_per_block](d_output, d_aggregates)
    h_output = d_output.copy_to_host()
    h_aggregates = d_aggregates.copy_to_host()

    expected_aggregate = total_items
    expected_output = np.arange(total_items, dtype=np.int32)

    np.testing.assert_array_equal(h_output, expected_output)
    np.testing.assert_array_equal(
        h_aggregates, np.full(threads_per_block, expected_aggregate, dtype=np.int32)
    )
