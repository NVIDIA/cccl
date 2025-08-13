# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Block-level scan examples demonstrating cooperative prefix scan algorithms within a CUDA block.
"""

import numba
import numpy as np
from numba import cuda

import cuda.cccl.cooperative.experimental as coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def exclusive_sum_multiple_items_example():
    """Demonstrate block exclusive sum with multiple items per thread."""
    items_per_thread = 4
    threads_per_block = 128

    # Specialize exclusive sum for a 1D block of 128 threads owning 4 integer items each
    block_exclusive_sum = coop.block.exclusive_sum(
        numba.int32, threads_per_block, items_per_thread
    )

    # Link the exclusive sum to a CUDA kernel
    @cuda.jit(link=block_exclusive_sum.files)
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

    tile_size = threads_per_block * items_per_thread

    h_keys = np.ones(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, threads_per_block](d_keys)
    h_keys = d_keys.copy_to_host()

    # Verify exclusive prefix sum (0, 1, 2, 3, ...)
    for i in range(tile_size):
        assert h_keys[i] == i

    print(f"Exclusive sum (multiple items): first 10 elements = {h_keys[:10]}")
    return h_keys


def exclusive_sum_single_item_example():
    """Demonstrate block exclusive sum with single item per thread."""
    items_per_thread = 1
    threads_per_block = 128

    # Specialize exclusive sum for a 1D block of 128 threads.  Each thread
    # owns a single integer item.
    block_exclusive_sum = coop.block.exclusive_sum(
        numba.int32, threads_per_block, items_per_thread
    )

    # Link the exclusive sum to a CUDA kernel
    @cuda.jit(link=block_exclusive_sum.files)
    def kernel(data):
        thread_data = 1

        # Collectively compute the block-wide exclusive prefix sum.
        result = block_exclusive_sum(thread_data)

        # Copy the result back to the output.
        data[cuda.threadIdx.x] = result

    tile_size = threads_per_block

    h_keys = np.ones(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, threads_per_block](d_keys)
    h_keys = d_keys.copy_to_host()

    # Verify exclusive prefix sum (0, 1, 2, 3, ...)
    for i in range(tile_size):
        assert h_keys[i] == i

    print(f"Exclusive sum (single item): first 10 elements = {h_keys[:10]}")
    return h_keys


def variable_input_scan_example():
    """Demonstrate block scan with variable input values."""
    items_per_thread = 2
    threads_per_block = 64

    block_exclusive_sum = coop.block.exclusive_sum(
        numba.int32, threads_per_block, items_per_thread
    )

    @cuda.jit(link=block_exclusive_sum.files)
    def kernel(input_data, output_data):
        # Each thread loads its items
        thread_data = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            thread_data[i] = input_data[cuda.threadIdx.x * items_per_thread + i]

        # Collectively compute the block-wide exclusive prefix sum
        block_exclusive_sum(thread_data, thread_data)

        # Copy the scanned results back to output
        for i in range(items_per_thread):
            output_data[cuda.threadIdx.x * items_per_thread + i] = thread_data[i]

    tile_size = threads_per_block * items_per_thread

    # Create input with pattern: 1, 2, 1, 2, 1, 2, ...
    h_input = np.tile([1, 2], tile_size // 2).astype(np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(tile_size, dtype=np.int32)

    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    # Verify the exclusive prefix sum
    expected = np.zeros(tile_size, dtype=np.int32)
    expected[1:] = np.cumsum(h_input[:-1])

    assert np.array_equal(h_output, expected)
    print(f"Variable input scan: input pattern = {h_input[:10]}")
    print(f"Variable input scan: output pattern = {h_output[:10]}")
    return h_output


if __name__ == "__main__":
    print("Running block scan examples...")
    exclusive_sum_multiple_items_example()
    exclusive_sum_single_item_example()
    variable_input_scan_example()
    print("All block scan examples completed successfully!")
