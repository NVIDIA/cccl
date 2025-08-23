# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Block-level reduction examples demonstrating cooperative algorithms within a CUDA block.
"""

import numba
import numpy as np
from numba import cuda

import cuda.cccl.cooperative.experimental as coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def custom_reduce_example():
    """Demonstrate block reduction with custom operator (maximum)."""

    def max_op(a, b):
        return a if a > b else b

    threads_per_block = 128
    block_reduce = coop.block.reduce(numba.int32, threads_per_block, max_op)

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        # Each thread contributes one element
        block_output = block_reduce(input[cuda.threadIdx.x])

        # Only thread 0 writes the result
        if cuda.threadIdx.x == 0:
            output[0] = block_output

    # Create test data
    h_input = np.random.randint(0, 100, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)

    # Launch kernel
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()
    h_expected = np.max(h_input)

    assert h_output[0] == h_expected
    print(f"Block max reduction: {h_output[0]} (expected: {h_expected})")
    return h_output[0]


def sum_reduce_example():
    """Demonstrate block sum reduction using built-in sum operation."""
    threads_per_block = 128
    block_sum = coop.block.sum(numba.int32, threads_per_block)

    @cuda.jit(link=block_sum.files)
    def kernel(input, output):
        # Each thread contributes one element
        block_output = block_sum(input[cuda.threadIdx.x])

        # Only thread 0 writes the result
        if cuda.threadIdx.x == 0:
            output[0] = block_output

    # Create test data (all ones for easy verification)
    h_input = np.ones(threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)

    # Launch kernel
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    assert h_output[0] == threads_per_block
    print(f"Block sum reduction: {h_output[0]} (expected: {threads_per_block})")
    return h_output[0]


def min_reduce_example():
    """Demonstrate block reduction with minimum operator."""

    def min_op(a, b):
        return a if a < b else b

    threads_per_block = 64
    block_reduce = coop.block.reduce(numba.int32, threads_per_block, min_op)

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        # Each thread contributes one element
        block_output = block_reduce(input[cuda.threadIdx.x])

        # Only thread 0 writes the result
        if cuda.threadIdx.x == 0:
            output[0] = block_output

    # Create test data
    h_input = np.random.randint(10, 50, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)

    # Launch kernel
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()
    h_expected = np.min(h_input)

    assert h_output[0] == h_expected
    print(f"Block min reduction: {h_output[0]} (expected: {h_expected})")
    return h_output[0]


def multi_block_example():
    """Demonstrate block reduction across multiple blocks."""

    def add_op(a, b):
        return a + b

    threads_per_block = 128
    num_blocks = 4
    block_reduce = coop.block.reduce(numba.int32, threads_per_block, add_op)

    @cuda.jit(link=block_reduce.files)
    def kernel(input, output):
        # Each thread contributes one element
        block_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        global_idx = block_idx * threads_per_block + thread_idx

        block_output = block_reduce(input[global_idx])

        # Only thread 0 of each block writes the result
        if thread_idx == 0:
            output[block_idx] = block_output

    # Create test data
    total_elements = threads_per_block * num_blocks
    h_input = np.arange(1, total_elements + 1, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(num_blocks, dtype=np.int32)

    # Launch kernel
    kernel[num_blocks, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    # Verify each block's sum
    for i in range(num_blocks):
        block_start = i * threads_per_block
        block_end = (i + 1) * threads_per_block
        expected_sum = np.sum(h_input[block_start:block_end])
        assert h_output[i] == expected_sum
        print(f"Block {i} sum: {h_output[i]} (expected: {expected_sum})")

    return h_output


if __name__ == "__main__":
    print("Running block reduce examples...")
    custom_reduce_example()
    sum_reduce_example()
    min_reduce_example()
    multi_block_example()
    print("All block reduce examples completed successfully!")
