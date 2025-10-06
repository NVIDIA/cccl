# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Warp-level reduction examples demonstrating cooperative algorithms within a CUDA warp.
"""

import numba
import numpy as np
from numba import cuda

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def custom_warp_reduce_example():
    """Demonstrate warp reduction with custom operator (maximum)."""

    def max_op(a, b):
        return a if a > b else b

    warp_reduce = coop.warp.reduce(numba.int32, max_op)

    @cuda.jit(link=warp_reduce.files)
    def kernel(input, output):
        # Each thread in the warp contributes one element
        warp_output = warp_reduce(input[cuda.threadIdx.x])

        # Only thread 0 writes the result
        if cuda.threadIdx.x == 0:
            output[0] = warp_output

    # Create test data (32 elements for a full warp)
    h_input = np.random.randint(0, 100, 32, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)

    # Launch kernel with 32 threads (one warp)
    kernel[1, 32](d_input, d_output)
    h_output = d_output.copy_to_host()
    h_expected = np.max(h_input)

    assert h_output[0] == h_expected
    print(f"Warp max reduction: {h_output[0]} (expected: {h_expected})")
    return h_output[0]


def warp_sum_example():
    """Demonstrate warp sum reduction using built-in sum operation."""
    warp_sum = coop.warp.sum(numba.int32)

    @cuda.jit(link=warp_sum.files)
    def kernel(input, output):
        # Each thread in the warp contributes one element
        warp_output = warp_sum(input[cuda.threadIdx.x])

        # Only thread 0 writes the result
        if cuda.threadIdx.x == 0:
            output[0] = warp_output

    # Create test data (all ones for easy verification)
    h_input = np.ones(32, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)

    # Launch kernel with 32 threads (one warp)
    kernel[1, 32](d_input, d_output)
    h_output = d_output.copy_to_host()

    assert h_output[0] == 32
    print(f"Warp sum reduction: {h_output[0]} (expected: 32)")
    return h_output[0]


def multi_warp_example():
    """Demonstrate warp reduction across multiple warps in a block."""

    def add_op(a, b):
        return a + b

    warp_reduce = coop.warp.reduce(numba.int32, add_op)

    @cuda.jit(link=warp_reduce.files)
    def kernel(input, output):
        # Each thread contributes one element
        thread_idx = cuda.threadIdx.x
        warp_id = thread_idx // 32
        lane_id = thread_idx % 32

        warp_output = warp_reduce(input[thread_idx])

        # Only the first thread of each warp writes the result
        if lane_id == 0:
            output[warp_id] = warp_output

    # Create test data for 4 warps (128 threads total)
    threads_per_block = 128
    num_warps = threads_per_block // 32
    h_input = np.arange(1, threads_per_block + 1, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(num_warps, dtype=np.int32)

    # Launch kernel with 128 threads (4 warps)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    # Verify each warp's sum
    for warp_id in range(num_warps):
        warp_start = warp_id * 32
        warp_end = (warp_id + 1) * 32
        expected_sum = np.sum(h_input[warp_start:warp_end])
        assert h_output[warp_id] == expected_sum
        print(f"Warp {warp_id} sum: {h_output[warp_id]} (expected: {expected_sum})")

    return h_output


def partial_warp_example():
    """Demonstrate warp reduction with partial warp (fewer than 32 threads)."""

    def add_op(a, b):
        return a + b

    warp_reduce = coop.warp.reduce(numba.int32, add_op)

    @cuda.jit(link=warp_reduce.files)
    def kernel(input, output, num_threads):
        thread_idx = cuda.threadIdx.x

        # Only active threads participate
        if thread_idx < num_threads:
            value = input[thread_idx]
        else:
            value = 0  # Inactive threads contribute 0

        warp_output = warp_reduce(value)

        # Only thread 0 writes the result
        if thread_idx == 0:
            output[0] = warp_output

    # Create test data for partial warp (20 threads)
    num_active_threads = 20
    h_input = np.arange(1, num_active_threads + 1, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)

    # Launch kernel with 32 threads but only 20 are active
    kernel[1, 32](d_input, d_output, num_active_threads)
    h_output = d_output.copy_to_host()

    expected_sum = np.sum(h_input)
    assert h_output[0] == expected_sum
    print(f"Partial warp sum: {h_output[0]} (expected: {expected_sum})")
    return h_output[0]


if __name__ == "__main__":
    print("Running warp reduce examples...")
    custom_warp_reduce_example()
    warp_sum_example()
    multi_warp_example()
    partial_warp_example()
    print("All warp reduce examples completed successfully!")
