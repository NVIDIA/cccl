# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin imports
import numba
import numpy as np
from numba import cuda

from cuda import coop
from cuda.coop.block import BlockShuffleType

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# example-end imports


def test_block_shuffle_offset_scalar():
    # example-begin offset-scalar
    threads_per_block = 64
    distance = 1

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        value = d_in[tid]
        shuffled = coop.block.shuffle(
            value,
            block_shuffle_type=BlockShuffleType.Offset,
            distance=distance,
        )
        d_out[tid] = -1
        if tid + distance < d_out.size:
            d_out[tid] = shuffled

    # example-end offset-scalar

    h_input = np.arange(threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = np.full_like(h_output, -1)
    expected[: threads_per_block // 2] = h_input[0 : threads_per_block // 2] * 2
    expected[threads_per_block // 2 : -1] = 0
    np.testing.assert_array_equal(h_output, expected)


def test_block_shuffle_rotate_scalar():
    # example-begin rotate-scalar
    threads_per_block = 64
    distance = 3

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        value = d_in[tid]
        shuffled = coop.block.shuffle(
            value,
            block_shuffle_type=BlockShuffleType.Rotate,
            distance=distance,
        )
        d_out[tid] = shuffled

    # example-end rotate-scalar

    h_input = np.arange(threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = np.empty_like(h_output)
    expected[: threads_per_block // 2] = h_input[0 : threads_per_block // 2] * 2
    expected[threads_per_block // 2 :] = h_input[0 : threads_per_block // 2] * 2
    np.testing.assert_array_equal(h_output, expected)


def test_block_shuffle_up_scalar():
    # example-begin up-scalar
    threads_per_block = 64
    distance = 2

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        if tid >= distance:
            d_out[tid] = d_in[tid - distance]
        else:
            d_out[tid] = d_in[tid]

    # example-end up-scalar

    h_input = np.arange(threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = np.empty_like(h_output)
    expected[:distance] = h_input[:distance]
    expected[distance:] = h_input[:-distance]
    np.testing.assert_array_equal(h_output, expected)


def test_block_shuffle_down_scalar():
    # example-begin down-scalar
    threads_per_block = 64
    distance = 2

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        if tid + distance < d_out.size:
            d_out[tid] = d_in[tid + distance]
        else:
            d_out[tid] = d_in[tid]

    # example-end down-scalar

    h_input = np.arange(threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = np.empty_like(h_output)
    expected[-distance:] = h_input[-distance:]
    expected[:-distance] = h_input[distance:]
    np.testing.assert_array_equal(h_output, expected)
