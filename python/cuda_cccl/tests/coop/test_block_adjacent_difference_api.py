# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin imports
import numba
import numpy as np
from numba import cuda

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# example-end imports


@cuda.jit(device=True)
def diff_op(lhs, rhs):
    return lhs - rhs


def test_block_adjacent_difference_subtract_left():
    # example-begin subtract-left
    threads_per_block = 128

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        items = cuda.local.array(1, dtype=numba.int32)
        output = cuda.local.array(1, dtype=numba.int32)

        items[0] = d_in[tid]
        coop.block.adjacent_difference(
            items,
            output,
            items_per_thread=1,
            difference_op=diff_op,
            block_adjacent_difference_type=coop.block.BlockAdjacentDifferenceType.SubtractLeft,
        )
        d_out[tid] = output[0]

    # example-end subtract-left

    h_input = np.random.randint(0, 32, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = np.zeros_like(h_output)
    expected[0] = h_input[0]
    expected[1:] = h_input[1:] - h_input[:-1]
    np.testing.assert_array_equal(h_output, expected)


def test_block_adjacent_difference_subtract_right():
    # example-begin subtract-right
    threads_per_block = 128

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        items = cuda.local.array(1, dtype=numba.int32)
        output = cuda.local.array(1, dtype=numba.int32)

        items[0] = d_in[tid]
        coop.block.adjacent_difference(
            items,
            output,
            items_per_thread=1,
            difference_op=diff_op,
            block_adjacent_difference_type=coop.block.BlockAdjacentDifferenceType.SubtractRight,
        )
        d_out[tid] = output[0]

    # example-end subtract-right

    h_input = np.random.randint(0, 32, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = np.zeros_like(h_output)
    expected[-1] = h_input[-1]
    expected[:-1] = h_input[:-1] - h_input[1:]
    np.testing.assert_array_equal(h_output, expected)
