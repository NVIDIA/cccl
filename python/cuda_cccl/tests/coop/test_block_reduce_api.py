# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


# example-begin imports
# example-end imports


def test_block_reduction_api_example():
    # example-begin reduce
    @cuda.jit(device=True)
    def op(a, b):
        return a if a > b else b

    threads_per_block = 128

    @cuda.jit
    def kernel(input, output):
        tid = cuda.threadIdx.x
        block_output = coop.block.reduce(
            input[tid],
            items_per_thread=1,
            binary_op=op,
        )
        if tid == 0:
            output[0] = block_output

    # example-end reduce

    h_input = np.random.randint(0, 42, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output)
    np.testing.assert_array_equal(d_output.copy_to_host(), np.array([np.max(h_input)]))


def test_block_sum_api_example():
    # example-begin sum
    threads_per_block = 128

    @cuda.jit
    def kernel(input, output):
        tid = cuda.threadIdx.x
        block_output = coop.block.sum(input[tid], items_per_thread=1)
        if tid == 0:
            output[0] = block_output

    # example-end sum

    h_input = np.ones(threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output)
    np.testing.assert_array_equal(
        d_output.copy_to_host(), np.array([threads_per_block])
    )
