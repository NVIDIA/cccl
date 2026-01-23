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
            block_shuffle_type=coop.block.BlockShuffleType.Offset,
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
    expected[:-distance] = h_input[distance:]
    np.testing.assert_array_equal(h_output, expected)
