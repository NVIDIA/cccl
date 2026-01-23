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
def flag_op(lhs, rhs):
    return numba.int32(1 if lhs != rhs else 0)


def test_block_discontinuity_flag_heads():
    # example-begin flag-heads
    threads_per_block = 128

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        items = cuda.local.array(1, dtype=numba.int32)
        flags = cuda.local.array(1, dtype=numba.int32)

        items[0] = d_in[tid]

        coop.block.discontinuity(
            items,
            flags,
            items_per_thread=1,
            flag_op=flag_op,
            block_discontinuity_type=coop.block.BlockDiscontinuityType.HEADS,
        )

        d_out[tid] = flags[0]

    # example-end flag-heads

    h_input = np.random.randint(0, 4, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()

    expected = np.zeros_like(h_output)
    expected[0] = 1
    expected[1:] = (h_input[1:] != h_input[:-1]).astype(np.int32)

    np.testing.assert_array_equal(h_output, expected)


def test_block_discontinuity_flag_tails():
    # example-begin flag-tails
    threads_per_block = 128

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        items = cuda.local.array(1, dtype=numba.int32)
        flags = cuda.local.array(1, dtype=numba.int32)

        items[0] = d_in[tid]

        coop.block.discontinuity(
            items,
            flags,
            items_per_thread=1,
            flag_op=flag_op,
            block_discontinuity_type=coop.block.BlockDiscontinuityType.TAILS,
        )

        d_out[tid] = flags[0]

    # example-end flag-tails

    h_input = np.random.randint(0, 4, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = np.zeros_like(h_output)
    expected[-1] = 1
    expected[:-1] = (h_input[1:] != h_input[:-1]).astype(np.int32)

    np.testing.assert_array_equal(h_output, expected)


def test_block_discontinuity_flag_heads_and_tails():
    # example-begin flag-heads-and-tails
    threads_per_block = 128

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        items = cuda.local.array(1, dtype=numba.int32)
        flags = cuda.local.array(1, dtype=numba.int32)

        items[0] = d_in[tid]

        coop.block.discontinuity(
            items,
            flags,
            items_per_thread=1,
            flag_op=flag_op,
            block_discontinuity_type=coop.block.BlockDiscontinuityType.HEADS_AND_TAILS,
        )

        d_out[tid] = flags[0]

    # example-end flag-heads-and-tails

    h_input = np.random.randint(0, 4, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = (h_input[1:] != h_input[:-1]).astype(np.int32)
    expected = np.concatenate(([1], expected))
    expected[-1] = 1

    np.testing.assert_array_equal(h_output, expected)
