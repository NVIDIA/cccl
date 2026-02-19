# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin imports
import numba
import numpy as np
from numba import cuda

from cuda import coop
from cuda.coop.block import BlockDiscontinuityType

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
            block_discontinuity_type=BlockDiscontinuityType.HEADS,
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
            block_discontinuity_type=BlockDiscontinuityType.TAILS,
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
    threads_per_block = 128

    @cuda.jit
    def kernel(d_in, d_head_out, d_tail_out):
        tid = cuda.threadIdx.x
        items = cuda.local.array(1, dtype=numba.int32)
        head_flags = cuda.local.array(1, dtype=numba.int32)
        tail_flags = cuda.local.array(1, dtype=numba.int32)

        items[0] = d_in[tid]

        coop.block.discontinuity(
            items,
            head_flags,
            tail_flags,
            items_per_thread=1,
            flag_op=flag_op,
            block_discontinuity_type=BlockDiscontinuityType.HEADS_AND_TAILS,
        )

        d_head_out[tid] = head_flags[0]
        d_tail_out[tid] = tail_flags[0]

    h_input = np.random.randint(0, 4, threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_head_out = cuda.device_array_like(d_input)
    d_tail_out = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_head_out, d_tail_out)
    cuda.synchronize()

    h_head = d_head_out.copy_to_host()
    h_tail = d_tail_out.copy_to_host()
    expected_heads = np.zeros_like(h_head)
    expected_tails = np.zeros_like(h_tail)
    prev = h_input[0]
    for idx in range(threads_per_block):
        if idx == 0:
            expected_heads[idx] = 1
        else:
            expected_heads[idx] = 1 if h_input[idx] != prev else 0
            prev = h_input[idx]

    for idx in range(threads_per_block):
        nxt = h_input[idx + 1] if idx < threads_per_block - 1 else h_input[idx]
        expected_tails[idx] = 1 if h_input[idx] != nxt else 0
    expected_tails[-1] = 1

    np.testing.assert_array_equal(h_head, expected_heads)
    np.testing.assert_array_equal(h_tail, expected_tails)
