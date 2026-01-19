# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import reduce
from operator import mul

import numba
import numpy as np
from helpers import row_major_tid
from numba import cuda

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_block_shuffle_offset_scalar():
    threads_per_block = 64
    distance = 1
    dtype = np.int32

    @cuda.jit
    def kernel(d_in, d_out):
        tid = row_major_tid()
        value = d_in[tid]
        shuffled = coop.block.shuffle(
            value,
            block_shuffle_type=coop.block.BlockShuffleType.Offset,
            distance=distance,
        )
        if tid + distance < d_out.shape[0]:
            d_out[tid] = shuffled

    num_threads = threads_per_block
    h_input = np.arange(num_threads, dtype=dtype)
    h_output = np.full(num_threads, -1, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(h_output)

    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    expected = np.full_like(h_output, -1)
    expected[:-distance] = h_input[distance:]
    np.testing.assert_array_equal(h_output, expected)


def test_block_shuffle_rotate_scalar():
    threads_per_block = 32
    distance = 3
    dtype = np.int32

    @cuda.jit
    def kernel(d_in, d_out):
        tid = row_major_tid()
        value = d_in[tid]
        shuffled = coop.block.shuffle(
            value,
            block_shuffle_type=coop.block.BlockShuffleType.Rotate,
            distance=distance,
        )
        d_out[tid] = shuffled

    num_threads = threads_per_block
    h_input = np.arange(num_threads, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    expected = np.roll(h_input, -distance)
    np.testing.assert_array_equal(h_output, expected)


def test_block_shuffle_up_down_arrays():
    threads_per_block = 128
    items_per_thread = 4
    dtype = np.int32

    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )
    total_items = num_threads * items_per_thread

    @cuda.jit
    def kernel_up(d_in, d_out):
        tid = row_major_tid()
        items = cuda.local.array(items_per_thread, numba.int32)

        for i in range(items_per_thread):
            items[i] = d_in[tid * items_per_thread + i]

        coop.block.shuffle(
            items,
            items,
            items_per_thread=items_per_thread,
            block_shuffle_type=coop.block.BlockShuffleType.Up,
        )

        for i in range(items_per_thread):
            d_out[tid * items_per_thread + i] = items[i]

    @cuda.jit
    def kernel_down(d_in, d_out):
        tid = row_major_tid()
        items = cuda.local.array(items_per_thread, numba.int32)

        for i in range(items_per_thread):
            items[i] = d_in[tid * items_per_thread + i]

        coop.block.shuffle(
            items,
            items,
            items_per_thread=items_per_thread,
            block_shuffle_type=coop.block.BlockShuffleType.Down,
        )

        for i in range(items_per_thread):
            d_out[tid * items_per_thread + i] = items[i]

    h_input = np.arange(total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_out_up = cuda.device_array_like(d_input)
    d_out_down = cuda.device_array_like(d_input)

    kernel_up[1, threads_per_block](d_input, d_out_up)
    kernel_down[1, threads_per_block](d_input, d_out_down)
    h_out_up = d_out_up.copy_to_host()
    h_out_down = d_out_down.copy_to_host()

    expected_up = h_input.copy()
    expected_up[1:] = h_input[:-1]

    expected_down = h_input.copy()
    expected_down[:-1] = h_input[1:]

    np.testing.assert_array_equal(h_out_up, expected_up)
    np.testing.assert_array_equal(h_out_down, expected_down)
