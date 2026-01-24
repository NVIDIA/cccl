# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin imports
from functools import reduce
from operator import mul

import numba
import numpy as np
from helpers import row_major_tid
from numba import cuda

from cuda import coop
from cuda.coop.block import BlockShuffleType

# example-end imports

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_block_shuffle_offset_scalar():
    dtype = np.int32

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


def test_block_shuffle_offset_scalar_two_phase():
    threads_per_block = 64
    distance = 1
    dtype = np.int32

    block_shuffle = coop.block.shuffle(
        BlockShuffleType.Offset,
        numba.int32,
        threads_per_block,
        distance=distance,
    )

    @cuda.jit
    def kernel(d_in, d_out):
        tid = row_major_tid()
        value = d_in[tid]
        shuffled = block_shuffle(
            value,
            block_shuffle_type=BlockShuffleType.Offset,
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
    # example-begin rotate-scalar
    threads_per_block = 32
    distance = 3
    dtype = np.int32

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

    num_threads = threads_per_block
    h_input = np.arange(num_threads, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    expected = np.roll(h_input, -distance)
    np.testing.assert_array_equal(h_output, expected)


def test_block_shuffle_up_scalar():
    # example-begin up-scalar
    threads_per_block = 64
    distance = 2

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        value = d_in[tid]
        shuffled = coop.block.shuffle(
            value,
            block_shuffle_type=BlockShuffleType.Up,
            distance=distance,
        )
        if tid >= distance:
            d_out[tid] = shuffled
        else:
            d_out[tid] = value

    # example-end up-scalar

    h_input = np.arange(threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
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
        value = d_in[tid]
        shuffled = coop.block.shuffle(
            value,
            block_shuffle_type=BlockShuffleType.Down,
            distance=distance,
        )
        if tid + distance < d_out.size:
            d_out[tid] = shuffled
        else:
            d_out[tid] = value

    # example-end down-scalar

    h_input = np.arange(threads_per_block, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    expected = np.empty_like(h_output)
    expected[-distance:] = h_input[-distance:]
    expected[:-distance] = h_input[distance:]
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
            block_shuffle_type=BlockShuffleType.Up,
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
            block_shuffle_type=BlockShuffleType.Down,
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


def test_block_shuffle_up_down_block_prefix_suffix():
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
    def kernel_up(d_in, d_out, d_suffix):
        tid = row_major_tid()
        items = cuda.local.array(items_per_thread, numba.int32)
        block_suffix = cuda.local.array(1, numba.int32)

        for i in range(items_per_thread):
            items[i] = d_in[tid * items_per_thread + i]

        coop.block.shuffle(
            items,
            items,
            items_per_thread=items_per_thread,
            block_shuffle_type=BlockShuffleType.Up,
            block_suffix=block_suffix,
        )

        for i in range(items_per_thread):
            d_out[tid * items_per_thread + i] = items[i]
        d_suffix[tid] = block_suffix[0]

    @cuda.jit
    def kernel_down(d_in, d_out, d_prefix):
        tid = row_major_tid()
        items = cuda.local.array(items_per_thread, numba.int32)
        block_prefix = cuda.local.array(1, numba.int32)

        for i in range(items_per_thread):
            items[i] = d_in[tid * items_per_thread + i]

        coop.block.shuffle(
            items,
            items,
            items_per_thread=items_per_thread,
            block_shuffle_type=BlockShuffleType.Down,
            block_prefix=block_prefix,
        )

        for i in range(items_per_thread):
            d_out[tid * items_per_thread + i] = items[i]
        d_prefix[tid] = block_prefix[0]

    h_input = np.arange(total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_out_up = cuda.device_array_like(d_input)
    d_out_down = cuda.device_array_like(d_input)
    d_suffix = cuda.device_array(num_threads, dtype=dtype)
    d_prefix = cuda.device_array(num_threads, dtype=dtype)

    kernel_up[1, threads_per_block](d_input, d_out_up, d_suffix)
    kernel_down[1, threads_per_block](d_input, d_out_down, d_prefix)

    h_out_up = d_out_up.copy_to_host()
    h_out_down = d_out_down.copy_to_host()
    h_suffix = d_suffix.copy_to_host()
    h_prefix = d_prefix.copy_to_host()

    expected_up = h_input.copy()
    expected_up[1:] = h_input[:-1]

    expected_down = h_input.copy()
    expected_down[:-1] = h_input[1:]

    np.testing.assert_array_equal(h_out_up, expected_up)
    np.testing.assert_array_equal(h_out_down, expected_down)
    np.testing.assert_array_equal(h_suffix, np.full(num_threads, h_input[-1]))
    np.testing.assert_array_equal(h_prefix, np.full(num_threads, h_input[0]))


def test_block_shuffle_up_down_block_prefix_suffix_two_phase():
    threads_per_block = 128
    items_per_thread = 4
    dtype = np.int32

    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )
    total_items = num_threads * items_per_thread

    block_shuffle_up = coop.block.shuffle(
        BlockShuffleType.Up,
        numba.int32,
        threads_per_block,
        items_per_thread=items_per_thread,
    )
    block_shuffle_down = coop.block.shuffle(
        BlockShuffleType.Down,
        numba.int32,
        threads_per_block,
        items_per_thread=items_per_thread,
    )

    @cuda.jit
    def kernel(d_in, d_out_up, d_out_down, d_suffix, d_prefix):
        tid = row_major_tid()
        items_up = cuda.local.array(items_per_thread, numba.int32)
        items_down = cuda.local.array(items_per_thread, numba.int32)
        block_suffix = cuda.local.array(1, numba.int32)
        block_prefix = cuda.local.array(1, numba.int32)

        for i in range(items_per_thread):
            value = d_in[tid * items_per_thread + i]
            items_up[i] = value
            items_down[i] = value

        block_shuffle_up(
            items_up,
            items_up,
            items_per_thread=items_per_thread,
            block_shuffle_type=BlockShuffleType.Up,
            block_suffix=block_suffix,
        )
        block_shuffle_down(
            items_down,
            items_down,
            items_per_thread=items_per_thread,
            block_shuffle_type=BlockShuffleType.Down,
            block_prefix=block_prefix,
        )

        for i in range(items_per_thread):
            d_out_up[tid * items_per_thread + i] = items_up[i]
            d_out_down[tid * items_per_thread + i] = items_down[i]
        d_suffix[tid] = block_suffix[0]
        d_prefix[tid] = block_prefix[0]

    h_input = np.arange(total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_out_up = cuda.device_array_like(d_input)
    d_out_down = cuda.device_array_like(d_input)
    d_suffix = cuda.device_array(num_threads, dtype=dtype)
    d_prefix = cuda.device_array(num_threads, dtype=dtype)

    kernel[1, threads_per_block](d_input, d_out_up, d_out_down, d_suffix, d_prefix)

    h_out_up = d_out_up.copy_to_host()
    h_out_down = d_out_down.copy_to_host()
    h_suffix = d_suffix.copy_to_host()
    h_prefix = d_prefix.copy_to_host()

    expected_up = h_input.copy()
    expected_up[1:] = h_input[:-1]

    expected_down = h_input.copy()
    expected_down[:-1] = h_input[1:]

    np.testing.assert_array_equal(h_out_up, expected_up)
    np.testing.assert_array_equal(h_out_down, expected_down)
    np.testing.assert_array_equal(h_suffix, np.full(num_threads, h_input[-1]))
    np.testing.assert_array_equal(h_prefix, np.full(num_threads, h_input[0]))
