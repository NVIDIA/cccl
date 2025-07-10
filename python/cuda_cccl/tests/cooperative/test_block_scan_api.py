# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda
from pynvjitlink import patch

import cuda.cccl.cooperative.experimental as coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# example-begin imports
patch.patch_numba_linker(lto=True)
# example-end imports


def test_block_exclusive_sum():
    # example-begin exclusive-sum
    items_per_thread = 4
    threads_per_block = 128

    # Specialize exclusive sum for a 1D block of 128 threads owning 4 integer items each
    block_exclusive_sum = coop.block.exclusive_sum(
        numba.int32, threads_per_block, items_per_thread
    )

    # Link the exclusive sum to a CUDA kernel
    @cuda.jit(link=block_exclusive_sum.files)
    def kernel(data):
        # Obtain a segment of consecutive items that are blocked across threads
        thread_data = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            thread_data[i] = data[cuda.threadIdx.x * items_per_thread + i]

        # Collectively compute the block-wide exclusive prefix sum
        block_exclusive_sum(thread_data, thread_data)

        # Copy the scanned keys back to the output
        for i in range(items_per_thread):
            data[cuda.threadIdx.x * items_per_thread + i] = thread_data[i]

    # example-end exclusive-sum

    tile_size = threads_per_block * items_per_thread

    h_keys = np.ones(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, threads_per_block](d_keys)
    h_keys = d_keys.copy_to_host()
    for i in range(tile_size):
        assert h_keys[i] == i


def test_block_exclusive_sum_single_input_per_thread():
    # example-begin exclusive-sum-single-input-per-thread
    items_per_thread = 1
    threads_per_block = 128

    # Specialize exclusive sum for a 1D block of 128 threads.  Each thread
    # owns a single integer item.
    block_exclusive_sum = coop.block.exclusive_sum(
        numba.int32, threads_per_block, items_per_thread
    )

    # Link the exclusive sum to a CUDA kernel
    @cuda.jit(link=block_exclusive_sum.files)
    def kernel(data):
        thread_data = 1

        # Collectively compute the block-wide exclusive prefix sum.
        result = block_exclusive_sum(thread_data)

        # Copy the result back to the output.
        data[cuda.threadIdx.x] = result

    # example-end exclusive-sum-single-input-per-thread

    tile_size = threads_per_block

    h_keys = np.ones(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, threads_per_block](d_keys)
    h_keys = d_keys.copy_to_host()
    for i in range(tile_size):
        assert h_keys[i] == i
