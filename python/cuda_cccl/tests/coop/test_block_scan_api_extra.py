# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda

from cuda import coop

# example-begin imports
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
# example-end imports


def test_block_inclusive_sum():
    # example-begin inclusive-sum
    items_per_thread = 4
    threads_per_block = 128

    block_inclusive_sum = coop.block.inclusive_sum(
        numba.int32, threads_per_block, items_per_thread
    )

    @cuda.jit
    def kernel(data):
        thread_data = cuda.local.array(items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            thread_data[i] = data[cuda.threadIdx.x * items_per_thread + i]

        block_inclusive_sum(thread_data, thread_data)

        for i in range(items_per_thread):
            data[cuda.threadIdx.x * items_per_thread + i] = thread_data[i]

    # example-end inclusive-sum

    tile_size = threads_per_block * items_per_thread
    h_keys = np.ones(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, threads_per_block](d_keys)
    h_keys = d_keys.copy_to_host()

    for i in range(tile_size):
        assert h_keys[i] == i + 1


def test_block_exclusive_scan():
    # example-begin exclusive-scan
    @cuda.jit(device=True)
    def op(a, b):
        return a + b

    items_per_thread = 4
    threads_per_block = 128

    block_exclusive_scan = coop.block.exclusive_scan(
        numba.int32, threads_per_block, items_per_thread, op
    )

    @cuda.jit
    def kernel(data):
        thread_data = cuda.local.array(items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            thread_data[i] = data[cuda.threadIdx.x * items_per_thread + i]

        block_exclusive_scan(thread_data, thread_data)

        for i in range(items_per_thread):
            data[cuda.threadIdx.x * items_per_thread + i] = thread_data[i]

    # example-end exclusive-scan

    tile_size = threads_per_block * items_per_thread
    h_keys = np.ones(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, threads_per_block](d_keys)
    h_keys = d_keys.copy_to_host()

    for i in range(tile_size):
        assert h_keys[i] == i


def test_block_inclusive_scan():
    # example-begin inclusive-scan
    @cuda.jit(device=True)
    def op(a, b):
        return a + b

    items_per_thread = 4
    threads_per_block = 128

    block_inclusive_scan = coop.block.inclusive_scan(
        numba.int32, threads_per_block, items_per_thread, op
    )

    @cuda.jit
    def kernel(data):
        thread_data = cuda.local.array(items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            thread_data[i] = data[cuda.threadIdx.x * items_per_thread + i]

        block_inclusive_scan(thread_data, thread_data)

        for i in range(items_per_thread):
            data[cuda.threadIdx.x * items_per_thread + i] = thread_data[i]

    # example-end inclusive-scan

    tile_size = threads_per_block * items_per_thread
    h_keys = np.ones(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, threads_per_block](d_keys)
    h_keys = d_keys.copy_to_host()

    for i in range(tile_size):
        assert h_keys[i] == i + 1
