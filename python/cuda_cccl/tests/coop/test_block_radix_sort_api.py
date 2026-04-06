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


def test_block_radix_sort_api_example():
    # example-begin radix-sort
    items_per_thread = 4
    threads_per_block = 128

    @cuda.jit
    def kernel(keys):
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            thread_keys[i] = keys[cuda.threadIdx.x * items_per_thread + i]

        coop.block.radix_sort_keys(thread_keys, items_per_thread)

        for i in range(items_per_thread):
            keys[cuda.threadIdx.x * items_per_thread + i] = thread_keys[i]

    # example-end radix-sort

    tile_size = threads_per_block * items_per_thread
    h_keys = np.arange(tile_size - 1, -1, -1, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, threads_per_block](d_keys)
    np.testing.assert_array_equal(
        d_keys.copy_to_host(), np.arange(tile_size, dtype=np.int32)
    )


def test_block_radix_sort_descending_api_example():
    # example-begin radix-sort-descending
    items_per_thread = 4
    threads_per_block = 128

    @cuda.jit
    def kernel(keys):
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=numba.int32)
        for i in range(items_per_thread):
            thread_keys[i] = keys[cuda.threadIdx.x * items_per_thread + i]

        coop.block.radix_sort_keys_descending(thread_keys, items_per_thread)

        for i in range(items_per_thread):
            keys[cuda.threadIdx.x * items_per_thread + i] = thread_keys[i]

    # example-end radix-sort-descending

    tile_size = threads_per_block * items_per_thread
    h_keys = np.arange(0, tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, threads_per_block](d_keys)
    np.testing.assert_array_equal(
        d_keys.copy_to_host(), np.arange(tile_size - 1, -1, -1, dtype=np.int32)
    )
